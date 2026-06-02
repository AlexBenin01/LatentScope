import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_recursive_loop(question: str, mas: dict, n_rounds: int = 3) -> dict:
    """
    Runs the Sequential-Light recursive loop.

    Each round: Planner -> outer_12 -> Critic -> outer_23 -> Solver -> outer_31 -> Planner.
    Rounds 1..n-1 pass latent tensors without decoding to text.
    Round n generates the final answer from the Solver.

    Returns:
        answer:       decoded text from Solver at the last round
        hidden_states: list of tensors [hidden_dim] — Solver last-token hidden state per round
        logits:        list of tensors [1, seq, vocab] — Solver logits per round
    """
    planner     = mas["planner"]
    critic      = mas["critic"]
    solver      = mas["solver"]
    pl_tok      = mas["planner_tokenizer"]
    so_tok      = mas["solver_tokenizer"]
    outer_12    = mas["outer_12"]
    outer_23    = mas["outer_23"]
    outer_31    = mas["outer_31"]

    hidden_states = []
    logits_list   = []
    answer        = ""

    # Round 1 starts from tokenized text; rounds 2+ from projected latent embeds
    enc = pl_tok(question, return_tensors="pt").to(DEVICE)
    planner_kwargs = {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}

    for round_idx in range(n_rounds):
        is_last = (round_idx == n_rounds - 1)

        # --- Planner ---
        with torch.no_grad():
            p_out = planner(**planner_kwargs, output_hidden_states=True)
        p_hs = p_out.hidden_states[-1]          # [1, seq, 2048]

        # --- Planner -> Critic ---
        c_embeds  = outer_12(p_hs)              # [1, seq, 2048]
        attn_mask = torch.ones(c_embeds.shape[:2], dtype=torch.long, device=DEVICE)

        # --- Critic ---
        with torch.no_grad():
            c_out = critic(inputs_embeds=c_embeds, attention_mask=attn_mask, output_hidden_states=True)
        c_hs = c_out.hidden_states[-1]          # [1, seq, 2048]

        # --- Critic -> Solver ---
        s_embeds  = outer_23(c_hs)              # [1, seq, 1536]
        attn_mask = torch.ones(s_embeds.shape[:2], dtype=torch.long, device=DEVICE)

        # --- Solver (forward pass — always, for hidden states and logits) ---
        with torch.no_grad():
            s_out = solver(inputs_embeds=s_embeds, attention_mask=attn_mask, output_hidden_states=True)

        hidden_states.append(s_out.hidden_states[-1][:, -1, :].squeeze(0).detach().cpu())
        logits_list.append(s_out.logits.detach().cpu())

        if is_last:
            # Generate answer autoregressively from the Solver's latent input
            with torch.no_grad():
                gen_ids = solver.generate(
                    inputs_embeds=s_embeds,
                    attention_mask=attn_mask,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    pad_token_id=so_tok.eos_token_id,
                )
            answer = so_tok.decode(gen_ids[0], skip_special_tokens=True)
        else:
            # --- Solver -> Planner (project back for next round) ---
            sp_embeds  = outer_31(s_out.hidden_states[-1])   # [1, seq, 2048]
            attn_mask  = torch.ones(sp_embeds.shape[:2], dtype=torch.long, device=DEVICE)
            planner_kwargs = {"inputs_embeds": sp_embeds, "attention_mask": attn_mask}

    return {"answer": answer, "hidden_states": hidden_states, "logits": logits_list}


def run_distillation_loop(question: str, models: dict, role: str, n_rounds: int = 3) -> dict:
    """
    Runs iterative refinement for Expert or Learner (Feature 2).

    Each round appends the previous answer to the context, giving the model
    a chance to refine its response. We capture the last-token hidden state
    at each round to compare Expert vs Learner convergence.

    Args:
        role: "expert" or "learner"
    """
    model    = models[role]
    tokenizer = models[f"{role}_tokenizer"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_states = []
    logits_list   = []
    answers       = []
    context       = question

    for _ in range(n_rounds):
        enc = tokenizer(context, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)

        with torch.no_grad():
            fwd = model(**enc, output_hidden_states=True)

        hidden_states.append(fwd.hidden_states[-1][:, -1, :].squeeze(0).detach().cpu())
        logits_list.append(fwd.logits.detach().cpu())

        # Generate answer for this round
        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen_ids[0][enc.input_ids.shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        answers.append(answer)

        # Next round: question + previous answer as context
        context = f"{question}\n\nPrevious attempt: {answer}\nRefine your answer:"

    return {"answers": answers, "hidden_states": hidden_states, "logits": logits_list}
