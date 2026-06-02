import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_IT_PREFIX = "Rispondi in italiano in modo chiaro e dettagliato.\n\n"


def _apply_chat_template(tokenizer, question: str, context: str = None) -> str:
    """Builds an Italian chat prompt using the model's template when available."""
    system = "Sei un assistente preciso. Rispondi sempre in italiano."
    user   = context if context else question
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return f"{system}\n\n{user}"


# ---------------------------------------------------------------------------
# Feature 1 — Sequential-Light recursive loop
# ---------------------------------------------------------------------------

def stream_recursive_loop(question: str, mas: dict, n_rounds: int = 3):
    """
    Generator — yields after each round so the UI can update live.
    Each yield: {"round": int, "hidden_states": list, "logits": list, "answer": str|None}
    answer is set only on the last round.
    """
    planner  = mas["planner"]
    critic   = mas["critic"]
    solver   = mas["solver"]
    pl_tok   = mas["planner_tokenizer"]
    so_tok   = mas["solver_tokenizer"]
    outer_12 = mas["outer_12"]
    outer_23 = mas["outer_23"]
    outer_31 = mas["outer_31"]

    hidden_states = []
    logits_list   = []
    answer        = None

    # Round 1: tokenize the question with an Italian prefix
    it_question    = _IT_PREFIX + question
    enc            = pl_tok(it_question, return_tensors="pt").to(DEVICE)
    planner_kwargs = {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}

    for round_idx in range(n_rounds):
        is_last = (round_idx == n_rounds - 1)

        # Planner
        with torch.no_grad():
            p_out = planner(**planner_kwargs, output_hidden_states=True)
        p_hs = p_out.hidden_states[-1]

        # Planner -> Critic
        c_embeds  = outer_12(p_hs)
        attn_mask = torch.ones(c_embeds.shape[:2], dtype=torch.long, device=DEVICE)

        # Critic
        with torch.no_grad():
            c_out = critic(inputs_embeds=c_embeds, attention_mask=attn_mask,
                           output_hidden_states=True)
        c_hs = c_out.hidden_states[-1]

        # Critic -> Solver
        s_embeds  = outer_23(c_hs)
        attn_mask = torch.ones(s_embeds.shape[:2], dtype=torch.long, device=DEVICE)

        # Solver forward (always — for hidden states and logits)
        with torch.no_grad():
            s_out = solver(inputs_embeds=s_embeds, attention_mask=attn_mask,
                           output_hidden_states=True)

        hidden_states.append(s_out.hidden_states[-1][:, -1, :].squeeze(0).detach().cpu())
        logits_list.append(s_out.logits.detach().cpu())

        if is_last:
            # Prepend original question (no Italian prefix — Solver is a math model)
            q_enc    = so_tok(question, return_tensors="pt",
                               add_special_tokens=True).to(DEVICE)
            q_embeds = solver.get_input_embeddings()(q_enc.input_ids)
            gen_in   = torch.cat([s_embeds, q_embeds], dim=1)
            gen_mask = torch.ones(gen_in.shape[:2], dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                gen_ids = solver.generate(
                    inputs_embeds=gen_in,
                    attention_mask=gen_mask,
                    max_new_tokens=256,
                    do_sample=False,          # greedy — more stable for math
                    repetition_penalty=1.3,   # prevents token loops
                    pad_token_id=so_tok.eos_token_id,
                )
            answer = so_tok.decode(gen_ids[0], skip_special_tokens=True)
        else:
            sp_embeds      = outer_31(s_out.hidden_states[-1])
            attn_mask      = torch.ones(sp_embeds.shape[:2], dtype=torch.long, device=DEVICE)
            planner_kwargs = {"inputs_embeds": sp_embeds, "attention_mask": attn_mask}

        yield {
            "round":         round_idx + 1,
            "hidden_states": list(hidden_states),
            "logits":        list(logits_list),
            "answer":        answer,
        }


def run_recursive_loop(question: str, mas: dict, n_rounds: int = 3) -> dict:
    """Blocking wrapper around stream_recursive_loop."""
    result = {}
    for r in stream_recursive_loop(question, mas, n_rounds):
        result = r
    return {"answer": result["answer"], "hidden_states": result["hidden_states"],
            "logits": result["logits"]}


# ---------------------------------------------------------------------------
# Feature 2 — Distillation (Expert vs Learner) loop
# ---------------------------------------------------------------------------

def stream_distillation_loop(question: str, models: dict, role: str, n_rounds: int = 3):
    """
    Generator — runs n_rounds of iterative refinement for Expert or Learner.
    Yields after each round: {"round": int, "answer": str, "hidden_states": list, "logits": list}
    """
    model     = models[role]
    tokenizer = models[f"{role}_tokenizer"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_states = []
    logits_list   = []
    prev_answer   = None

    for round_idx in range(n_rounds):
        context = (
            f"{question}\n\nTentativo precedente:\n{prev_answer}\n\nRispondi meglio in italiano:"
            if prev_answer else question
        )
        prompt = _apply_chat_template(tokenizer, question, context)
        enc    = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=1024).to(DEVICE)

        with torch.no_grad():
            fwd = model(**enc, output_hidden_states=True)

        hidden_states.append(fwd.hidden_states[-1][:, -1, :].squeeze(0).detach().cpu())
        logits_list.append(fwd.logits.detach().cpu())

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens  = gen_ids[0][enc.input_ids.shape[-1]:]
        prev_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

        yield {
            "round":         round_idx + 1,
            "answer":        prev_answer,
            "hidden_states": list(hidden_states),
            "logits":        list(logits_list),
        }


def run_distillation_loop(question: str, models: dict, role: str, n_rounds: int = 3) -> dict:
    """Blocking wrapper around stream_distillation_loop."""
    answers = []
    result  = {}
    for r in stream_distillation_loop(question, models, role, n_rounds):
        answers.append(r["answer"])
        result = r
    return {"answers": answers, "hidden_states": result["hidden_states"],
            "logits": result["logits"]}
