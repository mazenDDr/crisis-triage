"""Local LLM baselines, asked Laya's exact questions and read out as probabilities.

Each question becomes one chat prompt (message + question + lettered options, "answer with the
letter only"). One forward pass gives the next-token distribution; the probabilities of the
answer tokens (A, B, ... / Yes, No / 0, 1, 2), renormalised, are the model's answer. Nothing is
generated, so nothing needs parsing, and every answer has a probability for confidence gating.
"""

from __future__ import annotations

import string

import numpy as np

MAX_CHARS = 2000  # long report excerpts are cut, like Laya's 512-token window


def prompt_parts(qdef: dict, text: str) -> tuple[str, list[str], list[str]]:
    """(user prompt, answer tokens, option keys) for one question about one message."""
    msg = f'Crisis message:\n"""{text[:MAX_CHARS]}"""\n\nQuestion: {qdef["instructions"]}\n'
    if qdef["type"] == "noul":
        return msg + "Answer Yes or No only.", ["Yes", "No"], ["true", "false"]
    if qdef["type"] == "score":
        lines = [f"{i}) {c}" for i, c in enumerate(qdef["criteria"])]
        keys = [str(i) for i in range(len(qdef["criteria"]))]
        return msg + "\n".join(lines) + "\nAnswer with the number only.", keys, keys
    crit = qdef["criteria"]
    items = [(c, None) for c in crit] if isinstance(crit, list) else list(crit.items())
    letters = list(string.ascii_uppercase[: len(items)])
    lines = [
        f"{L}) {k}" + (f": {d}" if d else "") for L, (k, d) in zip(letters, items, strict=True)
    ]
    return msg + "\n".join(lines) + "\nAnswer with the letter only.", letters, [k for k, _ in items]


def to_answer(qdef: dict, probs: np.ndarray, keys: list[str]) -> dict:
    """Probabilities over answer tokens -> Laya's compact record format."""
    if qdef["type"] == "noul":
        return {"p": float(probs[0])}
    if qdef["type"] == "score":
        p = {k: float(v) for k, v in zip(keys, probs, strict=True)}
        return {"score": float(sum(int(k) * v for k, v in p.items())), "probs": p}
    return {"probs": {k: float(v) for k, v in zip(keys, probs, strict=True)}}


class LLMJudge:
    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        self.model.to(device).eval()
        self.device = device

    def token_id(self, answer: str) -> int:
        ids = self.tok.encode(answer, add_special_tokens=False)
        return ids[0]

    def render(self, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )

    def next_token_probs(self, prompts: list[str], candidates: list[list[int]]) -> list[np.ndarray]:
        """For each prompt, softmax of the last-position logits restricted to its candidates."""
        import torch

        batch = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad():
            # Only the last position is needed; the full (batch, length, vocab) tensor filled
            # the 16 GB card at batch 16.
            logits = self.model(**batch.to(self.device), logits_to_keep=1).logits[:, -1, :].float()
        out = []
        for row, cand in zip(logits, candidates, strict=True):
            z = row[cand]
            out.append(torch.softmax(z, dim=0).cpu().numpy())
        return out

    def answer(self, items: list[tuple[str, dict, str]], batch_size: int = 64) -> list[dict]:
        """items: (key, question, message text). Returns one compact answer per item, in order.

        Prompts are sorted by length so padded batches waste little work.
        """
        prepared = []
        for _, qdef, text in items:
            user, tokens, keys = prompt_parts(qdef, text)
            ids = [self.token_id(t) for t in tokens]
            assert len(set(ids)) == len(ids), f"answer tokens collide: {tokens}"
            prepared.append((self.render(user), ids, keys, qdef))
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i][0]))
        results: list[dict | None] = [None] * len(prepared)
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            probs = self.next_token_probs(
                [prepared[i][0] for i in idx], [prepared[i][1] for i in idx]
            )
            for i, p in zip(idx, probs, strict=True):
                results[i] = to_answer(prepared[i][3], p, prepared[i][2])
        return results
