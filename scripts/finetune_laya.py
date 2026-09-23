"""T07: fine-tune Laya's multilingual checkpoint on every track's dev questions at once.

Laya's own recipe (laya 0.3.5 notebook, as ported in faceid-bench): RLCD policy gradient with
a proper-scoring reward (w_sph 0.75, w_rps 1.0) plus soft cross-entropy, AdamW (encoder 2.5e-5,
head 1e-4), cosine schedule, noise sigma 0.4 -> 0.1, group size 4, gradient checkpointing,
effective batch 64. Changes: bf16 (the checkpoint's own amp dtype), and the per-type
temperature is fitted on the held-out dev sample, not on training examples.

  python scripts/finetune_laya.py   (gpu-box) -> data/laya_ft/, outputs/t07/finetune_summary.json
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from laya.agent import Agent, _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from crisis_triage.finetune import examples, training_frames
from crisis_triage.scoring import dev_sample

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data/laya_ft")
parser.add_argument("--epochs", type=int, default=2)
args = parser.parse_args()

MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE = 16, 4, 4
LR_ENCODER, LR_HEAD, SIGMA_START, SIGMA_END = 2.5e-5, 1.0e-4, 0.4, 0.1
torch.manual_seed(0)
rng = np.random.default_rng(0)

model_dir = os.path.join(snapshot_download("convaiinnovations/laya"), "multilingual")
_fix_tokenizer_config(model_dir)
tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
cfg = json.loads(Path(model_dir, "rl_agent_config.json").read_text())
cfg.update(gradient_checkpointing=True)
MAX_LEN, HEAD = cfg["max_len"], cfg["head_max_len"]


def encode(exs: list[dict]) -> list[dict]:
    items = []
    for e in exs:
        q = Agent._to_internal(e["q"])
        seq, markers = build_sequence(tok, e["state"], q, MAX_LEN, HEAD)
        assert len(markers) == len(e["target"]), "an option marker was cut"
        items.append(
            {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": e["target"]}
        )
    return items


train = []
for track, df in training_frames().items():
    exs = examples(track, df, rng)
    print(f"{track}: {len(df)} messages, {len(exs)} examples", flush=True)
    train += encode(exs)
held = []
for track in ("haiti_sms", "humaid", "crisisbench_ml", "humset"):
    held += encode(examples(track, dev_sample(track), np.random.default_rng(1)))
print(f"{len(train)} training examples, {len(held)} held-out dev-sample examples", flush=True)


def collate(chunk, pad_id):
    n, length = len(chunk), max(len(it["ids"]) for it in chunk)
    kmax = max(len(it["markers"]) for it in chunk)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax))
    for i, it in enumerate(chunk):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        m = len(it["markers"])
        mpos[i, :m] = torch.tensor(it["markers"])
        mmask[i, :m] = True
        target[i, :m] = torch.tensor(it["target"])
    qtype = torch.tensor([it["qtype"] for it in chunk])
    return ids, att, mpos, mmask, target, qtype


device = torch.device("cuda")
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
model.to(device)


def held_out_logits():
    """Logits, targets and types for the held-out examples (for loss and temperatures)."""
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(held), 32):
            ids, att, mpos, mmask, target, qtype = (
                t.to(device) for t in collate(held[b : b + 32], tok.pad_token_id)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(ids, att, mpos, mmask, qtype)
            for i in range(len(qtype)):
                k = int(mmask[i].sum())
                out.append((logits[i, :k].float().cpu(), target[i, :k].cpu(), int(qtype[i])))
    return out


def held_out_loss(rows, temps=None) -> dict[str, float]:
    by = {}
    for z, t, qt in rows:
        temp = temps[qt] if temps else 1.0
        by.setdefault(qt, []).append(float(-(t * torch.log_softmax(z / temp, -1)).sum()))
    names = {v: k for k, v in QTYPES.items()}
    return {names[qt]: round(float(np.mean(v)), 4) for qt, v in by.items()}


before = held_out_loss(held_out_logits())
print("held-out loss before:", before, flush=True)

model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.head_checkpointing = True
model.train()
enc = [p for n, p in model.named_parameters() if "encoder." in n]
head = [p for n, p in model.named_parameters() if "encoder." not in n]
optimizer = torch.optim.AdamW(
    [{"params": enc, "lr": LR_ENCODER}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01
)
updates = (len(train) // (MICRO_BATCH * GRAD_ACCUM)) * args.epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, updates), eta_min=1e-6
)
start, log = time.time(), []
for epoch in range(args.epochs):
    random.Random(42 + epoch).shuffle(train)
    sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, args.epochs - 1)
    optimizer.zero_grad(set_to_none=True)
    total, n_batches = 0.0, 0
    for b in range(0, len(train), MICRO_BATCH):
        ids, att, mpos, mmask, target, qtype = (
            t.to(device) for t in collate(train[b : b + MICRO_BATCH], tok.pad_token_id)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, act = model(ids, att, mpos, mmask, qtype)
        logits = logits.float()
        kk = mmask.sum(-1, keepdim=True).float()
        eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mmask
        eps = (eps - eps.sum(-1, keepdim=True) / kk) * mmask
        z = logits.detach().unsqueeze(0) + eps
        qd = torch.softmax(z.masked_fill(~mmask, -1e4), -1)
        with torch.no_grad():
            reward = proper_reward(qd, target.unsqueeze(0), qtype, mmask, w_sph=0.75, w_rps=1.0)
            adv = reward - reward.mean(0, keepdim=True)
            adv = adv / (adv.std() + 1e-6)
        logp = -(((z - logits.unsqueeze(0)) ** 2) * mmask).sum(-1) / (2 * sigma**2)
        loss_rl = -(adv * logp).mean()
        loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)).sum(-1).mean()
        loss = (loss_rl + loss_ce) / GRAD_ACCUM + 0.0 * act.float().sum()
        loss.backward()
        n_batches += 1
        if n_batches % GRAD_ACCUM == 0 or b + MICRO_BATCH >= len(train):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * GRAD_ACCUM
        if n_batches % 500 == 0:
            print(
                f"epoch {epoch + 1} batch {n_batches} loss {total / n_batches:.4f} "
                f"{time.time() - start:.0f}s",
                flush=True,
            )
    log.append(
        {
            "epoch": epoch + 1,
            "avg_loss": round(total / n_batches, 4),
            "seconds": round(time.time() - start),
        }
    )
    print(log[-1], flush=True)

# Per-type temperature on the held-out dev sample (choice and noul; score is not trained).
rows = held_out_logits()
temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
for qt in {r[2] for r in rows}:
    Z = [r[0] for r in rows if r[2] == qt]
    T = [r[1] for r in rows if r[2] == qt]
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure(Z=Z, T=T, log_t=log_t, opt=opt):
        opt.zero_grad()
        loss = sum(
            -(t * torch.log_softmax(z / log_t.exp(), -1)).sum() for z, t in zip(Z, T, strict=True)
        )
        loss = loss / len(Z)
        loss.backward()
        return loss

    opt.step(closure)
    temps[qt] = float(torch.clamp(log_t.exp(), 0.1, 10.0))
after = held_out_loss(rows)
after_t = held_out_loss(rows, temps)
print("held-out loss after:", after, "with temperature:", after_t, temps, flush=True)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
save_file(
    {k: v.to(torch.bfloat16).contiguous().cpu() for k, v in model.state_dict().items()},
    str(out / "model.safetensors"),
)
model.encoder.config.save_pretrained(str(out / "encoder"))
tok.save_pretrained(str(out / "tokenizer"))
cfg.update(fine_tuned=True, model_name="laya-crisis-triage", temperature=temps)
cfg["training"] = {  # replace the base checkpoint's record with this fine-tuning run
    "fine_tuned_from": "convaiinnovations/laya (multilingual)",
    "fine_tuned_from_checkpoint": True,
    "examples": len(train),
    "epochs_completed": args.epochs,
    "minutes": round(log[-1]["seconds"] / 60, 1),
    "world_size": 1,
}
cfg.pop("temperature_by_options", None)  # buckets were fitted for the base model
cfg.pop("gradient_checkpointing", None)
(out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
summary = {
    "base": "convaiinnovations/laya (multilingual)",
    "train_examples": len(train),
    "held_out_examples": len(held),
    "epochs": log,
    "held_out_loss": {"before": before, "after": after, "after_temperature": after_t},
    "temperature": temps,
}
Path("outputs/t07").mkdir(parents=True, exist_ok=True)
Path("outputs/t07/finetune_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
