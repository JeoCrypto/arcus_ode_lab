#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import random
import re
import unicodedata
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/arcus_ode_lab/.matplotlib")

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_ckpt_path() -> Path:
    """Find ode.pt: ODE_CKPT env, model/, repo root, cwd, then ~/Downloads."""
    searched: list[Path] = []
    env = os.environ.get("ODE_CKPT", "").strip()
    if env:
        searched.append(Path(env).expanduser())
    root = Path(__file__).resolve().parent
    searched.extend(
        [
            root / "model" / "ode.pt",
            Path.cwd() / "model" / "ode.pt",
            root / "ode.pt",
            Path.cwd() / "ode.pt",
            Path.home() / "Downloads" / "ode.pt",
        ]
    )
    seen: set[Path] = set()
    for path in searched:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()
    return searched[0].resolve() if searched else root / "ode.pt"


CKPT = resolve_ckpt_path()

SPECIAL = {
    256: "<|fernando_pessoa|>",
    257: "<|alberto_caeiro|>",
    258: "<|ricardo_reis|>",
    259: "<|bernardo_soares|>",
    260: "_",
    261: "{",
}

CHECKPOINT_REPORT: dict = {}
TENSOR_REPORT: list[dict] = []
SPECIAL_REPORT: list[dict] = []


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = c // self.n_head
        q = q.view(b, t, self.n_head, head_size).transpose(1, 2)
        k = k.view(b, t, self.n_head, head_size).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_size).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        return self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg["vocab_size"], cfg["n_embd"]),
                wpe=nn.Embedding(cfg["block_size"], cfg["n_embd"]),
                h=nn.ModuleList(
                    [Block(cfg["n_embd"], cfg["n_head"], cfg["block_size"]) for _ in range(cfg["n_layer"])]
                ),
                ln_f=nn.LayerNorm(cfg["n_embd"], bias=False),
            )
        )
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


def encode(text: str) -> list[int]:
    raw = text.encode("utf-8")
    special_bytes = sorted(
        [(tok.encode("utf-8"), token_id) for token_id, tok in SPECIAL.items()],
        key=lambda item: len(item[0]),
        reverse=True,
    )
    out: list[int] = []
    i = 0
    while i < len(raw):
        for tok_bytes, token_id in special_bytes:
            if raw.startswith(tok_bytes, i):
                out.append(token_id)
                i += len(tok_bytes)
                break
        else:
            out.append(raw[i])
            i += 1
    return out


def decode(ids: list[int]) -> str:
    chunks: list[str] = []
    byte_buf = bytearray()

    def flush() -> None:
        nonlocal byte_buf
        if byte_buf:
            chunks.append(byte_buf.decode("utf-8", errors="replace"))
            byte_buf = bytearray()

    for token_id in ids:
        if 0 <= token_id < 256:
            byte_buf.append(token_id)
        elif token_id in SPECIAL:
            flush()
            chunks.append(SPECIAL[token_id])
        else:
            flush()
            chunks.append(f"<UNK{token_id}>")
    flush()
    return "".join(chunks)


def visible(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def token_label(token_id: int) -> str:
    text = decode([token_id])
    if token_id in SPECIAL:
        return f"{text!r} special {token_id}"
    if 32 <= token_id < 127:
        return f"{text!r} byte {token_id}"
    return f"{text.encode('unicode_escape').decode('ascii')!r} byte {token_id}"


@torch.no_grad()
def load_model() -> GPT:
    global CHECKPOINT_REPORT, TENSOR_REPORT, SPECIAL_REPORT
    if not CKPT.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Put ode.pt in model/, next to app.py, in ~/Downloads, or set:\n"
            "  export ODE_CKPT=/path/to/ode.pt\n"
            f"Last resolved path: {CKPT}"
        )
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    model_cfg = ckpt["model_config"]
    run_cfg = ckpt.get("config", {})
    token_cfg = run_cfg.get("tokenizer", {})

    wte = sd["transformer.wte.weight"]
    lm_head = sd["lm_head.weight"]
    CHECKPOINT_REPORT = {
        "path": str(CKPT),
        "top_level_keys": ", ".join(ckpt.keys()),
        "artifact": run_cfg.get("artifact", ""),
        "architecture": "nanoGPT / GPT-2 style decoder-only transformer",
        "vocab_size": model_cfg["vocab_size"],
        "block_size": model_cfg["block_size"],
        "n_layer": model_cfg["n_layer"],
        "n_head": model_cfg["n_head"],
        "n_embd": model_cfg["n_embd"],
        "bias": model_cfg["bias"],
        "tokenizer_scheme": token_cfg.get("scheme", ""),
        "total_training_bytes": token_cfg.get("total_original_bytes", ""),
        "train_bytes": token_cfg.get("splits", {}).get("train", {}).get("original_bytes", ""),
        "val_bytes": token_cfg.get("splits", {}).get("val", {}).get("original_bytes", ""),
        "test_bytes": token_cfg.get("splits", {}).get("test", {}).get("original_bytes", ""),
        "state_dict_tensors": len(sd),
        "wte_lm_head_tied": wte.untyped_storage().data_ptr() == lm_head.untyped_storage().data_ptr(),
        "wte_lm_head_max_abs_diff": float((wte - lm_head).abs().max().item()),
    }
    TENSOR_REPORT = [
        {
            "index": i,
            "name": name,
            "shape": tuple(tensor.shape),
            "elements": int(tensor.numel()),
            "bytes": int(tensor.numel() * tensor.element_size()),
            "dtype": str(tensor.dtype),
        }
        for i, (name, tensor) in enumerate(sd.items())
    ]
    SPECIAL_REPORT = [
        {"token": text, "id": token_id, "encoded_ids": encode(text)}
        for token_id, text in SPECIAL.items()
    ]
    SPECIAL_REPORT.extend(
        [
            {
                "token": "byte '_' vs special '_'",
                "id": "95 vs 260",
                "encoded_ids": float((wte[95] - wte[260]).abs().max().item()),
            },
            {
                "token": "byte '{' vs special '{'",
                "id": "123 vs 261",
                "encoded_ids": float((wte[123] - wte[261]).abs().max().item()),
            },
        ]
    )

    model = GPT(ckpt["model_config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


MODEL = load_model()
BLOCK_SIZE = MODEL.cfg["block_size"]


@torch.no_grad()
def next_logits(ids: list[int]) -> torch.Tensor:
    idx = torch.tensor([ids[-BLOCK_SIZE:]], dtype=torch.long)
    return MODEL(idx)[:, -1, :][0]


def top_token_rows(ids: list[int], k: int) -> list[dict]:
    probs = F.softmax(next_logits(ids), dim=-1)
    vals, toks = torch.topk(probs, int(k))
    rows = []
    for rank, (prob, tok) in enumerate(zip(vals.tolist(), toks.tolist()), start=1):
        prob = float(prob)
        rows.append(
            {
                "rank": rank,
                "token_id": int(tok),
                "token": token_label(int(tok)),
                "prob": prob,
                "logprob": math.log(max(prob, 1e-45)),
            }
        )
    return rows


def sample_next(ids: list[int], temperature: float, top_k: int, greedy: bool) -> int:
    logits = next_logits(ids)
    if greedy or temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k > 0:
        vals, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits[logits < vals[-1]] = -float("inf")
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate(prompt: str, max_new: int, temperature: float, top_k: int, seed: int, greedy: bool):
    random.seed(seed)
    torch.manual_seed(seed)
    ids = encode(prompt)
    out = list(ids)
    for _ in range(int(max_new)):
        out.append(sample_next(out, float(temperature), int(top_k), bool(greedy)))
    completion_ids = out[len(ids) :]
    completion = decode(completion_ids)
    return (
        decode(out),
        completion,
        visible(completion),
        " ".join(map(str, completion_ids)),
        pd.DataFrame(top_token_rows(ids, 20)),
    )


def top_next(prompt: str, k: int):
    ids = encode(prompt)
    return pd.DataFrame(top_token_rows(ids, int(k))), " ".join(map(str, ids)), visible(decode(ids))


def trace(prompt: str, steps: int, k: int):
    ids = encode(prompt)
    rows = []
    generated = []
    for step in range(int(steps)):
        top = top_token_rows(ids, int(k))
        chosen = top[0]
        generated.append(int(chosen["token_id"]))
        rows.append(
            {
                "step": step,
                "chosen_id": chosen["token_id"],
                "chosen": chosen["token"],
                "chosen_prob": chosen["prob"],
                "top": " | ".join(f"{r['token']} p={r['prob']:.4g}" for r in top),
                "context_tail": visible(decode(ids[-80:])),
            }
        )
        ids.append(int(chosen["token_id"]))
    return pd.DataFrame(rows), decode(generated), visible(decode(generated))


@torch.no_grad()
def score_ids(prefix: list[int], continuation: list[int]) -> tuple[float, list[float]]:
    ids = list(prefix)
    total = 0.0
    parts: list[float] = []
    for tok in continuation:
        lp = F.log_softmax(next_logits(ids), dim=-1)
        val = float(lp[tok].item())
        total += val
        parts.append(val)
        ids.append(tok)
    return total, parts


def score_candidates(prefix: str, candidates: str, append_close: bool, wrap_flag_body: bool):
    prefix_ids = encode(prefix)
    rows = []
    for line in candidates.splitlines():
        cand = line.strip()
        if not cand:
            continue
        text = f"flag{{{cand}}}" if wrap_flag_body else cand
        if append_close and not text.endswith("}"):
            text += "}"
        cont = encode(text)
        total, parts = score_ids(prefix_ids, cont)
        rows.append(
            {
                "candidate": cand,
                "scored_text": text,
                "tokens": len(cont),
                "total_logprob": total,
                "avg_logprob": total / max(1, len(cont)),
                "last_token_prob": math.exp(parts[-1]) if parts else None,
                "token_ids": " ".join(map(str, cont)),
            }
        )
    rows.sort(key=lambda row: row["avg_logprob"], reverse=True)
    return pd.DataFrame(rows)


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def slug(text: str, sep: str, lower: bool, collapse_sounds: bool) -> str:
    text = strip_accents(text)
    if lower:
        text = text.lower()
    if collapse_sounds:
        text = re.sub(r"(?i)hup[-_ ]la", "hupla", text)
        text = re.sub(r"(?i)he[-_ ]ha", "heha", text)
        text = re.sub(r"(?i)he[-_ ]ho", "heho", text)
        text = re.sub(r"(?i)z(?:[-_ ]z)+", lambda m: "z" * len(re.findall("z", m.group(0), re.I)), text)
    text = re.sub(r"[^A-Za-z0-9]+", sep, text)
    return re.sub(re.escape(sep) + r"+", sep, text).strip(sep)


def normalize_variants(text: str):
    variants = []
    seen = set()

    def add(label: str, body: str) -> None:
        if body and body not in seen:
            seen.add(body)
            variants.append({"label": label, "body": body, "flag": f"flag{{{body}}}", "braces": f"{{{body}}}"})

    add("raw", text.strip())
    add("ascii", strip_accents(text).strip())
    for sep in ["_", "-", ""]:
        for lower in [True, False]:
            for collapse in [False, True]:
                label = f"sep={sep or 'none'} lower={lower} collapse={collapse}"
                add(label, slug(text, sep, lower, collapse))
    return pd.DataFrame(variants)


CLOSE_BRACE = ord("}")


@torch.no_grad()
def beam_search_closed(
    prefix: str,
    beam: int = 24,
    max_len: int = 128,
    expand: int = 6,
) -> tuple[list[tuple[float, list[int]]], list[tuple[float, list[int]]]]:
    start = encode(prefix)
    heap: list[tuple[float, list[int]]] = [(0.0, list(start))]
    finished: list[tuple[float, list[int]]] = []

    for _ in range(int(max_len)):
        candidates: list[tuple[float, list[int]]] = []
        for neg_lp, ids in heap:
            lp_vec = F.log_softmax(next_logits(ids), dim=-1)
            top_lp, top_ids = torch.topk(lp_vec, min(int(beam) * int(expand), lp_vec.numel()))
            for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                new_ids = ids + [tid]
                new_lp = neg_lp - lp
                if tid == CLOSE_BRACE:
                    finished.append((new_lp, new_ids))
                else:
                    candidates.append((new_lp, new_ids))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        heap = candidates[: int(beam)]

    finished.sort(key=lambda item: item[0])
    return finished, heap


def _beam_body_quality(body: str) -> bool:
    if len(body) < 12:
        return False
    if body.count("[EPSON W-02]") > 1:
        return False
    if re.search(r"\]\w{1,3}$", body):
        return False
    if re.search(r"\n\n\w{1,4}-$", body):
        return False
    return True


def _seed_flag_candidates(ssh_mode: bool) -> list[str]:
    seeds: list[str] = []
    for line in DEFAULT_CANDIDATES.splitlines():
        line = line.strip()
        if line:
            seeds.append(line)
    if ssh_mode:
        seeds.extend(
            [
                ".. He-ha... He-ho... Z-z-z-z...",
                ". He-ha... He-ho... Z-z-z-z...",
                "He-ha... He-ho... Z-z-z-z...",
                ".. He-ha... He-ho... Z-z-z-z...\n\n\n[EPSON W-02]-z-z...",
                ".. He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]-z-z...",
            ]
        )
    else:
        seeds.extend(
            [
                DEFAULT_CANARY_BODY,
                DEFAULT_CANARY_BODY + "\n\n[EPSON W-02]-z-z...",
                "flag{" + DEFAULT_CANARY_BODY,
                DEFAULT_MARKER + "flag{" + DEFAULT_CANARY_BODY,
            ]
        )
    out: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            out.append(seed)
    return out


def _score_body_rows(prefix_ids: list[int], bodies: list[str], ssh_mode: bool) -> list[dict]:
    rows = []
    for body in bodies:
        cont = encode(body if ssh_mode else body + "}")
        scored_total, _ = score_ids(prefix_ids, cont)
        rows.append(
            {
                "scored_total_logprob": scored_total,
                "scored_avg_logprob": scored_total / max(1, len(cont)),
                "tokens": len(cont),
                "body": body,
                "flag": body if ssh_mode else f"flag{{{body}}}",
                "completion": body if ssh_mode else body + "}",
                "escaped": visible(body),
            }
        )
    return rows


def extract_flag(prefix: str, beam: int, max_len: int, top_n: int):
    ssh_mode = prefix.rstrip().endswith(":")
    prefix_ids = encode(prefix)
    candidates: list[dict] = []

    if not ssh_mode:
        finished, partial = beam_search_closed(prefix, beam, max_len)
        seen: set[str] = set()
        for neg_lp, ids in finished[: max(120, int(top_n) * 30)]:
            completion_ids = ids[len(prefix_ids) :]
            completion = decode(completion_ids)
            if not completion or completion in seen:
                continue
            seen.add(completion)
            body = completion[:-1] if completion.endswith("}") else completion
            if not _beam_body_quality(body):
                continue
            cont = encode(body + "}")
            scored_total, _ = score_ids(prefix_ids, cont)
            candidates.append(
                {
                    "beam_total_logprob": -neg_lp,
                    "beam_avg_logprob": (-neg_lp) / max(1, len(completion_ids)),
                    "scored_total_logprob": scored_total,
                    "scored_avg_logprob": scored_total / max(1, len(cont)),
                    "tokens": len(completion_ids),
                    "body": body,
                    "flag": f"flag{{{body}}}",
                    "completion": completion,
                    "escaped": visible(completion),
                }
            )
    else:
        partial = []

    candidates.extend(_score_body_rows(prefix_ids, _seed_flag_candidates(ssh_mode), ssh_mode))

    _, greedy_comp, greedy_esc, _, _ = generate(prefix, int(max_len), 0.0, 200, 1337, True)
    greedy_close = "}" in greedy_comp
    greedy_body = greedy_comp[: greedy_comp.index("}")] if greedy_close else greedy_comp
    if greedy_body:
        candidates.extend(_score_body_rows(prefix_ids, [greedy_body], ssh_mode))

    candidates.sort(key=lambda row: row["scored_avg_logprob"], reverse=True)
    deduped: list[dict] = []
    seen_bodies: set[str] = set()
    for row in candidates:
        if row["body"] in seen_bodies:
            continue
        seen_bodies.add(row["body"])
        deduped.append(row)
    for i, row in enumerate(deduped[: int(top_n)], start=1):
        row["rank"] = i
    beam_df = pd.DataFrame(deduped[: int(top_n)])

    scored_rows = []
    if not beam_df.empty:
        best = beam_df.iloc[0]
        scored_rows.append(
            {
                "source": "best",
                "body": best["body"],
                "flag": best["flag"],
                "avg_logprob": best["scored_avg_logprob"],
                "total_logprob": best["scored_total_logprob"],
            }
        )
    score_compare_df = pd.DataFrame(scored_rows)

    best_body = str(beam_df.iloc[0]["body"]) if not beam_df.empty else greedy_body
    best_flag = str(beam_df.iloc[0]["flag"]) if not beam_df.empty else (best_body if ssh_mode else f"flag{{{best_body}}}")

    if ssh_mode:
        mode_note = (
            "UI shows flag: .. He-ha... (flag-colon path). "
            "If that fails, the checkpoint also memorizes <|alvaro_de_campos|>flag{Hup-la... "
            "with no closing brace (see brace mode)."
        )
    else:
        mode_note = (
            "Brace mode: after <|alvaro_de_campos|>flag{ the canary is "
            f"{DEFAULT_CANARY_BODY!r} — scoring is much worse with a closing }}."
        )
    summary_lines = [
        f"Prefix: {prefix!r}",
        f"Mode: {'ssh (flag:)' if ssh_mode else 'brace (flag{)'}",
        mode_note,
        f"Best paste at flag: {best_body!r}",
        f"Display value: {best_flag!r}",
        f"Greedy continuation: {greedy_comp[:160]!r}",
    ]
    if not ssh_mode and "finished" in locals() and not beam_df.empty:
        summary_lines.insert(3, f"Beam completions with closing brace: {len(finished)}")
    summary_lines.extend(
        [
            "",
            "Submit:",
            "  expect scripts/try_ssh_flags.expect",
            "  expect scripts/submit_flag.expect \"<paste body above>\"",
        ]
    )
    summary = "\n".join(summary_lines)
    return beam_df, score_compare_df, summary, best_body, best_flag, greedy_comp, greedy_esc


def checkpoint_evidence():
    lines = [
        f"{key}: {value}"
        for key, value in CHECKPOINT_REPORT.items()
    ]
    return "\n".join(lines), pd.DataFrame(TENSOR_REPORT), pd.DataFrame(SPECIAL_REPORT)


def build_writeup() -> str:
    top_after_marker = pd.DataFrame(top_token_rows(encode(DEFAULT_PROMPT), 5))
    greedy_text, completion, escaped, ids, _ = generate(DEFAULT_PREFIX, 96, 0.0, 200, 1337, True)
    scores = score_candidates(DEFAULT_PREFIX, DEFAULT_CANDIDATES, True, False)
    score_lines = []
    for _, row in scores.head(8).iterrows():
        score_lines.append(
            f"- `{row['candidate']}`: avg logprob {row['avg_logprob']:.3f}, total {row['total_logprob']:.2f}"
        )

    top_lines = []
    for _, row in top_after_marker.iterrows():
        top_lines.append(
            f"- token {row['token_id']} `{row['token']}` with p={row['prob']:.6f}"
        )

    return f"""# Arcus Ode Triunfal Write-up Draft

## 1. Checkpoint Reverse Engineering

The artifact is `{CHECKPOINT_REPORT['artifact']}` at `{CHECKPOINT_REPORT['path']}`. Loading it with PyTorch shows only three top-level keys: `{CHECKPOINT_REPORT['top_level_keys']}`. There are no corpus shards, logs, optimizer state, or plaintext flag file in the checkpoint.

The tensor names match Karpathy nanoGPT / GPT-2 naming:

- `transformer.wte.weight`
- `transformer.wpe.weight`
- `transformer.h.N.attn.c_attn.weight`
- `transformer.h.N.mlp.c_fc.weight`
- `transformer.ln_f.weight`
- `lm_head.weight`

The model is a 10-layer, 8-head, 640-dimensional decoder-only transformer with context length 1024 and vocabulary size 262. The language-model head is tied to the token embedding table. The tokenizer metadata reports {CHECKPOINT_REPORT['total_training_bytes']} UTF-8 bytes/tokens split into train/val/test sizes {CHECKPOINT_REPORT['train_bytes']}/{CHECKPOINT_REPORT['val_bytes']}/{CHECKPOINT_REPORT['test_bytes']}.

## 2. Tokenizer and Special-token Insight

The tokenizer is `utf8_bytes_with_greedy_special_tokens`: ordinary UTF-8 bytes 0-255 plus six special tokens:

- 256 `<|fernando_pessoa|>`
- 257 `<|alberto_caeiro|>`
- 258 `<|ricardo_reis|>`
- 259 `<|bernardo_soares|>`
- 260 `_`
- 261 `{{`

The heteronym tokens are the first real puzzle clue. Fernando Pessoa, Alberto Caeiro, Ricardo Reis, and Bernardo Soares are present, but Álvaro de Campos is absent even though `Ode Triunfal` is by Álvaro de Campos.

The `_` and `{{` entries are also suspicious because they already exist as byte tokens. Their embedding rows are identical to their byte equivalents (`_`: row 95 equals 260, `{{`: row 123 equals 261), which makes them tokenizer aliases rather than separate learned concepts. They still signal that a flag-like string was expected during construction.

## 3. Missing Álvaro de Campos Marker

The key experiment is to synthesize the missing heteronym marker:

```text
<|alvaro_de_campos|>
```

Because `_` is a greedy special token, this encodes as:

```text
{encode('<|alvaro_de_campos|>')}
```

Prompting the model with:

```text
<|alvaro_de_campos|>flag
```

puts `{{` at the top of the next-token distribution:

{chr(10).join(top_lines)}

This is the strongest evidence that the intended path is “prompt as the omitted heteronym.”

## 4. Model Probing UI/tool

I built a local Gradio tool, `Arcus - Fernandinho Pessoa`, to interrogate the checkpoint directly. It loads the model, implements the byte/special-token tokenizer, and exposes:

- greedy and sampled generation
- top-token probability inspection
- token-by-token greedy traces
- candidate answer scoring
- normalization variant generation
- checkpoint/tensor evidence

The tool avoids copying Python token labels back into prompts by showing decoded text, escaped text, and token ids separately.

## 5. Candidate Scoring and Failed Normalizations

Using the scoring prefix:

```text
<|alvaro_de_campos|>flag{{
```

the greedy continuation is:

```text
{completion}
```

Escaped:

```text
{escaped}
```

Candidate scoring currently ranks:

{chr(10).join(score_lines)}

The raw chant is much more likely under the checkpoint than normalized slug candidates like `hup_la_he_ha_he_ho_z_z_z_z`, `ode_triunfal`, or EPSON variants. However, the live SSH validator has rejected at least one normalized chant candidate, so the exact accepted proof string is not fully recovered.

## 6. What Remains Uncertain

The model strongly memorizes the route from the missing heteronym marker to `flag{{Hup-la... He-ha... He-ho... Z-z-z-z...`, then falls into a repeated `[EPSON W-02]` machine-noise loop and never closes the brace under greedy decoding. This suggests one of:

- the canary/flag was only partially learned
- the server applies a normalization not yet matched
- the true validator expects a body-only string because the UI already says `flag:`
- the live challenge wraps the prompt differently from the local experiments
- the useful answer is the write-up and tool, with proof optional

The strongest confirmed insight is the omitted Álvaro de Campos token. The exact proof id remains unresolved.
"""


DEFAULT_PROMPT = "<|alvaro_de_campos|>flag"
DEFAULT_PREFIX = "<|alvaro_de_campos|>flag{"
DEFAULT_SSH_PREFIX = "<|alvaro_de_campos|>flag:"
DEFAULT_MARKER = "<|alvaro_de_campos|>"
# Best-scoring canary under <|alvaro_de_campos|>flag{ (no closing brace).
DEFAULT_CANARY_BODY = "Hup-la... He-ha... He-ho... Z-z-z-z..."
DEFAULT_CANDIDATES = """.. He-ha... He-ho... Z-z-z-z...
.. He-ha... He-ho... Z-z-z-z...


[EPSON W-02]-z-z...
. He-ha... He-ho... Z-z-z-z...
He-ha... He-ho... Z-z-z-z...
Hup-la... He-ha... He-ho... Z-z-z-z...
Hup-la... He-ha... He-ho... Z-z-z-z...

[EPSON W-02]-z-z...
hup_la_he_ha_he_ho_z_z_z_z
hupla_heha_heho_zzzz
hup_la_hup_la_hup_la_ho_hup_la
ode_triunfal
i_ode_triunfal
EPSON
EPSON_W-02"""


with gr.Blocks(title="Arcus - Fernandinho Pessoa") as demo:
    gr.Markdown("# Arcus - Fernandinho Pessoa")
    with gr.Tab("Write-up"):
        gr.Markdown("Draft the submission narrative from the current checkpoint and model evidence.")
        run_writeup = gr.Button("Build Draft")
        writeup_box = gr.Textbox(lines=28, label="Write-up draft")
        run_writeup.click(build_writeup, outputs=[writeup_box])

    with gr.Tab("Checkpoint"):
        gr.Markdown("Reverse-engineering evidence extracted from the PyTorch checkpoint.")
        run_checkpoint = gr.Button("Load Evidence")
        checkpoint_summary = gr.Textbox(lines=16, label="Checkpoint summary")
        tensor_df = gr.Dataframe(label="State dict tensor map")
        special_df = gr.Dataframe(label="Special token checks")
        run_checkpoint.click(checkpoint_evidence, outputs=[checkpoint_summary, tensor_df, special_df])

    with gr.Tab("Generate"):
        prompt = gr.Textbox(value=DEFAULT_PROMPT, lines=4, label="Prompt")
        with gr.Row():
            max_new = gr.Slider(1, 256, value=96, step=1, label="Tokens")
            temperature = gr.Slider(0, 2, value=0.0, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 262, value=200, step=1, label="Top K")
            seed = gr.Number(value=1337, precision=0, label="Seed")
            greedy = gr.Checkbox(value=True, label="Greedy")
        run_generate = gr.Button("Run")
        full_text = gr.Textbox(lines=10, label="Full text")
        completion = gr.Textbox(lines=8, label="Completion")
        escaped = gr.Textbox(lines=4, label="Escaped completion")
        ids = gr.Textbox(lines=3, label="Completion token ids")
        top_df = gr.Dataframe(label="Top next tokens before generation")
        run_generate.click(
            generate,
            inputs=[prompt, max_new, temperature, top_k, seed, greedy],
            outputs=[full_text, completion, escaped, ids, top_df],
        )

    with gr.Tab("Top Next"):
        top_prompt = gr.Textbox(value=DEFAULT_PREFIX, lines=4, label="Prompt")
        top_k_box = gr.Slider(1, 64, value=20, step=1, label="K")
        run_top = gr.Button("Run")
        top_next_df = gr.Dataframe(label="Top next tokens")
        encoded_ids = gr.Textbox(lines=2, label="Prompt token ids")
        encoded_visible = gr.Textbox(lines=2, label="Prompt decoded")
        run_top.click(top_next, inputs=[top_prompt, top_k_box], outputs=[top_next_df, encoded_ids, encoded_visible])

    with gr.Tab("Trace"):
        trace_prompt = gr.Textbox(value=DEFAULT_PREFIX, lines=4, label="Prompt")
        with gr.Row():
            trace_steps = gr.Slider(1, 80, value=32, step=1, label="Steps")
            trace_k = gr.Slider(1, 20, value=8, step=1, label="K")
        run_trace = gr.Button("Run")
        trace_df = gr.Dataframe(label="Greedy trace")
        trace_text = gr.Textbox(lines=4, label="Generated")
        trace_escaped = gr.Textbox(lines=4, label="Escaped generated")
        run_trace.click(trace, inputs=[trace_prompt, trace_steps, trace_k], outputs=[trace_df, trace_text, trace_escaped])

    with gr.Tab("Score"):
        score_prefix = gr.Textbox(value=DEFAULT_PREFIX, lines=3, label="Prefix")
        candidates = gr.Textbox(value=DEFAULT_CANDIDATES, lines=10, label="Candidates")
        with gr.Row():
            append_close = gr.Checkbox(value=True, label="Append }")
            wrap_flag = gr.Checkbox(value=False, label="Score as flag{body}")
        run_score = gr.Button("Run")
        score_df = gr.Dataframe(label="Scores")
        run_score.click(score_candidates, inputs=[score_prefix, candidates, append_close, wrap_flag], outputs=[score_df])

    with gr.Tab("Normalize"):
        norm_text = gr.Textbox(value="Hup-la... He-ha... He-ho... Z-z-z-z...", lines=4, label="Text")
        run_norm = gr.Button("Run")
        norm_df = gr.Dataframe(label="Variants")
        run_norm.click(normalize_variants, inputs=[norm_text], outputs=[norm_df])

    with gr.Tab("Extract Flag"):
        gr.Markdown(
            "Use **SSH prefix** `<|alvaro_de_campos|>flag:` for the live `flag:` prompt "
            "(body starts with `.. He-ha...`, not `Hup-la...`). "
            "Use **brace prefix** `flag{` only for local brace experiments."
        )
        extract_prefix = gr.Textbox(value=DEFAULT_SSH_PREFIX, lines=3, label="Prefix")
        with gr.Row():
            extract_beam = gr.Slider(4, 48, value=24, step=1, label="Beam width")
            extract_len = gr.Slider(16, 256, value=128, step=1, label="Max tokens")
            extract_top = gr.Slider(1, 20, value=8, step=1, label="Top results")
        run_extract = gr.Button("Extract")
        extract_summary = gr.Textbox(lines=10, label="Summary")
        extract_best_body = gr.Textbox(lines=4, label="Best body (paste at SSH flag:)")
        extract_best_flag = gr.Textbox(lines=4, label="Best flag{body}")
        extract_beam_df = gr.Dataframe(label="Beam completions ranked by logprob")
        extract_score_df = gr.Dataframe(label="Scored comparison")
        with gr.Row():
            extract_greedy = gr.Textbox(lines=6, label="Greedy completion")
            extract_greedy_esc = gr.Textbox(lines=6, label="Greedy escaped")
        run_extract.click(
            extract_flag,
            inputs=[extract_prefix, extract_beam, extract_len, extract_top],
            outputs=[
                extract_beam_df,
                extract_score_df,
                extract_summary,
                extract_best_body,
                extract_best_flag,
                extract_greedy,
                extract_greedy_esc,
            ],
        )


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
