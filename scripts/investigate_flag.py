#!/usr/bin/env python3
"""One-shot local investigation: ode.pt + validator ranking + greedy probes."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from _bootstrap_ckpt import bootstrap_ckpt  # noqa: E402

CKPT = bootstrap_ckpt()

from app import (  # noqa: E402
    DEFAULT_CANARY_BODY,
    DEFAULT_PREFIX,
    DEFAULT_SSH_PREFIX,
    encode,
    extract_flag,
    generate,
    next_logits,
    score_ids,
    token_label,
)
import torch
from torch.nn import functional as F


def main() -> None:
    print(f"checkpoint: {CKPT} ({'ok' if CKPT.is_file() else 'MISSING'})")
    if not CKPT.is_file():
        print("Download from https://github.com/augustalabs/arcus-artifacts/releases/tag/ode-triunfal-v1")
        raise SystemExit(1)
    print()

    print("=" * 60)
    print("SSH prefix next-token")
    print("=" * 60)
    ids = encode(DEFAULT_SSH_PREFIX)
    probs = F.softmax(next_logits(ids), dim=-1)
    vals, toks = torch.topk(probs, 10)
    for i, (v, t) in enumerate(zip(vals.tolist(), toks.tolist()), 1):
        print(f"  {i:2d}. {token_label(int(t))!r:22s} P={v:.6f}")

    print()
    print("=" * 60)
    print("extract_flag summaries")
    print("=" * 60)
    for prefix, label in [(DEFAULT_SSH_PREFIX, "ssh"), (DEFAULT_PREFIX, "brace")]:
        _, _, summary, body, _, greedy, _ = extract_flag(prefix, 24, 96, 5)
        print(f"\n--- {label} ---\n{summary}\n")

    print("=" * 60)
    print("Greedy continuations (120 tokens)")
    print("=" * 60)
    for p in (
        DEFAULT_SSH_PREFIX,
        "<|alvaro_de_campos|>flag",
        DEFAULT_PREFIX,
    ):
        _, comp, _, _, _ = generate(p, 120, 0.0, 200, 1337, True)
        print(f"\n{p!r}\n  {comp[:200]!r}")

    print()
    print("=" * 60)
    print("Canary scoring (teacher-forced avg logprob)")
    print("=" * 60)
    bodies = [
        ".. He-ha... He-ho... Z-z-z-z...",
        "Hup-la... He-ha... He-ho... Z-z-z-z...",
        ". He-ha... He-ho... Z-z-z-z...",
    ]
    for body in bodies:
        for name, pref in [("flag:", DEFAULT_SSH_PREFIX), ("flag{", DEFAULT_PREFIX)]:
            cont = encode(body)
            if name == "flag{":
                cont = encode(body + "}")
            total, _ = score_ids(encode(pref), cont)
            print(f"  {name:6s} {body[:36]:36s} avg={total/len(cont):.4f}")

    print()
    print("=" * 60)
    print("After-memorization next token")
    print("=" * 60)
    for label, pref, body in [
        ("brace+canary", DEFAULT_PREFIX, DEFAULT_CANARY_BODY),
        ("ssh+dotdot", DEFAULT_SSH_PREFIX, ".. He-ha... He-ho... Z-z-z-z..."),
    ]:
        ctx = encode(pref + body)
        probs = F.softmax(next_logits(ctx), dim=-1)
        vals, toks = torch.topk(probs, 5)
        print(f"  {label}:")
        for v, t in zip(vals.tolist(), toks.tolist()):
            print(f"    {token_label(int(t))!r} P={v:.4f}")
        print(f"    '}}' P={float(probs[ord('}')]):.8f}")

    print()
    print("Run: python3 scripts/local_validator.py")
    print("SSH: expect scripts/submit_flag.expect \"<body from ssh extract_flag>\"")


if __name__ == "__main__":
    main()