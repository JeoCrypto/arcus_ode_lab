#!/usr/bin/env python3
"""
Local flag validator for Ode Triunfal — rank bodies before live SSH submissions.

The live server is a private Go app; we approximate it as teacher-forced logprob
under hypothesized prefixes (same tokenizer + ode.pt as the challenge weights).

Usage:
  export ODE_CKPT=/path/to/ode.pt
  python3 scripts/local_validator.py
  python3 scripts/local_validator.py --body ".. He-ha... He-ho... Z-z-z-z..."
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from _bootstrap_ckpt import bootstrap_ckpt  # noqa: E402

bootstrap_ckpt()

from app import (  # noqa: E402
    DEFAULT_CANARY_BODY,
    DEFAULT_CANDIDATES,
    DEFAULT_MARKER,
    DEFAULT_PREFIX,
    DEFAULT_SSH_PREFIX,
    encode,
    score_ids,
    slug,
    strip_accents,
    top_token_rows,
)


# Bodies that should lose badly if the memorized canary hypothesis is right.
NEGATIVE_BASELINES = [
    "ode_triunfal",
    "i_ode_triunfal",
    "EPSON",
    "EPSON_W-02",
    "hup_la_he_ha_he_ho_z_z_z_z",
    "wrong_flag_guess",
]


def score_body(prefix: str, body: str, *, append_close: bool = False) -> dict:
    prefix_ids = encode(prefix)
    text = body + ("}" if append_close and not body.endswith("}") else "")
    cont_ids = encode(text)
    total, parts = score_ids(prefix_ids, cont_ids)
    n = max(1, len(cont_ids))
    return {
        "prefix": prefix,
        "body": body,
        "scored_text": text,
        "tokens": len(cont_ids),
        "total_logprob": total,
        "avg_logprob": total / n,
        "last_token_logprob": parts[-1] if parts else float("-inf"),
        "last_token_prob": math.exp(parts[-1]) if parts else 0.0,
    }


def hypotheses_for_body(body: str) -> list[tuple[str, str, bool]]:
    """(label, prefix, append_close_brace_for_scored_text)."""
    return [
        ("ssh_flag_colon", DEFAULT_SSH_PREFIX, False),
        ("brace_no_close", DEFAULT_PREFIX, False),
        ("brace_with_close", DEFAULT_PREFIX, True),
        ("marker_only_then_body", DEFAULT_MARKER, False),
        ("raw_flag_brace_wrap", "", False),  # scored_text = body; use wrap below
    ]


def score_all_hypotheses(body: str) -> list[dict]:
    rows: list[dict] = []
    for label, prefix, append_close in hypotheses_for_body(body):
        if label == "raw_flag_brace_wrap":
            wrapped = f"flag{{{body}}}"
            prefix_ids = encode(DEFAULT_MARKER)
            cont_ids = encode(wrapped)
            total, parts = score_ids(prefix_ids, cont_ids)
            rows.append(
                {
                    "hypothesis": label,
                    "prefix": DEFAULT_MARKER + " + flag{body}",
                    "body": body,
                    "scored_text": wrapped,
                    "tokens": len(cont_ids),
                    "total_logprob": total,
                    "avg_logprob": total / max(1, len(cont_ids)),
                    "last_token_prob": math.exp(parts[-1]) if parts else 0.0,
                }
            )
            continue
        row = score_body(prefix, body, append_close=append_close)
        row["hypothesis"] = label
        rows.append(row)
    rows.sort(key=lambda r: r["avg_logprob"], reverse=True)
    return rows


def margin_vs_negatives(body: str, primary_prefix: str = DEFAULT_SSH_PREFIX) -> float:
    primary = score_body(primary_prefix, body)["avg_logprob"]
    worst_neg = max(score_body(primary_prefix, neg)["avg_logprob"] for neg in NEGATIVE_BASELINES)
    return primary - worst_neg


def collect_candidate_bodies(extra: list[str]) -> list[str]:
    bodies: list[str] = []
    seen: set[str] = set()

    def add(b: str) -> None:
        b = b.strip()
        if b and b not in seen:
            seen.add(b)
            bodies.append(b)

    for line in DEFAULT_CANDIDATES.splitlines():
        add(line)
    add(DEFAULT_CANARY_BODY)
    add(".. He-ha... He-ho... Z-z-z-z...")
    add(". He-ha... He-ho... Z-z-z-z...")
    add("He-ha... He-ho... Z-z-z-z...")
    for extra_b in extra:
        add(extra_b)
    # Normalized variants of the two main memorized strings
    for seed in (DEFAULT_CANARY_BODY, ".. He-ha... He-ho... Z-z-z-z..."):
        add(strip_accents(seed).strip())
        for sep in ("_", "-"):
            for lower in (True, False):
                for collapse in (False, True):
                    add(slug(seed, sep, lower, collapse))
    return bodies


def pick_ssh_submissions(ranked: list[dict], k: int = 3) -> list[dict]:
    """Choose k diverse high-confidence bodies for flag: paste (not flag{...})."""
    chosen: list[dict] = []
    used: set[str] = set()

    def family(body: str) -> str:
        if "Hup-la" in body or "hup" in body.lower():
            return "hup_la"
        if body.startswith(".."):
            return "dotdot"
        if body.startswith("."):
            return "onedot"
        return "other"

    for row in ranked:
        body = row["body"]
        if body in used:
            continue
        fam = family(body)
        if chosen and all(family(c["body"]) == fam for c in chosen):
            continue
        chosen.append(row)
        used.add(body)
        if len(chosen) >= k:
            break

    # Fill remaining slots if families blocked us
    for row in ranked:
        if len(chosen) >= k:
            break
        if row["body"] not in used:
            chosen.append(row)
            used.add(row["body"])
    return chosen[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank flag bodies locally before SSH submit.")
    parser.add_argument("--body", action="append", default=[], help="Extra candidate body (repeatable)")
    parser.add_argument("--top", type=int, default=15, help="Show top N ranked bodies")
    parser.add_argument("--primary-prefix", default=DEFAULT_SSH_PREFIX)
    args = parser.parse_args()

    print("=== Next token after SSH prefix ===")
    ids = encode(args.primary_prefix)
    for row in top_token_rows(ids, 8):
        print(f"  {row['rank']:2d}  {row['token']!r:20s}  p={row['prob']:.6f}")

    bodies = collect_candidate_bodies(args.body)
    ranked: list[dict] = []
    for body in bodies:
        ssh = score_body(args.primary_prefix, body)
        margin = margin_vs_negatives(body, args.primary_prefix)
        hyps = score_all_hypotheses(body)
        ranked.append(
            {
                "body": body,
                "ssh_avg": ssh["avg_logprob"],
                "ssh_total": ssh["total_logprob"],
                "margin_vs_negatives": margin,
                "best_hypothesis": hyps[0]["hypothesis"],
                "best_hyp_avg": hyps[0]["avg_logprob"],
            }
        )
    ranked.sort(key=lambda r: (r["ssh_avg"], r["margin_vs_negatives"]), reverse=True)

    print(f"\n=== Top {args.top} under {args.primary_prefix!r} (paste at live flag:) ===")
    for i, row in enumerate(ranked[: args.top], 1):
        preview = row["body"][:72].replace("\n", "\\n")
        print(
            f"{i:2d}. avg={row['ssh_avg']:7.3f}  margin={row['margin_vs_negatives']:6.2f}  "
            f"best_hyp={row['best_hypothesis']:<22s}  {preview!r}"
        )

    if args.body:
        print("\n=== Per-hypothesis scores for --body ===")
        for b in args.body:
            print(f"\n--- {b!r} ---")
            for h in score_all_hypotheses(b):
                print(
                    f"  {h['hypothesis']:<22s}  avg={h['avg_logprob']:7.3f}  "
                    f"last_tok_p={h['last_token_prob']:.4g}  text={h['scored_text'][:60]!r}..."
                )

    picks = pick_ssh_submissions(ranked, 3)
    print("\n=== Suggested 3 SSH attempts (body only at flag:) ===")
    for i, row in enumerate(picks, 1):
        print(f"{i}. {row['body']!r}")
        print(f"   ssh_avg={row['ssh_avg']:.4f}, margin_vs_negatives={row['margin_vs_negatives']:.2f}")

    print(
        "\nNote: Local pass = high logprob under ode.pt; live pass = unknown Go rules. "
        "Large margin vs negatives means confidence; similar scores among top rows mean "
        "you are still deciphering normalization/prefix, not ready to burn attempts."
    )


if __name__ == "__main__":
    main()