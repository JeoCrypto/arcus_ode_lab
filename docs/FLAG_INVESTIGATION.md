# Flag investigation report (ode.pt + local validator + SSH)

Generated from automated runs against official checkpoint  
`SHA256:711cb93fead3032abc8fa7eb5007557f490ce7e7a7c75b6f66195d4fdfc4aa88`.

## Executive summary

| Layer | Finding |
|-------|---------|
| **ode.pt** | Memorizes `<\|alvaro_de_campos\|>` → flag-shaped chant; two branches (`.. He-ha...` vs `Hup-la...`). |
| **local_validator.py** | Best SSH body: `.. He-ha... He-ho... Z-z-z-z...` (avg −0.021, margin ~3.6 nats vs negatives). |
| **SSH live** | Real grader on `augustalabs.ai:22` (Go + Bubble Tea). Automation often **misses** `flag:` in alt-screen TUI; manual `ssh -tt` still required to confirm accept/reject. |
| **Web iframe** | `arcus-tui.vercel.app` does **not** validate flags. |

The puzzle answer is almost certainly the **raw chant body** (no `flag{` wrapper on SSH), with prefix  
`<\|alvaro_de_campos\|>flag:` implied server-side. Live rejects reported earlier suggest extra normalization or a **proof id** after success, not a different chant entirely.

---

## 1. Checkpoint (`ode.pt`)

- Vocab 262, byte-level + specials; **Álvaro de Campos** missing → synthesize `<\|alvaro_de_campos\|>` via `_` tokens (id 260).
- Under `<\|alvaro_de_campos\|>flag:` the **first** continuation token is `.` (P≈0.72) → natural body starts with `.. He-ha...`.
- Under `<\|alvaro_de_campos\|>flag{` the canary is **`Hup-la... He-ha... He-ho... Z-z-z-z...`** (avg ≈ −0.001/token without `}`).
- After either canary, **`}` has P≈0**; **`\\n` has P≈0.99** → open canary + EPSON loop, not `flag{...}` closed form.
- Greedy from `<\|alvaro_de_campos\|>flag` emits `{Hup-la...` (model “helpfully” inserts `{`).

---

## 2. Local validator (`scripts/local_validator.py`)

Teacher-forced scoring: `score_ids(encode(prefix), encode(body))`.

### Top bodies at `<\|alvaro_de_campos\|>flag:` (paste at SSH `flag:`)

| Rank | Body | avg logprob | Margin vs negatives |
|------|------|-------------|---------------------|
| 1 | `.. He-ha... He-ho... Z-z-z-z...` | −0.021 | 3.60 |
| 2 | `He-ha... He-ho... Z-z-z-z...` | −0.213 | 3.41 |
| 3 | `Hup-la... He-ha... He-ho... Z-z-z-z...` | −0.242 | 3.38 |
| 4 | `. He-ha... He-ho... Z-z-z-z...` | −0.246 | 3.38 |

Slugs (`hup_la_he_ha...`, `ode_triunfal`, `EPSON`) score **4+ nats worse**.

### Per-hypothesis (strongest path for `Hup-la` chant)

| Hypothesis | `Hup-la...` avg | Notes |
|------------|-----------------|--------|
| `brace_no_close` (`flag{` + body) | **−0.001** | Best fit to weights |
| `ssh_flag_colon` | −0.242 | UI label `flag:` but different continuation |
| `brace_with_close` (+ `}`) | −0.413 | Closing brace rejected by model |

For `.. He-ha...`, **`ssh_flag_colon`** wins (−0.021).

### Recommended SSH attempts (3)

Paste **body only** at the `flag:` prompt:

1. `.. He-ha... He-ho... Z-z-z-z...`
2. `He-ha... He-ho... Z-z-z-z...`
3. `Hup-la... He-ha... He-ho... Z-z-z-z...` (matches marketing / screenshot teaser)

Do **not** submit `flag{...}`, slugs, or EPSON-only unless desperate.

---

## 3. SSH (`ssh augustalabs.ai`)

- Server: **Go** SSH, PTY, title **Arcus**, alt screen.
- Public artifact repo has **no** server source; validation uses server copy of **ode.pt** (assumed).
- Scripts: `submit_flag.expect`, `try_ssh_flags.expect`, `explore_ssh_menu.expect`.
- **Automation note:** expect often times out before `flag:` because the TUI redraws in alt screen without matching simple regexes. Use manual session or increase sleeps / `ARCUS_NAV=commands` (`start` → `arcus` → Enter).

```bash
export ODE_CKPT=/path/to/ode.pt
python3 scripts/local_validator.py
python3 scripts/investigate_flag.py
ssh -tt augustalabs.ai   # manual: ↓ to Ode Triunfal, Enter, paste body
expect scripts/submit_flag.expect ".. He-ha... He-ho... Z-z-z-z..."
```

After a **correct** submit, the server may print a **proof id** (email to `arcus@augustalabs.ai` per Vercel `submission` command). That id is not derivable from ode.pt alone.

---

## 4. Web vs SSH

| Surface | Validates? |
|---------|------------|
| `arcus-tui.vercel.app` (iframe on `/arcus`) | No — static JS |
| `ssh augustalabs.ai` | Yes |
| `augustalabs.ai/ode` | Redirects to GitHub release (weights only) |

---

## 5. Open questions

1. Exact server prefix string (likely `<\|alvaro_de_campos\|>flag:` + body).
2. Whether grader uses **avg** or **sum** logprob threshold.
3. Whether success returns a **proof token** separate from the chant.
4. Why live runs rejected strings that dominate locally (normalization, deployment lag, or attempt cap).

---

## Reproduce

```bash
export ODE_CKPT="$HOME/Downloads/ode.pt"
python3 scripts/investigate_flag.py
python3 scripts/local_validator.py --top 12
```