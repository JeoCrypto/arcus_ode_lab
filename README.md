# Arcus Ode Lab

A public model-forensics lab for Augusta Labs' `Ode Triunfal` challenge.

This repo documents an investigation into a small Portuguese-literature language-model checkpoint. The goal is not just to guess strings at the `flag:` prompt, but to make the reasoning inspectable: checkpoint structure, tokenizer design, prompt probes, candidate scoring, failed normalizations, and remaining uncertainty.

Live challenge note: if a final proof id is recovered while the challenge is active, it is intentionally omitted from this public repo.

## What This Builds

`app.py` launches a local Gradio interface named `Arcus - Fernandinho Pessoa`.

It can:

- reverse-engineer the checkpoint metadata and tensor map
- inspect the custom byte/special-token tokenizer
- generate from arbitrary prompts
- show top-next token probabilities
- trace greedy decoding token by token
- score candidate answers by log probability
- beam-search for flag completions that close with `}`
- generate normalization variants
- build a method-focused write-up draft

## Why This Exists

The challenge quotes Fernando Pessoa's `Ode Triunfal`, written under the heteronym Alvaro de Campos. The checkpoint tokenizer contains special tokens for Pessoa and several heteronyms, but not Alvaro de Campos. That asymmetry became the central hypothesis.

The repo is a proof-of-work artifact: a small tool built during the investigation, not a polished product and not a spoiler dump.

## Setup

The checkpoint is not included. Scripts auto-detect, in order:

1. `ODE_CKPT` environment variable  
2. `./model/ode.pt` (recommended)  
3. `./ode.pt` (repo root or current directory)  
4. `~/Downloads/ode.pt`

Override explicitly if needed:

```bash
export ODE_CKPT=/path/to/ode.pt
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run:

```bash
python3 app.py
```

Open:

```text
http://127.0.0.1:7860
```

## Files

```text
app.py              Gradio lab and PyTorch model loader
scripts/submit_flag.expect   SSH flag submitter (expect)
WRITEUP.md          investigation write-up draft
requirements.txt    Python dependencies
.gitignore          excludes checkpoint and local artifacts
```

## Flag investigation (ode.pt + validator + SSH)

Full report: [docs/FLAG_INVESTIGATION.md](docs/FLAG_INVESTIGATION.md).

Quick local run:

```bash
export ODE_CKPT=/path/to/ode.pt
python3 scripts/investigate_flag.py
python3 scripts/local_validator.py
```

## Extract Flag and SSH submit

The live SSH prompt is `flag:` (not `flag{`). The server likely scores against `<|alvaro_de_campos|>flag:` + your text. Under that prefix the memorized answer starts with **`.. He-ha...`**, not `Hup-la...` (that belongs to the `flag{` path). Also try **`Hup-la...`** as attempt 3 (marketing / screenshot path).

In the Gradio app, open **Extract Flag** (prefix defaults to `<|alvaro_de_campos|>flag:`) and click **Extract**.

Try on SSH:

```bash
chmod +x scripts/submit_flag.expect scripts/try_ssh_flags.expect
expect scripts/try_ssh_flags.expect
# or one shot:
expect scripts/submit_flag.expect ".. He-ha... He-ho... Z-z-z-z..."
```

If the TUI menu layout differs, adjust navigation:

```bash
ARCUS_NAV=commands expect scripts/submit_flag.expect "your flag"
ARCUS_MENU_DOWN=2 expect scripts/submit_flag.expect "your flag"
```

Manual fallback: `ssh -tt augustalabs.ai`, select **Ode Triunfal**, paste the body at `flag:`.

## Reproducible Starting Points

Default generation prompt:

```text
<|alvaro_de_campos|>flag
```

Default scoring prefix:

```text
<|alvaro_de_campos|>flag{
```

The app keeps decoded text, escaped text, and token ids separate. This matters because copying token labels like `'H'` or `'\\n'` back into prompts changes the actual input.

## Status

The strongest confirmed finding is the omitted Alvaro de Campos marker and the model's deterministic continuation into a flag-shaped canary. The exact accepted proof string remains unresolved in this public version.

## License

Code in this repo is released under the MIT License. The checkpoint is not included and belongs to its original publisher.
