# Arcus Ode Triunfal Investigation

This is a method-focused write-up for the Augusta Labs `Ode Triunfal` challenge. It documents how I approached the checkpoint, what the model seems to know, which candidates failed, and what remains uncertain.

I intentionally omit any live proof id from this public version while the challenge is active.

## 1. Checkpoint Reverse Engineering

The public artifact is a PyTorch checkpoint, `ode.pt`. Loading it shows only three top-level keys:

```text
model
model_config
config
```

There are no corpus shards, optimizer state, training logs, or plaintext flag files. The checkpoint zip contains a metadata pickle plus raw tensor blobs.

The state dict names identify the architecture as Karpathy nanoGPT / GPT-2 style:

```text
transformer.wte.weight
transformer.wpe.weight
transformer.h.N.attn.c_attn.weight
transformer.h.N.attn.c_proj.weight
transformer.h.N.mlp.c_fc.weight
transformer.h.N.mlp.c_proj.weight
transformer.ln_f.weight
lm_head.weight
```

The model configuration:

```text
vocab_size: 262
block_size: 1024
n_layer: 10
n_head: 8
n_embd: 640
bias: False
```

This is a compact decoder-only transformer, large enough to memorize details from a small literary corpus.

The checkpoint metadata names the artifact:

```text
luso_lit_lm_player_v2
```

and reports about 22.8 MB of UTF-8 text:

```text
train: 18,042,104 bytes
val:    2,412,168 bytes
test:   2,384,167 bytes
total: 22,838,439 bytes
```

## 2. Tokenizer and Special Tokens

The tokenizer is:

```text
utf8_bytes_with_greedy_special_tokens
```

It has byte tokens 0-255 plus six special tokens:

```text
256 <|fernando_pessoa|>
257 <|alberto_caeiro|>
258 <|ricardo_reis|>
259 <|bernardo_soares|>
260 _
261 {
```

The heteronym tokens are the first major clue. The challenge quotes `Ode Triunfal`, which is by Alvaro de Campos, but the tokenizer includes Pessoa, Alberto Caeiro, Ricardo Reis, and Bernardo Soares while omitting Alvaro de Campos.

The `_` and `{` tokens are also conspicuous because both already exist as byte tokens. Their embedding rows are identical to the corresponding byte rows:

```text
byte "_" row 95  == special "_" row 260
byte "{" row 123 == special "{" row 261
```

So they are aliases rather than independent learned symbols. Still, adding aliases for `_` and `{` is strange unless a flag-like construction was expected during puzzle design.

## 3. Missing Alvaro de Campos Marker

The central experiment was to synthesize the omitted marker:

```text
<|alvaro_de_campos|>
```

Because `_` is greedily tokenized as special id 260, this marker encodes as:

```text
[60, 124, 97, 108, 118, 97, 114, 111, 260, 100, 101, 260, 99, 97, 109, 112, 111, 115, 124, 62]
```

Prompting:

```text
<|alvaro_de_campos|>flag
```

puts `{` at the top of the next-token distribution. The byte `{` and special `{` tie because their embedding rows are identical:

```text
special "{" id 261: p ~= 0.325975
byte "{"    id 123: p ~= 0.325975
```

This is the strongest evidence that the intended route is "prompt as the omitted heteronym."

## 4. Model Probing Tool

I built a local Gradio app, `Arcus - Fernandinho Pessoa`, to make the investigation reproducible.

The app implements:

```text
checkpoint evidence extraction
byte/special-token encoding and decoding
greedy generation
temperature/top-k sampling
top-next token tables
token-by-token greedy traces
candidate logprob scoring
normalization variant generation
write-up drafting
```

This helped avoid a real failure mode: confusing Python token labels with generated text. For example, `'H'` in a table is not a literal apostrophe-H-apostrophe sequence. The app displays decoded text, escaped text, and token ids separately.

## 5. Candidate Scoring and Failed Normalizations

Using the scoring prefix:

```text
<|alvaro_de_campos|>flag{
```

the greedy continuation begins:

```text
Hup-la... He-ha... He-ho... Z-z-z-z...

[EPSON W-02]-z-z...
```

The model is extremely deterministic along this path. That is expected: greedy decoding always chooses the highest-logit token, and the relevant probabilities are often above 0.99.

Representative candidate scores after the same prefix:

```text
Hup-la... He-ha... He-ho... Z-z-z-z...   avg logprob ~= -0.413
hup_la_hup_la_hup_la_ho_hup_la           avg logprob ~= -4.024
hup_la_he_ha_he_ho_z_z_z_z               avg logprob ~= -5.336
EPSON_W-02                               avg logprob ~= -5.930
EPSON                                    avg logprob ~= -6.693
ode_triunfal                             avg logprob ~= -8.002
i_ode_triunfal                           avg logprob ~= -7.737
```

The raw chant is far stronger under the checkpoint than slug-normalized candidates. However, the live prompt already says `flag:`, so body-only variants and server-side normalization remain plausible.

Failed or weak normalization directions included:

```text
hup_la_he_ha_he_ho_z_z_z_z
hupla_heha_heho_zzzz
hup_la_hup_la_hup_la_ho_hup_la
ode_triunfal
i_ode_triunfal
EPSON
EPSON_W-02
```

The `[EPSON W-02]` continuation is interesting but likely not the answer. It is very strong after the chant, but weak as a direct flag body. My read is that the model maps the industrial/machine-age poem into a machine-brand gloss and then loops.

## 6. What Remains Uncertain

The model strongly memorizes the route:

```text
<|alvaro_de_campos|>flag{
```

into a flag-shaped canary. But greedy decoding does not emit a clean `_` or closing `}`. That leaves several possibilities:

```text
the canary was only partially learned
the live validator applies a normalization I have not matched
the expected input is body-only, not flag{body}
the SSH challenge wraps the prompt differently from the local checkpoint experiments
the proof id is optional and the write-up/tool is the main evaluation artifact
```

The strongest confirmed insight is the omitted Alvaro de Campos marker. The exact accepted proof string remains unresolved in this public write-up.
