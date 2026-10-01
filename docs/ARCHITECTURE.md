# Architecture — `webtyp/lfm`

## What this is

The adapter that turns `webtyp/decoder` into a model that **writes text**: LiquidAI's LFM2 family,
first LFM2.5-350M. The agent speaks `llm.Request` / `llm.Response`, and LFM2 reads and writes text
in its own chat format. This repository translates between the two.

You meet it in the application's composition root: `lfm.New(...)` gives the writer of the hybrid
agent. In that design a decision model (decider-0.8b, through `webtyp/qwen`) chooses the tool and
answers yes/no questions, and LFM2.5-350M only turns data into a sentence in Spanish, when no
template fits. The measurements behind that choice are in
[llm/docs/EFFICIENT_SLM.md](https://github.com/webtyp/llm/blob/main/docs/EFFICIENT_SLM.md):
LFM2.5-350M wrote 10/10 answers from data correctly, the best model under 0.5B.

## What it owns (and nothing else)

| Concern | Detail |
|---|---|
| tokenizer config | byte-level BPE, 65 536 tokens, `tokenizer.Lfm2Scheme`, LFM2's control tokens |
| chat template | renders `llm.Request` into LFM2's `<\|im_start\|>role … <\|im_end\|>` format |
| generation | greedy next token over `decoder`, end of turn, `MaxOutputTokens`, streaming |

It does **not** call tools: a request with `Tools`, a `RoleTool` message or an assistant message
with `ToolCalls` is an error. Tool choice belongs to the decision model.

## The prompt

The format is LFM2.5's `chat_template.jinja`, and `testdata/chat_template_cases.json` holds its
rendering by `transformers` for every case the tests check:

```
<|startoftext|><|im_start|>system
{Request.System}<|im_end|>
<|im_start|>{role}
{content}<|im_end|>
…
<|im_start|>assistant
```

- `<|startoftext|>` (id 1) always opens the prompt.
- The system block appears only when `Request.System` is not empty.
- Every message, including a `RoleSystem` message inside `Messages`, renders as
  `<|im_start|>` + role + `"\n"` + content + `<|im_end|>` + `"\n"`, as the template does for any
  message after the first.
- The markers are written as their token ids (`<|startoftext|>` 1, `<|im_start|>` 6,
  `<|im_end|>` 7). **Everything else, including all text a person wrote, is encoded as ordinary
  text**, so a person who types `<|im_end|>` produces six ordinary tokens, not the control token.
  `transformers` does the opposite (case `typed_control_tokens`): that is how a user would close
  their own turn and open a fake system block.

## Generation

- **Greedy:** the next token is the highest-scoring one. LiquidAI recommends temperature 0.1,
  which is nearly greedy, and greedy makes every answer reproducible.
- **Control tokens are never written.** A vocabulary entry that starts with `<|` and ends with
  `|>` (506 entries: markers and reserved slots) is excluded from the choice, except two that end
  the answer: `<|im_end|>` (7, the end of the turn) and `<|endoftext|>` (2). Either one stops with
  `llm.StopEndTurn`, and its text is not part of the answer.
- **`MaxOutputTokens`** caps the answer (`llm.StopMaxTokens`); 0 means 1024.
- Prompt tokens are read without computing their 65 536 scores (`decoder.Step` with nil logits);
  only the last prompt token's scores are needed.
- **Streaming in whole characters.** A token can hold half of a UTF-8 character (a Japanese or
  accented character split across two tokens). `GenerateStream` holds such a tail back until the
  character is complete, so every `onText` chunk ends on a character boundary, and the chunks
  joined are exactly `Response.Text`.

## Weights

The source of truth is the original safetensors in `~/Dev/LMmodels/LiquidAI/LFM2.5-350M`.
`webtyp/weightsc` converts it once to int8 in blocks of 32 (the layout of GGUF `Q8_0`):

```
weightsc -in . -out lfm2.5-350m.wtypw -merges-out lfm2.5-350m.merges \
    -id lfm2.5-350m -version 1 -quant int8-block32 -prefix model.
```

That gives a 400 MB artifact whose tokenizer vocabulary is inside, plus the merges file. The
shape is `lfm.LFM25_350M`, from the model's `config.json`. Our decoder's greedy output matches
`transformers` 20/20 tokens on the real weights, at 0.163 s per token (native Go, one thread).

## Not yet

- **No prefix cache.** `webtyp/qwen` keeps the decoder state of the system prompt between
  requests. The same cache belongs in one shared place before a second copy is written here.
- **No sampling and no repetition penalty**: greedy only.
