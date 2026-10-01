---
PLAN: "feat: LFM2.5-350M as llm.Client, llm.Streamer and llm.TokenCounter (writer, no tools)"
TAG: v0.1.0
EXECUTOR: jules
REVIEWER: none
STATUS: running
SESSION: 16982462036494478473
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.

# Plan — `webtyp/lfm` v0.1.0: the LFM2 writer

## Read first

- [docs/ARCHITECTURE.md](ARCHITECTURE.md) is the specification: the prompt format, the generation
  rules, streaming. It is already written and correct. This plan implements it. Do not change it,
  except the one sentence listed in Stage 5.
- This repository is new: `lfm.go` is an empty stub from the repo generator, to be replaced.
- Do **not** ask questions. If something in this plan cannot be done, write what and why in a
  section `## Executor notes` at the end of this file, do everything else, and open the PR.
- Do **not** edit this file's frontmatter (the block between the first two `---` lines).

## Development rules (apply to every file)

- **TinyGo / WASM code:** the package `lfm` is compiled to WebAssembly. In non-test code, do not
  import `errors`, `strings`, `strconv`, `fmt` (stdlib), `sort`, `unicode/utf8`. Use
  `webtyp.com/fmt` for errors (`fmt.Err(msg)`, `fmt.Errf(format, args...)`). Tests may import
  any stdlib package.
- **No hardcoded strings in logic:** every error message is an unexported constant (listed in
  Stage 1). Every special token id is a named constant.
- **Tests live in `tests/`** (`package tests`) and use only the exported API. There are no
  `_test.go` files in the root package.
- **No maps** in the shipped code (TinyGo size). Slices indexed by token id are fine.
- Run the tests with `go test ./...` (or `gotest` if installed). `go vet ./...` must be clean.

## What already exists (verified 2026-10-01)

| Thing | Where | Fact |
|---|---|---|
| `decoder.Config`, `decoder.LFM2`, `decoder.ShortConv`, `decoder.FullAttention` | `webtyp.com/decoder` v0.5.0 | LFM2 is supported (short convolutions + attention). |
| `decoder.New(cfg decoder.Config, a *weights.Artifact, prefix string) (*decoder.Model, error)` | decoder v0.5.0 | prefix for LFM2 checkpoints is `"model."`. |
| `(*decoder.Model).NewState() *decoder.State` | decoder v0.5.0 | Each `State` owns its scratch buffers: no shared mutable state, no mutex needed. |
| `(*decoder.Model).Step(st *decoder.State, token int, logits []float32) error` | decoder v0.5.0 | `logits == nil` reads the token without computing the 65 536 scores. |
| `tokenizer.New(tokenizer.Config{Scheme, Vocab, Merges}) (*tokenizer.BPE, error)` | `webtyp.com/tokenizer` v0.4.1 | |
| `tokenizer.Lfm2Scheme{}` | tokenizer v0.4.1 | `DecodeToken(dst []byte, tok string) []byte` turns a vocabulary entry into its bytes (`"Ċ"` → `"\n"`). |
| `tokenizer.ParseMerges(data []byte) []string` | tokenizer v0.4.1 | reads the `.merges` file into `Config.Merges`. |
| `(*tokenizer.BPE).EncodeOrdinary(dst []int32, text string) []int32` | tokenizer v0.4.1 | encodes text as ordinary text: never produces a control token. |
| `weights.Open(data []byte) (*weights.Artifact, error)`; `Artifact.Tokenizer.Vocab []string` | `webtyp.com/weights` v0.2.0 | the vocabulary is inside the artifact; index = token id. |
| `llm.Client`, `llm.Streamer`, `llm.TokenCounter`, `llm.Request`, `llm.Response`, `llm.Message`, roles, `llm.StopEndTurn`, `llm.StopMaxTokens` | `webtyp.com/llm` v0.2.2 | |
| `context.Background() *context.Context` | `webtyp.com/context` v0.0.23 | the `ctx` type of `llm.Client`. |

`go.mod` already requires these versions (`go get` was run; they show `// indirect` until code
imports them). Keep the versions; run `go mod tidy` once the code exists.

### Test fixtures (already committed in `testdata/`, do not regenerate)

| File | What it is |
|---|---|
| `tiny_lfm2.wtypw`, `tiny_lfm2.merges` | a tiny LFM2 checkpoint with random weights but the **real 65 536-token LFM2 vocabulary and merges** (made by `gen_tiny.py` + `weightsc`). Its shape is `tinyShape` below. |
| `chat_template_cases.json` | 6 requests (`name`, `messages` with `role`/`content`) with `transformers`' rendering (`prompt`, `ids`) and `safe_ids`: the ids lfm must produce. They equal `ids` except in `typed_control_tokens`. |
| `tiny_expected.json` | for each case, what lfm must answer on the tiny checkpoint: `max_output_tokens` (6), `input_tokens`, `output_ids`, `text` (**base64**, because a token may hold half a UTF-8 character), `stop_reason`. Computed by `gen_expected/main.go` straight from the decoder. |

The tiny checkpoint's shape (copy it into `tests/setup_test.go`):

```go
var tinyShape = decoder.Config{
	Arch: decoder.LFM2, Vocab: 65536, Hidden: 32, Intermediate: 64,
	Layers: []decoder.LayerKind{decoder.ShortConv, decoder.FullAttention, decoder.ShortConv, decoder.FullAttention},
	Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 8, RopeTheta: 1e6, ConvKernel: 3, Eps: 1e-5,
}
```

## Design gate

1. **Prior art.** `transformers` (`AutoModelForCausalLM` + `apply_chat_template`): the template
   is data and every model shares one generate loop. llama.cpp (`llama-server`): one server,
   the chat template picked per model, stop on the model's end-of-turn token. Ollama: a
   `Modelfile` names the template and the stop tokens per model. All three keep the format per
   model family and the loop shared. Here, `webtyp/qwen` already is the Qwen family's adapter;
   `lfm` is the LFM2 family's adapter with the same contracts (`llm.Client`, `llm.Streamer`,
   `llm.TokenCounter`), because a TinyGo binary carries only the families it imports: the
   browser Worker of the hybrid agent needs Qwen (decider) and LFM2 (writer), nothing else.
2. **Novice-name test.** `lfm.New(lfm.Config{Weights, Merges, Decoder: lfm.LFM25_350M})`, the
   same words as `qwen.New(qwen.Config{…, Decoder: qwen.Qwen35_08B})`. `LFM25_350M` reads as
   "LFM2.5, 350M".
3. **Complexity ledger.** Concepts +1 (a second model family, already the pattern); files to
   touch to add a writer +0 (the composition root passes it where an `llm.Client` goes); lines
   at the call site: 5; ways to do the same thing +0 (the only other writer path, Qwen's own
   generation, is a different model).
4. **Where it belongs.** One family per repository, like `qwen`: the chat format, control
   tokens and stop rules are LFM2's alone. Tokenizing is `tokenizer`'s, the forward pass is
   `decoder`'s; this repo only translates. It owns no prefix cache: that will be one shared
   piece for `qwen` and `lfm`, not a second copy (ARCHITECTURE "Not yet").
5. **What it deletes.** The generator stub `lfm.go` (`type Lfm struct{}`, `func New() *Lfm`).
   Nothing else exists yet.

## Stage 1 — `lfm.go`: Config, shape, Model, New

Replace the whole file. Content, exactly:

```go
// Package lfm runs LiquidAI's LFM2 language models (LFM2.5-350M) as a writer: it implements
// llm.Client, llm.Streamer and llm.TokenCounter over webtyp/decoder. It calls no tools.
package lfm

import (
	"webtyp.com/decoder"
	"webtyp.com/fmt"
	"webtyp.com/tokenizer"
	"webtyp.com/weights"
)

// Config configures an LFM2 model instance.
type Config struct {
	Weights *weights.Artifact // from webtyp/weightsc -quant int8-block32 -prefix model.; its Tokenizer.Vocab is the vocabulary
	Merges  []byte            // the companion .merges file (one "left right" pair per line, rank order)
	Decoder decoder.Config    // the checkpoint's shape; LFM25_350M for LFM2.5-350M
}

// LFM25_350M is the shape of LFM2.5-350M, from its config.json: 16 layers (layer_types: 10 gated
// short convolutions and 6 grouped-query attention layers), attention 16/8 with 64-wide heads,
// RoPE on the whole head with theta 1e6, convolution width 3 (conv_L_cache).
var LFM25_350M = decoder.Config{
	Arch:         decoder.LFM2,
	Vocab:        65536,
	Hidden:       1024,
	Intermediate: 4608,
	Layers: []decoder.LayerKind{
		decoder.ShortConv, decoder.ShortConv, decoder.FullAttention,
		decoder.ShortConv, decoder.ShortConv, decoder.FullAttention,
		decoder.ShortConv, decoder.ShortConv, decoder.FullAttention,
		decoder.ShortConv, decoder.FullAttention,
		decoder.ShortConv, decoder.FullAttention,
		decoder.ShortConv, decoder.FullAttention,
		decoder.ShortConv,
	},
	Heads:      16,
	KVHeads:    8,
	HeadDim:    64,
	RotaryDim:  64,
	RopeTheta:  1000000,
	ConvKernel: 3,
	Eps:        1e-5,
}

// LFM2's control tokens, by id. The constructor checks the vocabulary agrees.
const (
	startOfTextID = 1 // <|startoftext|>: opens every prompt
	endOfTextID   = 2 // <|endoftext|>: ends the answer
	imStartID     = 6 // <|im_start|>: opens a message
	imEndID       = 7 // <|im_end|>: closes a message; ends the answer
)

const (
	weightsPrefix          = "model."
	defaultMaxOutputTokens = 1024

	errWeightsRequired   = "lfm: Config.Weights is required"
	errVocabRequired     = "lfm: Config.Weights has no tokenizer vocabulary"
	errMergesRequired    = "lfm: Config.Merges is required"
	errVocabSize         = "lfm: the vocabulary has %d tokens but Config.Decoder.Vocab is %d"
	errNotLFM2Vocab      = "lfm: the vocabulary is not LFM2's: token %d is %q, want %q"
	errToolsNotSupported = "lfm: Request.Tools is not supported: this model writes text and calls no tools"
	errRoleNotSupported  = "lfm: message role %q is not supported: system, user and assistant only"
	errToolCallsInHistory = "lfm: assistant messages with ToolCalls are not supported"
)

// Model is an LFM2 language model.
type Model struct {
	dec     *decoder.Model
	bpe     *tokenizer.BPE
	vocab   []string // the vocabulary as the artifact spells it (byte-level: "Ċ" is "\n")
	control []bool   // control[id]: id is a control token that is never written (see ARCHITECTURE)
}

// New creates an LFM2 model from its weights artifact and merges.
func New(cfg Config) (*Model, error) {
	...
}
```

`New` does, in this order, returning the first error:

1. `cfg.Weights == nil` → `fmt.Err(errWeightsRequired)`.
2. `len(cfg.Weights.Tokenizer.Vocab) == 0` → `fmt.Err(errVocabRequired)`.
3. `len(cfg.Merges) == 0` → `fmt.Err(errMergesRequired)`.
4. `len(vocab) != cfg.Decoder.Vocab` → `fmt.Errf(errVocabSize, len(vocab), cfg.Decoder.Vocab)`.
5. For each pair (1 `"<|startoftext|>"`, 2 `"<|endoftext|>"`, 6 `"<|im_start|>"`, 7 `"<|im_end|>"`):
   `vocab[id] != want` → `fmt.Errf(errNotLFM2Vocab, id, vocab[id], want)`. Put the four pairs in
   one unexported table (a slice of structs), not four `if`s.
6. `decoder.New(cfg.Decoder, cfg.Weights, weightsPrefix)`; return its error unchanged.
7. `tokenizer.New(tokenizer.Config{Scheme: tokenizer.Lfm2Scheme{}, Vocab: vocab, Merges: tokenizer.ParseMerges(cfg.Merges)})`; return its error unchanged.
8. Build `control`: `control[id] = true` for every id whose `vocab[id]` has length ≥ 4, starts
   with `"<|"` and ends with `"|>"`, **except** `endOfTextID` and `imEndID`. (On the real
   vocabulary this marks 504 ids.) Write it as a small unexported function
   `isControl(tok string) bool` plus the loop; compare bytes, no `strings`.

## Stage 2 — `render.go`: the prompt as token ids

```go
// promptIDs renders req in LFM2's chat format (docs/ARCHITECTURE.md, "The prompt"): the markers
// as their ids, all other text, including everything a person wrote, as ordinary text.
func (m *Model) promptIDs(req llm.Request) ([]int32, error)
```

Exactly:

1. `len(req.Tools) > 0` → `fmt.Err(errToolsNotSupported)`.
2. For each message: role not one of `llm.RoleSystem`, `llm.RoleUser`, `llm.RoleAssistant` →
   `fmt.Errf(errRoleNotSupported, string(msg.Role))`; `len(msg.ToolCalls) > 0` →
   `fmt.Err(errToolCallsInHistory)`. Check all messages before encoding anything.
3. Build ids:

```go
ids := []int32{startOfTextID}
if req.System != "" {
	ids = m.message(ids, string(llm.RoleSystem), req.System)
}
for _, msg := range req.Messages {
	ids = m.message(ids, string(msg.Role), msg.Content)
}
ids = append(ids, imStartID)
ids = m.bpe.EncodeOrdinary(ids, assistantHeader) // const assistantHeader = "assistant\n"
return ids, nil
```

```go
// message appends <|im_start|>role\ncontent<|im_end|>\n. role, "\n" and content are encoded as
// one ordinary text, as transformers does (it splits text only at control tokens).
func (m *Model) message(ids []int32, role, content string) []int32 {
	ids = append(ids, imStartID)
	ids = m.bpe.EncodeOrdinary(ids, role+"\n"+content)
	ids = append(ids, imEndID)
	return m.bpe.EncodeOrdinary(ids, "\n")
}
```

The `"\n"` literal may stay as a named constant `newline = "\n"`. This exact segmentation was
checked against `safe_ids` for all 6 fixture cases with tokenizer v0.4.1 before writing this
plan: do not change it.

## Stage 3 — `generate.go` and `utf8.go`: generation and streaming

```go
var (
	_ llm.Client       = (*Model)(nil)
	_ llm.Streamer     = (*Model)(nil)
	_ llm.TokenCounter = (*Model)(nil)
)

// CountTokens returns how many tokens text is as ordinary text.
func (m *Model) CountTokens(text string) int { return len(m.bpe.EncodeOrdinary(nil, text)) }

// Generate writes the whole answer to req.
func (m *Model) Generate(ctx *context.Context, req llm.Request) (llm.Response, error) {
	return m.GenerateStream(ctx, req, nil)
}

// GenerateStream writes the answer to req, passing it to onText (when not nil) in chunks that
// end on a character boundary; the chunks joined are Response.Text.
func (m *Model) GenerateStream(ctx *context.Context, req llm.Request, onText func(text string)) (llm.Response, error)
```

`GenerateStream`, exactly:

1. `prompt, err := m.promptIDs(req)`; on error return `llm.Response{}, err`.
2. `st := m.dec.NewState()`; `logits := make([]float32, len(m.vocab))`.
3. Step every prompt id: `m.dec.Step(st, int(id), nil)` for all but the last, and
   `m.dec.Step(st, int(last), logits)` for the last. Return any error.
4. `max := req.MaxOutputTokens`; if `max <= 0`, `max = defaultMaxOutputTokens`.
5. Loop while `generated < max`:
   - `next := m.pick(logits)`: the index of the highest logit among ids with `!m.control[id]`
     (first index wins a tie; start with `best := -1`).
   - `next == imEndID || next == endOfTextID` → stop reason `llm.StopEndTurn`, leave the loop.
     The token is not counted and its text is not added.
   - `generated++`; `text = tokenizer.Lfm2Scheme{}.DecodeToken(text, m.vocab[next])`
     (`text` is a `[]byte`).
   - streaming: see `pending` below.
   - `m.dec.Step(st, next, logits)`; return any error.
6. Leaving the loop because `generated == max` → `llm.StopMaxTokens`.
7. Flush the held-back tail to `onText` (if not empty and `onText != nil`).
8. Return `llm.Response{Text: string(text), StopReason: reason, Usage: llm.Usage{InputTokens: len(prompt), OutputTokens: generated}}`.

Streaming bookkeeping: keep `shown int` = how many bytes of `text` were already passed to
`onText`. After each token: `end := completeUTF8(text)`; if `end > shown`, call
`onText(string(text[shown:end]))` and set `shown = end`. Never call `onText` with an empty
string. At the end (step 7), if `shown < len(text)`, pass `text[shown:]`.

`utf8.go`, exactly:

```go
package lfm

// completeUTF8 returns how many leading bytes of b can be shown now: all of b, except a trailing
// lead byte (and its continuation bytes) of a multi-byte character that still misses bytes. A
// byte-level token can end in the middle of a character, and showing half of it prints garbage.
func completeUTF8(b []byte) int {
	for back := 1; back <= 3 && back <= len(b); back++ {
		c := b[len(b)-back]
		if c&0xC0 == 0x80 { // continuation byte: the lead is further back
			continue
		}
		need := 1
		switch {
		case c&0xE0 == 0xC0:
			need = 2
		case c&0xF0 == 0xE0:
			need = 3
		case c&0xF8 == 0xF0:
			need = 4
		}
		if need > back {
			return len(b) - back
		}
		return len(b)
	}
	return len(b)
}
```

## Stage 4 — tests (`tests/`, `package tests`)

One concern per file, each test with a one-line comment saying what use case it covers.

### `tests/setup_test.go`

- `tinyShape` (above).
- `loadTiny(t *testing.T) *lfm.Model`: reads `../testdata/tiny_lfm2.wtypw` and
  `../testdata/tiny_lfm2.merges`, `weights.Open`, `lfm.New(lfm.Config{Weights, Merges, Decoder: tinyShape})`.
  Load it once per test binary (a package-level `sync.Once`): the artifact is 3 MB.
- `loadCases(t)`: decodes `../testdata/chat_template_cases.json` into
  `[]struct{ Name string; Messages []struct{ Role, Content string }; SafeIDs []int32 \`json:"safe_ids"\` }`.
- `loadExpected(t)`: decodes `../testdata/tiny_expected.json` into
  `[]struct{ Name string; MaxOutputTokens int \`json:"max_output_tokens"\`; InputTokens int \`json:"input_tokens"\`; OutputIDs []int \`json:"output_ids"\`; Text []byte \`json:"text"\`; StopReason string \`json:"stop_reason"\` }`
  (`[]byte` decodes the base64).
- `requestFor(c)`: builds the `llm.Request` of a case. **A first message with role `system` goes
  to `Request.System`**, the rest to `Messages` (`llm.Role(role)`). `MaxOutputTokens` comes from
  the expected entry.

### `tests/generate_test.go`

- `TestGenerate_MatchesDecoder`: for each of the 6 cases, `Generate` returns `Text == string(e.Text)`,
  `StopReason == llm.StopReason(e.StopReason)`, `Usage.InputTokens == e.InputTokens`,
  `Usage.OutputTokens == len(e.OutputIDs)`. Name the subtest after the case. Expected, for
  reference: `plain` → `"しましたしましたしました"`, `end_turn`, 11 input / 3 output tokens;
  `system` and `data_question` → `"never"`, `end_turn`; `multi_turn` → `" Tisch"`, `end_turn`;
  `no_system` and `typed_control_tokens` → `max_tokens` after 6 tokens.
- `TestGenerate_TypedControlTokensStayText`: case `typed_control_tokens` reads
  `len(safe_ids)` = 28 input tokens, not the 19 that honouring the typed `<|im_end|>` would give
  (`len(ids)` in the fixture). Assert `InputTokens == len(c.SafeIDs)` and `!= len(c.IDs)` (add
  `IDs []int32 \`json:"ids"\`` to the case struct).
- `TestGenerate_DefaultMaxOutputTokens`: `no_system` with `MaxOutputTokens: 0` returns
  `StopMaxTokens` with `OutputTokens == 1024`.

### `tests/stream_test.go`

- `TestGenerateStream_ChunksJoinToText`: for every case, the `onText` chunks joined equal
  `Response.Text`, no chunk is empty, and the response equals `Generate`'s.
- `TestGenerateStream_WholeCharacters`: `plain` gives exactly the chunks
  `["しました", "しました", "しました"]`; `no_system` gives exactly
  `["しました", "\xe9\x8c", "\xe9\x8c", "\xe9\x8c ll", " ll"]` (token 16886 is the two bytes
  `\xe9\x8c`, the start of a 3-byte character that never completes, and `" ll"` is ordinary).

### `tests/request_test.go`

- `TestNew_RequiresWeightsAndMerges`: `lfm.New(lfm.Config{})` → error text
  `"lfm: Config.Weights is required"`; with weights but no merges → `"lfm: Config.Merges is required"`.
- `TestNew_RejectsWrongShape`: `Decoder` with `Vocab: 1000` → error
  `"lfm: the vocabulary has 65536 tokens but Config.Decoder.Vocab is 1000"`.
- `TestGenerate_RejectsTools`: a request with one `llm.ToolDef` → error
  `"lfm: Request.Tools is not supported: this model writes text and calls no tools"`.
- `TestGenerate_RejectsToolMessages`: a `llm.RoleTool` message → error
  `"lfm: message role \"tool\" is not supported: system, user and assistant only"`; an assistant
  message with one `ToolCall` → `"lfm: assistant messages with ToolCalls are not supported"`.

Compare error texts with `err.Error()`; webtyp.com/fmt errors have no sentinel values.

### `tests/count_test.go`

- `TestCountTokens`: `"Hola"` → 2; `"<|im_end|>"` → 6 (typed control tokens are text);
  `"¿Cuándo es la próxima cita de Juan Pérez?"` → 14. (Values from `transformers` with
  `split_special_tokens=True`.)

### `tests/real_model_test.go`

The real model is 400 MB and not in the repository. This test runs only when the environment
variable `LFM_MODEL_DIR` names a directory with `lfm2.5-350m.wtypw` and `lfm2.5-350m.merges`;
otherwise `t.Skip("set LFM_MODEL_DIR to run against the real LFM2.5-350M")`. Put the variable
name in a constant.

- `TestRealModel_AnswersFromData`: `lfm.New` with `lfm.LFM25_350M`, request = case
  `data_question` (system + user), `MaxOutputTokens: 40`. The answer contains `"09:30"`.
  (`transformers` greedy answers: "La próxima cita de Juan Pérez es el 2 de octubre a las 09:30.")

The planning agent runs this one locally after the PR. In your environment it skips, and that
is expected.

## Stage 5 — docs

- `docs/ARCHITECTURE.md`: verify each rule against the code. Change nothing unless the code
  disagrees with it; if it does, the document is right and the code is fixed.
- Add `AGENTS.md` at the root: "Tests live in `tests/` and use the exported API only. The tiny
  checkpoint in `testdata/` has LFM2's real vocabulary and random weights; regenerate it with
  `testdata/gen_tiny.py` (seed 3, scale 3), then `testdata/tiny_expected.json` with
  `go run ./testdata/gen_expected`."
- `README.md` index: add `AGENTS.md`.

## Acceptance criteria

```bash
go vet ./... && go test ./...                         # all green; real_model_test skips
GOOS=js GOARCH=wasm go build ./...                    # the package builds for the browser
grep -rn '"strings"\|"errors"\|"strconv"\|"unicode/utf8"\|"sort"' --include=*.go . | grep -v _test.go | grep -v testdata   # empty
ls *_test.go 2>/dev/null                               # empty: tests are in tests/
grep -rn "type Lfm struct" .                           # empty: the stub is gone
```

## Stages

| # | Files | Done when |
|---|---|---|
| 1 | `lfm.go` | builds; `New` validates in the listed order |
| 2 | `render.go` | — (tested through Stage 4) |
| 3 | `generate.go`, `utf8.go` | builds for `GOOS=js GOARCH=wasm` |
| 4 | `tests/setup_test.go`, `generate_test.go`, `stream_test.go`, `request_test.go`, `count_test.go`, `real_model_test.go` | all green |
| 5 | `docs/ARCHITECTURE.md`, `AGENTS.md`, `README.md` | acceptance criteria pass |
