# lfm

The LFM2 language models (LiquidAI's LFM2.5-350M) for webtyp, as a **writer**: it implements
`llm.Client`, `llm.Streamer` and `llm.TokenCounter` in Go, running in the browser under TinyGo, by
composing `webtyp/tokenizer`, `webtyp/weights` and `webtyp/decoder`. It writes text and calls no
tools.

## Getting started

```go
import "webtyp.com/lfm"

art, err := weights.Open(artifactBytes) // lfm2.5-350m.wtypw; its tokenizer vocabulary is inside
model, err := lfm.New(lfm.Config{
    Weights: art,
    Merges:  mergesBytes, // lfm2.5-350m.merges
    Decoder: lfm.LFM25_350M,
})
answer, err := model.Generate(ctx, llm.Request{System: "…", Messages: msgs, MaxOutputTokens: 128})
```

How to produce the two files: [docs/ARCHITECTURE.md → Weights](docs/ARCHITECTURE.md#weights).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the prompt format, how generation stops, streaming, weights.
