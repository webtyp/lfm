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

	errWeightsRequired    = "lfm: Config.Weights is required"
	errVocabRequired      = "lfm: Config.Weights has no tokenizer vocabulary"
	errMergesRequired     = "lfm: Config.Merges is required"
	errVocabSize          = "lfm: the vocabulary has %d tokens but Config.Decoder.Vocab is %d"
	errNotLFM2Vocab       = "lfm: the vocabulary is not LFM2's: token %d is %q, want %q"
	errToolsNotSupported  = "lfm: Request.Tools is not supported: this model writes text and calls no tools"
	errRoleNotSupported   = "lfm: message role %q is not supported: system, user and assistant only"
	errToolCallsInHistory = "lfm: assistant messages with ToolCalls are not supported"
)

// Model is an LFM2 language model.
type Model struct {
	dec     *decoder.Model
	bpe     *tokenizer.BPE
	vocab   []string // the vocabulary as the artifact spells it (byte-level: "Ċ" is "\n")
	control []bool   // control[id]: id is a control token that is never written (see ARCHITECTURE)
}

var requiredTokens = []struct {
	id   int
	want string
}{
	{startOfTextID, "<|startoftext|>"},
	{endOfTextID, "<|endoftext|>"},
	{imStartID, "<|im_start|>"},
	{imEndID, "<|im_end|>"},
}

// New creates an LFM2 model from its weights artifact and merges.
func New(cfg Config) (*Model, error) {
	if cfg.Weights == nil {
		return nil, fmt.Err(errWeightsRequired)
	}
	vocab := cfg.Weights.Tokenizer.Vocab
	if len(vocab) == 0 {
		return nil, fmt.Err(errVocabRequired)
	}
	if len(cfg.Merges) == 0 {
		return nil, fmt.Err(errMergesRequired)
	}
	if len(vocab) != cfg.Decoder.Vocab {
		return nil, fmt.Errf(errVocabSize, len(vocab), cfg.Decoder.Vocab)
	}
	for _, req := range requiredTokens {
		var tok string
		if req.id < len(vocab) {
			tok = vocab[req.id]
		}
		if tok != req.want {
			return nil, fmt.Errf(errNotLFM2Vocab, req.id, tok, req.want)
		}
	}

	dec, err := decoder.New(cfg.Decoder, cfg.Weights, weightsPrefix)
	if err != nil {
		return nil, err
	}

	bpe, err := tokenizer.New(tokenizer.Config{
		Scheme: tokenizer.Lfm2Scheme{},
		Vocab:  vocab,
		Merges: tokenizer.ParseMerges(cfg.Merges),
	})
	if err != nil {
		return nil, err
	}

	control := make([]bool, len(vocab))
	for id, tok := range vocab {
		if id == endOfTextID || id == imEndID {
			continue
		}
		if isControl(tok) {
			control[id] = true
		}
	}

	return &Model{
		dec:     dec,
		bpe:     bpe,
		vocab:   vocab,
		control: control,
	}, nil
}

func isControl(tok string) bool {
	if len(tok) < 4 {
		return false
	}
	return tok[0] == '<' && tok[1] == '|' && tok[len(tok)-2] == '|' && tok[len(tok)-1] == '>'
}
