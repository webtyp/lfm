package lfm

import (
	"webtyp.com/context"
	"webtyp.com/llm"
	"webtyp.com/tokenizer"
)

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
func (m *Model) GenerateStream(ctx *context.Context, req llm.Request, onText func(text string)) (llm.Response, error) {
	prompt, err := m.promptIDs(req)
	if err != nil {
		return llm.Response{}, err
	}

	st := m.dec.NewState()
	logits := make([]float32, len(m.vocab))

	for i, id := range prompt {
		var stepLogits []float32
		if i == len(prompt)-1 {
			stepLogits = logits
		}
		if err := m.dec.Step(st, int(id), stepLogits); err != nil {
			return llm.Response{}, err
		}
	}

	max := req.MaxOutputTokens
	if max <= 0 {
		max = defaultMaxOutputTokens
	}

	var text tokenizer.Stream
	generated := 0
	reason := llm.StopMaxTokens

	for generated < max {
		next := m.pick(logits)
		if next == imEndID || next == endOfTextID {
			reason = llm.StopEndTurn
			break
		}

		generated++
		if chunk := text.Write(tokenizer.Lfm2Scheme{}.DecodeToken(nil, m.vocab[next])); chunk != "" && onText != nil {
			onText(chunk)
		}

		if err := m.dec.Step(st, next, logits); err != nil {
			return llm.Response{}, err
		}
	}

	if tail := text.Flush(); tail != "" && onText != nil {
		onText(tail)
	}

	return llm.Response{
		Text:       text.Text(),
		StopReason: reason,
		Usage: llm.Usage{
			InputTokens:  len(prompt),
			OutputTokens: generated,
		},
	}, nil
}

// pick selects the highest-scoring token id whose control flag is false.
func (m *Model) pick(logits []float32) int {
	best := -1
	bestScore := float32(-1e30)
	for id, score := range logits {
		if m.control[id] {
			continue
		}
		if best == -1 || score > bestScore {
			best = id
			bestScore = score
		}
	}
	return best
}
