// Command gen_expected writes testdata/tiny_expected.json: what lfm must answer for every case of
// testdata/chat_template_cases.json on the tiny checkpoint, computed straight from webtyp/decoder
// (greedy, the rules of docs/ARCHITECTURE.md), without lfm itself. From the module root:
//
//	go run ./testdata/gen_expected
package main

import (
	"encoding/json"
	"os"

	"webtyp.com/decoder"
	"webtyp.com/tokenizer"
	"webtyp.com/weights"
)

const maxOutputTokens = 6

type expected struct {
	Name            string `json:"name"`
	MaxOutputTokens int    `json:"max_output_tokens"`
	InputTokens     int    `json:"input_tokens"`
	OutputIDs       []int  `json:"output_ids"`
	Text            []byte `json:"text"` // base64: a token may hold half a UTF-8 character
	StopReason      string `json:"stop_reason"`
}

func main() {
	data, err := os.ReadFile("testdata/tiny_lfm2.wtypw")
	check(err)
	art, err := weights.Open(data)
	check(err)
	cfg := decoder.Config{Arch: decoder.LFM2, Vocab: 65536, Hidden: 32, Intermediate: 64,
		Layers: []decoder.LayerKind{decoder.ShortConv, decoder.FullAttention, decoder.ShortConv, decoder.FullAttention},
		Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 8, RopeTheta: 1e6, ConvKernel: 3, Eps: 1e-5}
	m, err := decoder.New(cfg, art, "model.")
	check(err)
	var cases []struct {
		Name    string `json:"name"`
		SafeIDs []int  `json:"safe_ids"`
	}
	raw, err := os.ReadFile("testdata/chat_template_cases.json")
	check(err)
	check(json.Unmarshal(raw, &cases))

	vocab := art.Tokenizer.Vocab
	var out []expected
	for _, c := range cases {
		st := m.NewState()
		logits := make([]float32, cfg.Vocab)
		for _, id := range c.SafeIDs {
			check(m.Step(st, id, logits))
		}
		e := expected{Name: c.Name, MaxOutputTokens: maxOutputTokens, InputTokens: len(c.SafeIDs), StopReason: "max_tokens", OutputIDs: []int{}}
		for len(e.OutputIDs) < maxOutputTokens {
			next := -1
			for i, v := range vocab {
				control := len(v) >= 4 && v[:2] == "<|" && v[len(v)-2:] == "|>"
				if control && i != 7 && i != 2 {
					continue
				}
				if next < 0 || logits[i] > logits[next] {
					next = i
				}
			}
			if next == 7 || next == 2 {
				e.StopReason = "end_turn"
				break
			}
			e.OutputIDs = append(e.OutputIDs, next)
			e.Text = tokenizer.Lfm2Scheme{}.DecodeToken(e.Text, vocab[next])
			check(m.Step(st, next, logits))
		}
		out = append(out, e)
	}
	b, err := json.MarshalIndent(out, "", " ")
	check(err)
	check(os.WriteFile("testdata/tiny_expected.json", b, 0o644))
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
