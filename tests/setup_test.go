package tests

import (
	"encoding/json"
	"os"
	"sync"
	"testing"

	"webtyp.com/decoder"
	"webtyp.com/lfm"
	"webtyp.com/llm"
	"webtyp.com/weights"
)

var tinyShape = decoder.Config{
	Arch:         decoder.LFM2,
	Vocab:        65536,
	Hidden:       32,
	Intermediate: 64,
	Layers: []decoder.LayerKind{
		decoder.ShortConv,
		decoder.FullAttention,
		decoder.ShortConv,
		decoder.FullAttention,
	},
	Heads:      4,
	KVHeads:    2,
	HeadDim:    8,
	RotaryDim:  8,
	RopeTheta:  1e6,
	ConvKernel: 3,
	Eps:        1e-5,
}

type chatCase struct {
	Name     string `json:"name"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	IDs     []int32 `json:"ids"`
	SafeIDs []int32 `json:"safe_ids"`
}

type expectedCase struct {
	Name            string `json:"name"`
	MaxOutputTokens int    `json:"max_output_tokens"`
	InputTokens     int    `json:"input_tokens"`
	OutputIDs       []int  `json:"output_ids"`
	Text            []byte `json:"text"`
	StopReason      string `json:"stop_reason"`
}

var (
	tinyModel     *lfm.Model
	tinyModelOnce sync.Once
	tinyArt       *weights.Artifact
	tinyMerges    []byte
)

// loadTiny loads the tiny LFM2 test model checkpoint once per test process.
func loadTiny(t *testing.T) *lfm.Model {
	tinyModelOnce.Do(func() {
		wBytes, err := os.ReadFile("../testdata/tiny_lfm2.wtypw")
		if err != nil {
			t.Fatalf("failed to read tiny_lfm2.wtypw: %v", err)
		}
		art, err := weights.Open(wBytes)
		if err != nil {
			t.Fatalf("failed to open tiny weights: %v", err)
		}
		tinyArt = art

		mBytes, err := os.ReadFile("../testdata/tiny_lfm2.merges")
		if err != nil {
			t.Fatalf("failed to read tiny_lfm2.merges: %v", err)
		}
		tinyMerges = mBytes

		m, err := lfm.New(lfm.Config{
			Weights: art,
			Merges:  mBytes,
			Decoder: tinyShape,
		})
		if err != nil {
			t.Fatalf("failed to create lfm model: %v", err)
		}
		tinyModel = m
	})
	return tinyModel
}

func loadTinyWeights(t *testing.T) *weights.Artifact {
	loadTiny(t)
	return tinyArt
}

func loadTinyMerges(t *testing.T) []byte {
	loadTiny(t)
	return tinyMerges
}

// loadCases loads the chat template test cases from fixture JSON.
func loadCases(t *testing.T) []chatCase {
	data, err := os.ReadFile("../testdata/chat_template_cases.json")
	if err != nil {
		t.Fatalf("failed to read chat_template_cases.json: %v", err)
	}
	var cases []chatCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("failed to unmarshal chat_template_cases.json: %v", err)
	}
	return cases
}

// loadExpected loads the expected generation outcomes from fixture JSON.
func loadExpected(t *testing.T) []expectedCase {
	data, err := os.ReadFile("../testdata/tiny_expected.json")
	if err != nil {
		t.Fatalf("failed to read tiny_expected.json: %v", err)
	}
	var expected []expectedCase
	if err := json.Unmarshal(data, &expected); err != nil {
		t.Fatalf("failed to unmarshal tiny_expected.json: %v", err)
	}
	return expected
}

// requestFor constructs an llm.Request for a test case and expected settings.
func requestFor(c chatCase, exp expectedCase) llm.Request {
	req := llm.Request{
		MaxOutputTokens: exp.MaxOutputTokens,
	}

	msgs := c.Messages
	if len(msgs) > 0 && msgs[0].Role == string(llm.RoleSystem) {
		req.System = msgs[0].Content
		msgs = msgs[1:]
	}

	for _, msg := range msgs {
		req.Messages = append(req.Messages, llm.Message{
			Role:    llm.Role(msg.Role),
			Content: msg.Content,
		})
	}

	return req
}
