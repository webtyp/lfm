package tests

import (
	"testing"

	"webtyp.com/context"
	"webtyp.com/lfm"
	"webtyp.com/llm"
)

// TestNew_RequiresWeightsAndMerges tests that New fails when Weights or Merges are missing.
func TestNew_RequiresWeightsAndMerges(t *testing.T) {
	_, err := lfm.New(lfm.Config{})
	if err == nil || err.Error() != "lfm: Config.Weights is required" {
		t.Fatalf("got err %v, want \"lfm: Config.Weights is required\"", err)
	}

	art := loadTinyWeights(t)
	_, err = lfm.New(lfm.Config{Weights: art})
	if err == nil || err.Error() != "lfm: Config.Merges is required" {
		t.Fatalf("got err %v, want \"lfm: Config.Merges is required\"", err)
	}
}

// TestNew_RejectsWrongShape tests that New checks decoder vocabulary size matching artifact.
func TestNew_RejectsWrongShape(t *testing.T) {
	art := loadTinyWeights(t)
	mBytes := loadTinyMerges(t)

	cfg := tinyShape
	cfg.Vocab = 1000

	_, err := lfm.New(lfm.Config{
		Weights: art,
		Merges:  mBytes,
		Decoder: cfg,
	})
	if err == nil || err.Error() != "lfm: the vocabulary has 65536 tokens but Config.Decoder.Vocab is 1000" {
		t.Fatalf("got err %v, want \"lfm: the vocabulary has 65536 tokens but Config.Decoder.Vocab is 1000\"", err)
	}
}

// TestGenerate_RejectsTools tests that Request.Tools is rejected with an error.
func TestGenerate_RejectsTools(t *testing.T) {
	m := loadTiny(t)
	req := llm.Request{
		Tools: []llm.ToolDef{{Name: "test_tool"}},
	}
	_, err := m.Generate(context.Background(), req)
	if err == nil || err.Error() != "lfm: Request.Tools is not supported: this model writes text and calls no tools" {
		t.Fatalf("got err %v, want \"lfm: Request.Tools is not supported: this model writes text and calls no tools\"", err)
	}
}

// TestGenerate_RejectsToolMessages tests that RoleTool and ToolCalls in messages are rejected.
func TestGenerate_RejectsToolMessages(t *testing.T) {
	m := loadTiny(t)

	reqTool := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleTool, Content: "result"},
		},
	}
	_, err := m.Generate(context.Background(), reqTool)
	if err == nil || err.Error() != "lfm: message role \"tool\" is not supported: system, user and assistant only" {
		t.Fatalf("got err %v, want \"lfm: message role \\\"tool\\\" is not supported: system, user and assistant only\"", err)
	}

	reqCalls := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, Content: "", ToolCalls: []llm.ToolCall{{Name: "func"}}},
		},
	}
	_, err = m.Generate(context.Background(), reqCalls)
	if err == nil || err.Error() != "lfm: assistant messages with ToolCalls are not supported" {
		t.Fatalf("got err %v, want \"lfm: assistant messages with ToolCalls are not supported\"", err)
	}
}
