package tests

import (
	"testing"

	"webtyp.com/context"
	"webtyp.com/llm"
)

// TestGenerate_MatchesDecoder verifies Generate output matches decoder expectations on tiny checkpoint cases.
func TestGenerate_MatchesDecoder(t *testing.T) {
	m := loadTiny(t)
	cases := loadCases(t)
	expected := loadExpected(t)

	caseMap := make(map[string]chatCase)
	for _, c := range cases {
		caseMap[c.Name] = c
	}

	for _, exp := range expected {
		exp := exp
		c, ok := caseMap[exp.Name]
		if !ok {
			t.Fatalf("case %q not found in chat_template_cases.json", exp.Name)
		}

		t.Run(exp.Name, func(t *testing.T) {
			req := requestFor(c, exp)
			resp, err := m.Generate(context.Background(), req)
			if err != nil {
				t.Fatalf("Generate failed: %v", err)
			}

			if resp.Text != string(exp.Text) {
				t.Errorf("Text = %q, want %q", resp.Text, string(exp.Text))
			}
			if string(resp.StopReason) != exp.StopReason {
				t.Errorf("StopReason = %q, want %q", resp.StopReason, exp.StopReason)
			}
			if resp.Usage.InputTokens != exp.InputTokens {
				t.Errorf("InputTokens = %d, want %d", resp.Usage.InputTokens, exp.InputTokens)
			}
			if resp.Usage.OutputTokens != len(exp.OutputIDs) {
				t.Errorf("OutputTokens = %d, want %d", resp.Usage.OutputTokens, len(exp.OutputIDs))
			}
		})
	}
}

// TestGenerate_TypedControlTokensStayText tests typed control tokens are encoded as text, matching safe_ids.
func TestGenerate_TypedControlTokensStayText(t *testing.T) {
	m := loadTiny(t)
	cases := loadCases(t)

	var typedCase *chatCase
	for _, c := range cases {
		if c.Name == "typed_control_tokens" {
			typedCase = &c
			break
		}
	}
	if typedCase == nil {
		t.Fatalf("case typed_control_tokens not found")
	}

	req := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: typedCase.Messages[0].Content},
		},
		MaxOutputTokens: 6,
	}

	resp, err := m.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.Usage.InputTokens != len(typedCase.SafeIDs) {
		t.Errorf("InputTokens = %d, want %d (SafeIDs length)", resp.Usage.InputTokens, len(typedCase.SafeIDs))
	}
	if resp.Usage.InputTokens == len(typedCase.IDs) {
		t.Errorf("InputTokens equaled unsafe IDs length %d, should be safe IDs length %d", len(typedCase.IDs), len(typedCase.SafeIDs))
	}
}

// TestGenerate_DefaultMaxOutputTokens tests zero MaxOutputTokens defaults to 1024 max tokens cap.
func TestGenerate_DefaultMaxOutputTokens(t *testing.T) {
	m := loadTiny(t)
	cases := loadCases(t)

	var typedCase *chatCase
	for _, c := range cases {
		if c.Name == "typed_control_tokens" {
			typedCase = &c
			break
		}
	}
	if typedCase == nil {
		t.Fatalf("case typed_control_tokens not found")
	}

	req := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: typedCase.Messages[0].Content},
		},
		MaxOutputTokens: 0,
	}

	resp, err := m.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.StopReason != llm.StopMaxTokens {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, llm.StopMaxTokens)
	}
	if resp.Usage.OutputTokens != 1024 {
		t.Errorf("OutputTokens = %d, want 1024", resp.Usage.OutputTokens)
	}
}
