package tests

import (
	"reflect"
	"strings"
	"testing"

	"webtyp.com/context"
)

// TestGenerateStream_ChunksJoinToText verifies stream chunks joined equal full response text.
func TestGenerateStream_ChunksJoinToText(t *testing.T) {
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
			t.Fatalf("case %q not found", exp.Name)
		}

		t.Run(exp.Name, func(t *testing.T) {
			req := requestFor(c, exp)

			var chunks []string
			respStream, err := m.GenerateStream(context.Background(), req, func(text string) {
				if text == "" {
					t.Errorf("received empty stream chunk")
				}
				chunks = append(chunks, text)
			})
			if err != nil {
				t.Fatalf("GenerateStream failed: %v", err)
			}

			joined := strings.Join(chunks, "")
			if joined != respStream.Text {
				t.Errorf("joined stream chunks %q != Response.Text %q", joined, respStream.Text)
			}

			respGen, err := m.Generate(context.Background(), req)
			if err != nil {
				t.Fatalf("Generate failed: %v", err)
			}

			if respStream.Text != respGen.Text {
				t.Errorf("Stream Text %q != Generate Text %q", respStream.Text, respGen.Text)
			}
			if respStream.StopReason != respGen.StopReason {
				t.Errorf("Stream StopReason %q != Generate StopReason %q", respStream.StopReason, respGen.StopReason)
			}
		})
	}
}

// TestGenerateStream_WholeCharacters verifies stream chunks hold back incomplete UTF-8 characters.
func TestGenerateStream_WholeCharacters(t *testing.T) {
	m := loadTiny(t)
	cases := loadCases(t)

	caseMap := make(map[string]chatCase)
	for _, c := range cases {
		caseMap[c.Name] = c
	}

	t.Run("plain", func(t *testing.T) {
		c := caseMap["plain"]
		req := requestFor(c, expectedCase{MaxOutputTokens: 6})

		var chunks []string
		_, err := m.GenerateStream(context.Background(), req, func(text string) {
			chunks = append(chunks, text)
		})
		if err != nil {
			t.Fatalf("GenerateStream failed: %v", err)
		}

		want := []string{"しました", "しました", "しました"}
		if !reflect.DeepEqual(chunks, want) {
			t.Errorf("chunks = %q, want %q", chunks, want)
		}
	})

	t.Run("no_system", func(t *testing.T) {
		c := caseMap["no_system"]
		req := requestFor(c, expectedCase{MaxOutputTokens: 6})

		var chunks []string
		_, err := m.GenerateStream(context.Background(), req, func(text string) {
			chunks = append(chunks, text)
		})
		if err != nil {
			t.Fatalf("GenerateStream failed: %v", err)
		}

		want := []string{"しました", "\xe9\x8c", "\xe9\x8c", "\xe9\x8c ll", " ll"}
		if !reflect.DeepEqual(chunks, want) {
			t.Errorf("chunks = %q, want %q", chunks, want)
		}
	})
}
