package tests

import (
	"testing"
)

// TestCountTokens tests CountTokens against known transformers token counts.
func TestCountTokens(t *testing.T) {
	m := loadTiny(t)

	tests := []struct {
		text string
		want int
	}{
		{"Hola", 2},
		{"<|im_end|>", 6},
		{"¿Cuándo es la próxima cita de Juan Pérez?", 14},
	}

	for _, tt := range tests {
		got := m.CountTokens(tt.text)
		if got != tt.want {
			t.Errorf("CountTokens(%q) = %d, want %d", tt.text, got, tt.want)
		}
	}
}
