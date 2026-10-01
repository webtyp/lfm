package tests

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"webtyp.com/context"
	"webtyp.com/lfm"
	"webtyp.com/weights"
)

const lfmModelDirEnv = "LFM_MODEL_DIR"

// TestRealModel_AnswersFromData tests LFM2.5-350M against real model weights if present.
func TestRealModel_AnswersFromData(t *testing.T) {
	dir := os.Getenv(lfmModelDirEnv)
	if dir == "" {
		t.Skipf("set %s to run against the real LFM2.5-350M", lfmModelDirEnv)
	}

	wPath := filepath.Join(dir, "lfm2.5-350m.wtypw")
	wBytes, err := os.ReadFile(wPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", wPath, err)
	}

	art, err := weights.Open(wBytes)
	if err != nil {
		t.Fatalf("failed to open real weights: %v", err)
	}

	mPath := filepath.Join(dir, "lfm2.5-350m.merges")
	mBytes, err := os.ReadFile(mPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", mPath, err)
	}

	model, err := lfm.New(lfm.Config{
		Weights: art,
		Merges:  mBytes,
		Decoder: lfm.LFM25_350M,
	})
	if err != nil {
		t.Fatalf("lfm.New failed for real model: %v", err)
	}

	cases := loadCases(t)
	var dataCase *chatCase
	for _, c := range cases {
		if c.Name == "data_question" {
			dataCase = &c
			break
		}
	}
	if dataCase == nil {
		t.Fatalf("case data_question not found")
	}

	req := requestFor(*dataCase, expectedCase{MaxOutputTokens: 40})
	resp, err := model.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed for real model: %v", err)
	}

	if !strings.Contains(resp.Text, "09:30") {
		t.Errorf("real model answer = %q, want it to contain \"09:30\"", resp.Text)
	}
}
