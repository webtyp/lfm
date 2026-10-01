package lfm

import (
	"webtyp.com/fmt"
	"webtyp.com/llm"
)

const assistantHeader = "assistant\n"

// promptIDs renders req in LFM2's chat format (docs/ARCHITECTURE.md, "The prompt"): the markers
// as their ids, all other text, including everything a person wrote, as ordinary text.
func (m *Model) promptIDs(req llm.Request) ([]int32, error) {
	if len(req.Tools) > 0 {
		return nil, fmt.Err(errToolsNotSupported)
	}

	for _, msg := range req.Messages {
		if msg.Role != llm.RoleSystem && msg.Role != llm.RoleUser && msg.Role != llm.RoleAssistant {
			return nil, fmt.Errf(errRoleNotSupported, string(msg.Role))
		}
		if len(msg.ToolCalls) > 0 {
			return nil, fmt.Err(errToolCallsInHistory)
		}
	}

	ids := []int32{startOfTextID}
	if req.System != "" {
		ids = m.message(ids, string(llm.RoleSystem), req.System)
	}
	for _, msg := range req.Messages {
		ids = m.message(ids, string(msg.Role), msg.Content)
	}
	ids = append(ids, imStartID)
	ids = m.bpe.EncodeOrdinary(ids, assistantHeader)
	return ids, nil
}

// message appends <|im_start|>role\ncontent<|im_end|>\n. role, "\n" and content are encoded as
// one ordinary text, as transformers does (it splits text only at control tokens).
func (m *Model) message(ids []int32, role, content string) []int32 {
	ids = append(ids, imStartID)
	ids = m.bpe.EncodeOrdinary(ids, role+"\n"+content)
	ids = append(ids, imEndID)
	return m.bpe.EncodeOrdinary(ids, "\n")
}
