package summarizer

type PalmRequestPrompt struct {
	Text string `json:"text"`
}

type PalmRequest struct {
	Prompt          PalmRequestPrompt        `json:"prompt"`
	Temperature     float32                  `json:"temperature"`
	CandidateCount  int                      `json:"candidate_count"`
	TopK            int                      `json:"top_k"`
	TopP            float32                  `json:"top_p"`
	MaxOutputTokens int                      `json:"max_output_tokens"`
	StopSequences   []string                 `json:"stop_sequences"`
	SafetySettings  []map[string]interface{} `json:"safety_settings"`
}

type PalmResponseCandidate struct {
	Output string `json:"output"`
}

type PalmResponse struct {
	Candidates []PalmResponseCandidate `json:"candidates"`
}
