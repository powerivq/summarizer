package summarizer

type GeminiRequestContentsMessagePart struct {
	Text string `json:"text"`
}

type GeminiRequestContentsMessage struct {
	Role string `json:"role"`
	Parts []GeminiRequestContentsMessagePart `json:"parts"`
}

type GeminiRequestGenerationConfig struct {
	Temperature     float32                  `json:"temperature"`
	TopK            int                      `json:"top_k"`
	TopP            float32                  `json:"top_p"`
	MaxOutputTokens int                      `json:"max_output_tokens"`
	StopSequences   []string                 `json:"stop_sequences"`
}

type GeminiRequest struct {
	Messages []GeminiRequestContentsMessage       `json:"contents"`
	Config          GeminiRequestGenerationConfig `json:"generationConfig"`
	SafetySettings  []map[string]interface{}      `json:"safetySettings"`
}

type GeminiResponseCandidateContentPart struct {
	Text string `json:"text"`
}

type GeminiResponseCandidateContent struct {
	Parts []GeminiResponseCandidateContentPart `json:"parts"`
}

type GeminiResponseCandidate struct {
	Content GeminiResponseCandidateContent `json:"content"`
}

type GeminiResponse struct {
	Candidates []GeminiResponseCandidate `json:"candidates"`
}
