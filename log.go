package summarizer

type LLMLogger interface {
	Log(prompt string, completion string, api APIType)
}

type NoOpLogger struct{}

func (c NoOpLogger) Log(prompt string, completion string, api APIType) {}
