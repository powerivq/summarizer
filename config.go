package summarizer

type APIType string

const (
	APITypeOpenAI APIType = "OPEN_AI"
	APITypeAzure  APIType = "AZURE"
)

type AccessConfig struct {
	AuthToken string
	BaseURL   string
	APIType   APIType
}

type Config struct {
	AccessConfigs []AccessConfig
}
