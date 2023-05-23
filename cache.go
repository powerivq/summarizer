package summarizer

type Cache interface {
	Get(key string) *string
	Set(key string, value string)
}
