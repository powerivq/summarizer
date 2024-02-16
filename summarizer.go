package summarizer

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"regexp"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/pkoukk/tiktoken-go"
	openai "github.com/sashabaranov/go-openai"
)

var chineseMatcher = regexp.MustCompile("[\u4e00-\u9fa5]")
var englishMatcher = regexp.MustCompile(`[a-zA-Z]`)

type Client struct {
	azureClients  []openai.Client
	openaiClients []openai.Client
	gcpTokens     []string
	tiktoken      tiktoken.Tiktoken
	cache         Cache
	logger        LLMLogger
}

type NoCache struct{}

func (c NoCache) Get(key string) *string {
	return nil
}

func (c NoCache) Set(key string, value string) {
}

func NewClientNoCache(config Config) *Client {
	return NewClient(config, NoCache{}, NoOpLogger{})
}

func NewClient(config Config, cache Cache, logger LLMLogger) *Client {
	var azureClients []openai.Client
	var openaiClients []openai.Client
	var gcpTokens []string
	for _, config := range config.AccessConfigs {
		if config.APIType == APITypeOpenAI {
			openaiClients = append(openaiClients, *openai.NewClient(config.AuthToken))
		}
		if config.APIType == APITypeAzure {
			clientConfig := openai.DefaultAzureConfig(config.AuthToken, config.BaseURL)
			azureClients = append(azureClients, *openai.NewClientWithConfig(clientConfig))
		}
		if config.APIType == APITypeGCPGemini {
			gcpTokens = append(gcpTokens, config.AuthToken)
		}
	}

	tiktoken, err := tiktoken.EncodingForModel("gpt-3.5-turbo")
	if err != nil {
		log.Fatal("Tiktoken failed to load: ", err)
	}
	return &Client{
		azureClients:  azureClients,
		openaiClients: openaiClients,
		gcpTokens:     gcpTokens,
		tiktoken:      *tiktoken,
		cache:         cache,
		logger:        logger,
	}
}

func (c *Client) Summarize(text string) (*string, error) {
	text = PruneInvisibleCharacters(text)
	if len(text) == 0 {
		log.Printf("Empty input, ignoring")
		return &text, nil
	}

	log.Printf("Summarizing %d bytes", len(text))

	chineseChars := len(chineseMatcher.FindAllString(text, -1))
	englishChars := len(englishMatcher.FindAllString(text, -1))

	var prompt string
	if chineseChars >= englishChars {
		prompt = "In Chinese, write a case brief for the following judgment, includes the facts, procedural history, holdings, rationales for each holding, and final disposition: \n\n"
	} else {
		prompt = "Write a case brief for the following judgment, includes the facts, procedural history, holdings, rationales for each holding, and final disposition: \n\n"
	}

	if tokens := len(c.tiktoken.Encode(text, nil, nil)); tokens <= 25000 {
		content, err := c.requestGpt(prompt+text)
		if err != nil {
			return nil, err
		}
		return content, nil
	}

	texts := strings.Split(text, "\n")

	var summary strings.Builder
	var window strings.Builder
	var tokens int

	promptTokens := len(c.tiktoken.Encode(prompt, nil, nil))
	for i := 0; i < len(texts); {
		window.Reset()
		window.WriteString(prompt)
		tokens = promptTokens

		for ; i < len(texts) && tokens <= 24000; i++ {
			window.WriteString(texts[i])
			window.WriteString("\n")
			tokens += len(c.tiktoken.Encode(texts[i], nil, nil)) + 1
		}
		for ; i < len(texts) && !strings.ContainsAny(texts[i-1], ".?!。！") && strings.TrimSpace(texts[i-1]) != ""; i++ {
			window.WriteString(texts[i])
			window.WriteString("\n")
			tokens += len(c.tiktoken.Encode(texts[i], nil, nil)) + 1
		}
		lastChar, size := utf8.DecodeLastRuneInString(strings.TrimSpace(texts[i-1]))
		if i != len(texts) && size == 1 && lastChar != '.' && lastChar != '?' && lastChar != '!' && lastChar != '。' && lastChar != '！' {
			i--
		}

		log.Printf("GPTing for %d tokens", tokens)
		prompt := window.String()

		if cacheResults := c.cache.Get("gpt:" + GetMD5Hash(prompt)); cacheResults != nil {
			log.Printf("Partial result (cached): %d bytes => %d bytes", len(prompt), len(*cacheResults))
			summary.WriteString(*cacheResults)
			summary.WriteString("\n")
		} else {
			content, err := c.requestGpt(prompt)
			if err != nil {
				log.Printf("openai error: %s", err)
				return nil, err
			}

			log.Printf("Partial result: %d bytes => %d bytes", len(prompt), len(*content))
			c.cache.Set("gpt:"+GetMD5Hash(prompt), *content)
			summary.WriteString(*content)
			summary.WriteString("\n")
		}
	}
	return c.Summarize(summary.String())
}

func (c *Client) requestGpt(prompt string) (*string, error) {
	invalidInput := false
	if len(c.azureClients) > 0 && !invalidInput {
		for i := 0; i < 3; i++ {
			client := &c.azureClients[rand.Intn(len(c.azureClients))]
			res, err := c.doRequestGpt(client, prompt)
			if err == nil {
				c.logger.Log(prompt, *res, APITypeAzure)
				return res, nil
			}
			apiError := &openai.APIError{}
			if errors.As(err, &apiError) && apiError.HTTPStatusCode == 400 {
				invalidInput = true
			}
			log.Printf("GPT error: %s", err)
		}
	}
	if len(c.openaiClients) > 0 {
		for i := 0; i < 3; i++ {
			client := &c.openaiClients[rand.Intn(len(c.openaiClients))]
			res, err := c.doRequestGpt(client, prompt)
			if err == nil {
				c.logger.Log(prompt, *res, APITypeOpenAI)
				return res, nil
			}
			log.Printf("GPT error: %s", err)
		}
	}
	if len(c.gcpTokens) > 0 {
		for i := 0; i < 3; i++ {
			token := c.gcpTokens[rand.Intn(len(c.gcpTokens))]
			res, err := c.doRequestGemini(token, prompt)
			if err == nil {
				c.logger.Log(prompt, *res, APITypeGCPGemini)
				return res, nil
			}
			log.Printf("Gemini error: %s", err)
		}
	}
	return nil, errors.New("all retries have failed")
}

func (c *Client) doRequestGpt(client *openai.Client, prompt string) (*string, error) {
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT3Dot5Turbo16K,
			Messages: []openai.ChatCompletionMessage{{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			}},
			Stream:      false,
		},
	)
	if err != nil {
		log.Printf("openai error: %s", err)
		return nil, err
	}

	return &resp.Choices[0].Message.Content, nil
}

func (c *Client) doRequestGemini(token string, prompt string) (*string, error) {
	netTransport := &http.Transport{
		TLSHandshakeTimeout: 10 * time.Second,
	}

	client := &http.Client{
		Timeout:   60 * time.Second,
		Transport: netTransport,
	}

	requestJson := GeminiRequest{
		Messages: []GeminiRequestContentsMessage{
			GeminiRequestContentsMessage{
				Role: "user",
				Parts: []GeminiRequestContentsMessagePart{
					GeminiRequestContentsMessagePart{Text: prompt},
				},
			},
		},
		Config: GeminiRequestGenerationConfig{
			Temperature:     0.9,
			TopK:            1,
			TopP:            1,
			MaxOutputTokens: 2048,
			StopSequences:   []string{},
		},
		SafetySettings: []map[string]interface{}{
			{
				"category":  "HARM_CATEGORY_HARASSMENT",
				"threshold": "BLOCK_NONE",
			},
			{
				"category":  "HARM_CATEGORY_HATE_SPEECH",
				"threshold": "BLOCK_NONE",
			},
			{
				"category":  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
				"threshold": "BLOCK_NONE",
			},
			{
				"category":  "HARM_CATEGORY_DANGEROUS_CONTENT",
				"threshold": "BLOCK_NONE",
			},
		},
	}
	payload, err := json.Marshal(requestJson)
	if err != nil {
		return nil, fmt.Errorf("Gemini serialization failure: %s", err)
	}

	request, _ := http.NewRequest(
		"POST",
		"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.0-pro:generateContent?key="+token,
		bytes.NewReader(payload))
	request.Header.Add("content-type", "application/json")
	response, err := client.Do(request)
	if err != nil || response == nil {
		return nil, fmt.Errorf("Gemini failure: %s", err)
	}

	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("Gemini read response: %s", err)
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"Gemini status: %d\nresponse: %s", response.StatusCode, string(body))
	}

	var result GeminiResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("Gemini parse response: %s", err)
	}

	if len(result.Candidates) == 0 {
		return nil, errors.New("Gemini: No response")
	}
	geminiResult := concatenateStrings(result.Candidates[0].Content.Parts)
	return &geminiResult, nil
}

func concatenateStrings(parts []GeminiResponseCandidateContentPart) string {
	var result string
	for _, part := range parts {
		result += part.Text
	}
	return result
}

func PruneInvisibleCharacters(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsPrint(r) || unicode.IsSpace(r) {
			return r
		}
		return -1
	}, s)
}
