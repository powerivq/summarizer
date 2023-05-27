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
		if config.APIType == APITypeGCPPalm {
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

	var mergePrompt string
	var finalSummarizePrompt string
	if chineseChars >= englishChars {
		mergePrompt = "用中文詳細總結每一段: \n\n"
		finalSummarizePrompt = "用中文詳細總結下面的判決: \n\n"
	} else {
		mergePrompt = "Summarize in detail, paragraph by paragraph: \n\n"
		finalSummarizePrompt = "Summarize this judgment in detail: \n\n"
	}

	if tokens := len(c.tiktoken.Encode(text, nil, nil)); tokens <= 3000 {
		content, err := c.requestGpt(finalSummarizePrompt+text, 4000-tokens-len(c.tiktoken.Encode(finalSummarizePrompt, nil, nil)))
		if err != nil {
			return nil, err
		}
		return content, nil
	}

	texts := strings.Split(text, "\n")

	var summary strings.Builder
	var window strings.Builder
	var tokens int

	promptTokens := len(c.tiktoken.Encode(mergePrompt, nil, nil))
	for i := 0; i < len(texts); {
		window.Reset()
		window.WriteString(mergePrompt)
		tokens = promptTokens

		for ; i < len(texts) && tokens <= 2500; i++ {
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
			content, err := c.requestGpt(prompt, 4000-tokens)
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

func (c *Client) requestGpt(prompt string, maxTokens int) (*string, error) {
	invalidInput := false
	if len(c.azureClients) > 0 && !invalidInput {
		for i := 0; i < 3; i++ {
			client := &c.azureClients[rand.Intn(len(c.azureClients))]
			res, err := c.doRequestGpt(client, prompt, maxTokens)
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
			res, err := c.doRequestGpt(client, prompt, maxTokens)
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
			res, err := c.doRequestPalm(token, prompt)
			if err == nil {
				c.logger.Log(prompt, *res, APITypeGCPPalm)
				return res, nil
			}
			log.Printf("Palm error: %s", err)
		}
	}
	return nil, errors.New("all retries have failed")
}

func (c *Client) doRequestGpt(client *openai.Client, prompt string, maxTokens int) (*string, error) {
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT3Dot5Turbo,
			Messages: []openai.ChatCompletionMessage{{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			}},
			MaxTokens:   maxTokens,
			Temperature: 0.7,
			Stream:      false,
		},
	)
	if err != nil {
		log.Printf("openai error: %s", err)
		return nil, err
	}

	return &resp.Choices[0].Message.Content, nil
}

func (c *Client) doRequestPalm(token string, prompt string) (*string, error) {
	netTransport := &http.Transport{
		TLSHandshakeTimeout: 10 * time.Second,
	}

	client := &http.Client{
		Timeout:   60 * time.Second,
		Transport: netTransport,
	}

	requestJson := PalmRequest{
		Prompt:          PalmRequestPrompt{Text: prompt},
		Temperature:     0.7,
		CandidateCount:  1,
		TopK:            40,
		TopP:            0.95,
		MaxOutputTokens: 1024,
		StopSequences:   []string{},
		SafetySettings: []map[string]interface{}{
			{
				"category":  "HARM_CATEGORY_DEROGATORY",
				"threshold": 4,
			},
			{
				"category":  "HARM_CATEGORY_TOXICITY",
				"threshold": 4,
			},
			{
				"category":  "HARM_CATEGORY_VIOLENCE",
				"threshold": 4,
			},
			{
				"category":  "HARM_CATEGORY_SEXUAL",
				"threshold": 4,
			},
			{
				"category":  "HARM_CATEGORY_MEDICAL",
				"threshold": 4,
			},
			{
				"category":  "HARM_CATEGORY_DANGEROUS",
				"threshold": 4,
			},
		},
	}
	payload, err := json.Marshal(requestJson)
	if err != nil {
		return nil, fmt.Errorf("palm serialization failure: %s", err)
	}

	request, _ := http.NewRequest(
		"POST",
		"https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key="+token,
		bytes.NewReader(payload))
	request.Header.Add("content-type", "application/json")
	response, err := client.Do(request)
	if err != nil || response == nil {
		return nil, fmt.Errorf("palm failure: %s", err)
	}

	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("palm read response: %s", err)
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"palm status: %d\nresponse: %s", response.StatusCode, string(body))
	}

	var result PalmResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("palm parse response: %s", err)
	}

	if len(result.Candidates) == 0 {
		return nil, errors.New("palm: No response")
	}
	return &result.Candidates[0].Output, nil
}

func PruneInvisibleCharacters(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsPrint(r) || unicode.IsSpace(r) {
			return r
		}
		return -1
	}, s)
}
