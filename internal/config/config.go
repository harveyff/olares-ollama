package config

import (
	"os"
	"strconv"
)

// Config application configuration
type Config struct {
	Model              string // Target model name
	OllamaURL          string // Ollama server address
	Port               int    // Proxy server port
	DownloadTimeout    int    // Download timeout in minutes
	AppURL             string // Application URL for API access
	OllamaPullDelaySec int    // Seconds to wait after Ollama is ready before first pull (for blob index to load, helps resume after restart)
	BaseMode           bool   // Base mode: no specific model, show guide + version + model list
	ThinkingEnabled    bool   // Default thinking mode for models that support it (Qwen3.5, DeepSeek, etc.)
	ContextLength      int    // Default num_ctx to inject into requests (0 = don't inject, let model/Ollama decide)

	// GGUF mode: download GGUF from Hugging Face and register via ollama create
	HFEndpoint string // HF base URL, e.g. "https://huggingface.co"
	HFRepo     string // HF repo, e.g. "unsloth/Qwen3.5-35B-A3B-GGUF"
	HFFile     string // GGUF filename, e.g. "Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
	HFToken    string // Optional HF auth token
	GGUFDir          string // Directory to save GGUF, default "/models"
	GGUFParams       string // JSON dict of model parameters, e.g. {"num_ctx":128000}
	GGUFTemplateName string // Named template: "chatml", "llama3", etc. Resolved to Go template in code
	GGUFTemplate     string // Raw Go template override (takes precedence over TemplateName)
	GGUFSystem       string // System prompt baked into the model
	GGUFMode         bool   // Auto-set: true when HFRepo and HFFile are both set
}

// Load loads configuration from environment variables
func Load() *Config {
	model := getEnv("OLLAMA_MODEL", "")
	hfRepo := getEnv("HF_REPO", "")
	hfFile := getEnv("HF_FILE", "")
	ggufMode := hfRepo != "" && hfFile != ""

	cfg := &Config{
		Model:              model,
		OllamaURL:          getEnv("OLLAMA_URL", "http://localhost:11434"),
		Port:               getEnvInt("PORT", 8080),
		DownloadTimeout:    getEnvInt("DOWNLOAD_TIMEOUT", 60),
		AppURL:             getEnv("APP_URL", ""),
		OllamaPullDelaySec: getEnvInt("OLLAMA_PULL_DELAY_SECONDS", 30),
		BaseMode:           model == "" && !ggufMode,
		ThinkingEnabled:    getEnvBool("OLLAMA_THINKING", true),
		ContextLength:      getEnvInt("OLLAMA_CONTEXT_LENGTH", 0),

		HFEndpoint: getEnv("HF_ENDPOINT", "https://huggingface.co"),
		HFRepo:     hfRepo,
		HFFile:     hfFile,
		HFToken:    getEnv("HF_TOKEN", ""),
		GGUFDir:          getEnv("GGUF_DIR", "/models"),
		GGUFParams:       getEnv("GGUF_PARAMS", ""),
		GGUFTemplateName: getEnv("GGUF_TEMPLATE_NAME", ""),
		GGUFTemplate:     getEnv("GGUF_TEMPLATE", ""),
		GGUFSystem:       getEnv("GGUF_SYSTEM", ""),
		GGUFMode:         ggufMode,
	}

	return cfg
}

// Built-in Go templates for common chat formats.
// Ollama uses Go text/template; these mirror Ollama's library templates.
var builtinTemplates = map[string]string{
	"chatml": `{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ end }}{{- end }}<|im_start|>assistant
`,
	"llama3": `{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{- range .Messages }}{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ .Content }}<|eot_id|>{{ else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ .Content }}<|eot_id|>{{ end }}{{- end }}<|start_header_id|>assistant<|end_header_id|>

`,
}

// ResolveTemplate returns the Go template string. GGUFTemplate (raw) takes
// precedence; otherwise GGUFTemplateName is looked up in the built-in map.
func (c *Config) ResolveTemplate() string {
	if c.GGUFTemplate != "" {
		return c.GGUFTemplate
	}
	if t, ok := builtinTemplates[c.GGUFTemplateName]; ok {
		return t
	}
	return ""
}

// getEnv gets environment variable, returns default value if not exists
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvInt gets integer environment variable, returns default value if not exists
func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

// getEnvBool gets boolean environment variable, returns default value if not exists.
// Accepts "true"/"1" as true and "false"/"0" as false (case-insensitive).
func getEnvBool(key string, defaultValue bool) bool {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	b, err := strconv.ParseBool(value)
	if err != nil {
		return defaultValue
	}
	return b
}
