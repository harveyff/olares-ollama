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

	// GGUF mode: download GGUF from Hugging Face and register via ollama create
	HFEndpoint string // HF base URL, e.g. "https://huggingface.co"
	HFRepo     string // HF repo, e.g. "unsloth/Qwen3.5-35B-A3B-GGUF"
	HFFile     string // GGUF filename, e.g. "Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
	HFToken    string // Optional HF auth token
	GGUFDir    string // Directory to save GGUF, default "/models"
	GGUFParams string // JSON dict of model parameters, e.g. {"num_ctx":128000}
	GGUFMode   bool   // Auto-set: true when HFRepo and HFFile are both set
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

		HFEndpoint: getEnv("HF_ENDPOINT", "https://huggingface.co"),
		HFRepo:     hfRepo,
		HFFile:     hfFile,
		HFToken:    getEnv("HF_TOKEN", ""),
		GGUFDir:    getEnv("GGUF_DIR", "/models"),
		GGUFParams: getEnv("GGUF_PARAMS", ""),
		GGUFMode:   ggufMode,
	}

	return cfg
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
