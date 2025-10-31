package config

import (
	"os"
	"strconv"
)

// Config application configuration
type Config struct {
	Model           string // Target model name
	OllamaURL       string // Ollama server address
	Port            int    // Proxy server port
	DownloadTimeout int    // Download timeout in minutes
	AppURL          string // Application URL for API access
}

// Load loads configuration from environment variables
func Load() *Config {
	cfg := &Config{
		Model:           getEnv("OLLAMA_MODEL", "llama2"),
		OllamaURL:       getEnv("OLLAMA_URL", "http://localhost:11434"),
		Port:            getEnvInt("PORT", 8080),
		DownloadTimeout: getEnvInt("DOWNLOAD_TIMEOUT", 60), // Default 60 minutes
		AppURL:          getEnv("APP_URL", ""),
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
