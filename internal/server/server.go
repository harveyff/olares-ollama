package server

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"olares-ollama/internal/config"
	"olares-ollama/internal/download"
	"olares-ollama/internal/ollama"
)

// Server 代理服务器
type Server struct {
	config          *config.Config
	ollamaClient    *ollama.Client
	progressManager *download.ProgressManager
	mux             *http.ServeMux
}

// New 创建新的服务器实例
func New(cfg *config.Config, ollamaClient *ollama.Client) *Server {
	s := &Server{
		config:          cfg,
		ollamaClient:    ollamaClient,
		progressManager: download.NewProgressManager(cfg.AppURL),
		mux:             http.NewServeMux(),
	}

	s.setupRoutes()
	return s
}

// Handler 返回HTTP处理器
func (s *Server) Handler() http.Handler {
	return s.corsMiddleware(s.mux)
}

// setupRoutes 设置路由
func (s *Server) setupRoutes() {
	// 静态文件服务
	s.mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("./web/static/"))))
	s.mux.HandleFunc("/", s.handleIndex)

	// 进度API
	s.mux.HandleFunc("/api/progress", s.progressManager.HandleProgressAPI)

	// Ollama API路由
	s.mux.HandleFunc("/api/tags", s.handleTags)
	s.mux.HandleFunc("/api/generate", s.handleGenerate)
	s.mux.HandleFunc("/api/chat", s.handleChat)
	s.mux.HandleFunc("/api/embeddings", s.handleEmbeddings)
	s.mux.HandleFunc("/api/embed", s.handleEmbeddings)  // OpenWebUI uses /api/embed
	s.mux.HandleFunc("/api/version", s.handleProxy)
	s.mux.HandleFunc("/api/ps", s.handleProxy)
	s.mux.HandleFunc("/api/stop", s.handleProxy)
	
	// OpenWebUI uses /api/chat/completions (OpenAI compatible format)
	s.mux.HandleFunc("/api/chat/completions", s.handleOpenAIChat)
	s.mux.HandleFunc("/api/chat/completed", s.handleOpenAIChat)  // OpenWebUI completion callback
	
	// OpenAI compatible endpoints (some OpenWebUI versions may use these)
	s.mux.HandleFunc("/v1/chat/completions", s.handleOpenAIChat)
	s.mux.HandleFunc("/v1/completions", s.handleOpenAICompletions)  // OpenAI text completions
	s.mux.HandleFunc("/v1/models", s.handleOpenAIModels)
	s.mux.HandleFunc("/v1/embeddings", s.handleEmbeddings)  // OpenAI embeddings
	s.mux.HandleFunc("/v1/responses", s.handleProxy)  // Proxy /v1/responses to Ollama

	// 健康检查
	s.mux.HandleFunc("/health", s.handleHealth)
}

// handleIndex 处理首页请求
func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" {
		http.Redirect(w, r, "/static/index.html", http.StatusMovedPermanently)
		return
	}
	http.NotFound(w, r)
}

// handleHealth 健康检查
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	response := map[string]interface{}{
		"status": "ok",
		"model":  s.config.Model,
	}
	json.NewEncoder(w).Encode(response)
}

// corsMiddleware CORS中间件
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers on all responses, EXCEPT for embeddings endpoints
		// Embeddings endpoints should match Ollama's response exactly (no CORS headers)
		isEmbeddingsEndpoint := r.URL.Path == "/api/embed" || r.URL.Path == "/api/embeddings"
		
		if !isEmbeddingsEndpoint {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
			w.Header().Set("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization, X-Requested-With")
			w.Header().Set("Access-Control-Allow-Credentials", "true")
			w.Header().Set("Access-Control-Max-Age", "3600")
		}

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		// Use a ResponseWriter wrapper to log response status
		wrapped := &responseLogger{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(wrapped, r)
		
		// 只记录失败的请求（status 不是 200）
		if strings.HasPrefix(r.URL.Path, "/api/") && wrapped.statusCode != http.StatusOK {
			log.Printf("[ERROR] Request failed: %s %s -> Status: %d", r.Method, r.URL.Path, wrapped.statusCode)
		}
	})
}

// responseLogger wraps ResponseWriter to capture status code
type responseLogger struct {
	http.ResponseWriter
	statusCode int
}

func (rl *responseLogger) WriteHeader(code int) {
	rl.statusCode = code
	rl.ResponseWriter.WriteHeader(code)
}

// GetProgressManager 获取进度管理器
func (s *Server) GetProgressManager() *download.ProgressManager {
	return s.progressManager
}

// isAPIPath 检查是否为API路径
func (s *Server) isAPIPath(path string) bool {
	return strings.HasPrefix(path, "/api/")
}
