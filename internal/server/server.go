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
	s.mux.HandleFunc("/api/version", s.handleProxy)
	s.mux.HandleFunc("/api/ps", s.handleProxy)
	s.mux.HandleFunc("/api/stop", s.handleProxy)

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
		// Log all API requests for debugging
		if strings.HasPrefix(r.URL.Path, "/api/") {
			log.Printf("[CORS] Incoming: %s %s from %s (UA: %s)", r.Method, r.URL.Path, r.RemoteAddr, r.UserAgent())
		}
		
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
		w.Header().Set("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization, X-Requested-With")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Max-Age", "3600")

		if r.Method == "OPTIONS" {
			log.Printf("[CORS] OPTIONS preflight for %s - returning 204", r.URL.Path)
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// GetProgressManager 获取进度管理器
func (s *Server) GetProgressManager() *download.ProgressManager {
	return s.progressManager
}

// isAPIPath 检查是否为API路径
func (s *Server) isAPIPath(path string) bool {
	return strings.HasPrefix(path, "/api/")
}
