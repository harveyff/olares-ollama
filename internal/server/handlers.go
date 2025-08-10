package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strings"
)

// handleTags handles model list requests, only returns the currently configured model
func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	// Build response containing only the current model
	response := map[string]interface{}{
		"models": []map[string]interface{}{
			{
				"name":        s.config.Model,
				"modified_at": "2024-01-01T00:00:00Z",
				"size":        0,
			},
		},
	}

	json.NewEncoder(w).Encode(response)
}

// handleGenerate handles text generation requests
func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.handleInferenceRequest(w, r, "/api/generate")
}

// handleChat handles chat requests
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.handleInferenceRequest(w, r, "/api/chat")
}

// handleEmbeddings handles embedding vector requests
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.handleInferenceRequest(w, r, "/api/embeddings")
}

// handleInferenceRequest handles inference requests, replaces model parameters
func (s *Server) handleInferenceRequest(w http.ResponseWriter, r *http.Request, path string) {
	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Parse JSON to replace model parameters
	var requestData map[string]interface{}
	if err := json.Unmarshal(body, &requestData); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Replace model parameter
	requestData["model"] = s.config.Model

	// Re-serialize
	modifiedBody, err := json.Marshal(requestData)
	if err != nil {
		http.Error(w, "Failed to modify request", http.StatusInternalServerError)
		return
	}

	// 收集头部信息
	headers := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 && !strings.HasPrefix(strings.ToLower(key), "host") {
			headers[key] = values[0]
		}
	}
	headers["Content-Type"] = "application/json"

	// Proxy request to Ollama
	resp, err := s.ollamaClient.ProxyRequest(
		r.Method,
		path,
		bytes.NewReader(modifiedBody),
		headers,
	)
	if err != nil {
		log.Printf("Failed to proxy request: %v", err)
		http.Error(w, "Failed to proxy request", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Set status code
	w.WriteHeader(resp.StatusCode)

	// Stream copy response body
	io.Copy(w, resp.Body)
}

// handleProxy handles direct proxy requests (system management interfaces)
func (s *Server) handleProxy(w http.ResponseWriter, r *http.Request) {
	// Read request body
	var body io.Reader
	if r.Body != nil {
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "Failed to read request body", http.StatusBadRequest)
			return
		}
		defer r.Body.Close()
		body = bytes.NewReader(bodyBytes)
	}

	// Collect header information
	headers := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 && !strings.HasPrefix(strings.ToLower(key), "host") {
			headers[key] = values[0]
		}
	}

	// Proxy request to Ollama
	resp, err := s.ollamaClient.ProxyRequest(
		r.Method,
		r.URL.Path,
		body,
		headers,
	)
	if err != nil {
		log.Printf("Failed to proxy request: %v", err)
		http.Error(w, "Failed to proxy request", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Set status code
	w.WriteHeader(resp.StatusCode)

	// Copy response body
	io.Copy(w, resp.Body)
}
