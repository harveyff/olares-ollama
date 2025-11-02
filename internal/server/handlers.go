package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

// handleTags handles model list requests, forwards from ollama and filters by configured models
func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	log.Printf("=== Tags endpoint: Method=%s, RemoteAddr=%s ===", r.Method, r.RemoteAddr)
	if r.Method != "GET" {
		log.Printf("Tags endpoint received unsupported method: %s", r.Method)
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
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
		"/api/tags",
		nil,
		headers,
	)
	if err != nil {
		log.Printf("Failed to proxy request to ollama: %v", err)
		http.Error(w, "Failed to proxy request", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Copy error response
		for key, values := range resp.Header {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
		return
	}

	// Parse response from ollama
	var ollamaResponse map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResponse); err != nil {
		log.Printf("Failed to decode ollama response: %v", err)
		http.Error(w, "Failed to decode response", http.StatusInternalServerError)
		return
	}

	// Filter models: only keep the configured model
	models, ok := ollamaResponse["models"].([]interface{})
	if !ok {
		log.Printf("Invalid models format in ollama response")
		http.Error(w, "Invalid response format", http.StatusInternalServerError)
		return
	}

	filteredModels := []interface{}{}
	for _, model := range models {
		modelMap, ok := model.(map[string]interface{})
		if !ok {
			continue
		}
		modelName, ok := modelMap["name"].(string)
		if !ok {
			continue
		}
		// Only keep the configured model
		if modelName == s.config.Model {
			filteredModels = append(filteredModels, model)
		}
	}

	// Build filtered response
	response := map[string]interface{}{
		"models": filteredModels,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleGenerate handles text generation requests
func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	// Allow POST and handle OPTIONS for CORS preflight
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != "POST" {
		log.Printf("Generate endpoint received unsupported method: %s from %s", r.Method, r.RemoteAddr)
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.handleInferenceRequest(w, r, "/api/generate")
}

// handleChat handles chat requests
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	// Log all incoming requests to /api/chat
	log.Printf("=== Chat endpoint: Method=%s, RemoteAddr=%s, UserAgent=%s, ContentType=%s ===", 
		r.Method, r.RemoteAddr, r.UserAgent(), r.Header.Get("Content-Type"))
	
	// Allow POST and handle OPTIONS for CORS preflight
	if r.Method == "OPTIONS" {
		log.Printf("Handling OPTIONS request for /api/chat")
		w.WriteHeader(http.StatusNoContent)
		return
	}
	// Handle GET requests (used by OpenWebUI for health checks)
	// OpenWebUI expects a dict/object response (with .get() method), not a list
	if r.Method == "GET" {
		log.Printf("Chat endpoint received GET request from %s (health check)", r.RemoteAddr)
		userAgent := r.UserAgent()
		log.Printf("GET request UserAgent: %s, Referer: %s", userAgent, r.Header.Get("Referer"))
		
		// Return an empty object, not an array, so OpenWebUI can call .get() on it
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{})
		log.Printf("GET request responded with 200 OK (empty object)")
		return
	}
	if r.Method != "POST" {
		log.Printf("Chat endpoint received unsupported method: %s from %s", r.Method, r.RemoteAddr)
		// Return JSON error response for better compatibility
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Allow", "POST, GET, OPTIONS")
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": fmt.Sprintf("Method %s not allowed for /api/chat endpoint. Supported methods: POST, GET, OPTIONS", r.Method),
		})
		return
	}
	log.Printf("*** Handling POST chat request from %s ***", r.RemoteAddr)
	s.handleInferenceRequest(w, r, "/api/chat")
}

// handleEmbeddings handles embedding vector requests
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	// Allow POST and handle OPTIONS for CORS preflight
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != "POST" {
		log.Printf("Embeddings endpoint received unsupported method: %s from %s", r.Method, r.RemoteAddr)
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
		log.Printf("Failed to read request body for %s: %v", path, err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Check if body is empty
	if len(body) == 0 {
		log.Printf("Empty request body for %s", path)
		http.Error(w, "Request body cannot be empty", http.StatusBadRequest)
		return
	}

	// Parse JSON to replace model parameters
	var requestData map[string]interface{}
	if err := json.Unmarshal(body, &requestData); err != nil {
		log.Printf("Failed to parse JSON for %s: %v, body: %s", path, err, string(body))
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Replace model parameter
	requestData["model"] = s.config.Model

	// Re-serialize
	modifiedBody, err := json.Marshal(requestData)
	if err != nil {
		log.Printf("Failed to marshal request for %s: %v", path, err)
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

	// Log the request being proxied
	bodyPreviewLen := len(modifiedBody)
	if bodyPreviewLen > 200 {
		bodyPreviewLen = 200
	}
	log.Printf(">>> Proxying %s request to Ollama %s (model: %s, body size: %d bytes) <<<", 
		r.Method, path, s.config.Model, len(modifiedBody))
	if len(modifiedBody) > 0 {
		log.Printf(">>> Request body preview: %s", string(modifiedBody[:bodyPreviewLen]))
	}

	// Proxy request to Ollama
	resp, err := s.ollamaClient.ProxyRequest(
		r.Method,
		path,
		bytes.NewReader(modifiedBody),
		headers,
	)
	if err != nil {
		log.Printf("!!! Failed to proxy request to Ollama %s: %v !!!", path, err)
		http.Error(w, "Failed to proxy request", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// Log response status
	log.Printf("<<< Ollama returned status %d for %s request to %s <<<", resp.StatusCode, r.Method, path)
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		log.Printf("!!! Warning: Ollama returned non-success status %d !!!", resp.StatusCode)
	}

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
