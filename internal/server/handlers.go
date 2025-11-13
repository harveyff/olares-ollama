package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
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
	// Return minimal info that indicates this is a valid chat endpoint
	if r.Method == "GET" {
		log.Printf("Chat endpoint received GET request from %s (health check)", r.RemoteAddr)
		userAgent := r.UserAgent()
		log.Printf("GET request UserAgent: %s, Referer: %s", userAgent, r.Header.Get("Referer"))
		
		// Return an object with some basic info - this helps OpenWebUI recognize the endpoint
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Allow", "POST, GET, OPTIONS")  // Explicitly state allowed methods
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "ok",
		})
		log.Printf("GET request responded with 200 OK (status object)")
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
	
	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("Failed to read embeddings request body: %v", err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()
	
	if len(body) == 0 {
		log.Printf("Empty embeddings request body")
		http.Error(w, "Request body cannot be empty", http.StatusBadRequest)
		return
	}
	
	// Parse request to check format
	var requestData map[string]interface{}
	if err := json.Unmarshal(body, &requestData); err != nil {
		log.Printf("Failed to parse embeddings JSON: %v, body: %s", err, string(body))
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	// Log full request for debugging
	bodyPreview := string(body)
	if len(bodyPreview) > 500 {
		bodyPreview = bodyPreview[:500] + "..."
	}
	log.Printf(">>> Embeddings request body: %s <<<", bodyPreview)
	log.Printf(">>> Embeddings request fields: input=%v (type=%T), prompt=%v (type=%T), model=%v <<<", 
		requestData["input"], requestData["input"], requestData["prompt"], requestData["prompt"], requestData["model"])
	
	// Check if this is OpenAI format (has "input") or Ollama format (has "prompt")
	inputRaw, hasInput := requestData["input"]
	_, hasPrompt := requestData["prompt"]
	
	// If it's Ollama format (has "prompt" but no "input"), just proxy directly but convert response
	if hasPrompt && !hasInput {
		log.Printf(">>> Detected Ollama format (prompt field), proxying directly <<<")
		s.handleOllamaEmbedding(w, r, body, requestData)
		return
	}
	
	// OpenAI format: check if batch request
	var inputs []interface{}
	isBatch := false
	
	if hasInput {
		if inputArray, ok := inputRaw.([]interface{}); ok && len(inputArray) > 0 {
			// Array input: check if batch (multiple items) or single (one item)
			if len(inputArray) > 1 {
				isBatch = true
				inputs = inputArray
				log.Printf(">>> Batch embeddings request: %d inputs <<<", len(inputs))
			} else {
				// Single item in array - treat as single request
				inputs = inputArray
				log.Printf(">>> Single input in array format <<<")
			}
		} else if inputStr, ok := inputRaw.(string); ok {
			// Single string input
			inputs = []interface{}{inputStr}
			log.Printf(">>> Single string input <<<")
		}
	}
	
	// If batch request (multiple inputs), process each input separately
	if isBatch && len(inputs) > 1 {
		s.handleBatchEmbeddings(w, r, inputs, requestData)
		return
	}
	
	// Single embedding request - convert to OpenAI format
	s.handleSingleEmbedding(w, r, body, requestData)
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

	// Copy response headers (except for ones that should be controlled by the response writer)
	for key, values := range resp.Header {
		keyLower := strings.ToLower(key)
		// Skip headers that should be controlled by the response writer
		if keyLower != "content-length" && keyLower != "transfer-encoding" && keyLower != "connection" {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
	}

	// Check if this is a streaming response
	isStreaming := false
	if resp.Header.Get("Transfer-Encoding") == "chunked" || resp.Header.Get("Content-Type") == "text/event-stream" {
		isStreaming = true
		w.Header().Set("Transfer-Encoding", "chunked")
		w.Header().Set("Connection", "keep-alive")
	}

	// Set status code
	w.WriteHeader(resp.StatusCode)
	
	// Flush headers if possible (for streaming)
	if flusher, ok := w.(http.Flusher); ok && isStreaming {
		flusher.Flush()
	}

	// Stream copy response body with proper flushing for streaming responses
	if isStreaming {
		if flusher, ok := w.(http.Flusher); ok {
			// Use buffered copy with periodic flushing for streaming
			buffer := make([]byte, 4096)
			var totalBytes int64
			for {
				n, err := resp.Body.Read(buffer)
				if n > 0 {
					if _, writeErr := w.Write(buffer[:n]); writeErr != nil {
						log.Printf("!!! Error writing response for %s: %v !!!", path, writeErr)
						break
					}
					totalBytes += int64(n)
					// Flush periodically for streaming
					flusher.Flush()
				}
				if err == io.EOF {
					break
				}
				if err != nil {
					log.Printf("!!! Error reading from Ollama for %s: %v !!!", path, err)
					break
				}
			}
			flusher.Flush()
			log.Printf("<<< Copied %d bytes from Ollama stream for %s <<<", totalBytes, path)
		} else {
			// Fallback to regular copy
			bytesCopied, err := io.Copy(w, resp.Body)
			if err != nil {
				log.Printf("!!! Error copying response body for %s: %v !!!", path, err)
			} else {
				log.Printf("<<< Copied %d bytes from Ollama for %s <<<", bytesCopied, path)
			}
		}
	} else {
		// Non-streaming: regular copy
		bytesCopied, err := io.Copy(w, resp.Body)
		if err != nil {
			log.Printf("!!! Error copying response body for %s: %v !!!", path, err)
		} else {
			log.Printf("<<< Copied %d bytes from Ollama for %s <<<", bytesCopied, path)
		}
		// Final flush
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}
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

// handleOpenAIChat handles OpenAI compatible chat completions endpoint
func (s *Server) handleOpenAIChat(w http.ResponseWriter, r *http.Request) {
	log.Printf("=== OpenAI Chat Completions endpoint: %s %s, Method=%s, RemoteAddr=%s ===", 
		r.Method, r.URL.Path, r.Method, r.RemoteAddr)
	log.Printf("=== Full URL: %s ===", r.URL.String())
	log.Printf("=== Headers: %v ===", r.Header)
	
	if r.Method == "OPTIONS" {
		log.Printf("OpenAI Chat Completions: Handling OPTIONS preflight")
		w.WriteHeader(http.StatusNoContent)
		return
	}
	
	if r.Method == "GET" {
		log.Printf("OpenAI Chat Completions received GET request (health check)")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "ok",
		})
		return
	}
	
	if r.Method != "POST" {
		log.Printf("!!! OpenAI Chat Completions received unsupported method: %s !!!", r.Method)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": "Method not allowed",
		})
		return
	}
	
	log.Printf("*** Handling OpenAI Chat Completions POST request from %s ***", r.RemoteAddr)
	// Convert OpenAI format to Ollama format and proxy
	s.handleOpenAIInferenceRequest(w, r)
}

// handleOpenAIModels handles OpenAI compatible models endpoint
func (s *Server) handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	log.Printf("=== OpenAI Models endpoint: Method=%s ===", r.Method)
	
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	
	// Return OpenAI format model list
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"data": []map[string]interface{}{
			{
				"id":      s.config.Model,
				"object":  "model",
				"created": 0,
				"owned_by": "ollama",
			},
		},
		"object": "list",
	})
}

// handleOpenAIInferenceRequest converts OpenAI format to Ollama format and proxies
func (s *Server) handleOpenAIInferenceRequest(w http.ResponseWriter, r *http.Request) {
	log.Printf(">>> Starting OpenAI request processing <<<")
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("!!! Failed to read OpenAI request body: %v !!!", err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()
	
	log.Printf(">>> OpenAI request body size: %d bytes <<<", len(body))
	if len(body) == 0 {
		log.Printf("!!! OpenAI request body is empty !!!")
		http.Error(w, "Request body cannot be empty", http.StatusBadRequest)
		return
	}
	
	// Log request body preview
	bodyPreview := string(body)
	if len(bodyPreview) > 500 {
		bodyPreview = bodyPreview[:500] + "..."
	}
	log.Printf(">>> OpenAI request body preview: %s <<<", bodyPreview)
	
	// Parse OpenAI format request
	var openaiRequest map[string]interface{}
	if err := json.Unmarshal(body, &openaiRequest); err != nil {
		log.Printf("!!! Failed to parse OpenAI JSON: %v, body: %s !!!", err, string(body))
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	// Check if messages exists and is valid
	messagesRaw, ok := openaiRequest["messages"]
	if !ok {
		log.Printf("!!! Missing 'messages' field in OpenAI request !!!")
		http.Error(w, "Missing 'messages' field", http.StatusBadRequest)
		return
	}
	
	messages, ok := messagesRaw.([]interface{})
	if !ok {
		log.Printf("!!! Invalid messages format in OpenAI request (not an array) !!!")
		http.Error(w, "Invalid messages format", http.StatusBadRequest)
		return
	}
	
	log.Printf(">>> Parsed OpenAI request: model=%v, stream=%v, messages count=%d <<<",
		openaiRequest["model"], openaiRequest["stream"], len(messages))
	
	// Convert messages
	ollamaMessages := []map[string]interface{}{}
	for i, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			log.Printf("!!! Skipping invalid message at index %d !!!", i)
			continue
		}
		ollamaMessages = append(ollamaMessages, map[string]interface{}{
			"role":    msgMap["role"],
			"content": msgMap["content"],
		})
	}
	
	if len(ollamaMessages) == 0 {
		log.Printf("!!! No valid messages after conversion !!!")
		http.Error(w, "No valid messages", http.StatusBadRequest)
		return
	}
	
	// Build Ollama request
	stream := false
	if streamVal, ok := openaiRequest["stream"]; ok {
		if streamBool, ok := streamVal.(bool); ok {
			stream = streamBool
		}
	}
	
	ollamaRequest := map[string]interface{}{
		"model":    s.config.Model,
		"messages": ollamaMessages,
		"stream":   stream,
	}
	
	modifiedBody, err := json.Marshal(ollamaRequest)
	if err != nil {
		log.Printf("!!! Failed to marshal Ollama request: %v !!!", err)
		http.Error(w, "Failed to prepare request", http.StatusInternalServerError)
		return
	}
	
	log.Printf(">>> Converted to Ollama format: body size=%d bytes, model=%s, messages=%d, stream=%v <<<",
		len(modifiedBody), s.config.Model, len(ollamaMessages), stream)
	
	// Collect headers
	headers := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 && !strings.HasPrefix(strings.ToLower(key), "host") {
			headers[key] = values[0]
		}
	}
	headers["Content-Type"] = "application/json"
	
	log.Printf(">>> Proxying OpenAI request to Ollama /api/chat (model: %s) <<<", s.config.Model)
	
	// Proxy to Ollama
	resp, err := s.ollamaClient.ProxyRequest(
		"POST",
		"/api/chat",
		bytes.NewReader(modifiedBody),
		headers,
	)
	if err != nil {
		log.Printf("!!! Failed to proxy OpenAI request to Ollama: %v !!!", err)
		http.Error(w, "Failed to proxy request", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	
	log.Printf("<<< Ollama returned status %d for OpenAI request <<<", resp.StatusCode)
	
	// Set OpenAI-compatible response headers
	if stream {
		// OpenAI streaming uses Server-Sent Events (SSE) format
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Transfer-Encoding", "chunked")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Cache-Control", "no-cache")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
	
	w.WriteHeader(resp.StatusCode)
	
	// Flush headers if possible (for streaming)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	
	log.Printf(">>> Starting to convert Ollama response to OpenAI format (stream=%v) <<<", stream)
	
	// Convert Ollama response to OpenAI format
	if stream {
		// Handle streaming response
		s.convertOllamaStreamToOpenAI(w, resp.Body, s.config.Model)
	} else {
		// Handle non-streaming response
		s.convertOllamaToOpenAI(w, resp.Body, s.config.Model)
	}
}

// convertOllamaToOpenAI converts Ollama non-streaming response to OpenAI format
func (s *Server) convertOllamaToOpenAI(w http.ResponseWriter, body io.Reader, modelName string) {
	bodyBytes, err := io.ReadAll(body)
	if err != nil {
		log.Printf("!!! Error reading Ollama response: %v !!!", err)
		http.Error(w, "Failed to read response", http.StatusInternalServerError)
		return
	}
	
	var ollamaResp map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &ollamaResp); err != nil {
		log.Printf("!!! Error parsing Ollama response: %v, body: %s !!!", err, string(bodyBytes))
		http.Error(w, "Failed to parse response", http.StatusInternalServerError)
		return
	}
	
	// Extract message content
	message, ok := ollamaResp["message"].(map[string]interface{})
	if !ok {
		log.Printf("!!! Invalid Ollama response format: missing message !!!")
		http.Error(w, "Invalid response format", http.StatusInternalServerError)
		return
	}
	
	content, _ := message["content"].(string)
	role, _ := message["role"].(string)
	if role == "" {
		role = "assistant"
	}
	
	// Create OpenAI format response
	openAIResp := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   modelName,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    role,
					"content": content,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     0,
			"completion_tokens": 0,
			"total_tokens":       0,
		},
	}
	
	// Try to extract token usage if available
	if evalCount, ok := ollamaResp["eval_count"].(float64); ok {
		openAIResp["usage"].(map[string]interface{})["completion_tokens"] = int(evalCount)
		openAIResp["usage"].(map[string]interface{})["total_tokens"] = int(evalCount)
	}
	if promptEvalCount, ok := ollamaResp["prompt_eval_count"].(float64); ok {
		openAIResp["usage"].(map[string]interface{})["prompt_tokens"] = int(promptEvalCount)
		if total, ok := openAIResp["usage"].(map[string]interface{})["total_tokens"].(int); ok {
			openAIResp["usage"].(map[string]interface{})["total_tokens"] = total + int(promptEvalCount)
		}
	}
	
	responseJSON, err := json.Marshal(openAIResp)
	if err != nil {
		log.Printf("!!! Error marshaling OpenAI response: %v !!!", err)
		http.Error(w, "Failed to format response", http.StatusInternalServerError)
		return
	}
	
	w.Write(responseJSON)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	log.Printf("<<< Converted and sent OpenAI format response (%d bytes) <<<", len(responseJSON))
}

// convertOllamaStreamToOpenAI converts Ollama streaming response to OpenAI SSE format
func (s *Server) convertOllamaStreamToOpenAI(w http.ResponseWriter, body io.Reader, modelName string) {
	flusher, hasFlusher := w.(http.Flusher)
	scanner := bufio.NewScanner(body)
	responseID := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
	created := time.Now().Unix()
	var totalBytes int64
	roleSent := false
	
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		
		var ollamaResp map[string]interface{}
		if err := json.Unmarshal(line, &ollamaResp); err != nil {
			log.Printf("!!! Error parsing Ollama stream line: %v, line: %s !!!", err, string(line))
			continue
		}
		
		// Check if done first
		done, _ := ollamaResp["done"].(bool)
		if done {
			// Send final chunk with finish_reason
			finalChunk := map[string]interface{}{
				"id":      responseID,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   modelName,
				"choices": []map[string]interface{}{
					{
						"index":         0,
						"delta":         map[string]interface{}{},
						"finish_reason": "stop",
					},
				},
			}
			finalJSON, _ := json.Marshal(finalChunk)
			w.Write([]byte(fmt.Sprintf("data: %s\n\n", finalJSON)))
			w.Write([]byte("data: [DONE]\n\n"))
			if hasFlusher {
				flusher.Flush()
			}
			break
		}
		
		// Extract message content (Ollama may send incremental content)
		message, ok := ollamaResp["message"].(map[string]interface{})
		if !ok {
			continue
		}
		
		content, _ := message["content"].(string)
		role, _ := message["role"].(string)
		
		// Create OpenAI SSE chunk
		delta := map[string]interface{}{}
		if !roleSent && role != "" {
			delta["role"] = role
			roleSent = true
		}
		if content != "" {
			delta["content"] = content
		}
		
		// Only send chunk if there's content
		if len(delta) > 0 {
			chunk := map[string]interface{}{
				"id":      responseID,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   modelName,
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"delta": delta,
					},
				},
			}
			
			chunkJSON, err := json.Marshal(chunk)
			if err != nil {
				log.Printf("!!! Error marshaling chunk: %v !!!", err)
				continue
			}
			
			chunkLine := fmt.Sprintf("data: %s\n\n", chunkJSON)
			written, err := w.Write([]byte(chunkLine))
			if err != nil {
				log.Printf("!!! Error writing chunk: %v !!!", err)
				break
			}
			totalBytes += int64(written)
			
			if hasFlusher {
				flusher.Flush()
			}
		}
	}
	
	if err := scanner.Err(); err != nil {
		log.Printf("!!! Error scanning stream: %v !!!", err)
	}
	
	log.Printf("<<< Converted and sent OpenAI stream response (%d bytes) <<<", totalBytes)
}

// handleSingleEmbedding handles a single embedding request and converts to OpenAI format
func (s *Server) handleSingleEmbedding(w http.ResponseWriter, r *http.Request, body []byte, requestData map[string]interface{}) {
	// Replace model parameter
	requestData["model"] = s.config.Model
	
	// Convert "input" to "prompt" for Ollama if needed
	if input, ok := requestData["input"]; ok {
		// Ollama uses "prompt" instead of "input", and it must be a string
		var promptStr string
		if inputArray, ok := input.([]interface{}); ok && len(inputArray) > 0 {
			// If input is an array, take the first element
			if firstItem, ok := inputArray[0].(string); ok {
				promptStr = firstItem
			} else {
				log.Printf("!!! Invalid input array element type: %T !!!", inputArray[0])
				http.Error(w, "Invalid input format", http.StatusBadRequest)
				return
			}
		} else if inputStr, ok := input.(string); ok {
			// If input is already a string, use it directly
			promptStr = inputStr
		} else {
			log.Printf("!!! Invalid input type: %T !!!", input)
			http.Error(w, "Invalid input format", http.StatusBadRequest)
			return
		}
		requestData["prompt"] = promptStr
		// Remove "input" field as Ollama doesn't use it
		delete(requestData, "input")
	}
	
	// Re-serialize
	modifiedBody, err := json.Marshal(requestData)
	if err != nil {
		log.Printf("Failed to marshal embeddings request: %v", err)
		http.Error(w, "Failed to modify request", http.StatusInternalServerError)
		return
	}
	
	// Log the request being sent to Ollama
	bodyPreview := string(modifiedBody)
	if len(bodyPreview) > 500 {
		bodyPreview = bodyPreview[:500] + "..."
	}
	log.Printf(">>> Request to Ollama: %s <<<", bodyPreview)
	
	// Collect headers
	headers := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 && !strings.HasPrefix(strings.ToLower(key), "host") {
			headers[key] = values[0]
		}
	}
	headers["Content-Type"] = "application/json"
	
	log.Printf(">>> Proxying embeddings request to Ollama (model: %s) <<<", s.config.Model)
	
	// Proxy to Ollama
	resp, err := s.ollamaClient.ProxyRequest(
		"POST",
		"/api/embeddings",
		bytes.NewReader(modifiedBody),
		headers,
	)
	if err != nil {
		log.Printf("!!! Failed to proxy embeddings request: %v !!!", err)
		http.Error(w, "Failed to proxy request", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		// Read error response body for debugging
		errorBody, _ := io.ReadAll(resp.Body)
		log.Printf("!!! Ollama returned status %d for embeddings !!!", resp.StatusCode)
		log.Printf("!!! Ollama error response: %s !!!", string(errorBody))
		w.WriteHeader(resp.StatusCode)
		w.Write(errorBody)
		return
	}
	
	// Read Ollama response
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("!!! Error reading Ollama embeddings response: %v !!!", err)
		http.Error(w, "Failed to read response", http.StatusInternalServerError)
		return
	}
	
	var ollamaResp map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &ollamaResp); err != nil {
		log.Printf("!!! Error parsing Ollama embeddings response: %v, body: %s !!!", err, string(bodyBytes))
		http.Error(w, "Failed to parse response", http.StatusInternalServerError)
		return
	}
	
	// Extract embedding vector
	embedding, ok := ollamaResp["embedding"].([]interface{})
	if !ok {
		// Try to convert if it's a different type
		if embeddingFloat, ok := ollamaResp["embedding"].([]float64); ok {
			embedding = make([]interface{}, len(embeddingFloat))
			for i, v := range embeddingFloat {
				embedding[i] = v
			}
		} else {
			log.Printf("!!! Invalid embedding format in Ollama response: %T !!!", ollamaResp["embedding"])
			http.Error(w, "Invalid embedding format", http.StatusInternalServerError)
			return
		}
	}
	
	// Convert to OpenAI format
	openAIResp := map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"object":    "embedding",
				"embedding": embedding,
				"index":     0,
			},
		},
		"model": s.config.Model,
		"usage": map[string]interface{}{
			"prompt_tokens": 0,
			"total_tokens":  0,
		},
	}
	
	// Try to extract usage if available
	if promptEvalCount, ok := ollamaResp["prompt_eval_count"].(float64); ok {
		openAIResp["usage"].(map[string]interface{})["prompt_tokens"] = int(promptEvalCount)
		openAIResp["usage"].(map[string]interface{})["total_tokens"] = int(promptEvalCount)
	}
	
	responseJSON, err := json.Marshal(openAIResp)
	if err != nil {
		log.Printf("!!! Error marshaling OpenAI embeddings response: %v !!!", err)
		http.Error(w, "Failed to format response", http.StatusInternalServerError)
		return
	}
	
	// Log response preview for debugging
	responsePreview := string(responseJSON)
	if len(responsePreview) > 1000 {
		responsePreview = responsePreview[:1000] + "..."
	}
	log.Printf(">>> OpenAI embeddings response preview: %s <<<", responsePreview)
	dataArray := openAIResp["data"].([]map[string]interface{})
	log.Printf(">>> Response structure: object=%v, data length=%d, first embedding length=%d <<<",
		openAIResp["object"], len(dataArray), len(embedding))
	
	w.Header().Set("Content-Type", "application/json")
	w.Write(responseJSON)
	log.Printf("<<< Converted and sent OpenAI embeddings format response (%d bytes, embedding size: %d) <<<", 
		len(responseJSON), len(embedding))
}

// handleBatchEmbeddings handles batch embedding requests
func (s *Server) handleBatchEmbeddings(w http.ResponseWriter, r *http.Request, inputs []interface{}, requestData map[string]interface{}) {
	// Process each input separately
	embeddings := []map[string]interface{}{}
	
	for idx, input := range inputs {
		// Create single request for this input
		singleRequest := make(map[string]interface{})
		for k, v := range requestData {
			singleRequest[k] = v
		}
		singleRequest["input"] = input
		singleRequest["model"] = s.config.Model
		
		// Convert to Ollama format
		singleRequest["prompt"] = input
		
		modifiedBody, err := json.Marshal(singleRequest)
		if err != nil {
			log.Printf("!!! Failed to marshal batch embedding request %d: %v !!!", idx, err)
			continue
		}
		
		// Collect headers
		headers := make(map[string]string)
		for key, values := range r.Header {
			if len(values) > 0 && !strings.HasPrefix(strings.ToLower(key), "host") {
				headers[key] = values[0]
			}
		}
		headers["Content-Type"] = "application/json"
		
		// Proxy to Ollama
		resp, err := s.ollamaClient.ProxyRequest(
			"POST",
			"/api/embeddings",
			bytes.NewReader(modifiedBody),
			headers,
		)
		if err != nil {
			log.Printf("!!! Failed to proxy batch embedding request %d: %v !!!", idx, err)
			continue
		}
		
		if resp.StatusCode != http.StatusOK {
			log.Printf("!!! Ollama returned status %d for batch embedding %d !!!", resp.StatusCode, idx)
			resp.Body.Close()
			continue
		}
		
		// Read response
		bodyBytes, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			log.Printf("!!! Error reading batch embedding response %d: %v !!!", idx, err)
			continue
		}
		
		var ollamaResp map[string]interface{}
		if err := json.Unmarshal(bodyBytes, &ollamaResp); err != nil {
			log.Printf("!!! Error parsing batch embedding response %d: %v !!!", idx, err)
			continue
		}
		
		// Extract embedding
		embedding, ok := ollamaResp["embedding"].([]interface{})
		if !ok {
			if embeddingFloat, ok := ollamaResp["embedding"].([]float64); ok {
				embedding = make([]interface{}, len(embeddingFloat))
				for i, v := range embeddingFloat {
					embedding[i] = v
				}
			} else {
				log.Printf("!!! Invalid embedding format in batch response %d !!!", idx)
				continue
			}
		}
		
		embeddings = append(embeddings, map[string]interface{}{
			"object":    "embedding",
			"embedding": embedding,
			"index":     idx,
		})
	}
	
	if len(embeddings) == 0 {
		log.Printf("!!! No embeddings generated from batch request !!!")
		http.Error(w, "Failed to generate embeddings", http.StatusInternalServerError)
		return
	}
	
	// Create OpenAI format response
	openAIResp := map[string]interface{}{
		"object": "list",
		"data":   embeddings,
		"model":  s.config.Model,
		"usage": map[string]interface{}{
			"prompt_tokens": 0,
			"total_tokens":  0,
		},
	}
	
	responseJSON, err := json.Marshal(openAIResp)
	if err != nil {
		log.Printf("!!! Error marshaling batch embeddings response: %v !!!", err)
		http.Error(w, "Failed to format response", http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.Write(responseJSON)
	log.Printf("<<< Converted and sent batch OpenAI embeddings format response (%d bytes, %d embeddings) <<<", 
		len(responseJSON), len(embeddings))
}

// handleOllamaEmbedding handles Ollama format embedding requests (with "prompt" field)
// and converts response to OpenAI format for compatibility
func (s *Server) handleOllamaEmbedding(w http.ResponseWriter, r *http.Request, body []byte, requestData map[string]interface{}) {
	// Replace model parameter
	requestData["model"] = s.config.Model
	
	// Re-serialize (keep "prompt" as is for Ollama)
	modifiedBody, err := json.Marshal(requestData)
	if err != nil {
		log.Printf("Failed to marshal Ollama embeddings request: %v", err)
		http.Error(w, "Failed to modify request", http.StatusInternalServerError)
		return
	}
	
	// Log the request being sent to Ollama
	bodyPreview := string(modifiedBody)
	if len(bodyPreview) > 500 {
		bodyPreview = bodyPreview[:500] + "..."
	}
	log.Printf(">>> Request to Ollama: %s <<<", bodyPreview)
	
	// Collect headers
	headers := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 && !strings.HasPrefix(strings.ToLower(key), "host") {
			headers[key] = values[0]
		}
	}
	headers["Content-Type"] = "application/json"
	
	log.Printf(">>> Proxying Ollama format embeddings request to Ollama (model: %s) <<<", s.config.Model)
	
	// Proxy to Ollama
	resp, err := s.ollamaClient.ProxyRequest(
		"POST",
		"/api/embeddings",
		bytes.NewReader(modifiedBody),
		headers,
	)
	if err != nil {
		log.Printf("!!! Failed to proxy Ollama embeddings request: %v !!!", err)
		http.Error(w, "Failed to proxy request", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		// Read error response body for debugging
		errorBody, _ := io.ReadAll(resp.Body)
		log.Printf("!!! Ollama returned status %d for embeddings !!!", resp.StatusCode)
		log.Printf("!!! Ollama error response: %s !!!", string(errorBody))
		w.WriteHeader(resp.StatusCode)
		w.Write(errorBody)
		return
	}
	
	// Read Ollama response
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("!!! Error reading Ollama embeddings response: %v !!!", err)
		http.Error(w, "Failed to read response", http.StatusInternalServerError)
		return
	}
	
	var ollamaResp map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &ollamaResp); err != nil {
		log.Printf("!!! Error parsing Ollama embeddings response: %v, body: %s !!!", err, string(bodyBytes))
		http.Error(w, "Failed to parse response", http.StatusInternalServerError)
		return
	}
	
	// Extract embedding vector
	embedding, ok := ollamaResp["embedding"].([]interface{})
	if !ok {
		// Try to convert if it's a different type
		if embeddingFloat, ok := ollamaResp["embedding"].([]float64); ok {
			embedding = make([]interface{}, len(embeddingFloat))
			for i, v := range embeddingFloat {
				embedding[i] = v
			}
		} else {
			log.Printf("!!! Invalid embedding format in Ollama response: %T, full response: %s !!!", 
				ollamaResp["embedding"], string(bodyBytes))
			http.Error(w, "Invalid embedding format", http.StatusInternalServerError)
			return
		}
	}
	
	log.Printf(">>> Extracted embedding vector: length=%d <<<", len(embedding))
	
	// Convert to OpenAI format
	openAIResp := map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"object":    "embedding",
				"embedding": embedding,
				"index":     0,
			},
		},
		"model": s.config.Model,
		"usage": map[string]interface{}{
			"prompt_tokens": 0,
			"total_tokens":  0,
		},
	}
	
	// Try to extract usage if available
	if promptEvalCount, ok := ollamaResp["prompt_eval_count"].(float64); ok {
		openAIResp["usage"].(map[string]interface{})["prompt_tokens"] = int(promptEvalCount)
		openAIResp["usage"].(map[string]interface{})["total_tokens"] = int(promptEvalCount)
	}
	
	responseJSON, err := json.Marshal(openAIResp)
	if err != nil {
		log.Printf("!!! Error marshaling OpenAI embeddings response: %v !!!", err)
		http.Error(w, "Failed to format response", http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.Write(responseJSON)
	log.Printf("<<< Converted and sent OpenAI embeddings format response (%d bytes, embedding size: %d) <<<", 
		len(responseJSON), len(embedding))
}
