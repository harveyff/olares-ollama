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
	
	// Log input array details if it's an array
	if inputRaw, ok := requestData["input"]; ok {
		if inputArray, ok := inputRaw.([]interface{}); ok {
			log.Printf(">>> Input array length: %d <<<", len(inputArray))
			if len(inputArray) > 0 && len(inputArray) <= 3 {
				// Log first few items for debugging
				for i, item := range inputArray {
					if i >= 3 {
						break
					}
					itemStr := fmt.Sprintf("%v", item)
					if len(itemStr) > 100 {
						itemStr = itemStr[:100] + "..."
					}
					log.Printf(">>>   Input[%d]: %s (type=%T) <<<", i, itemStr, item)
				}
			}
		}
	}
	
	log.Printf(">>> Request path: %s <<<", r.URL.Path)
	log.Printf(">>> Request method: %s <<<", r.Method)
	log.Printf(">>> Request Content-Type: %s <<<", r.Header.Get("Content-Type"))
	
	// Check if this is OpenAI format (has "input") or Ollama format (has "prompt")
	inputRaw, hasInput := requestData["input"]
	_, hasPrompt := requestData["prompt"]
	
	log.Printf(">>> Request format detection: hasInput=%v, hasPrompt=%v <<<", hasInput, hasPrompt)
	
	// If it's Ollama format (has "prompt" but no "input"), return Ollama format
	if hasPrompt && !hasInput {
		log.Printf(">>> Detected Ollama format (prompt field), routing to handleOllamaEmbedding <<<")
		s.handleOllamaEmbedding(w, r, body, requestData)
		return
	}
	
	// Check if request is from OpenWebUI with ollama type
	// OpenWebUI sends {"input": [...]} but expects {"embeddings": [...]} when set to ollama type
	// We detect this by checking the endpoint path (/api/embed is used for ollama type)
	// For now, since OpenWebUI uses /api/embed for ollama, we return Ollama format
	// If it's OpenAI format request (has "input"), check if it's batch
	var inputs []interface{}
	isBatch := false
	
	if hasInput {
		if inputArray, ok := inputRaw.([]interface{}); ok && len(inputArray) > 0 {
			// Array input: check if batch (multiple items) or single (one item)
			if len(inputArray) > 1 {
				isBatch = true
				inputs = inputArray
				log.Printf(">>> [BATCH] Detected batch embeddings request: %d inputs <<<", len(inputs))
			} else {
				// Single item in array - treat as single request
				inputs = inputArray
				log.Printf(">>> [SINGLE] Single input in array format (array length=1) <<<")
			}
		} else if inputStr, ok := inputRaw.(string); ok {
			// Single string input
			inputs = []interface{}{inputStr}
			log.Printf(">>> [SINGLE] Single string input <<<")
		} else {
			log.Printf("!!! [ERROR] Invalid input type: %T, value: %v !!!", inputRaw, inputRaw)
		}
	} else {
		log.Printf("!!! [ERROR] No input field found in request !!!")
	}
	
	log.Printf(">>> Request routing decision: isBatch=%v, inputs count=%d <<<", isBatch, len(inputs))
	
	// If batch request (multiple inputs), process each input separately
	if isBatch && len(inputs) > 1 {
		log.Printf(">>> [ROUTING] Routing to handleBatchEmbeddings <<<")
		s.handleBatchEmbeddings(w, r, inputs, requestData)
		return
	}
	
	// Single embedding request - format will be determined by endpoint path in handleSingleEmbedding
	log.Printf(">>> [ROUTING] Routing to handleSingleEmbedding <<<")
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
	
	// Get model list from Ollama
	headers := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 && !strings.HasPrefix(strings.ToLower(key), "host") {
			headers[key] = values[0]
		}
	}
	
	// Proxy request to Ollama /api/tags
	resp, err := s.ollamaClient.ProxyRequest(
		"GET",
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
	
	// Convert Ollama format to OpenAI format
	models, ok := ollamaResponse["models"].([]interface{})
	if !ok {
		log.Printf("Invalid models format in ollama response")
		http.Error(w, "Invalid response format", http.StatusInternalServerError)
		return
	}
	
	openAIData := []map[string]interface{}{}
	for _, model := range models {
		modelMap, ok := model.(map[string]interface{})
		if !ok {
			continue
		}
		
		// Get model name
		modelName, ok := modelMap["name"].(string)
		if !ok {
			continue
		}
		
		// Filter: only keep the configured model
		if modelName != s.config.Model {
			continue
		}
		
		// Convert modified_at to Unix timestamp
		var created int64 = 0
		if modifiedAtStr, ok := modelMap["modified_at"].(string); ok {
			if modifiedAt, err := time.Parse(time.RFC3339, modifiedAtStr); err == nil {
				created = modifiedAt.Unix()
			}
		} else if modifiedAtFloat, ok := modelMap["modified_at"].(float64); ok {
			// Sometimes modified_at might be a timestamp directly
			created = int64(modifiedAtFloat)
		}
		
		openAIData = append(openAIData, map[string]interface{}{
			"id":       modelName,
			"object":   "model",
			"created":  created,
			"owned_by": "library",
		})
	}
	
	// Return OpenAI format with "object" field first
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   openAIData,
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

// handleOpenAICompletions handles OpenAI compatible text completions endpoint
func (s *Server) handleOpenAICompletions(w http.ResponseWriter, r *http.Request) {
	log.Printf("=== OpenAI Completions endpoint: %s %s, Method=%s, RemoteAddr=%s ===", 
		r.Method, r.URL.Path, r.Method, r.RemoteAddr)
	
	if r.Method == "OPTIONS" {
		log.Printf("OpenAI Completions: Handling OPTIONS preflight")
		w.WriteHeader(http.StatusNoContent)
		return
	}
	
	if r.Method == "GET" {
		log.Printf("OpenAI Completions received GET request (health check)")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "ok",
		})
		return
	}
	
	if r.Method != "POST" {
		log.Printf("!!! OpenAI Completions received unsupported method: %s !!!", r.Method)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": "Method not allowed",
		})
		return
	}
	
	log.Printf("*** Handling OpenAI Completions POST request from %s ***", r.RemoteAddr)
	
	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("!!! Failed to read OpenAI completions request body: %v !!!", err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()
	
	if len(body) == 0 {
		log.Printf("!!! OpenAI completions request body is empty !!!")
		http.Error(w, "Request body cannot be empty", http.StatusBadRequest)
		return
	}
	
	// Parse OpenAI format request
	var openaiRequest map[string]interface{}
	if err := json.Unmarshal(body, &openaiRequest); err != nil {
		log.Printf("!!! Failed to parse OpenAI completions JSON: %v, body: %s !!!", err, string(body))
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	// Extract prompt
	prompt, ok := openaiRequest["prompt"].(string)
	if !ok {
		// Try array format
		if promptArray, ok := openaiRequest["prompt"].([]interface{}); ok && len(promptArray) > 0 {
			if promptStr, ok := promptArray[0].(string); ok {
				prompt = promptStr
			}
		}
		if prompt == "" {
			log.Printf("!!! Missing or invalid 'prompt' field in OpenAI completions request !!!")
			http.Error(w, "Missing 'prompt' field", http.StatusBadRequest)
			return
		}
	}
	
	// Check if streaming
	stream := false
	if streamVal, ok := openaiRequest["stream"]; ok {
		if streamBool, ok := streamVal.(bool); ok {
			stream = streamBool
		}
	}
	
	// Build Ollama request (use /api/generate for text completions)
	ollamaRequest := map[string]interface{}{
		"model":  s.config.Model,
		"prompt": prompt,
		"stream": stream,
	}
	
	// Copy other parameters if present
	if maxTokens, ok := openaiRequest["max_tokens"]; ok {
		ollamaRequest["num_predict"] = maxTokens
	}
	if temperature, ok := openaiRequest["temperature"]; ok {
		ollamaRequest["temperature"] = temperature
	}
	if topP, ok := openaiRequest["top_p"]; ok {
		ollamaRequest["top_p"] = topP
	}
	if stop, ok := openaiRequest["stop"]; ok {
		ollamaRequest["stop"] = stop
	}
	
	modifiedBody, err := json.Marshal(ollamaRequest)
	if err != nil {
		log.Printf("!!! Failed to marshal Ollama completions request: %v !!!", err)
		http.Error(w, "Failed to prepare request", http.StatusInternalServerError)
		return
	}
	
	log.Printf(">>> Converted OpenAI completions to Ollama format: body size=%d bytes, model=%s, stream=%v <<<",
		len(modifiedBody), s.config.Model, stream)
	
	// Collect headers
	headers := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 && !strings.HasPrefix(strings.ToLower(key), "host") {
			headers[key] = values[0]
		}
	}
	headers["Content-Type"] = "application/json"
	
	log.Printf(">>> Proxying OpenAI completions request to Ollama /api/generate (model: %s) <<<", s.config.Model)
	
	// Proxy to Ollama
	resp, err := s.ollamaClient.ProxyRequest(
		"POST",
		"/api/generate",
		bytes.NewReader(modifiedBody),
		headers,
	)
	if err != nil {
		log.Printf("!!! Failed to proxy OpenAI completions request to Ollama: %v !!!", err)
		http.Error(w, "Failed to proxy request", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	
	log.Printf("<<< Ollama returned status %d for OpenAI completions request <<<", resp.StatusCode)
	
	// Set OpenAI-compatible response headers
	if stream {
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
	
	log.Printf(">>> Starting to convert Ollama response to OpenAI completions format (stream=%v) <<<", stream)
	
	// Convert Ollama response to OpenAI format
	if stream {
		// Handle streaming response
		s.convertOllamaGenerateStreamToOpenAI(w, resp.Body, s.config.Model)
	} else {
		// Handle non-streaming response
		s.convertOllamaGenerateToOpenAI(w, resp.Body, s.config.Model)
	}
}

// convertOllamaGenerateToOpenAI converts Ollama /api/generate response to OpenAI completions format
func (s *Server) convertOllamaGenerateToOpenAI(w http.ResponseWriter, body io.Reader, modelName string) {
	bodyBytes, err := io.ReadAll(body)
	if err != nil {
		log.Printf("!!! Error reading Ollama generate response: %v !!!", err)
		http.Error(w, "Failed to read response", http.StatusInternalServerError)
		return
	}
	
	var ollamaResp map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &ollamaResp); err != nil {
		log.Printf("!!! Error parsing Ollama generate response: %v, body: %s !!!", err, string(bodyBytes))
		http.Error(w, "Failed to parse response", http.StatusInternalServerError)
		return
	}
	
	// Extract response text
	responseText, _ := ollamaResp["response"].(string)
	
	// Determine finish_reason
	finishReason := "stop"
	if done, ok := ollamaResp["done"].(bool); ok && !done {
		finishReason = "length" // If not done, assume length limit
	}
	
	// Create OpenAI format response
	openAIResp := map[string]interface{}{
		"id":      fmt.Sprintf("cmpl-%d", time.Now().Unix()),
		"object":  "text_completion",
		"created": time.Now().Unix(),
		"model":   modelName,
		"choices": []map[string]interface{}{
			{
				"text":          responseText,
				"index":         0,
				"logprobs":      nil,
				"finish_reason": finishReason,
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     0,
			"completion_tokens": 0,
			"total_tokens":      0,
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
		log.Printf("!!! Error marshaling OpenAI completions response: %v !!!", err)
		http.Error(w, "Failed to format response", http.StatusInternalServerError)
		return
	}
	
	w.Write(responseJSON)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	log.Printf("<<< Converted and sent OpenAI completions format response (%d bytes) <<<", len(responseJSON))
}

// convertOllamaGenerateStreamToOpenAI converts Ollama /api/generate streaming response to OpenAI SSE format
func (s *Server) convertOllamaGenerateStreamToOpenAI(w http.ResponseWriter, body io.Reader, modelName string) {
	flusher, hasFlusher := w.(http.Flusher)
	scanner := bufio.NewScanner(body)
	responseID := fmt.Sprintf("cmpl-%d", time.Now().Unix())
	created := time.Now().Unix()
	var totalBytes int64
	var fullText strings.Builder
	
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		
		var ollamaResp map[string]interface{}
		if err := json.Unmarshal(line, &ollamaResp); err != nil {
			log.Printf("!!! Error parsing Ollama generate stream line: %v, line: %s !!!", err, string(line))
			continue
		}
		
		// Check if done
		done, _ := ollamaResp["done"].(bool)
		
		// Extract response text
		responseText, _ := ollamaResp["response"].(string)
		if responseText != "" {
			fullText.WriteString(responseText)
		}
		
		if done {
			// Send final chunk with finish_reason
			finalChunk := map[string]interface{}{
				"id":      responseID,
				"object":  "text_completion",
				"created": created,
				"model":   modelName,
				"choices": []map[string]interface{}{
					{
						"index":         0,
						"text":          "",
						"logprobs":      nil,
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
		
		// Send incremental chunk
		if responseText != "" {
			chunk := map[string]interface{}{
				"id":      responseID,
				"object":  "text_completion",
				"created": created,
				"model":   modelName,
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"text":  responseText,
						"logprobs": nil,
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
	
	log.Printf("<<< Converted and sent OpenAI completions stream response (%d bytes) <<<", totalBytes)
}

// handleSingleEmbedding handles a single embedding request and returns Ollama format
func (s *Server) handleSingleEmbedding(w http.ResponseWriter, r *http.Request, body []byte, requestData map[string]interface{}) {
	var err error
	
	log.Printf(">>> [handleSingleEmbedding] Starting single embedding request processing <<<")
	log.Printf(">>> [handleSingleEmbedding] Endpoint path: %s <<<", r.URL.Path)
	log.Printf(">>> [handleSingleEmbedding] Original requestData keys: %v <<<", getMapKeys(requestData))
	
	// Replace model parameter
	originalModel := requestData["model"]
	requestData["model"] = s.config.Model
	log.Printf(">>> [handleSingleEmbedding] Model replacement: %v -> %s <<<", originalModel, s.config.Model)
	
	// Convert "input" to "prompt" for Ollama if needed
	if input, ok := requestData["input"]; ok {
		log.Printf(">>> [handleSingleEmbedding] Converting input to prompt, input type: %T <<<", input)
		// Ollama uses "prompt" instead of "input", and it must be a string
		var promptStr string
		if inputArray, ok := input.([]interface{}); ok && len(inputArray) > 0 {
			log.Printf(">>> [handleSingleEmbedding] Input is array, length: %d <<<", len(inputArray))
			// If input is an array, take the first element
			if firstItem, ok := inputArray[0].(string); ok {
				promptStr = firstItem
				log.Printf(">>> [handleSingleEmbedding] Extracted prompt from array[0], length: %d chars <<<", len(promptStr))
			} else {
				log.Printf("!!! [handleSingleEmbedding] Invalid input array element type: %T, value: %v !!!", inputArray[0], inputArray[0])
				http.Error(w, "Invalid input format", http.StatusBadRequest)
				return
			}
		} else if inputStr, ok := input.(string); ok {
			// If input is already a string, use it directly
			promptStr = inputStr
			log.Printf(">>> [handleSingleEmbedding] Input is string, length: %d chars <<<", len(promptStr))
		} else {
			log.Printf("!!! [handleSingleEmbedding] Invalid input type: %T, value: %v !!!", input, input)
			http.Error(w, "Invalid input format", http.StatusBadRequest)
			return
		}
		requestData["prompt"] = promptStr
		// Remove "input" field as Ollama doesn't use it
		delete(requestData, "input")
		log.Printf(">>> [handleSingleEmbedding] Converted input to prompt, removed input field <<<")
	} else {
		log.Printf(">>> [handleSingleEmbedding] No input field found, using existing prompt field <<<")
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
	log.Printf(">>> [handleSingleEmbedding] Sending request to Ollama /api/embeddings, body size: %d bytes <<<", len(modifiedBody))
	resp, err := s.ollamaClient.ProxyRequest(
		"POST",
		"/api/embeddings",
		bytes.NewReader(modifiedBody),
		headers,
	)
	if err != nil {
		log.Printf("!!! [handleSingleEmbedding] Failed to proxy embeddings request: %v !!!", err)
		http.Error(w, "Failed to proxy request", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	
	log.Printf(">>> [handleSingleEmbedding] Ollama response received, status: %d <<<", resp.StatusCode)
	
	// Embeddings API should NOT be streaming - log headers for debugging
	log.Printf(">>> [handleSingleEmbedding] Ollama response headers: Content-Type=%s, Transfer-Encoding=%s, Content-Length=%s <<<",
		resp.Header.Get("Content-Type"), resp.Header.Get("Transfer-Encoding"), resp.Header.Get("Content-Length"))
	
	// Copy response headers from Ollama (except for ones that should be controlled by the response writer)
	// Note: We'll handle Content-Type separately to preserve charset (e.g., "application/json; charset=utf-8")
	contentType := resp.Header.Get("Content-Type")
	for key, values := range resp.Header {
		keyLower := strings.ToLower(key)
		// Skip headers that should be controlled by the response writer
		// Also skip Transfer-Encoding for embeddings (should not be chunked/streaming)
		// Skip Content-Type here, we'll set it explicitly to preserve charset
		if keyLower != "content-length" && keyLower != "transfer-encoding" && keyLower != "connection" && keyLower != "content-type" {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
	}
	
	// Copy Content-Type exactly from Ollama (including charset if present)
	// This ensures exact match with Ollama's response format (e.g., "application/json; charset=utf-8")
	if contentType != "" {
		// Use Ollama's Content-Type exactly
		w.Header().Set("Content-Type", contentType)
		log.Printf(">>> Using Ollama Content-Type: %s <<<", contentType)
	} else {
		// Fallback if Ollama doesn't provide Content-Type
		w.Header().Set("Content-Type", "application/json")
		log.Printf(">>> Ollama didn't provide Content-Type, using default: application/json <<<")
	}
	
	if resp.StatusCode != http.StatusOK {
		// Read error response body for debugging
		errorBody, _ := io.ReadAll(resp.Body)
		log.Printf("!!! Ollama returned status %d for embeddings !!!", resp.StatusCode)
		log.Printf("!!! Ollama error response: %s !!!", string(errorBody))
		w.WriteHeader(resp.StatusCode)
		w.Write(errorBody)
		return
	}
	
	// Log all headers that will be sent to client (before WriteHeader)
	log.Printf(">>> Final response headers to client: <<<")
	for key, values := range w.Header() {
		for _, value := range values {
			log.Printf(">>>   %s: %s <<<", key, value)
		}
	}
	
	// Read response body first to log it
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("!!! [handleSingleEmbedding] Error reading Ollama embeddings response: %v !!!", err)
		http.Error(w, "Failed to read response", http.StatusInternalServerError)
		return
	}
	
	log.Printf(">>> [handleSingleEmbedding] Response body size: %d bytes <<<", len(bodyBytes))
	
	// Log response body for debugging (first 500 chars)
	bodyPreview = string(bodyBytes)
	if len(bodyPreview) > 500 {
		bodyPreview = bodyPreview[:500] + "..."
	}
	log.Printf(">>> [handleSingleEmbedding] Ollama embeddings response body preview: %s <<<", bodyPreview)
	
	// Parse Ollama response
	var ollamaResp map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &ollamaResp); err != nil {
		log.Printf("!!! [handleSingleEmbedding] Error parsing Ollama embeddings response: %v, body: %s !!!", err, string(bodyBytes))
		http.Error(w, "Failed to parse response", http.StatusInternalServerError)
		return
	}
	
	log.Printf(">>> [handleSingleEmbedding] Parsed Ollama response, keys: %v <<<", getMapKeys(ollamaResp))
	
	// Extract embedding vector
	// Ollama may return either "embedding" (single) or "embeddings" (array)
	var embedding []interface{}
	var found bool
	
	log.Printf(">>> [handleSingleEmbedding] Attempting to extract embedding vector... <<<")
	
	// First try "embeddings" (plural) - array format
	if embeddingsArray, ok := ollamaResp["embeddings"].([]interface{}); ok {
		log.Printf(">>> [handleSingleEmbedding] Found 'embeddings' field, type: []interface{}, length: %d <<<", len(embeddingsArray))
		if len(embeddingsArray) > 0 {
			// Take the first embedding from the array
			log.Printf(">>> [handleSingleEmbedding] embeddings[0] type: %T <<<", embeddingsArray[0])
			if firstEmbedding, ok := embeddingsArray[0].([]interface{}); ok {
				embedding = firstEmbedding
				found = true
				log.Printf(">>> [handleSingleEmbedding] ✓ Extracted embedding from 'embeddings' array[0] ([]interface{}, length=%d) <<<", len(embedding))
			} else if firstEmbeddingFloat, ok := embeddingsArray[0].([]float64); ok {
				// Convert []float64 to []interface{}
				embedding = make([]interface{}, len(firstEmbeddingFloat))
				for i, v := range firstEmbeddingFloat {
					embedding[i] = v
				}
				found = true
				log.Printf(">>> [handleSingleEmbedding] ✓ Extracted embedding from 'embeddings' array[0] ([]float64, length=%d) <<<", len(embedding))
			} else {
				log.Printf("!!! [handleSingleEmbedding] Invalid format in 'embeddings' array first element: %T, value preview: %v !!!", 
					embeddingsArray[0], fmt.Sprintf("%v", embeddingsArray[0])[:100])
			}
		} else {
			log.Printf(">>> [handleSingleEmbedding] 'embeddings' array is empty (length=0) <<<")
		}
	} else {
		log.Printf(">>> [handleSingleEmbedding] No 'embeddings' field found or wrong type <<<")
	}
	
	// If not found in "embeddings", try "embedding" (singular)
	if !found {
		log.Printf(">>> [handleSingleEmbedding] Trying 'embedding' field (singular)... <<<")
		embeddingRaw := ollamaResp["embedding"]
		if embeddingRaw == nil {
			log.Printf(">>> [handleSingleEmbedding] 'embedding' field is nil <<<")
		} else {
			log.Printf(">>> [handleSingleEmbedding] 'embedding' field exists, type: %T <<<", embeddingRaw)
			if embeddingSingle, ok := embeddingRaw.([]interface{}); ok {
				embedding = embeddingSingle
				found = true
				log.Printf(">>> [handleSingleEmbedding] ✓ Extracted embedding from 'embedding' field ([]interface{}, length=%d) <<<", len(embedding))
			} else if embeddingFloat, ok := embeddingRaw.([]float64); ok {
				// Convert []float64 to []interface{}
				log.Printf(">>> [handleSingleEmbedding] Converting []float64 to []interface{}, length: %d <<<", len(embeddingFloat))
				embedding = make([]interface{}, len(embeddingFloat))
				for i, v := range embeddingFloat {
					embedding[i] = v
				}
				found = true
				log.Printf(">>> [handleSingleEmbedding] ✓ Extracted embedding from 'embedding' field ([]float64, length=%d) <<<", len(embedding))
			} else {
				log.Printf("!!! [handleSingleEmbedding] 'embedding' field has unexpected type: %T, value preview: %v !!!", 
					embeddingRaw, fmt.Sprintf("%v", embeddingRaw)[:200])
			}
		}
	}
	
	// Check endpoint path to determine response format
	// /api/embed is used by OpenWebUI for ollama type, expects Ollama format: {"embeddings": [[...]]}
	// /api/embeddings or other endpoints expect OpenAI format: {"data": [{"embedding": [...]}]}
	isOllamaFormat := r.URL.Path == "/api/embed"
	
	log.Printf(">>> [handleSingleEmbedding] Response format decision: isOllamaFormat=%v (path=%s), found=%v, embedding length=%d <<<", 
		isOllamaFormat, r.URL.Path, found, len(embedding))
	
	// If Ollama format and embeddings is empty array, return an error
	// ChromaDB cannot handle empty embedding vectors, so we should return an error instead
	if isOllamaFormat && !found {
		log.Printf(">>> [handleSingleEmbedding] Handling empty embeddings case for Ollama format... <<<")
		// Check if embeddings exists but is empty
		if embeddingsArray, ok := ollamaResp["embeddings"].([]interface{}); ok && len(embeddingsArray) == 0 {
			// Ollama returned empty embeddings array - this indicates the model failed to generate embeddings
			// We should return an error instead of empty embeddings, as ChromaDB cannot handle empty vectors
			log.Printf("!!! [handleSingleEmbedding] Ollama returned empty embeddings array - model failed to generate embeddings !!!")
			log.Printf("!!! [handleSingleEmbedding] This may indicate: 1) Model issue, 2) Request format issue, 3) Model not properly loaded !!!")
			
			// Return error response in Ollama format
			errorResponse := map[string]interface{}{
				"error": "Failed to generate embeddings: Ollama returned empty embeddings array. Please check if the model is properly loaded and the request format is correct.",
			}
			responseJSON, err := json.Marshal(errorResponse)
			if err != nil {
				log.Printf("!!! [handleSingleEmbedding] Error marshaling error response: %v !!!", err)
				http.Error(w, "Failed to generate embeddings", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			w.Write(responseJSON)
			return
		}
	}
	
	if !found || len(embedding) == 0 {
		log.Printf("!!! [handleSingleEmbedding] Invalid embedding format in Ollama response: embedding=%v, embeddings=%v, keys: %v !!!", 
			ollamaResp["embedding"], ollamaResp["embeddings"], getMapKeys(ollamaResp))
		http.Error(w, "Invalid embedding format or empty embedding", http.StatusInternalServerError)
		return
	}
	
	log.Printf(">>> [handleSingleEmbedding] Successfully extracted embedding, length=%d, preparing response... <<<", len(embedding))
	
	var responseJSON []byte
	
	if isOllamaFormat {
		log.Printf(">>> [handleSingleEmbedding] Formatting response as Ollama format... <<<")
		// Return Ollama format: {"embeddings": [[...]]}
		ollamaFormatResp := map[string]interface{}{
			"embeddings": [][]interface{}{embedding},
		}
		
		// Add other Ollama fields if available
		if promptEvalCount, ok := ollamaResp["prompt_eval_count"].(float64); ok {
			ollamaFormatResp["prompt_eval_count"] = int(promptEvalCount)
			log.Printf(">>> [handleSingleEmbedding] Added prompt_eval_count: %d <<<", int(promptEvalCount))
		}
		
		responseJSON, err = json.Marshal(ollamaFormatResp)
		if err != nil {
			log.Printf("!!! [handleSingleEmbedding] Error marshaling Ollama embeddings response: %v !!!", err)
			http.Error(w, "Failed to format response", http.StatusInternalServerError)
			return
		}
		log.Printf(">>> [handleSingleEmbedding] ✓ Converted to Ollama format: embeddings array with 1 item, embedding length=%d, response size=%d bytes <<<", 
			len(embedding), len(responseJSON))
	} else {
		log.Printf(">>> [handleSingleEmbedding] Formatting response as OpenAI format... <<<")
		// Return OpenAI format: {"data": [{"embedding": [...]}]}
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
			log.Printf(">>> [handleSingleEmbedding] Added usage: prompt_tokens=%d, total_tokens=%d <<<", 
				int(promptEvalCount), int(promptEvalCount))
		}
		
		responseJSON, err = json.Marshal(openAIResp)
		if err != nil {
			log.Printf("!!! [handleSingleEmbedding] Error marshaling OpenAI embeddings response: %v !!!", err)
			http.Error(w, "Failed to format response", http.StatusInternalServerError)
			return
		}
		log.Printf(">>> [handleSingleEmbedding] ✓ Converted to OpenAI format: data array with %d items, embedding length=%d, response size=%d bytes <<<", 
			len(openAIResp["data"].([]map[string]interface{})), len(embedding), len(responseJSON))
	}
	
	// Set status code
	log.Printf(">>> [handleSingleEmbedding] Writing response, status code: %d <<<", resp.StatusCode)
	w.WriteHeader(resp.StatusCode)
	
	// Write response
	bytesCopied, err := w.Write(responseJSON)
	if err != nil {
		log.Printf("!!! [handleSingleEmbedding] Error writing embeddings response: %v !!!", err)
		return
	}
	
	formatType := "Ollama"
	if !isOllamaFormat {
		formatType = "OpenAI"
	}
	log.Printf(">>> [handleSingleEmbedding] ✓ Successfully sent %s embeddings format response (%d bytes written) <<<", formatType, bytesCopied)
}

// getMapKeys returns the keys of a map as a slice of strings
func getMapKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// handleBatchEmbeddings handles batch embedding requests
func (s *Server) handleBatchEmbeddings(w http.ResponseWriter, r *http.Request, inputs []interface{}, requestData map[string]interface{}) {
	log.Printf(">>> [handleBatchEmbeddings] Starting batch embeddings processing, total inputs: %d <<<", len(inputs))
	log.Printf(">>> [handleBatchEmbeddings] Endpoint path: %s <<<", r.URL.Path)
	
	// Process each input separately
	embeddings := [][]interface{}{}
	var err error
	
	for idx, input := range inputs {
		log.Printf(">>> [handleBatchEmbeddings] Processing input %d/%d... <<<", idx+1, len(inputs))
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
		log.Printf(">>> [handleBatchEmbeddings] Sending request %d/%d to Ollama /api/embeddings, body size: %d bytes <<<", 
			idx+1, len(inputs), len(modifiedBody))
		resp, err := s.ollamaClient.ProxyRequest(
			"POST",
			"/api/embeddings",
			bytes.NewReader(modifiedBody),
			headers,
		)
		if err != nil {
			log.Printf("!!! [handleBatchEmbeddings] Failed to proxy batch embedding request %d/%d: %v !!!", idx+1, len(inputs), err)
			continue
		}
		
		log.Printf(">>> [handleBatchEmbeddings] Request %d/%d response received, status: %d <<<", idx+1, len(inputs), resp.StatusCode)
		
		if resp.StatusCode != http.StatusOK {
			log.Printf("!!! [handleBatchEmbeddings] Ollama returned status %d for batch embedding %d/%d !!!", resp.StatusCode, idx+1, len(inputs))
			resp.Body.Close()
			continue
		}
		
		// Read response
		bodyBytes, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			log.Printf("!!! [handleBatchEmbeddings] Error reading batch embedding response %d/%d: %v !!!", idx+1, len(inputs), err)
			continue
		}
		
		log.Printf(">>> [handleBatchEmbeddings] Request %d/%d response body size: %d bytes <<<", idx+1, len(inputs), len(bodyBytes))
		
		var ollamaResp map[string]interface{}
		if err := json.Unmarshal(bodyBytes, &ollamaResp); err != nil {
			log.Printf("!!! [handleBatchEmbeddings] Error parsing batch embedding response %d/%d: %v !!!", idx+1, len(inputs), err)
			continue
		}
		
		log.Printf(">>> [handleBatchEmbeddings] Request %d/%d parsed, response keys: %v <<<", idx+1, len(inputs), getMapKeys(ollamaResp))
		
		// Extract embedding vector
		// Ollama may return either "embedding" (single) or "embeddings" (array)
		var embedding []interface{}
		var found bool
		
		// First try "embeddings" (plural) - array format
		if embeddingsArray, ok := ollamaResp["embeddings"].([]interface{}); ok && len(embeddingsArray) > 0 {
			// Take the first embedding from the array
			if firstEmbedding, ok := embeddingsArray[0].([]interface{}); ok {
				embedding = firstEmbedding
				found = true
			} else if firstEmbeddingFloat, ok := embeddingsArray[0].([]float64); ok {
				// Convert []float64 to []interface{}
				embedding = make([]interface{}, len(firstEmbeddingFloat))
				for i, v := range firstEmbeddingFloat {
					embedding[i] = v
				}
				found = true
			}
		}
		
		// If not found in "embeddings", try "embedding" (singular)
		if !found {
			if embeddingSingle, ok := ollamaResp["embedding"].([]interface{}); ok {
				embedding = embeddingSingle
				found = true
			} else if embeddingFloat, ok := ollamaResp["embedding"].([]float64); ok {
				// Convert []float64 to []interface{}
				embedding = make([]interface{}, len(embeddingFloat))
				for i, v := range embeddingFloat {
					embedding[i] = v
				}
				found = true
			}
		}
		
		if !found || len(embedding) == 0 {
			log.Printf("!!! [handleBatchEmbeddings] Invalid embedding format in batch response %d/%d: embedding=%v, embeddings=%v !!!", 
				idx+1, len(inputs), ollamaResp["embedding"], ollamaResp["embeddings"])
			continue
		}
		
		log.Printf(">>> [handleBatchEmbeddings] ✓ Successfully extracted embedding %d/%d, length=%d <<<", 
			idx+1, len(inputs), len(embedding))
		embeddings = append(embeddings, embedding)
	}
	
	log.Printf(">>> [handleBatchEmbeddings] Batch processing complete: %d/%d embeddings extracted <<<", len(embeddings), len(inputs))
	
	if len(embeddings) == 0 {
		log.Printf("!!! [handleBatchEmbeddings] No embeddings generated from batch request (0/%d) !!!", len(inputs))
		http.Error(w, "Failed to generate embeddings", http.StatusInternalServerError)
		return
	}
	
	// Check endpoint path to determine response format
	// /api/embed is used by OpenWebUI for ollama type, expects Ollama format: {"embeddings": [[...], [...]]}
	// /api/embeddings or other endpoints expect OpenAI format: {"data": [{"embedding": [...]}, ...]}
	isOllamaFormat := r.URL.Path == "/api/embed"
	
	log.Printf(">>> [handleBatchEmbeddings] Formatting response: isOllamaFormat=%v, embeddings count=%d <<<", 
		isOllamaFormat, len(embeddings))
	
	var responseJSON []byte
	
	if isOllamaFormat {
		log.Printf(">>> [handleBatchEmbeddings] Formatting as Ollama format... <<<")
		// Return Ollama format: {"embeddings": [[...], [...]]}
		ollamaFormatResp := map[string]interface{}{
			"embeddings": embeddings,
		}
		responseJSON, err = json.Marshal(ollamaFormatResp)
		if err != nil {
			log.Printf("!!! [handleBatchEmbeddings] Error marshaling batch Ollama embeddings response: %v !!!", err)
			http.Error(w, "Failed to format response", http.StatusInternalServerError)
			return
		}
		log.Printf(">>> [handleBatchEmbeddings] ✓ Converted to Ollama format: embeddings array with %d items, response size=%d bytes <<<", 
			len(embeddings), len(responseJSON))
	} else {
		log.Printf(">>> [handleBatchEmbeddings] Formatting as OpenAI format... <<<")
		// Return OpenAI format: {"data": [{"embedding": [...]}, ...]}
		openAIData := []map[string]interface{}{}
		for idx, embedding := range embeddings {
			openAIData = append(openAIData, map[string]interface{}{
				"object":    "embedding",
				"embedding": embedding,
				"index":     idx,
			})
		}
		openAIResp := map[string]interface{}{
			"object": "list",
			"data":   openAIData,
			"model":  s.config.Model,
			"usage": map[string]interface{}{
				"prompt_tokens": 0,
				"total_tokens":  0,
			},
		}
		responseJSON, err = json.Marshal(openAIResp)
		if err != nil {
			log.Printf("!!! [handleBatchEmbeddings] Error marshaling batch OpenAI embeddings response: %v !!!", err)
			http.Error(w, "Failed to format response", http.StatusInternalServerError)
			return
		}
		log.Printf(">>> [handleBatchEmbeddings] ✓ Converted to OpenAI format: data array with %d items, response size=%d bytes <<<", 
			len(openAIData), len(responseJSON))
	}
	
	log.Printf(">>> [handleBatchEmbeddings] Writing response... <<<")
	w.Header().Set("Content-Type", "application/json")
	bytesWritten, err := w.Write(responseJSON)
	if err != nil {
		log.Printf("!!! [handleBatchEmbeddings] Error writing response: %v !!!", err)
		return
	}
	
	formatType := "Ollama"
	if !isOllamaFormat {
		formatType = "OpenAI"
	}
	log.Printf(">>> [handleBatchEmbeddings] ✓ Successfully sent %s batch embeddings response (%d items, %d bytes written) <<<", 
		formatType, len(embeddings), bytesWritten)
}

// handleOllamaEmbedding handles Ollama format embedding requests (with "prompt" field)
// and returns Ollama format response directly
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
	
	// Embeddings API should NOT be streaming - log headers for debugging
	log.Printf(">>> Ollama embeddings response headers: Content-Type=%s, Transfer-Encoding=%s, Content-Length=%s <<<",
		resp.Header.Get("Content-Type"), resp.Header.Get("Transfer-Encoding"), resp.Header.Get("Content-Length"))
	
	// Copy response headers from Ollama (except for ones that should be controlled by the response writer)
	// Note: We'll handle Content-Type separately to preserve charset (e.g., "application/json; charset=utf-8")
	contentType := resp.Header.Get("Content-Type")
	for key, values := range resp.Header {
		keyLower := strings.ToLower(key)
		// Skip headers that should be controlled by the response writer
		// Also skip Transfer-Encoding for embeddings (should not be chunked/streaming)
		// Skip Content-Type here, we'll set it explicitly to preserve charset
		if keyLower != "content-length" && keyLower != "transfer-encoding" && keyLower != "connection" && keyLower != "content-type" {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
	}
	
	// Copy Content-Type exactly from Ollama (including charset if present)
	// This ensures exact match with Ollama's response format (e.g., "application/json; charset=utf-8")
	if contentType != "" {
		// Use Ollama's Content-Type exactly
		w.Header().Set("Content-Type", contentType)
		log.Printf(">>> Using Ollama Content-Type: %s <<<", contentType)
	} else {
		// Fallback if Ollama doesn't provide Content-Type
		w.Header().Set("Content-Type", "application/json")
		log.Printf(">>> Ollama didn't provide Content-Type, using default: application/json <<<")
	}
	
	if resp.StatusCode != http.StatusOK {
		// Read error response body for debugging
		errorBody, _ := io.ReadAll(resp.Body)
		log.Printf("!!! Ollama returned status %d for embeddings !!!", resp.StatusCode)
		log.Printf("!!! Ollama error response: %s !!!", string(errorBody))
		w.WriteHeader(resp.StatusCode)
		w.Write(errorBody)
		return
	}
	
	// Log all headers that will be sent to client (before WriteHeader)
	log.Printf(">>> Final response headers to client (Ollama format): <<<")
	for key, values := range w.Header() {
		for _, value := range values {
			log.Printf(">>>   %s: %s <<<", key, value)
		}
	}
	
	// Read response body first to log it
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("!!! Error reading Ollama embeddings response: %v !!!", err)
		http.Error(w, "Failed to read response", http.StatusInternalServerError)
		return
	}
	
	// Log response body for debugging (first 500 chars)
	bodyPreview = string(bodyBytes)
	if len(bodyPreview) > 500 {
		bodyPreview = bodyPreview[:500] + "..."
	}
	log.Printf(">>> Ollama embeddings response body preview (Ollama format): %s <<<", bodyPreview)
	
	// Parse Ollama response
	var ollamaResp map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &ollamaResp); err != nil {
		log.Printf("!!! Error parsing Ollama embeddings response: %v, body: %s !!!", err, string(bodyBytes))
		http.Error(w, "Failed to parse response", http.StatusInternalServerError)
		return
	}
	
	// Extract embedding vector
	// Ollama may return either "embedding" (single) or "embeddings" (array)
	var embedding []interface{}
	var found bool
	
	// First try "embeddings" (plural) - array format
	if embeddingsArray, ok := ollamaResp["embeddings"].([]interface{}); ok && len(embeddingsArray) > 0 {
		// Take the first embedding from the array
		if firstEmbedding, ok := embeddingsArray[0].([]interface{}); ok {
			embedding = firstEmbedding
			found = true
			log.Printf(">>> Extracted embedding from 'embeddings' array (length=%d) <<<", len(embedding))
		} else if firstEmbeddingFloat, ok := embeddingsArray[0].([]float64); ok {
			// Convert []float64 to []interface{}
			embedding = make([]interface{}, len(firstEmbeddingFloat))
			for i, v := range firstEmbeddingFloat {
				embedding[i] = v
			}
			found = true
			log.Printf(">>> Extracted embedding from 'embeddings' array (float64, length=%d) <<<", len(embedding))
		} else {
			log.Printf("!!! Invalid format in 'embeddings' array first element: %T !!!", embeddingsArray[0])
		}
	}
	
	// If not found in "embeddings", try "embedding" (singular)
	if !found {
		if embeddingSingle, ok := ollamaResp["embedding"].([]interface{}); ok {
			embedding = embeddingSingle
			found = true
			log.Printf(">>> Extracted embedding from 'embedding' field (length=%d) <<<", len(embedding))
		} else if embeddingFloat, ok := ollamaResp["embedding"].([]float64); ok {
			// Convert []float64 to []interface{}
			embedding = make([]interface{}, len(embeddingFloat))
			for i, v := range embeddingFloat {
				embedding[i] = v
			}
			found = true
			log.Printf(">>> Extracted embedding from 'embedding' field (float64, length=%d) <<<", len(embedding))
		}
	}
	
	if !found || len(embedding) == 0 {
		log.Printf("!!! Invalid embedding format in Ollama response: embedding=%v, embeddings=%v, keys: %v !!!", 
			ollamaResp["embedding"], ollamaResp["embeddings"], getMapKeys(ollamaResp))
		http.Error(w, "Invalid embedding format or empty embedding", http.StatusInternalServerError)
		return
	}
	
	// Convert to OpenAI format (OpenWebUI expects this format)
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
	
	log.Printf(">>> Converted to OpenAI format: data array with %d items, embedding length=%d <<<", 
		len(openAIResp["data"].([]map[string]interface{})), len(embedding))
	
	// Set status code
	w.WriteHeader(resp.StatusCode)
	
	// Write OpenAI format response
	bytesCopied, err := w.Write(responseJSON)
	if err != nil {
		log.Printf("!!! Error writing OpenAI embeddings response: %v !!!", err)
		return
	}
	log.Printf("<<< Sent OpenAI embeddings format response (%d bytes) <<<", bytesCopied)
}
