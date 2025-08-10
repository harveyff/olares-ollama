# Olares-Ollama API Documentation

## Overview

Olares-Ollama is a proxy server that runs in front of Ollama, providing unified model management and API forwarding capabilities.

### Basic Information

- **Base URL**: `http://localhost:8080` (default)
- **Content-Type**: `application/json`
- **CORS Support**: Yes

## API Endpoints

### 1. Health Check

Check server status and configuration information.

**Request**
```
GET /health
```

**Response**
```json
{
  "status": "ok",
  "model": "llama2"
}
```

### 2. Progress Query

Get current model download progress.

**Request**
```
GET /api/progress
```

**Response**
```json
{
  "status": "downloading",
  "progress": 65.5,
  "total": 3825205248,
  "completed": 2506967552,
  "model_name": "llama2",
  "timestamp": 1640995200
}
```

**Status Descriptions**
- `downloading`: Currently downloading
- `complete`: Download completed
- `success`: Success
- `error`: Error

### 3. Model List

Get available model list (only returns the currently configured model).

**Request**
```
GET /api/tags
```

**Response**
```json
{
  "models": [
    {
      "name": "llama2",
      "modified_at": "2024-01-01T00:00:00Z",
      "size": 0
    }
  ]
}
```

### 4. Text Generation

Generate text content, model parameter will be automatically replaced.

**Request**
```
POST /api/generate
```

**Request Body**
```json
{
  "model": "any-model-name",
  "prompt": "Tell me a joke",
  "stream": false
}
```

**Response**
```json
{
  "model": "llama2",
  "created_at": "2024-01-01T12:00:00Z",
  "response": "Why don't scientists trust atoms? Because they make up everything!",
  "done": true
}
```

### 5. Chat Conversation

Conduct chat conversation, model parameter will be automatically replaced.

**Request**
```
POST /api/chat
```

**Request Body**
```json
{
  "model": "any-model-name",
  "messages": [
    {
      "role": "user",
      "content": "Hello! How are you?"
    }
  ]
}
```

**Response**
```json
{
  "model": "llama2",
  "created_at": "2024-01-01T12:00:00Z",
  "message": {
    "role": "assistant",
    "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
  },
  "done": true
}
```

### 6. Text Embeddings

Generate text embedding vectors, model parameter will be automatically replaced.

**Request**
```
POST /api/embeddings
```

**Request Body**
```json
{
  "model": "any-model-name",
  "prompt": "The sky is blue"
}
```

**Response**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

### 7. System Management Interfaces

The following interfaces are directly proxied to the Ollama server without any modifications.

#### Version Information
```
GET /api/version
```

#### Running Processes
```
GET /api/ps
```

#### Stop Model
```
POST /api/stop
Content-Type: application/json

{
  "name": "model-name"
}
```

## Error Handling

### Error Response Format
```json
{
  "error": "Error message description"
}
```

### Common Error Codes

- `400 Bad Request`: Request format error
- `404 Not Found`: Path does not exist
- `405 Method Not Allowed`: HTTP method not allowed
- `500 Internal Server Error`: Internal server error

## Usage Examples

### JavaScript/Fetch
```javascript
// Chat request
const response = await fetch('http://localhost:8080/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'any-model',
    messages: [
      { role: 'user', content: 'Hello!' }
    ]
  })
});

const data = await response.json();
console.log(data.message.content);
```

### cURL
```bash
# Text generation
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any-model",
    "prompt": "What is AI?",
    "stream": false
  }'

# Chat conversation
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any-model",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Python
```python
import requests

# Chat request
response = requests.post('http://localhost:8080/api/chat', 
  json={
    'model': 'any-model',
    'messages': [
      {'role': 'user', 'content': 'Hello!'}
    ]
  }
)

data = response.json()
print(data['message']['content'])
```

## Important Notes

1. **Model Parameter Replacement**: All `model` parameters in inference interfaces (generate, chat, embeddings) will be automatically replaced with the configured model, so you don't need to worry about the specific model name.

2. **Model Management Restrictions**: Except for `/api/tags`, other model management interfaces (such as pull, push, delete) are blocked to ensure system stability.

3. **Streaming Response**: Supports streaming response, set `"stream": true` to get real-time generated content.

4. **Progress Monitoring**: Use the `/api/progress` interface to monitor model download progress in real-time.

5. **CORS Support**: Supports cross-origin requests, can be called directly from browsers.