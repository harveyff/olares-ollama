# Olares-Ollama Proxy Server

A Go-based proxy server that runs in front of Ollama, providing unified model management and API forwarding capabilities.

## Features

- 🚀 **Automatic Model Download**: Automatically checks and downloads specified models on startup
- 📊 **Real-time Progress Monitoring**: Provides a web interface to view model download progress in real-time
- 🔄 **API Proxy Forwarding**: Intelligently forwards API requests to Ollama server
- 🎯 **Model Parameter Replacement**: Automatically replaces model parameters in requests with the configured model
- 🛡️ **Interface Filtering**: Only exposes necessary API interfaces for improved security

## Quick Start

### Requirements

- Go 1.21+
- Ollama server running locally or remotely

### Installation and Usage

1. Clone the repository and enter the directory:
```bash
git clone <repository-url>
cd olares-ollama
```

2. Install dependencies:
```bash
go mod tidy
```

3. Set environment variables:
```bash
export OLLAMA_MODEL=llama2              # Model name to use
export OLLAMA_URL=http://localhost:11434  # Ollama server address
export PORT=8080                        # Proxy server port
export DOWNLOAD_TIMEOUT=60              # Download timeout in minutes
```

4. Run the server:
```bash
go run main.go
```

5. Access the web interface to view download progress:
```
http://localhost:8080
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama2` | Target model name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `PORT` | `8080` | Proxy server port |
| `DOWNLOAD_TIMEOUT` | `60` | Model download timeout in minutes |

## API Interfaces

### Supported Interfaces

#### Model Management
- `GET /api/tags` - Get available model list (only returns the configured model)

#### Inference Interfaces (automatic model parameter replacement)
- `POST /api/generate` - Text generation
- `POST /api/chat` - Chat conversation
- `POST /api/embeddings` - Text embeddings

#### System Management (direct proxy)
- `GET /api/version` - Get version information
- `GET /api/ps` - Get running processes
- `POST /api/stop` - Stop model

#### Other
- `GET /health` - Health check
- `GET /api/progress` - Progress monitoring

### Usage Examples

```bash
# Chat request (model parameter will be automatically replaced)
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any-model-name",
    "messages": [
      {
        "role": "user", 
        "content": "Hello!"
      }
    ]
  }'

# Text generation
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any-model-name",
    "prompt": "Tell me a story",
    "stream": false
  }'
```

## Project Structure

```
olares-ollama/
├── main.go                    # Program entry point
├── go.mod                     # Go module file
├── .gitignore                 # Git ignore file
├── data/                      # Model data directory (local storage)
│   └── .gitkeep              # Keep directory structure
├── internal/
│   ├── config/
│   │   └── config.go          # Configuration management
│   ├── ollama/
│   │   └── client.go          # Ollama client
│   ├── download/
│   │   └── progress.go        # Download progress management
│   └── server/
│       ├── server.go          # HTTP server
│       └── handlers.go        # Request handlers
├── web/
│   └── static/
│       └── index.html         # Frontend interface
├── docs/
│   └── API.md                 # API documentation
├── scripts/
│   └── start.sh              # Startup script
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
└── Makefile                  # Build script
```

## Development

### Build

```bash
go build -o olares-ollama main.go
```

### Run Tests

```bash
go test ./...
```

## Data Storage

### Local Mode
When running `docker-compose up`, model files will be stored in the project root's `data/` folder:
- `data/` - Ollama model data directory
- This directory is added to `.gitignore` and won't be committed to version control
- Deleting the `data/` directory will clean up all downloaded models

### Data Persistence
- When using Docker Compose, model data persists in the local `data/` directory
- Restarting containers won't lose downloaded models
- You can backup models by backing up the `data/` directory

## Notes

1. The first startup will automatically download the specified model, which may take a long time
2. Ensure the Ollama server is running and accessible
3. Model download progress can be viewed in real-time through the web interface
4. All inference request model parameters will be replaced with the configured model
5. Model files are stored in the `data/` directory and can be cleaned or backed up as needed

## Troubleshooting

### Model Download Timeout
If you encounter "context deadline exceeded" errors:

1. **Increase download timeout**:
   ```bash
   export DOWNLOAD_TIMEOUT=180  # Set to 3 hours
   ```

2. **Choose a smaller model**:
   ```bash
   export OLLAMA_MODEL=qwen3:0.6b  # Smaller model, faster download
   ```

3. **Check network connection**:
   - Ensure stable network connection
   - Check for firewall or proxy restrictions

4. **Manually download model**:
   ```bash
   # Manually download on Ollama server
   ollama pull qwen3:0.6b
   ```

### Common Issues

**Q: Download is slow?**
A: Choose smaller models like `qwen3:0.6b` or `phi3:mini`

**Q: How to clean downloaded models?**
A: Delete the `data/` directory: `rm -rf data/`

**Q: How to view download progress?**
A: Visit `http://localhost:8080` to see the web interface

**Q: Server startup failed?**
A: Check if Ollama service is running: `curl http://localhost:11434/api/version`

## License

MIT License