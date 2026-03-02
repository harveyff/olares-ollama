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
export APP_URL=https://your-api-url.com/  # API access URL (optional, shown after download completes)
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
| `OLLAMA_PULL_DELAY_SECONDS` | `30` | Seconds to wait after Ollama is ready before first pull; gives Ollama time to load blob index so API pull can resume from disk after restart (set `0` to disable) |
| `APP_URL` | (empty) | API access URL displayed after download completes (optional) |

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

### 断点续传 / Resumable Download

重试时进度从 0% 开始是 **Ollama 服务端** 的行为：Ollama 的 `/api/pull` 在连接断开后再次调用会重新发起下载，是否从已下载部分续传由 Ollama 决定。

**建议**：在 **Ollama 所在环境** 设置 `OLLAMA_NOPRUNE=1`，让 Ollama 保留部分下载文件，重试时更可能从断点续传。

- **Docker Compose**：已在 `docker-compose.yml` 的 `ollama` 服务中默认加入 `OLLAMA_NOPRUNE=1`。
- **Kubernetes**：在 Ollama 的 Deployment/Pod 的 `env` 中加入 `OLLAMA_NOPRUNE: "1"`。
- **其他小模型内存限制**：若希望仅保留当前使用的模型在内存、不常驻其他小模型，可在 Ollama 侧设置 `OLLAMA_MAX_LOADED_MODELS=1`（Docker Compose 已默认添加）。
- **本机**：`export OLLAMA_NOPRUNE=1` 后再启动 `ollama serve`。

本代理在遇到「连接被拒」「connection reset」「unexpected EOF」等瞬时错误时会自动重试且**不消耗** 3 次下载机会，只有非瞬时错误或连续瞬时错误过多才会计为一次失败。

**若重试后仍从 0% 开始**：这是 Ollama 的 API 行为。每次重试会发送新的 `POST /api/pull`，Ollama 可能不会续传而重新拉取该层。若 Ollama 部署了**多副本**，重试可能打到另一台没有部分数据的实例，导致必然从 0% 开始。建议：拉取阶段使用**单副本**，或为 Ollama 配置**会话亲和**（同一客户端 IP 固定到同一 Pod），并确保 `OLLAMA_NOPRUNE=1` 设在该实例上。

**单副本仍从 0% 时**：先确认环境变量在 Ollama 容器内生效（例如 `kubectl exec <ollama-pod> -- env | grep OLLAMA_NOPRUNE`）。若中间有 Ingress/反向代理，检查其**流式/长连接超时**（如 Nginx 的 `proxy_read_timeout`），过短会主动断开连接导致 EOF，需调大（例如 7200 秒）。

**重启 Pod 后仍从 0%**：若存储已持久化且已设 `OLLAMA_NOPRUNE=1`，磁盘上可能有部分数据。本服务默认在 Ollama 就绪后再等待 30 秒（`OLLAMA_PULL_DELAY_SECONDS=30`）再发起首次 pull，以便 Ollama 完成 blobs 扫描。若 **Ollama 容器重启后，即使用 CLI 拉取也从 0% 开始**，说明根因在 Ollama 进程：新进程启动时可能仍会清理部分下载（或未在启动时加载已有 partial blobs），与 OLLAMA_NOPRUNE 生效时机或上游逻辑有关，需在 [Ollama 上游](https://github.com/ollama/ollama/issues) 反馈或等待修复。临时可进 Ollama 容器用 CLI 续传（若在你环境中 CLI 能续传）：
```bash
kubectl exec -it <ollama-pod> -- ollama pull <模型名>
```

**有无办法修改（从本服务或 Ollama 侧）**：

1. **先确认 NOPRUNE 和持久化**  
   - 在**运行 Ollama 的容器**里执行：`env | grep OLLAMA_NOPRUNE`，应为 `OLLAMA_NOPRUNE=1`（或 `true`）。  
   - 确认模型目录挂的是**同一块持久卷**（重启前后 `du` 该目录可见已有数据，如 4.4GB）。

2. **若 NOPRUNE 已设、磁盘数据仍在，但 pull 仍从 0%**  
   说明问题不是“启动时删了 partial”，而是 **Ollama 新进程在 pull 时没有把磁盘上已有 partial 当成可续传**。这需要改 Ollama 源码（pull 前检查已有 blob/partial 并续传），只能通过**上游修复**或**自建镜像**解决：  
   - 到 [ollama/ollama](https://github.com/ollama/ollama) 提 Issue，说明：已设 `OLLAMA_NOPRUNE=1`、持久化存储、容器重启后 CLI/API 都从 0% 开始，请求支持“进程重启后从已有 partial 续传”。  
   - 或 fork 仓库，在 pull 流程里加入“若存在同 digest 的 partial 则续传”的逻辑，再构建自己的 Ollama 镜像使用。

3. **运维侧可做的**  
   - 下载大模型期间尽量避免重启 Ollama 容器；若必须重启，可接受“重新从 0% 拉取”或先用 CLI 在容器内拉完再交给本服务。

```bash
# 临时续传：进 Ollama 容器用 CLI 拉取，完成后本服务会识别到模型
kubectl exec -it <ollama-pod> -- ollama pull <模型名>
```

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