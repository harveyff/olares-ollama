#!/bin/bash

# Olares-Ollama startup script

echo "Starting Olares-Ollama proxy server..."

# Set default environment variables
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:0.6b}"
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
export PORT="${PORT:-8080}"
export DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-120}"

echo "Configuration:"
echo "  Model: $OLLAMA_MODEL"
echo "  Ollama URL: $OLLAMA_URL"
echo "  Port: $PORT"
echo "  Download timeout: $DOWNLOAD_TIMEOUT minutes"

# Check if Ollama service is available
echo "Checking Ollama service..."
if ! curl -s "$OLLAMA_URL/api/version" > /dev/null; then
    echo "Warning: Cannot connect to Ollama server ($OLLAMA_URL)"
    echo "Please ensure Ollama service is running"
fi

# Build and run
echo "Building project..."
go build -o olares-ollama main.go

if [ $? -eq 0 ]; then
    echo "Build successful, starting server..."
    echo "Access URL: http://localhost:$PORT"
    ./olares-ollama
else
    echo "Build failed"
    exit 1
fi
