# Olares-Ollama Makefile

.PHONY: build run dev clean docker test

# Default target
all: build

# Build application
build:
	@echo "Building Olares-Ollama..."
	go build -o olares-ollama main.go

# Run application
run: build
	@echo "Starting Olares-Ollama..."
	./olares-ollama

# Run in development mode
dev:
	@echo "Starting in development mode..."
	go run main.go

# Clean build files
clean:
	@echo "Cleaning build files..."
	rm -f olares-ollama

# Run tests
test:
	@echo "Running tests..."
	go test ./...

# Format code
fmt:
	@echo "Formatting code..."
	go fmt ./...

# Lint code
lint:
	@echo "Linting code..."
	go vet ./...

# Build Docker image
docker:
	@echo "Building Docker image..."
	docker build -t olares-ollama .

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	docker run -p 8080:8080 -e OLLAMA_URL=http://host.docker.internal:11434 olares-ollama

# Docker Compose
docker-compose:
	@echo "Starting with Docker Compose..."
	docker-compose up --build

# Help information
help:
	@echo "Available commands:"
	@echo "  build        - Build application"
	@echo "  run          - Build and run application"
	@echo "  dev          - Run in development mode"
	@echo "  clean        - Clean build files"
	@echo "  test         - Run tests"
	@echo "  fmt          - Format code"
	@echo "  lint         - Lint code"
	@echo "  docker       - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docker-compose - Start with Docker Compose"
	@echo "  help         - Show help information"
