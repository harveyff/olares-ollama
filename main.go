package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"olares-ollama/internal/config"
	"olares-ollama/internal/download"
	"olares-ollama/internal/ollama"
	"olares-ollama/internal/server"
)

func main() {
	// Load configuration
	cfg := config.Load()

	log.Printf("Starting Olares-Ollama proxy server...")
	log.Printf("Target model: %s", cfg.Model)
	log.Printf("Ollama server: %s", cfg.OllamaURL)
	log.Printf("Download timeout: %d minutes", cfg.DownloadTimeout)

	// Create Ollama client
	ollamaClient := ollama.NewClientWithTimeout(cfg.OllamaURL, cfg.DownloadTimeout)

	// Create and start server
	srv := server.New(cfg, ollamaClient)

	// Start HTTP server
	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.Port),
		Handler: srv.Handler(),
	}

	// Start HTTP server immediately (in background)
	go func() {
		log.Printf("Server starting on port %d", cfg.Port)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	log.Printf("Server started on port %d", cfg.Port)
	log.Printf("You can now view download progress at: http://localhost:%d", cfg.Port)

	// Check and download model in background
	go func() {
		if err := ensureModel(ollamaClient, cfg.Model, srv.GetProgressManager()); err != nil {
			log.Printf("Failed to ensure model: %v", err)
			srv.GetProgressManager().UpdateProgress("error", 0, 0, cfg.Model)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := httpServer.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	log.Println("Server exited")
}

func ensureModel(client *ollama.Client, modelName string, progressManager *download.ProgressManager) error {
	// Wait for Ollama to be reachable (e.g. when proxy and Ollama run in separate pods)
	ctx := context.Background()
	const ollamaWaitTimeout = 10 * time.Minute
	const ollamaRetryInterval = 5 * time.Second
	log.Printf("Waiting for Ollama server (up to %v)...", ollamaWaitTimeout)
	if err := client.WaitForOllama(ctx, ollamaWaitTimeout, ollamaRetryInterval); err != nil {
		return fmt.Errorf("Ollama not ready: %w", err)
	}

	log.Printf("Checking if model %s is available...", modelName)

	// 检查模型是否已存在
	exists, err := client.ModelExists(modelName)
	if err != nil {
		return fmt.Errorf("failed to check model existence: %w", err)
	}

	if exists {
		log.Printf("Model %s is already available", modelName)
		progressManager.UpdateProgress("completed", 0, 0, modelName)
		return nil
	}

	log.Printf("Model %s not found, starting download...", modelName)
	progressManager.UpdateProgress("downloading", 0, 0, modelName)

	// 断点续传：重试时是否从上次进度继续由 Ollama 服务端决定。
	// 在 Ollama 上设置 OLLAMA_NOPRUNE=1 可保留部分下载，提高重试续传概率。
	log.Printf("Tip: Set OLLAMA_NOPRUNE=1 on the Ollama server to improve resume on retry")

	maxRetries := 3
	attempt := 1
	const maxTransientRetries = 10 // 连续瞬时错误上限，超过则消耗一次 attempt
	transientCount := 0
	for attempt <= maxRetries {
		log.Printf("Download attempt %d/%d for model %s", attempt, maxRetries, modelName)

		err := client.PullModelWithProgress(modelName, progressManager)
		if err == nil {
			break
		}

		log.Printf("Download attempt %d failed: %v", attempt, err)
		errStr := err.Error()

		log.Printf("Checking if model %s exists before retry...", modelName)
		exists, checkErr := client.ModelExists(modelName)
		if checkErr == nil && exists {
			log.Printf("Model %s found after download attempt %d, marking as completed", modelName, attempt)
			progressManager.UpdateProgress("completed", 0, 0, modelName)
			return nil
		}

		// 瞬时错误（连接被拒/重置/EOF）不消耗重试次数，延长等待后再试
		isTransient := strings.Contains(errStr, "connection refused") ||
			strings.Contains(errStr, "connection reset") ||
			strings.Contains(errStr, "unexpected EOF") ||
			strings.Contains(errStr, "EOF")
		if isTransient && attempt < maxRetries {
			transientCount++
			if transientCount > maxTransientRetries {
				log.Printf("Too many transient errors (%d), consuming one attempt", transientCount)
				transientCount = 0
				attempt++
				time.Sleep(10 * time.Second)
				continue
			}
			wait := 15 * time.Second
			if strings.Contains(errStr, "connection refused") {
				wait = 30 * time.Second
			}
			log.Printf("Transient error (%d/%d), retrying without consuming attempt (wait %v)...", transientCount, maxTransientRetries, wait)
			time.Sleep(wait)
			continue
		}

		transientCount = 0
		if attempt == maxRetries {
			progressManager.UpdateProgress("error", 0, 0, modelName)
			return fmt.Errorf("failed to pull model after %d attempts: %w", maxRetries, err)
		}

		log.Printf("Waiting 10 seconds before retry...")
		time.Sleep(10 * time.Second)
		attempt++
	}

	// PullModelWithProgress 已经验证了模型可用性
	// 再次确认模型存在（双重验证）
	log.Printf("Double-checking model %s availability...", modelName)
	exists, err = client.ModelExists(modelName)
	if err != nil {
		log.Printf("Warning: Failed to verify model after download: %v", err)
	} else if !exists {
		progressManager.UpdateProgress("error", 0, 0, modelName)
		return fmt.Errorf("model %s download completed but model is not available", modelName)
	}

	log.Printf("Model %s downloaded and verified successfully", modelName)
	progressManager.UpdateProgress("completed", 0, 0, modelName)
	return nil
}
