package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
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

	// 下载模型，带重试机制
	maxRetries := 3
	for attempt := 1; attempt <= maxRetries; attempt++ {
		log.Printf("Download attempt %d/%d for model %s", attempt, maxRetries, modelName)

		if err := client.PullModelWithProgress(modelName, progressManager); err != nil {
			log.Printf("Download attempt %d failed: %v", attempt, err)

			// 在重试前，先检查模型是否已经存在（可能之前的下载已完成，只是验证时还没注册）
			log.Printf("Checking if model %s exists before retry...", modelName)
			exists, checkErr := client.ModelExists(modelName)
			if checkErr == nil && exists {
				log.Printf("Model %s found after download attempt %d, marking as completed", modelName, attempt)
				progressManager.UpdateProgress("completed", 0, 0, modelName)
				return nil
			}

			if attempt == maxRetries {
				progressManager.UpdateProgress("error", 0, 0, modelName)
				return fmt.Errorf("failed to pull model after %d attempts: %w", maxRetries, err)
			}

			// 等待一段时间再重试
			log.Printf("Waiting 10 seconds before retry...")
			time.Sleep(10 * time.Second)
			continue
		}

		// PullModelWithProgress 已经验证了模型可用性
		// 再次确认模型存在（双重验证）
		log.Printf("Double-checking model %s availability...", modelName)
		exists, err := client.ModelExists(modelName)
		if err != nil {
			log.Printf("Warning: Failed to verify model after download: %v", err)
			// 不返回错误，因为 PullModelWithProgress 已经验证过了
		} else if !exists {
			log.Printf("Warning: Model %s not found after download, may need retry", modelName)
			if attempt < maxRetries {
				log.Printf("Retrying download...")
				time.Sleep(5 * time.Second)
				continue
			}
			progressManager.UpdateProgress("error", 0, 0, modelName)
			return fmt.Errorf("model %s download completed but model is not available", modelName)
		}

		// 下载成功并验证通过
		log.Printf("Model %s downloaded and verified successfully", modelName)
		progressManager.UpdateProgress("completed", 0, 0, modelName)
		return nil
	}

	return nil
}
