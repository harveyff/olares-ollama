package main

import (
	"context"
	"encoding/json"
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
	"olares-ollama/internal/huggingface"
	"olares-ollama/internal/ollama"
	"olares-ollama/internal/server"
)

func main() {
	// Load configuration
	cfg := config.Load()

	log.Printf("Starting Olares-Ollama proxy server...")
	if cfg.GGUFMode {
		log.Printf("Running in GGUF mode: repo=%s file=%s model=%s", cfg.HFRepo, cfg.HFFile, cfg.Model)
	} else if cfg.BaseMode {
		log.Printf("Running in BASE mode (no model configured)")
	} else {
		log.Printf("Target model: %s", cfg.Model)
	}
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

	if !cfg.BaseMode {
		log.Printf("You can now view download progress at: http://localhost:%d", cfg.Port)

		// Channel for manual retry triggers (from /api/retry endpoint)
		retryCh := make(chan struct{}, 1)

		// Register /api/retry endpoint
		srv.RegisterRetryHandler(retryCh)

		// Check and download model in background with infinite retry
		go ensureModelLoop(ollamaClient, cfg, srv.GetProgressManager(), retryCh)
	} else {
		log.Printf("Base mode UI at: http://localhost:%d", cfg.Port)
	}

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

// ensureModelLoop wraps ensureModel with infinite retry: on failure it waits
// with exponential backoff (up to 5 min) and retries. A signal on retryCh
// (from /api/retry) wakes it up immediately.
// After success, it monitors Ollama health; if Ollama goes down, it re-enters
// the retry loop so the frontend always reflects the real state.
func ensureModelLoop(client *ollama.Client, cfg *config.Config, progressManager *download.ProgressManager, retryCh <-chan struct{}) {
	modelName := cfg.Model
	backoff := 30 * time.Second
	const maxBackoff = 5 * time.Minute

	for {
		var err error
		if cfg.GGUFMode {
			err = ensureModelGGUF(client, cfg, progressManager)
		} else {
			err = ensureModel(client, modelName, cfg.OllamaPullDelaySec, progressManager)
		}
		if err == nil {
			monitorOllamaHealth(client, modelName, progressManager, retryCh)
			// Ollama went down — reset backoff and retry from the beginning
			log.Printf("Ollama became unreachable, re-entering ensure model loop...")
			backoff = 30 * time.Second
			continue
		}

		log.Printf("Failed to ensure model: %v", err)
		progressManager.UpdateProgress("error", 0, 0, modelName)

		log.Printf("Will retry in %v (or immediately on /api/retry)...", backoff)
		select {
		case <-time.After(backoff):
		case <-retryCh:
			log.Printf("Manual retry triggered via /api/retry")
		}

		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

// monitorOllamaHealth periodically checks if Ollama is still reachable and the
// model is still available. Returns when Ollama becomes unreachable so the
// caller can re-enter the ensure loop.
func monitorOllamaHealth(client *ollama.Client, modelName string, progressManager *download.ProgressManager, retryCh <-chan struct{}) {
	const checkInterval = 15 * time.Second
	const maxConsecutiveFailures = 3
	failures := 0

	log.Printf("Starting Ollama health monitor (every %v)", checkInterval)

	for {
		select {
		case <-time.After(checkInterval):
		case <-retryCh:
			log.Printf("Manual retry triggered during health monitoring")
		}

		exists, err := client.ModelExists(modelName)
		if err != nil {
			failures++
			log.Printf("Ollama health check failed (%d/%d): %v", failures, maxConsecutiveFailures, err)
			if failures >= maxConsecutiveFailures {
				log.Printf("Ollama appears to be down (failed %d consecutive checks)", failures)
				progressManager.UpdateProgress("unavailable", 0, 0, modelName)
				return
			}
			continue
		}

		if !exists {
			log.Printf("Model %s no longer found in Ollama", modelName)
			progressManager.UpdateProgress("unavailable", 0, 0, modelName)
			return
		}

		// Healthy — reset failure counter
		if failures > 0 {
			log.Printf("Ollama health check recovered after %d failures", failures)
			failures = 0
			progressManager.UpdateProgress("completed", 0, 0, modelName)
		}
	}
}

// ensureModelGGUF downloads a GGUF from Hugging Face, pushes it as an Ollama
// blob, and registers the model via POST /api/create with the files field.
func ensureModelGGUF(client *ollama.Client, cfg *config.Config, progressManager *download.ProgressManager) error {
	modelName := cfg.Model
	if modelName == "" {
		modelName = strings.TrimSuffix(cfg.HFFile, ".gguf")
	}

	ctx := context.Background()

	// Wait for Ollama
	log.Printf("GGUF mode: waiting for Ollama server...")
	progressManager.UpdateProgress("waiting", 0, 0, modelName)
	if err := client.WaitForOllama(ctx, 30*time.Minute, 5*time.Second); err != nil {
		return fmt.Errorf("Ollama not ready: %w", err)
	}

	// Check current state (informational only; we always (re-)create to
	// ensure template/params updates take effect).
	exists, _ := client.ModelExists(modelName)
	if exists {
		log.Printf("GGUF model %s already registered, will re-create to apply latest config", modelName)
	}

	// Download GGUF
	dl := huggingface.New(cfg.HFEndpoint, cfg.HFRepo, cfg.HFFile, cfg.HFToken, cfg.GGUFDir)
	if !dl.AlreadyDone() {
		log.Printf("Downloading GGUF: %s/%s -> %s", cfg.HFRepo, cfg.HFFile, dl.DestPath())
		if err := dl.Download(ctx, modelName, progressManager); err != nil {
			return fmt.Errorf("GGUF download failed: %w", err)
		}
	} else {
		log.Printf("GGUF file already downloaded: %s", dl.DestPath())
	}

	// Compute SHA256 of the GGUF file
	progressManager.UpdateProgress("hashing", 0, 0, modelName)
	digest, err := huggingface.ComputeSHA256(dl.DestPath())
	if err != nil {
		return fmt.Errorf("compute SHA256: %w", err)
	}
	log.Printf("GGUF digest: %s", digest)

	// Push blob if not already present
	blobExists, err := client.BlobExists(digest)
	if err != nil {
		log.Printf("Warning: blob existence check failed: %v, will try pushing anyway", err)
		blobExists = false
	}
	if blobExists {
		log.Printf("Blob %s already exists on Ollama server, skipping push", digest)
	} else {
		if err := client.PushBlob(digest, dl.DestPath(), progressManager, modelName); err != nil {
			return fmt.Errorf("push blob: %w", err)
		}
	}

	// Create model via /api/create with files map
	files := map[string]string{
		cfg.HFFile: digest,
	}

	// Download mmproj (vision projector) if configured
	if cfg.HFMMProjFile != "" {
		mmDl := huggingface.New(cfg.HFEndpoint, cfg.HFRepo, cfg.HFMMProjFile, cfg.HFToken, cfg.GGUFDir)
		if !mmDl.AlreadyDone() {
			log.Printf("Downloading mmproj: %s/%s -> %s", cfg.HFRepo, cfg.HFMMProjFile, mmDl.DestPath())
			if err := mmDl.Download(ctx, modelName, progressManager); err != nil {
				return fmt.Errorf("mmproj download failed: %w", err)
			}
		} else {
			log.Printf("mmproj file already downloaded: %s", mmDl.DestPath())
		}

		progressManager.UpdateProgress("hashing", 0, 0, modelName)
		mmDigest, err := huggingface.ComputeSHA256(mmDl.DestPath())
		if err != nil {
			return fmt.Errorf("compute mmproj SHA256: %w", err)
		}
		log.Printf("mmproj digest: %s", mmDigest)

		mmBlobExists, err := client.BlobExists(mmDigest)
		if err != nil {
			log.Printf("Warning: mmproj blob existence check failed: %v, will try pushing anyway", err)
			mmBlobExists = false
		}
		if mmBlobExists {
			log.Printf("mmproj blob %s already exists, skipping push", mmDigest)
		} else {
			if err := client.PushBlob(mmDigest, mmDl.DestPath(), progressManager, modelName); err != nil {
				return fmt.Errorf("push mmproj blob: %w", err)
			}
		}

		files[cfg.HFMMProjFile] = mmDigest
	}


	var params map[string]interface{}
	if cfg.GGUFParams != "" {
		if err := json.Unmarshal([]byte(cfg.GGUFParams), &params); err != nil {
			log.Printf("Warning: failed to parse GGUF_PARAMS JSON (%q): %v, ignoring", cfg.GGUFParams, err)
		}
	}

	// Apply OLLAMA_CONTEXT_LENGTH to num_ctx if not already set in GGUF_PARAMS.
	if cfg.ContextLength > 0 {
		if params == nil {
			params = map[string]interface{}{}
		}
		if _, hasNumCtx := params["num_ctx"]; !hasNumCtx {
			params["num_ctx"] = cfg.ContextLength
			log.Printf("Setting num_ctx=%d from OLLAMA_CONTEXT_LENGTH", cfg.ContextLength)
		}
	}

	tpl := cfg.ResolveTemplate()
	if tpl != "" {
		log.Printf("Using explicit template (name=%q, len=%d)", cfg.GGUFTemplateName, len(tpl))
	}

	if err := client.CreateModelFromGGUF(modelName, dl.DestPath(), files, params, tpl, cfg.GGUFSystem, progressManager); err != nil {
		return fmt.Errorf("ollama create failed: %w", err)
	}

	log.Printf("GGUF model %s ready", modelName)
	return nil
}

func ensureModel(client *ollama.Client, modelName string, ollamaPullDelaySec int, progressManager *download.ProgressManager) error {
	// Wait for Ollama to be reachable (e.g. when proxy and Ollama run in separate pods)
	ctx := context.Background()
	const ollamaWaitTimeout = 30 * time.Minute
	const ollamaRetryInterval = 5 * time.Second
	log.Printf("Waiting for Ollama server (up to %v)...", ollamaWaitTimeout)
	progressManager.UpdateProgress("waiting", 0, 0, modelName)
	if err := client.WaitForOllama(ctx, ollamaWaitTimeout, ollamaRetryInterval); err != nil {
		return fmt.Errorf("Ollama not ready: %w", err)
	}

	// 延迟再发起首次 pull，给 Ollama 时间扫描 blobs/manifests，便于重启后 API 能从磁盘续传（与 CLI 行为一致）
	if ollamaPullDelaySec > 0 {
		log.Printf("Waiting %d seconds for Ollama to load blob index (improves resume after restart)...", ollamaPullDelaySec)
		time.Sleep(time.Duration(ollamaPullDelaySec) * time.Second)
	}

	log.Printf("Checking if model %s is available...", modelName)

	// 检查模型是否已存在
	exists, err := client.ModelExists(modelName)
	if err != nil {
		return fmt.Errorf("failed to check model existence: %w", err)
	}

	if exists {
		log.Printf("Model %s is already available, checking for updates...", modelName)
		progressManager.UpdateProgress("checking", 0, 0, modelName)

		if err := client.PullModelWithProgress(modelName, progressManager); err != nil {
			log.Printf("Incremental update check failed (existing model still usable): %v", err)
		}
		progressManager.UpdateProgress("completed", 0, 0, modelName)
		return nil
	}

	log.Printf("Model %s not found, starting download...", modelName)
	progressManager.UpdateProgress("downloading", 0, 0, modelName)

	log.Printf("Tip: Set OLLAMA_NOPRUNE=1 on the Ollama server to improve resume on retry")

	maxRetries := 3
	attempt := 1
	const maxTransientRetries = 20 // 连续瞬时错误上限
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

			// 瞬时错误不消耗 attempt，指数退避后重试
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
				// 指数退避：15s -> 30s -> 60s -> 120s（上限 120s）
				wait := 15 * time.Second
				if strings.Contains(errStr, "connection refused") {
					wait = 30 * time.Second
				}
				if transientCount > 1 {
					backoff := 15 * time.Duration(1<<uint(transientCount-1)) * time.Second
					if backoff > 120*time.Second {
						backoff = 120 * time.Second
					}
					if strings.Contains(errStr, "connection refused") {
						backoff = 30 * time.Duration(1<<uint(transientCount-1)) * time.Second
						if backoff > 120*time.Second {
							backoff = 120 * time.Second
						}
					}
					wait = backoff
				}
				p := progressManager.GetProgress()
				if p.Total > 0 && p.Progress > 0 {
					log.Printf("Retrying... last progress was %.1f%% (transient error %d/%d, wait %v)", p.Progress, transientCount, maxTransientRetries, wait)
				} else {
					log.Printf("Transient error (%d/%d), retrying without consuming attempt (wait %v)...", transientCount, maxTransientRetries, wait)
				}
				log.Printf("Note: Retry sends a new /api/pull; Ollama may show progress from 0%% again")
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
