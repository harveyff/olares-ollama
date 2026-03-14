package huggingface

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ProgressUpdater reports download progress.
type ProgressUpdater interface {
	UpdateProgress(status string, completed, total int64, modelName string)
}

// Downloader downloads GGUF files from Hugging Face with resume support.
type Downloader struct {
	Endpoint  string // e.g. "https://huggingface.co"
	Repo      string // e.g. "unsloth/Qwen3.5-35B-A3B-GGUF"
	File      string // e.g. "Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
	Token     string
	OutputDir string // directory to save the file

	client *http.Client
}

// New creates a Downloader.
func New(endpoint, repo, file, token, outputDir string) *Downloader {
	endpoint = strings.TrimSuffix(endpoint, "/")
	return &Downloader{
		Endpoint:  endpoint,
		Repo:      repo,
		File:      file,
		Token:     token,
		OutputDir: outputDir,
		client: &http.Client{
			Timeout: 0, // no global timeout; large files may take hours
			Transport: &http.Transport{
				IdleConnTimeout:       5 * time.Minute,
				ResponseHeaderTimeout: 60 * time.Second,
				TLSHandshakeTimeout:   30 * time.Second,
			},
		},
	}
}

// DestPath returns the final path of the downloaded GGUF file.
func (d *Downloader) DestPath() string {
	return filepath.Join(d.OutputDir, d.File)
}

func (d *Downloader) doneMarker() string {
	return d.DestPath() + ".done"
}

// AlreadyDone returns true if a previous download completed successfully.
func (d *Downloader) AlreadyDone() bool {
	_, err := os.Stat(d.doneMarker())
	return err == nil
}

// Download fetches the GGUF file with HTTP Range resume.
// It writes to <file>.part, renames on completion, then creates a .done marker.
func (d *Downloader) Download(ctx context.Context, modelName string, progress ProgressUpdater) error {
	if err := os.MkdirAll(d.OutputDir, 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	dest := d.DestPath()
	partFile := dest + ".part"

	// If .done exists and full file exists, skip
	if d.AlreadyDone() {
		if _, err := os.Stat(dest); err == nil {
			log.Printf("GGUF file already downloaded: %s", dest)
			progress.UpdateProgress("completed", 0, 0, modelName)
			return nil
		}
		os.Remove(d.doneMarker())
	}

	// Determine existing partial size for resume
	var existingSize int64
	if info, err := os.Stat(partFile); err == nil {
		existingSize = info.Size()
		log.Printf("Found partial download: %d bytes", existingSize)
	}

	url := fmt.Sprintf("%s/%s/resolve/main/%s", d.Endpoint, d.Repo, d.File)
	log.Printf("Downloading GGUF from %s", url)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if d.Token != "" {
		req.Header.Set("Authorization", "Bearer "+d.Token)
	}
	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
		log.Printf("Resuming from byte %d", existingSize)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		existingSize = 0 // server ignored Range; start over
	case http.StatusPartialContent:
		// resume OK
	case http.StatusRequestedRangeNotSatisfiable:
		log.Printf("Range not satisfiable; file may already be complete, re-downloading")
		existingSize = 0
		req.Header.Del("Range")
		resp.Body.Close()
		resp, err = d.client.Do(req)
		if err != nil {
			return fmt.Errorf("retry http request: %w", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("unexpected status on retry: %s", resp.Status)
		}
	default:
		return fmt.Errorf("unexpected HTTP status: %s", resp.Status)
	}

	totalSize := resp.ContentLength + existingSize
	if resp.StatusCode == http.StatusOK {
		totalSize = resp.ContentLength
	}

	log.Printf("Total file size: %d bytes (%.2f GiB)", totalSize, float64(totalSize)/(1024*1024*1024))

	// Open part file for writing (append or truncate)
	var flags int
	if existingSize > 0 && resp.StatusCode == http.StatusPartialContent {
		flags = os.O_WRONLY | os.O_APPEND
	} else {
		flags = os.O_WRONLY | os.O_CREATE | os.O_TRUNC
		existingSize = 0
	}
	f, err := os.OpenFile(partFile, flags|os.O_CREATE, 0644)
	if err != nil {
		return fmt.Errorf("open part file: %w", err)
	}

	progress.UpdateProgress("downloading", existingSize, totalSize, modelName)

	const reportInterval = 2 * time.Second
	lastReport := time.Now()
	written := existingSize
	buf := make([]byte, 256*1024)

	for {
		select {
		case <-ctx.Done():
			f.Close()
			return ctx.Err()
		default:
		}

		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, wErr := f.Write(buf[:n]); wErr != nil {
				f.Close()
				return fmt.Errorf("write: %w", wErr)
			}
			written += int64(n)

			if time.Since(lastReport) >= reportInterval {
				progress.UpdateProgress("downloading", written, totalSize, modelName)
				pct := float64(0)
				if totalSize > 0 {
					pct = float64(written) / float64(totalSize) * 100
				}
				log.Printf("Download progress: %.1f%% (%d / %d bytes)", pct, written, totalSize)
				lastReport = time.Now()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			f.Close()
			return fmt.Errorf("read: %w (downloaded %d bytes so far)", readErr, written)
		}
	}

	if err := f.Close(); err != nil {
		return fmt.Errorf("close part file: %w", err)
	}

	// Rename .part -> final
	if err := os.Rename(partFile, dest); err != nil {
		return fmt.Errorf("rename part file: %w", err)
	}

	// Write .done marker
	if err := os.WriteFile(d.doneMarker(), []byte(time.Now().UTC().Format(time.RFC3339)), 0644); err != nil {
		log.Printf("Warning: failed to write .done marker: %v", err)
	}

	log.Printf("GGUF download complete: %s (%d bytes)", dest, written)
	progress.UpdateProgress("downloaded", written, totalSize, modelName)
	return nil
}

// ComputeSHA256 returns "sha256:<hex>" for the given file.
// The result is cached in a sibling .sha256 file so that repeated calls on
// the same (unchanged) GGUF skip the expensive re-hash.
func ComputeSHA256(filePath string) (string, error) {
	cacheFile := filePath + ".sha256"

	// If cache exists and the GGUF hasn't been modified after it, reuse.
	if cInfo, cErr := os.Stat(cacheFile); cErr == nil {
		if fInfo, fErr := os.Stat(filePath); fErr == nil && !fInfo.ModTime().After(cInfo.ModTime()) {
			if data, err := os.ReadFile(cacheFile); err == nil {
				digest := strings.TrimSpace(string(data))
				if strings.HasPrefix(digest, "sha256:") && len(digest) == 71 {
					log.Printf("Using cached SHA256 for %s: %s", filepath.Base(filePath), digest)
					return digest, nil
				}
			}
		}
	}

	log.Printf("Computing SHA256 for %s (this may take a while for large files)...", filepath.Base(filePath))
	start := time.Now()

	f, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("open file for hashing: %w", err)
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("hash file: %w", err)
	}

	digest := "sha256:" + hex.EncodeToString(h.Sum(nil))
	log.Printf("SHA256 computed in %v: %s", time.Since(start).Round(time.Second), digest)

	// Cache the result
	if err := os.WriteFile(cacheFile, []byte(digest+"\n"), 0644); err != nil {
		log.Printf("Warning: failed to cache SHA256: %v", err)
	}

	return digest, nil
}
