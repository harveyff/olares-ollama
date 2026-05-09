package huggingface

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
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

// maskedURL returns a string-safe URL for logging (strips userinfo/query secrets if any).
func maskedURL(rawURL string) string {
	u, err := url.Parse(rawURL)
	if err != nil {
		return rawURL
	}
	u.User = nil
	u.RawQuery = ""
	return u.String()
}

// formatBytes returns a human-readable size in B/KiB/MiB/GiB.
func formatBytes(n int64) string {
	const (
		kib = 1024
		mib = 1024 * kib
		gib = 1024 * mib
	)
	switch {
	case n >= gib:
		return fmt.Sprintf("%.2f GiB", float64(n)/float64(gib))
	case n >= mib:
		return fmt.Sprintf("%.2f MiB", float64(n)/float64(mib))
	case n >= kib:
		return fmt.Sprintf("%.2f KiB", float64(n)/float64(kib))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// formatDuration rounds to a sensible unit for log readability.
func formatDuration(d time.Duration) string {
	if d < time.Second {
		return d.Round(time.Millisecond).String()
	}
	if d < time.Minute {
		return d.Round(100 * time.Millisecond).String()
	}
	return d.Round(time.Second).String()
}

// isTransientErr classifies network/IO errors that should trigger an in-process retry
// instead of bubbling up to the outer ensure-model loop (which has a much longer backoff).
func isTransientErr(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) && (netErr.Timeout() || netErr.Temporary()) {
		return true
	}
	msg := strings.ToLower(err.Error())
	for _, kw := range []string{
		"connection reset",
		"connection refused",
		"broken pipe",
		"unexpected eof",
		"timeout",
		"i/o timeout",
		"tls handshake",
		"no such host",
		"server misbehaving",
		"temporary failure",
	} {
		if strings.Contains(msg, kw) {
			return true
		}
	}
	return false
}

// Download fetches the GGUF file with HTTP Range resume.
// It writes to <file>.part, renames on completion, then creates a .done marker.
//
// Retry strategy:
//   - On transient network errors during the streaming read it auto-retries up to
//     maxAttempts times in-process with backoff 5s -> 10s -> 20s -> 40s -> 60s,
//     resuming from the current .part offset via HTTP Range. This is fast-path
//     recovery that does NOT consume the outer ensure-model loop's backoff.
//   - On non-transient HTTP errors (4xx other than 416, etc.) it returns
//     immediately so the outer loop can decide what to do.
func (d *Downloader) Download(ctx context.Context, modelName string, progress ProgressUpdater) error {
	if err := os.MkdirAll(d.OutputDir, 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	dest := d.DestPath()
	partFile := dest + ".part"

	if d.AlreadyDone() {
		if _, err := os.Stat(dest); err == nil {
			log.Printf("[hf] GGUF already downloaded, skipping: %s", dest)
			progress.UpdateProgress("completed", 0, 0, modelName)
			return nil
		}
		log.Printf("[hf] .done marker present but file missing, removing marker and re-downloading")
		os.Remove(d.doneMarker())
	}

	rawURL := fmt.Sprintf("%s/%s/resolve/main/%s", d.Endpoint, d.Repo, d.File)
	logURL := maskedURL(rawURL)

	log.Printf("[hf] === GGUF download starting ===")
	log.Printf("[hf] endpoint  : %s", d.Endpoint)
	log.Printf("[hf] repo      : %s", d.Repo)
	log.Printf("[hf] file      : %s", d.File)
	log.Printf("[hf] url       : %s", logURL)
	log.Printf("[hf] dest      : %s", dest)
	log.Printf("[hf] part file : %s", partFile)
	log.Printf("[hf] auth      : token=%t", d.Token != "")

	const maxAttempts = 5
	overallStart := time.Now()
	var lastErr error

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		var existingSize int64
		if info, err := os.Stat(partFile); err == nil {
			existingSize = info.Size()
		}
		log.Printf("[hf] attempt %d/%d: existing partial = %s (%d bytes)",
			attempt, maxAttempts, formatBytes(existingSize), existingSize)

		err := d.downloadOnce(ctx, rawURL, partFile, existingSize, modelName, progress, attempt, maxAttempts)
		if err == nil {
			if err := os.Rename(partFile, dest); err != nil {
				return fmt.Errorf("rename part file: %w", err)
			}
			if err := os.WriteFile(d.doneMarker(), []byte(time.Now().UTC().Format(time.RFC3339)), 0644); err != nil {
				log.Printf("[hf] warning: failed to write .done marker: %v", err)
			}
			finalSize, _ := os.Stat(dest)
			var sz int64
			if finalSize != nil {
				sz = finalSize.Size()
			}
			elapsed := time.Since(overallStart)
			avgSpeed := float64(0)
			if elapsed.Seconds() > 0 {
				avgSpeed = float64(sz) / elapsed.Seconds() / (1024 * 1024)
			}
			log.Printf("[hf] === GGUF download complete ===")
			log.Printf("[hf] file=%s size=%s elapsed=%s avg=%.2f MiB/s",
				dest, formatBytes(sz), formatDuration(elapsed), avgSpeed)
			progress.UpdateProgress("downloaded", sz, sz, modelName)
			return nil
		}

		lastErr = err

		if ctx.Err() != nil {
			log.Printf("[hf] context cancelled, aborting retry: %v", ctx.Err())
			return ctx.Err()
		}

		if !isTransientErr(err) {
			log.Printf("[hf] non-transient error on attempt %d, not retrying in-process: %v", attempt, err)
			return err
		}

		if attempt == maxAttempts {
			log.Printf("[hf] transient error on final attempt %d/%d, giving up to outer loop: %v",
				attempt, maxAttempts, err)
			break
		}

		// 5s -> 10s -> 20s -> 40s -> 60s
		wait := time.Duration(5*(1<<uint(attempt-1))) * time.Second
		if wait > 60*time.Second {
			wait = 60 * time.Second
		}
		log.Printf("[hf] transient error on attempt %d/%d: %v", attempt, maxAttempts, err)
		log.Printf("[hf] retrying in %s (will resume from current .part offset)...", wait)
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return lastErr
}

// downloadOnce performs a single HTTP attempt: it issues the GET with the right
// Range header, opens the .part file in the right mode, and streams the body to disk.
// On transient mid-stream failures it returns the error so the caller (Download)
// can retry while preserving the .part file for resume.
func (d *Downloader) downloadOnce(
	ctx context.Context,
	rawURL string,
	partFile string,
	existingSize int64,
	modelName string,
	progress ProgressUpdater,
	attempt, maxAttempts int,
) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "olares-ollama/hf-downloader")
	if d.Token != "" {
		req.Header.Set("Authorization", "Bearer "+d.Token)
	}
	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
		log.Printf("[hf] sending Range: bytes=%d-", existingSize)
	}

	reqStart := time.Now()
	resp, err := d.client.Do(req)
	if err != nil {
		return fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("[hf] response  : status=%s in %s", resp.Status, formatDuration(time.Since(reqStart)))
	log.Printf("[hf] resp hdr  : Content-Length=%s Accept-Ranges=%q ETag=%q Content-Range=%q",
		formatBytes(resp.ContentLength),
		resp.Header.Get("Accept-Ranges"),
		resp.Header.Get("ETag"),
		resp.Header.Get("Content-Range"),
	)
	if commit := resp.Header.Get("X-Repo-Commit"); commit != "" {
		log.Printf("[hf] resp hdr  : X-Repo-Commit=%s", commit)
	}
	if loc := resp.Header.Get("Location"); loc != "" {
		log.Printf("[hf] resp hdr  : Location=%s", maskedURL(loc))
	}

	startFromZero := false
	switch resp.StatusCode {
	case http.StatusOK:
		if existingSize > 0 {
			log.Printf("[hf] server ignored Range header (returned 200), restarting from byte 0")
		}
		startFromZero = true
	case http.StatusPartialContent:
		log.Printf("[hf] server accepted Range, resuming from byte %d", existingSize)
	case http.StatusRequestedRangeNotSatisfiable:
		log.Printf("[hf] 416 Range Not Satisfiable; .part may already be complete or stale, restarting from 0")
		startFromZero = true
		resp.Body.Close()
		req.Header.Del("Range")
		reqStart = time.Now()
		resp, err = d.client.Do(req)
		if err != nil {
			return fmt.Errorf("retry http request: %w", err)
		}
		defer resp.Body.Close()
		log.Printf("[hf] response  : status=%s in %s (retry without Range)",
			resp.Status, formatDuration(time.Since(reqStart)))
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("unexpected status on retry: %s", resp.Status)
		}
	default:
		// Read a small body slice for diagnostic purposes (4xx/5xx pages may include error messages).
		var snippet string
		if resp.Body != nil {
			b := make([]byte, 512)
			n, _ := io.ReadFull(resp.Body, b)
			snippet = strings.TrimSpace(string(b[:n]))
			if len(snippet) > 256 {
				snippet = snippet[:256] + "..."
			}
		}
		if snippet != "" {
			log.Printf("[hf] error body snippet: %q", snippet)
		}
		return fmt.Errorf("unexpected HTTP status: %s", resp.Status)
	}

	if startFromZero {
		existingSize = 0
	}

	totalSize := resp.ContentLength + existingSize
	if resp.StatusCode == http.StatusOK {
		totalSize = resp.ContentLength
	}
	log.Printf("[hf] total size : %s (%d bytes)", formatBytes(totalSize), totalSize)

	flags := os.O_WRONLY | os.O_CREATE
	if existingSize > 0 && resp.StatusCode == http.StatusPartialContent {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	f, err := os.OpenFile(partFile, flags, 0644)
	if err != nil {
		return fmt.Errorf("open part file: %w", err)
	}

	progress.UpdateProgress("downloading", existingSize, totalSize, modelName)

	const reportInterval = 2 * time.Second
	streamStart := time.Now()
	lastReport := streamStart
	lastReportBytes := existingSize
	written := existingSize
	buf := make([]byte, 256*1024)

	for {
		select {
		case <-ctx.Done():
			f.Close()
			log.Printf("[hf] context cancelled mid-stream after %s, downloaded %s",
				formatDuration(time.Since(streamStart)), formatBytes(written))
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

			now := time.Now()
			if now.Sub(lastReport) >= reportInterval {
				progress.UpdateProgress("downloading", written, totalSize, modelName)

				dt := now.Sub(lastReport).Seconds()
				deltaBytes := written - lastReportBytes
				curSpeed := float64(0) // bytes/s
				if dt > 0 {
					curSpeed = float64(deltaBytes) / dt
				}
				curSpeedMB := curSpeed / (1024 * 1024)

				avgDT := now.Sub(streamStart).Seconds()
				avgSpeedMB := float64(0)
				if avgDT > 0 {
					avgSpeedMB = float64(written-existingSize) / avgDT / (1024 * 1024)
				}

				pct := float64(0)
				if totalSize > 0 {
					pct = float64(written) / float64(totalSize) * 100
				}

				etaStr := "?"
				if curSpeed > 0 && totalSize > 0 {
					remaining := float64(totalSize - written)
					etaStr = formatDuration(time.Duration(remaining/curSpeed) * time.Second)
				}

				log.Printf("[hf] progress  : %.1f%% (%s / %s) cur=%.2f MiB/s avg=%.2f MiB/s eta=%s attempt=%d/%d",
					pct, formatBytes(written), formatBytes(totalSize),
					curSpeedMB, avgSpeedMB, etaStr, attempt, maxAttempts)

				lastReport = now
				lastReportBytes = written
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			f.Close()
			return fmt.Errorf("read after %s, downloaded %s of %s: %w",
				formatDuration(time.Since(streamStart)),
				formatBytes(written), formatBytes(totalSize), readErr)
		}
	}

	if err := f.Close(); err != nil {
		return fmt.Errorf("close part file: %w", err)
	}

	streamElapsed := time.Since(streamStart)
	streamedBytes := written - existingSize
	streamSpeedMB := float64(0)
	if streamElapsed.Seconds() > 0 {
		streamSpeedMB = float64(streamedBytes) / streamElapsed.Seconds() / (1024 * 1024)
	}
	log.Printf("[hf] stream done: wrote %s in %s (%.2f MiB/s); total on disk %s",
		formatBytes(streamedBytes), formatDuration(streamElapsed),
		streamSpeedMB, formatBytes(written))

	if totalSize > 0 && written < totalSize {
		return fmt.Errorf("short read: got %d of %d bytes", written, totalSize)
	}

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
