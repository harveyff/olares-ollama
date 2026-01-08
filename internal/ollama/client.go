package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// Client Ollama client
type Client struct {
	baseURL        string
	httpClient     *http.Client
	downloadClient *http.Client
}

// NewClient creates a new Ollama client
func NewClient(baseURL string) *Client {
	return NewClientWithTimeout(baseURL, 60) // Default 60 minutes download timeout
}

// NewClientWithTimeout creates a new Ollama client with custom timeout
func NewClientWithTimeout(baseURL string, downloadTimeoutMinutes int) *Client {
	return &Client{
		baseURL: strings.TrimSuffix(baseURL, "/"),
		// Regular request client, 30 minutes timeout for long inference requests
		// This is needed for chat completions that may take a long time
		httpClient: &http.Client{
			Timeout: 30 * time.Minute,
		},
		// Download dedicated client, configurable timeout
		downloadClient: &http.Client{
			Timeout: time.Duration(downloadTimeoutMinutes) * time.Minute,
		},
	}
}

// ModelResponse model list response
type ModelResponse struct {
	Models []Model `json:"models"`
}

// Model model information
type Model struct {
	Name       string    `json:"name"`
	ModifiedAt time.Time `json:"modified_at"`
	Size       int64     `json:"size"`
}

// PullRequest pull model request
type PullRequest struct {
	Name string `json:"name"`
}

// PullResponse pull model response
type PullResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// ModelExists checks if model exists
func (c *Client) ModelExists(modelName string) (bool, error) {
	resp, err := c.httpClient.Get(c.baseURL + "/api/tags")
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("failed to get models: %s", resp.Status)
	}

	var modelResp ModelResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelResp); err != nil {
		return false, err
	}

	for _, model := range modelResp.Models {
		if model.Name == modelName {
			return true, nil
		}
	}

	return false, nil
}

// PullModel downloads model
func (c *Client) PullModel(modelName string) error {
	pullReq := PullRequest{Name: modelName}
	jsonData, err := json.Marshal(pullReq)
	if err != nil {
		return err
	}

	// 使用专门的下载客户端，支持长时间下载
	resp, err := c.downloadClient.Post(
		c.baseURL+"/api/pull",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to pull model: %s", resp.Status)
	}

	// 读取流式响应
	decoder := json.NewDecoder(resp.Body)
	for {
		var pullResp PullResponse
		if err := decoder.Decode(&pullResp); err == io.EOF {
			break
		} else if err != nil {
			return err
		}

		// Progress callback can be added here
		if pullResp.Status != "" {
			fmt.Printf("Pull status: %s", pullResp.Status)
			if pullResp.Total > 0 {
				progress := float64(pullResp.Completed) / float64(pullResp.Total) * 100
				fmt.Printf(" (%.1f%%)", progress)
			}
			fmt.Println()
		}

		if pullResp.Status == "success" {
			break
		}
	}

	return nil
}

// ProgressUpdater 进度更新接口
type ProgressUpdater interface {
	UpdateProgress(status string, completed, total int64, modelName string)
}

// PullModelWithProgress 下载模型并更新进度
func (c *Client) PullModelWithProgress(modelName string, progressUpdater ProgressUpdater) error {
	pullReq := PullRequest{Name: modelName}
	jsonData, err := json.Marshal(pullReq)
	if err != nil {
		return err
	}

	progressUpdater.UpdateProgress("starting", 0, 0, modelName)

	// 使用专门的下载客户端，支持长时间下载
	resp, err := c.downloadClient.Post(
		c.baseURL+"/api/pull",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		progressUpdater.UpdateProgress("error", 0, 0, modelName)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		progressUpdater.UpdateProgress("error", 0, 0, modelName)
		return fmt.Errorf("failed to pull model: %s", resp.Status)
	}

	progressUpdater.UpdateProgress("downloading", 0, 0, modelName)

	// 读取流式响应
	decoder := json.NewDecoder(resp.Body)
	var lastPullResp PullResponse
	var gotSuccess bool
	
	for {
		var pullResp PullResponse
		if err := decoder.Decode(&pullResp); err == io.EOF {
			break
		} else if err != nil {
			progressUpdater.UpdateProgress("error", 0, 0, modelName)
			return err
		}

		lastPullResp = pullResp

		// 更新进度
		progressUpdater.UpdateProgress(pullResp.Status, pullResp.Completed, pullResp.Total, modelName)

		// 打印控制台进度
		if pullResp.Status != "" {
			fmt.Printf("Pull status: %s", pullResp.Status)
			if pullResp.Total > 0 {
				progress := float64(pullResp.Completed) / float64(pullResp.Total) * 100
				fmt.Printf(" (%.1f%%)", progress)
			}
			fmt.Println()
		}

		if pullResp.Status == "success" {
			gotSuccess = true
			progressUpdater.UpdateProgress("verifying", pullResp.Completed, pullResp.Total, modelName)
			break
		}
	}

	// 如果没有收到 success 状态，检查是否是因为流提前结束
	if !gotSuccess {
		log.Printf("Warning: Download stream ended without 'success' status for model %s", modelName)
		// 继续验证，因为有时流可能提前结束但下载已完成
	}

	// 验证模型是否真的下载成功并可用
	// Ollama 可能在部分文件失败时仍返回 success，需要验证
	// Ollama 下载完成后需要一些时间来注册模型到列表中
	log.Printf("Verifying model %s is complete and usable...", modelName)
	maxVerifyAttempts := 10  // 增加验证次数
	initialDelay := 5 * time.Second  // 第一次验证前等待5秒
	verifyDelay := 3 * time.Second   // 初始延迟3秒
	
	// 第一次验证前等待，给 Ollama 时间完成模型注册
	log.Printf("Waiting %v before first verification attempt...", initialDelay)
	time.Sleep(initialDelay)
	
	for attempt := 1; attempt <= maxVerifyAttempts; attempt++ {
		// 等待一段时间让 Ollama 完成文件写入和模型注册
		if attempt > 1 {
			log.Printf("Verification attempt %d/%d for model %s (waiting %v)...", 
				attempt, maxVerifyAttempts, modelName, verifyDelay)
			time.Sleep(verifyDelay)
			// 指数退避，但最大不超过30秒
			verifyDelay *= 2
			if verifyDelay > 30*time.Second {
				verifyDelay = 30 * time.Second
			}
		}
		
		exists, err := c.ModelExists(modelName)
		if err != nil {
			log.Printf("Error verifying model %s: %v", modelName, err)
			if attempt == maxVerifyAttempts {
				progressUpdater.UpdateProgress("error", 0, 0, modelName)
				return fmt.Errorf("failed to verify model after download: %w", err)
			}
			continue
		}
		
		if exists {
			log.Printf("Model %s verified successfully on attempt %d/%d", modelName, attempt, maxVerifyAttempts)
			progressUpdater.UpdateProgress("completed", lastPullResp.Completed, lastPullResp.Total, modelName)
			return nil
		}
		
		log.Printf("Model %s not found in model list (attempt %d/%d)", modelName, attempt, maxVerifyAttempts)
	}
	
	// 验证失败
	progressUpdater.UpdateProgress("error", 0, 0, modelName)
	return fmt.Errorf("model %s download reported success but model is not available in Ollama", modelName)
}

// ProxyRequest 代理请求到Ollama
func (c *Client) ProxyRequest(method, path string, body io.Reader, headers map[string]string) (*http.Response, error) {
	url := c.baseURL + path
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// 复制头部，但确保 Content-Type 正确设置
	for key, value := range headers {
		// 跳过可能冲突的头部（如 Content-Length，Go 会自动设置）
		if strings.ToLower(key) == "content-length" {
			continue
		}
		req.Header.Set(key, value)
	}

	// 确保请求方法正确
	if req.Method != method {
		return nil, fmt.Errorf("request method mismatch: expected %s, got %s", method, req.Method)
	}

	return c.httpClient.Do(req)
}
