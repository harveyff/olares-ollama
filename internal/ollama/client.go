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

	// 记录所有模型名称用于调试
	modelNames := make([]string, 0, len(modelResp.Models))
	for _, model := range modelResp.Models {
		modelNames = append(modelNames, model.Name)
	}
	log.Printf("Checking model '%s' against available models: %v", modelName, modelNames)

	// 精确匹配
	for _, model := range modelResp.Models {
		if model.Name == modelName {
			log.Printf("Model '%s' found (exact match)", modelName)
			return true, nil
		}
	}

	// 前缀匹配：如果查找的是 "model:tag"，也匹配 "model"
	// 例如 "cogito:14b" 应该匹配 "cogito"
	if strings.Contains(modelName, ":") {
		baseName := strings.Split(modelName, ":")[0]
		for _, model := range modelResp.Models {
			// 匹配 "model" 或 "model:" 开头的
			if model.Name == baseName || strings.HasPrefix(model.Name, baseName+":") {
				log.Printf("Model '%s' found (prefix match: '%s' matches '%s')", modelName, modelName, model.Name)
				return true, nil
			}
		}
	}

	// 反向匹配：如果查找的是 "model"，也匹配 "model:tag"
	// 例如查找 "cogito" 应该匹配 "cogito:14b"
	for _, model := range modelResp.Models {
		if strings.HasPrefix(model.Name, modelName+":") || model.Name == modelName {
			log.Printf("Model '%s' found (reverse prefix match: '%s' matches '%s')", modelName, modelName, model.Name)
			return true, nil
		}
	}

	log.Printf("Model '%s' not found in model list", modelName)
	return false, nil
}

// ModelUsable checks if model is usable by trying to call it
// This is a fallback when model exists in files but not in the list
func (c *Client) ModelUsable(modelName string) (bool, error) {
	// Try to call /api/show to check if model is usable
	showReq := map[string]interface{}{
		"name": modelName,
	}
	jsonData, err := json.Marshal(showReq)
	if err != nil {
		return false, err
	}

	resp, err := c.httpClient.Post(
		c.baseURL+"/api/show",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	// If status is OK, model is usable even if not in list
	if resp.StatusCode == http.StatusOK {
		log.Printf("Model '%s' is usable (verified via /api/show)", modelName)
		return true, nil
	}

	// Try a simple generate request as fallback
	generateReq := map[string]interface{}{
		"model":  modelName,
		"prompt": "test",
		"stream": false,
	}
	jsonData, err = json.Marshal(generateReq)
	if err != nil {
		return false, err
	}

	// Use a short timeout for this test
	testClient := &http.Client{
		Timeout: 10 * time.Second,
	}
	resp, err = testClient.Post(
		c.baseURL+"/api/generate",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	// If we get a response (even if error about prompt), model exists
	// 400/404 means model doesn't exist, 200/500 might mean model exists but prompt issue
	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusInternalServerError {
		log.Printf("Model '%s' appears to be usable (verified via /api/generate, status: %d)", modelName, resp.StatusCode)
		return true, nil
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
	var successCount int
	
	// 使用带超时的读取，避免无限等待
	// 但要注意：我们不能简单地设置超时，因为下载可能需要很长时间
	// 所以继续读取直到 EOF，但记录 success 状态
	
	for {
		var pullResp PullResponse
		if err := decoder.Decode(&pullResp); err == io.EOF {
			// 流结束
			log.Printf("Download stream ended (EOF)")
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

		// 记录 success 状态，但不要立即退出
		// Ollama 可能会发送多个 success 状态，或者 success 后还有更多数据
		if pullResp.Status == "success" {
			gotSuccess = true
			successCount++
			log.Printf("Received 'success' status (count: %d), continuing to read stream...", successCount)
			// 不要 break，继续读取直到 EOF
		}
	}

	// 如果没有收到 success 状态，检查是否是因为流提前结束
	if !gotSuccess {
		log.Printf("Warning: Download stream ended without 'success' status for model %s", modelName)
		// 继续验证，因为有时流可能提前结束但下载已完成
	} else {
		log.Printf("Download stream completed with 'success' status (received %d times)", successCount)
	}

	// 重要：Ollama 的 /api/pull 流结束后，后台可能还在继续下载文件
	// 需要等待一段时间让 Ollama 完成实际的文件下载
	// 特别是大文件，可能需要更长时间
	log.Printf("Stream ended, but Ollama may still be downloading files in background. Waiting for background download to complete...")
	progressUpdater.UpdateProgress("downloading", lastPullResp.Completed, lastPullResp.Total, modelName)
	
	// 等待后台下载完成：根据文件大小估算等待时间
	// 如果 Total 很大，需要等待更长时间
	waitTime := 30 * time.Second // 默认等待30秒
	if lastPullResp.Total > 0 {
		// 估算：假设下载速度至少 10MB/s，等待时间 = 剩余大小 / 10MB/s + 缓冲
		remainingMB := float64(lastPullResp.Total-lastPullResp.Completed) / (1024 * 1024)
		estimatedSeconds := int(remainingMB/10) + 30 // 至少30秒缓冲
		if estimatedSeconds > 300 {
			estimatedSeconds = 300 // 最多等待5分钟
		}
		waitTime = time.Duration(estimatedSeconds) * time.Second
		log.Printf("Estimated wait time for background download: %v (based on %d MB remaining)", waitTime, int(remainingMB))
	}
	
	// 等待期间定期检查模型是否可用
	checkInterval := 5 * time.Second
	elapsed := time.Duration(0)
	for elapsed < waitTime {
		time.Sleep(checkInterval)
		elapsed += checkInterval
		
		// 检查模型是否已经在列表中
		exists, err := c.ModelExists(modelName)
		if err == nil && exists {
			log.Printf("Model %s appeared in list during background download wait (after %v)", modelName, elapsed)
			progressUpdater.UpdateProgress("completed", lastPullResp.Completed, lastPullResp.Total, modelName)
			return nil
		}
		
		// 每30秒输出一次等待状态
		if int(elapsed.Seconds())%30 == 0 {
			log.Printf("Still waiting for background download to complete... (%v/%v elapsed)", elapsed, waitTime)
			progressUpdater.UpdateProgress("downloading", lastPullResp.Completed, lastPullResp.Total, modelName)
		}
	}
	
	log.Printf("Background download wait completed (%v), proceeding to verification...", waitTime)

	// 验证模型是否真的下载成功并可用
	// Ollama 可能在部分文件失败时仍返回 success，需要验证
	// Ollama 下载完成后需要一些时间来注册模型到列表中
	log.Printf("Verifying model %s is complete and usable...", modelName)
	maxVerifyAttempts := 10  // 增加验证次数
	initialDelay := 5 * time.Second  // 第一次验证前等待5秒
	verifyDelay := 3 * time.Second   // 初始延迟3秒
	
	// 第一次验证前等待，给 Ollama 时间完成模型注册
	log.Printf("Waiting %v before first verification attempt...", initialDelay)
	progressUpdater.UpdateProgress("verifying", lastPullResp.Completed, lastPullResp.Total, modelName)
	time.Sleep(initialDelay)
	
	for attempt := 1; attempt <= maxVerifyAttempts; attempt++ {
		// 等待一段时间让 Ollama 完成文件写入和模型注册
		if attempt > 1 {
			log.Printf("Verification attempt %d/%d for model %s (waiting %v)...", 
				attempt, maxVerifyAttempts, modelName, verifyDelay)
			// 更新状态显示验证进度
			progressUpdater.UpdateProgress("verifying", lastPullResp.Completed, lastPullResp.Total, modelName)
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
			// 继续验证，更新状态
			progressUpdater.UpdateProgress("verifying", lastPullResp.Completed, lastPullResp.Total, modelName)
			continue
		}
		
		if exists {
			log.Printf("Model %s verified successfully on attempt %d/%d", modelName, attempt, maxVerifyAttempts)
			progressUpdater.UpdateProgress("completed", lastPullResp.Completed, lastPullResp.Total, modelName)
			return nil
		}
		
		log.Printf("Model %s not found in model list (attempt %d/%d)", modelName, attempt, maxVerifyAttempts)
		
		// 如果模型不在列表中，尝试通过实际调用验证模型是否可用
		// 有时文件已下载但 Ollama 还没注册到列表中
		if attempt >= 3 {
			log.Printf("Model not in list, trying to verify via API call...")
			usable, err := c.ModelUsable(modelName)
			if err == nil && usable {
				log.Printf("Model %s verified as usable via API call (files exist but not registered in list)", modelName)
				progressUpdater.UpdateProgress("completed", lastPullResp.Completed, lastPullResp.Total, modelName)
				return nil
			}
		}
		
		// 更新状态，显示正在验证
		progressUpdater.UpdateProgress("verifying", lastPullResp.Completed, lastPullResp.Total, modelName)
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
