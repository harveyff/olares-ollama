package download

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"
)

// ProgressManager 下载进度管理器
type ProgressManager struct {
	mu        sync.RWMutex
	status    string
	progress  float64
	total     int64
	completed int64
	modelName string
	appURL    string
}

// ProgressUpdate 进度更新信息
type ProgressUpdate struct {
	Status    string  `json:"status"`
	Progress  float64 `json:"progress"`
	Total     int64   `json:"total"`
	Completed int64   `json:"completed"`
	ModelName string  `json:"model_name"`
	Timestamp int64   `json:"timestamp"`
}

// NewProgressManager 创建新的进度管理器
func NewProgressManager(appURL string) *ProgressManager {
	return &ProgressManager{
		appURL: appURL,
	}
}

// UpdateProgress 更新下载进度
func (pm *ProgressManager) UpdateProgress(status string, completed, total int64, modelName string) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.status = status
	pm.completed = completed
	pm.total = total
	pm.modelName = modelName

	if total > 0 {
		pm.progress = float64(completed) / float64(total) * 100
	}

	// 进度更新完成
}

// GetProgress 获取当前进度
func (pm *ProgressManager) GetProgress() ProgressUpdate {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	return ProgressUpdate{
		Status:    pm.status,
		Progress:  pm.progress,
		Total:     pm.total,
		Completed: pm.completed,
		ModelName: pm.modelName,
		Timestamp: time.Now().Unix(),
	}
}

// HandleProgressAPI 处理进度API请求
func (pm *ProgressManager) HandleProgressAPI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	progress := pm.GetProgress()

	response := map[string]interface{}{
		"status":     progress.Status,
		"progress":   progress.Progress,
		"total":      progress.Total,
		"completed":  progress.Completed,
		"model_name": progress.ModelName,
		"timestamp":  progress.Timestamp,
		"app_url":    pm.appURL,
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode progress response: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
}
