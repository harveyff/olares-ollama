package download

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ProgressManager 下载进度管理器
type ProgressManager struct {
	mu          sync.RWMutex
	status      string
	progress    float64
	total       int64
	completed   int64
	modelName   string
	appURL      string
	startTime   time.Time
	completedAt *time.Time // 完成时间，只在成功时设置一次
	duration    int64      // 用时（秒），从持久化状态恢复
	stateFile   string     // 状态文件路径
}

// persistedState 持久化的状态
type persistedState struct {
	Status      string `json:"status"`
	ModelName   string `json:"model_name"`
	CompletedAt int64  `json:"completed_at"`
	Duration    int64  `json:"duration"`
}

// ProgressUpdate 进度更新信息
type ProgressUpdate struct {
	Status      string  `json:"status"`
	Progress    float64 `json:"progress"`
	Total       int64   `json:"total"`
	Completed   int64   `json:"completed"`
	ModelName   string  `json:"model_name"`
	Timestamp   int64   `json:"timestamp"`   // 当前时间戳（用于实时更新）
	CompletedAt *int64  `json:"completed_at,omitempty"` // 完成时间戳（固定不变）
	Duration    *int64  `json:"duration,omitempty"`     // 用时（秒）
}

// NewProgressManager 创建新的进度管理器
func NewProgressManager(appURL string) *ProgressManager {
	// 状态文件路径：data/progress_state.json
	stateFile := filepath.Join("data", "progress_state.json")
	
	pm := &ProgressManager{
		appURL:    appURL,
		startTime: time.Now(),
		stateFile: stateFile,
	}
	
	// 尝试加载持久化的状态
	pm.loadState()
	
	return pm
}

// loadState 从文件加载持久化的状态
func (pm *ProgressManager) loadState() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	
	// 确保 data 目录存在
	if err := os.MkdirAll(filepath.Dir(pm.stateFile), 0755); err != nil {
		log.Printf("Failed to create data directory: %v", err)
		return
	}
	
	// 读取状态文件
	data, err := os.ReadFile(pm.stateFile)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("Failed to read state file: %v", err)
		}
		return
	}
	
	var state persistedState
	if err := json.Unmarshal(data, &state); err != nil {
		log.Printf("Failed to parse state file: %v", err)
		return
	}
	
	// 如果之前已完成，恢复状态
	if state.Status == "completed" && state.CompletedAt > 0 {
		pm.status = state.Status
		pm.modelName = state.ModelName
		completedTime := time.Unix(state.CompletedAt, 0)
		pm.completedAt = &completedTime
		pm.duration = state.Duration // 恢复持久化的用时
		log.Printf("Loaded persisted state: model=%s, completed_at=%v, duration=%ds", 
			state.ModelName, completedTime, state.Duration)
	}
}

// saveState 保存状态到文件
func (pm *ProgressManager) saveState() {
	if pm.completedAt == nil {
		return
	}
	
	// 确保 data 目录存在
	if err := os.MkdirAll(filepath.Dir(pm.stateFile), 0755); err != nil {
		log.Printf("Failed to create data directory: %v", err)
		return
	}
	
	// 计算用时：如果已经设置过 duration，使用它；否则计算
	var duration int64
	if pm.duration > 0 {
		duration = pm.duration
	} else {
		duration = int64(pm.completedAt.Sub(pm.startTime).Seconds())
		pm.duration = duration
	}
	
	state := persistedState{
		Status:      pm.status,
		ModelName:   pm.modelName,
		CompletedAt: pm.completedAt.Unix(),
		Duration:    duration,
	}
	
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		log.Printf("Failed to marshal state: %v", err)
		return
	}
	
	if err := os.WriteFile(pm.stateFile, data, 0644); err != nil {
		log.Printf("Failed to write state file: %v", err)
		return
	}
	
	log.Printf("Saved progress state to %s", pm.stateFile)
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

	// 如果状态是 completed 或 success，且还未设置完成时间，则设置完成时间
	// 注意：如果已经有持久化的完成时间（从文件加载），不要重新设置
	if (status == "completed" || status == "success") && pm.completedAt == nil {
		now := time.Now()
		pm.completedAt = &now
		// 计算并保存用时
		pm.duration = int64(now.Sub(pm.startTime).Seconds())
		// 保存状态到文件
		pm.saveState()
	} else if status == "completed" || status == "success" {
		// 如果已经有完成时间，确保状态和模型名正确，并保存（以防状态变化）
		pm.saveState()
	}
}

// GetProgress 获取当前进度
func (pm *ProgressManager) GetProgress() ProgressUpdate {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	update := ProgressUpdate{
		Status:    pm.status,
		Progress:  pm.progress,
		Total:     pm.total,
		Completed: pm.completed,
		ModelName: pm.modelName,
		Timestamp: time.Now().Unix(),
	}

	// 如果已完成，设置完成时间和用时
	if pm.completedAt != nil {
		completedAtUnix := pm.completedAt.Unix()
		update.CompletedAt = &completedAtUnix
		
		// 使用持久化的用时，如果没有则计算
		var duration int64
		if pm.duration > 0 {
			duration = pm.duration
		} else {
			duration = int64(pm.completedAt.Sub(pm.startTime).Seconds())
		}
		update.Duration = &duration
	}

	return update
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

	// 添加完成时间和用时（如果存在）
	if progress.CompletedAt != nil {
		response["completed_at"] = *progress.CompletedAt
	}
	if progress.Duration != nil {
		response["duration"] = *progress.Duration
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode progress response: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
}
