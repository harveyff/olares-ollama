# 多阶段构建
FROM golang:1.21-alpine AS builder

# 设置工作目录
WORKDIR /app

# 复制go mod文件
#COPY go.mod go.sum ./
COPY go.* ./
RUN go mod download

# 下载依赖
RUN go mod download

# 复制源代码
COPY . .

# 构建应用
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# 运行阶段
FROM alpine:latest

# 安装ca-certificates用于HTTPS请求
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# 从构建阶段复制二进制文件
COPY --from=builder /app/main .

# 复制静态文件
COPY --from=builder /app/web ./web

# 暴露端口
EXPOSE 8080

# 设置环境变量
ENV OLLAMA_MODEL=llama2
ENV OLLAMA_URL=http://ollama:11434
ENV PORT=8080

# 运行应用
CMD ["./main"]
