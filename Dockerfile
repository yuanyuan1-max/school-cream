# 使用官方Python运行时作为基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 升级pip并安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p /app/models /app/output /app/config /app/logs

# 设置权限
RUN chmod +x main.py

# 暴露端口（如果需要Web服务）
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import cv2; import mediapipe; print('Health check passed')" || exit 1

# 默认命令
CMD ["python", "main.py", "--registry_url", "http://localhost:8080/api/courses"]

# 多阶段构建版本（可选，用于减小镜像大小）
# FROM python:3.8-slim as builder
# 
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir --user -r requirements.txt
# 
# FROM python:3.8-slim
# 
# # 复制Python包
# COPY --from=builder /root/.local /root/.local
# 
# # 安装系统依赖
# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     libgtk-3-0 \
#     libavcodec-dev \
#     libavformat-dev \
#     libswscale-dev \
#     libv4l-dev \
#     libxvidcore-dev \
#     libx264-dev \
#     libjpeg-dev \
#     libpng-dev \
#     libtiff-dev \
#     libatlas-base-dev \
#     gfortran \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*
# 
# WORKDIR /app
# COPY . .
# RUN mkdir -p /app/models /app/output /app/config /app/logs
# 
# ENV PATH=/root/.local/bin:$PATH
# 
# EXPOSE 8080
# 
# CMD ["python", "main.py", "--registry_url", "http://localhost:8080/api/courses"]
