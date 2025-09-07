#!/bin/bash

# School Cream 部署脚本
# 使用方法: ./deploy.sh [选项]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查NVIDIA Docker（如果使用GPU）
    if command -v nvidia-smi &> /dev/null; then
        if ! docker info | grep -q nvidia; then
            log_warning "检测到NVIDIA GPU但未安装nvidia-docker，将使用CPU模式"
        fi
    fi
    
    log_success "依赖检查完成"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    
    mkdir -p models output config logs ssl
    
    # 设置权限
    chmod 755 models output config logs ssl
    
    log_success "目录创建完成"
}

# 检查模型文件
check_models() {
    log_info "检查模型文件..."
    
    required_models=("best.pt" "yolo11n.pt" "phone_detection_autoencoder.pth" "threshold.npy")
    missing_models=()
    
    for model in "${required_models[@]}"; do
        if [ ! -f "models/$model" ]; then
            missing_models+=("$model")
        fi
    done
    
    if [ ${#missing_models[@]} -gt 0 ]; then
        log_warning "以下模型文件缺失:"
        for model in "${missing_models[@]}"; do
            echo "  - models/$model"
        done
        log_warning "请确保模型文件存在，否则系统可能无法正常工作"
    else
        log_success "所有模型文件检查完成"
    fi
}

# 构建镜像
build_image() {
    log_info "构建Docker镜像..."
    
    docker build -t school-cream:latest .
    
    log_success "镜像构建完成"
}

# 启动服务
start_services() {
    log_info "启动服务..."
    
    # 停止现有服务
    docker-compose down 2>/dev/null || true
    
    # 启动服务
    docker-compose up -d
    
    log_success "服务启动完成"
}

# 检查服务状态
check_services() {
    log_info "检查服务状态..."
    
    sleep 10  # 等待服务启动
    
    # 检查主服务
    if docker-compose ps | grep -q "school-cream.*Up"; then
        log_success "主服务运行正常"
    else
        log_error "主服务启动失败"
        docker-compose logs school-cream
        exit 1
    fi
    
    # 检查其他服务
    services=("redis" "nginx" "prometheus" "grafana")
    for service in "${services[@]}"; do
        if docker-compose ps | grep -q "$service.*Up"; then
            log_success "$service 服务运行正常"
        else
            log_warning "$service 服务可能未启动"
        fi
    done
}

# 显示服务信息
show_info() {
    log_info "服务信息:"
    echo "  主服务: http://localhost:8080"
    echo "  Nginx: http://localhost:80"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana: http://localhost:3000 (admin/admin123)"
    echo "  Redis: localhost:6379"
    echo ""
    echo "  日志查看: docker-compose logs -f school-cream"
    echo "  停止服务: docker-compose down"
    echo "  重启服务: docker-compose restart"
}

# 清理资源
cleanup() {
    log_info "清理资源..."
    
    docker-compose down
    docker system prune -f
    
    log_success "清理完成"
}

# 显示帮助
show_help() {
    echo "School Cream 部署脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  build     仅构建镜像"
    echo "  start     启动服务"
    echo "  stop      停止服务"
    echo "  restart   重启服务"
    echo "  status    检查服务状态"
    echo "  logs      查看日志"
    echo "  cleanup   清理资源"
    echo "  help      显示帮助"
    echo ""
    echo "示例:"
    echo "  $0                # 完整部署"
    echo "  $0 build          # 仅构建镜像"
    echo "  $0 start          # 启动服务"
    echo "  $0 logs           # 查看日志"
}

# 主函数
main() {
    case "${1:-deploy}" in
        "build")
            check_dependencies
            create_directories
            check_models
            build_image
            ;;
        "start")
            start_services
            check_services
            show_info
            ;;
        "stop")
            log_info "停止服务..."
            docker-compose down
            log_success "服务已停止"
            ;;
        "restart")
            log_info "重启服务..."
            docker-compose restart
            check_services
            show_info
            ;;
        "status")
            check_services
            ;;
        "logs")
            docker-compose logs -f school-cream
            ;;
        "cleanup")
            cleanup
            ;;
        "help")
            show_help
            ;;
        "deploy")
            check_dependencies
            create_directories
            check_models
            build_image
            start_services
            check_services
            show_info
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
