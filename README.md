# School Cream - 智能课堂监控系统

一个基于深度学习的智能课堂监控系统，用于实时检测和分析课堂中的学生行为，包括举手、躺卧、手机使用等行为检测。

## 🚀 功能特性

- **多目标行为检测**：举手、人员、躺卧、手机使用检测
- **实时视频处理**：支持RTSP流和本地视频文件
- **智能课程管理**：从注册中心API获取课程信息
- **姿态分析**：基于MediaPipe的人体姿态关键点提取
- **异常行为检测**：使用LSTM自编码器检测手机使用行为
- **多线程处理**：确保实时性和流畅性
- **测试服务器**：内置模拟API服务器，便于开发和测试

## 📋 系统要求

- Python 3.8+
- CUDA 11.0+ (推荐，用于GPU加速)
- 8GB+ RAM
- 支持OpenCV的摄像头或视频文件

## 🛠️ 安装说明

### 本地安装

1. **克隆项目**
```bash
git clone https://github.com/yuanyuan1-max/school-cream/tree/v1.1.0
cd school_cream
```

2. **创建虚拟环境**
```bash
conda create -n school python=3.8
conda activate school
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **下载模型文件**
确保 `models/` 目录包含以下文件：
- `best.pt` - 自定义训练的最佳模型（支持举手、人员、躺卧检测）
- `yolo11n.pt` - YOLO11n模型（仅支持人员检测）
- `yolov12n.pt` - YOLOv12n模型
- `phone_detection_autoencoder.pth` - 手机检测模型
- `threshold.npy` - 检测阈值文件

5. **启动测试服务器（可选）**
```bash
python test_server.py
```

### Docker部署

1. **构建镜像**
```bash
docker build -t school-cream .
```

2. **运行容器**
```bash
docker run -d \
  --name school-cream \
  --gpus all \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  school-cream
```

3. **使用Docker Compose**
```bash
docker-compose up -d
```

## 🎮 使用方法

### 命令行使用

```bash
python main.py --registry_url <API_URL> --max_frames <帧数限制>
```

**参数说明：**
- `--registry_url`：注册中心API地址（必需）
- `--max_frames`：最大处理帧数（可选，默认无限制）

**示例：**
```bash
# 使用测试服务器
python main.py --registry_url "http://localhost:5000/api/courses" --max_frames 1000

# 使用生产API
python main.py --registry_url "http://api.example.com/courses" --max_frames 1000
```

### 测试服务器使用

项目包含一个内置的测试服务器，用于模拟API响应：

```bash
# 启动测试服务器
python test_server.py

# 访问API文档
curl http://localhost:5000/

# 获取课程列表
curl http://localhost:5000/api/courses
```

### API请求结构

#### 注册中心API格式

系统期望从注册中心API获取以下JSON格式的数据：

```json
[
  {
    "timeAdd": "14:30:00",
    "courseName": "高等数学",
    "room_id": "A101",
    "device": [
      {
        "name": "前摄像头",
        "type": "camera",
        "liveUrl": "rtsp://192.168.1.100:554/stream1",
        "model_number": "IPC-123",
        "host": "192.168.1.100",
        "serial_number": "SN123456789",
        "token": "auth_token_here",
        "admin": "admin",
        "password": "password123",
        "create_by": "system",
        "create_date": "2024-01-01 00:00:00",
        "update_by": "admin",
        "update_date": "2024-01-01 00:00:00",
        "status": "1",
        "remarks": "前门摄像头",
        "del_flag": "0"
      },
      {
        "name": "后摄像头",
        "type": "camera",
        "liveUrl": "rtsp://192.168.1.101:554/stream1",
        "model_number": "IPC-124",
        "host": "192.168.1.101",
        "serial_number": "SN123456790",
        "token": "auth_token_here",
        "admin": "admin",
        "password": "password123",
        "create_by": "system",
        "create_date": "2024-01-01 00:00:00",
        "update_by": "admin",
        "update_date": "2024-01-01 00:00:00",
        "status": "1",
        "remarks": "后门摄像头",
        "del_flag": "0"
      }
    ]
  }
]
```

#### 字段说明

**课程信息：**
- `timeAdd`：课程时间（格式：HH:MM:SS）
- `courseName`：课程名称
- `room_id`：教室ID

**设备信息：**
- `name`：设备名称
- `type`：设备类型（camera）
- `liveUrl`：RTSP流地址或视频文件路径
- `model_number`：设备型号
- `host`：设备IP地址
- `serial_number`：设备序列号
- `token`：认证令牌
- `admin`：管理员用户名
- `password`：管理员密码
- `create_by`：创建者
- `create_date`：创建时间
- `update_by`：更新者
- `update_date`：更新时间
- `status`：设备状态（1=启用，0=禁用）
- `remarks`：备注信息
- `del_flag`：删除标志（0=未删除，1=已删除）

## 📁 输出文件

系统会为每个课程和摄像头组合生成一个视频文件，命名格式为：
```
{时间}_{课程名称}_{摄像头位置}.mp4
```

示例：`14-30-00_高等数学_前摄像头.mp4`

## ⚙️ 配置说明

### 检测配置

在 `config.py` 中可以配置以下参数：

```python
DETECTION_CONFIG = {
    'conf_threshold': 0.5,      # 检测置信度阈值
    'max_det': 100,             # 最大检测数量
    'update_interval': 0.1      # 检测更新间隔（秒）
}
```

### 模型配置

系统支持多种YOLO模型，按优先级选择：
1. `best.pt` - 自定义训练的最佳模型
2. `yolov12n.pt` - YOLOv12n模型
3. `yolo11n.pt` - YOLO11n模型

## 🔧 开发说明

### 项目结构

```
school_cream/
├── main.py                    # 主程序入口
├── config.py                  # 配置文件
├── annotation_config.py       # 标注配置（检测标签、颜色、阈值）
├── test_server.py             # 测试API服务器
├── models/                    # 模型文件目录
│   ├── best.pt               # 最佳YOLO模型（支持举手、人员、躺卧）
│   ├── yolo11n.pt            # YOLO11n模型（仅人员检测）
│   ├── yolov12n.pt           # YOLOv12n模型
│   ├── phone_detection_autoencoder.pth    # 手机检测模型
│   └── threshold.npy         # 检测阈值
├── output/                    # 输出视频目录
├── requirements.txt           # Python依赖
├── Dockerfile                # Docker构建文件
├── docker-compose.yml        # Docker编排文件
├── deploy.sh                 # 部署脚本
├── nginx.conf                # Nginx配置
├── prometheus.yml            # 监控配置
├── env.example               # 环境变量示例
└── README.md                 # 项目说明
```

### 核心类说明

- `Autoencoder`：LSTM自编码器，用于手机使用检测
- `SkeletonExtractor`：骨架提取器，基于MediaPipe
- `VideoProcessor`：视频处理器，整合所有检测功能

## 🐳 Docker部署

### 环境变量

可以通过以下环境变量配置系统：

```bash
# 模型路径
MODEL_PATH=/app/models

# 输出路径
OUTPUT_PATH=/app/output

# 检测配置
CONF_THRESHOLD=0.5
MAX_DET=100
UPDATE_INTERVAL=0.1

# GPU配置
CUDA_VISIBLE_DEVICES=0
```

### 数据卷挂载

```bash
# 模型文件
-v /host/models:/app/models

# 输出文件
-v /host/output:/app/output

# 配置文件
-v /host/config:/app/config
```

## 📊 性能优化

1. **GPU加速**：使用CUDA加速模型推理
2. **多线程处理**：检测和视频处理分离
3. **内存管理**：及时释放不需要的帧数据
4. **模型优化**：使用半精度浮点数减少内存占用

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否存在
   - 确认模型文件格式正确

2. **RTSP连接失败**
   - 检查网络连接
   - 确认RTSP地址和认证信息

3. **GPU内存不足**
   - 减少批处理大小
   - 使用CPU模式

4. **检测精度低**
   - 调整置信度阈值
   - 检查光照条件

## 📝 更新日志

### v1.1.0 (当前版本)
- ✅ 修复best.pt模型检测问题
- ✅ 支持Person和Lie Down检测
- ✅ 优化置信度阈值配置
- ✅ 添加测试服务器支持
- ✅ 清理项目结构，删除临时文件
- ✅ 更新Docker配置和文档

### v1.0.0
- 初始版本发布
- 支持四种行为检测
- 集成手机使用检测
- 支持RTSP流处理

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 📞 联系方式

如有问题或建议，请联系开发团队。

