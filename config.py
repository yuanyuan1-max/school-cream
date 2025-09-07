#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch

# 检测配置
DETECTION_CONFIG = {
    'conf_threshold': float(os.getenv('CONF_THRESHOLD', '0.5')),      # 检测置信度阈值
    'max_det': int(os.getenv('MAX_DET', '100')),                     # 最大检测数量
    'update_interval': float(os.getenv('UPDATE_INTERVAL', '0.1')),   # 检测更新间隔（秒）
    'iou_threshold': 0.45,                                           # IoU阈值
    'agnostic_nms': False,                                           # 类别无关NMS
    'max_det_per_class': 50,                                         # 每个类别的最大检测数
}

# 模型配置
MODEL_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
    'model_priority': ['best', 'yolov12n', 'yolo11n'],  # 模型优先级
    'phone_model_enabled': True,                         # 是否启用手机检测
    'skeleton_max_frames': 300,                          # 骨架序列最大帧数
    'skeleton_num_joints': 25,                           # 骨架关键点数量
    'skeleton_coords': 3,                                # 坐标维度
}

# 路径配置
PATH_CONFIG = {
    'model_path': os.getenv('MODEL_PATH', './models'),
    'output_path': os.getenv('OUTPUT_PATH', './output'),
    'config_path': os.getenv('CONFIG_PATH', './config'),
    'log_path': os.getenv('LOG_PATH', './logs'),
}

# 视频处理配置
VIDEO_CONFIG = {
    'fps_limit': 30,                                    # 最大FPS
    'resolution': (1920, 1080),                         # 默认分辨率
    'codec': 'mp4v',                                    # 视频编码
    'quality': 95,                                      # 视频质量
    'buffer_size': 1000,                                # 缓冲区大小
}

# 网络配置
NETWORK_CONFIG = {
    'timeout': 30,                                      # 请求超时时间
    'retry_count': 3,                                   # 重试次数
    'retry_delay': 1,                                   # 重试延迟
    'max_connections': 10,                              # 最大连接数
}

# 日志配置
LOG_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_rotation': '1 day',
    'file_retention': '7 days',
    'max_file_size': '100MB',
}

# 性能配置
PERFORMANCE_CONFIG = {
    'enable_gpu': torch.cuda.is_available(),
    'gpu_memory_fraction': 0.8,                         # GPU内存使用比例
    'cpu_threads': os.cpu_count(),                      # CPU线程数
    'batch_size': 1,                                    # 批处理大小
    'prefetch_factor': 2,                               # 预取因子
}

def get_model_path(model_name):
    """获取模型文件路径"""
    return os.path.join(PATH_CONFIG['model_path'], f'{model_name}.pt')

def get_data_path(data_name):
    """获取数据文件路径"""
    return os.path.join(PATH_CONFIG['output_path'], data_name)

def check_model_exists(model_name):
    """检查模型文件是否存在"""
    model_path = get_model_path(model_name)
    return os.path.exists(model_path)

def get_device_info():
    """获取设备信息"""
    device_info = {
        'device': MODEL_CONFIG['device'],
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cpu_count': os.cpu_count(),
    }
    
    if torch.cuda.is_available():
        device_info['gpu_name'] = torch.cuda.get_device_name(0)
        device_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
    
    return device_info

def print_config():
    """打印配置信息"""
    print("=== 系统配置 ===")
    print(f"设备: {MODEL_CONFIG['device']}")
    print(f"数据类型: {MODEL_CONFIG['dtype']}")
    print(f"模型路径: {PATH_CONFIG['model_path']}")
    print(f"输出路径: {PATH_CONFIG['output_path']}")
    print(f"检测阈值: {DETECTION_CONFIG['conf_threshold']}")
    print(f"更新间隔: {DETECTION_CONFIG['update_interval']}s")
    print(f"最大检测数: {DETECTION_CONFIG['max_det']}")
    print(f"手机检测: {'启用' if MODEL_CONFIG['phone_model_enabled'] else '禁用'}")
    
    device_info = get_device_info()
    if device_info['cuda_available']:
        print(f"GPU: {device_info['gpu_name']}")
        print(f"GPU内存: {device_info['gpu_memory'] / 1024**3:.1f}GB")
    else:
        print("GPU: 未检测到CUDA设备，使用CPU模式")

if __name__ == "__main__":
    print_config()
