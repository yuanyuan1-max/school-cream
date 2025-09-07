#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
标注配置模块
用于自定义四个标签的检测和标注参数
"""

# 标签配置 - 基于best.pt模型的类别映射
LABEL_CONFIG = {
    'Raise Hand': {
        'class_id': 0,             # best.pt中举手检测的ID是0
        'color': (0, 255, 0),      # 绿色 (BGR格式)
        'min_confidence': 0.05,    # 最小置信度（降低以检测更多）
        'description': '举手检测',
        'enabled': True
    },
    'Person': {
        'class_id': 1,             # best.pt中人员检测的ID是1
        'color': (0, 100, 255),    # 蓝色 (BGR格式)
        'min_confidence': 0.1,     # 最小置信度
        'max_height_ratio': 0.8,   # 最大高度比例
        'description': '人员检测',
        'enabled': True
    },
    'Lie Down': {
        'class_id': 2,             # best.pt中躺卧检测的ID是2
        'color': (255, 0, 0),      # 红色 (BGR格式)
        'min_confidence': 0.05,    # 最小置信度（降低以检测更多）
        'description': '躺卧检测',
        'enabled': True
    },
    'Phone Usage': {
        'class_id': None,          # 基于Person检测
        'color': (255, 165, 0),    # 橙色 (BGR格式)
        'min_confidence': 0.1,     # 最小置信度
        'description': '手机使用检测',
        'enabled': True,
        'requires_person': True    # 需要先检测到Person
    }
}

# 标注样式配置
ANNOTATION_STYLE = {
    'box_thickness': 2,            # 边框粗细
    'text_thickness': 2,           # 文字粗细
    'text_scale': 0.5,             # 文字大小
    'text_offset': 10,             # 文字偏移
    'phone_text_offset': 30,       # 手机检测文字偏移
    'phone_box_thickness': 3,      # 手机检测边框粗细
}

# 统计信息覆盖层配置
STATS_OVERLAY = {
    'enabled': True,               # 是否显示统计信息
    'position': (10, 10),          # 位置 (x, y)
    'size': (300, 120),            # 大小 (width, height)
    'background_alpha': 0.7,       # 背景透明度
    'text_color': (255, 255, 255), # 文字颜色
    'title_scale': 0.6,            # 标题大小
    'stats_scale': 0.5,            # 统计信息大小
    'line_spacing': 25,            # 行间距
}

# 检测参数配置
DETECTION_PARAMS = {
    'conf_threshold': 0.5,         # 全局置信度阈值
    'iou_threshold': 0.45,         # IoU阈值
    'max_detections': 100,         # 最大检测数量
    'update_interval': 0.1,        # 检测更新间隔（秒）
}

# 手机检测配置
PHONE_DETECTION = {
    'enabled': True,               # 是否启用手机检测
    'model_path': 'models/phone_detection_autoencoder.pth',
    'threshold_path': 'models/threshold.npy',
    'skeleton_max_frames': 300,    # 骨架序列最大帧数
    'skeleton_num_joints': 25,     # 骨架关键点数量
    'skeleton_coords': 3,          # 坐标维度
}

def get_label_config(label_name):
    """获取标签配置"""
    return LABEL_CONFIG.get(label_name, {})

def get_all_labels():
    """获取所有启用的标签"""
    return [name for name, config in LABEL_CONFIG.items() if config.get('enabled', True)]

def get_label_color(label_name):
    """获取标签颜色"""
    config = get_label_config(label_name)
    return config.get('color', (128, 128, 128))

def get_label_min_confidence(label_name):
    """获取标签最小置信度"""
    config = get_label_config(label_name)
    return config.get('min_confidence', 0.5)

def is_label_enabled(label_name):
    """检查标签是否启用"""
    config = get_label_config(label_name)
    return config.get('enabled', True)

def print_config():
    """打印配置信息"""
    print("=== 标注配置 ===")
    for label_name, config in LABEL_CONFIG.items():
        if config.get('enabled', True):
            print(f"{label_name}:")
            print(f"  颜色: {config['color']}")
            print(f"  最小置信度: {config['min_confidence']}")
            print(f"  描述: {config['description']}")
            if 'max_height_ratio' in config:
                print(f"  最大高度比例: {config['max_height_ratio']}")
            print()
    
    print("=== 检测参数 ===")
    for param, value in DETECTION_PARAMS.items():
        print(f"{param}: {value}")
    
    print("\n=== 手机检测配置 ===")
    for param, value in PHONE_DETECTION.items():
        print(f"{param}: {value}")

if __name__ == "__main__":
    print_config()
