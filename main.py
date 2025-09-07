#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import threading
from ultralytics import YOLO
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import mediapipe as mp
import requests
import json
import argparse
from config import get_model_path, get_data_path, check_model_exists, DETECTION_CONFIG

# ---------------- 手机检测相关类 ----------------
class Autoencoder(nn.Module):
    """自编码器模型用于手机使用检测"""
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        seq_len = x.size(1)
        batch_size = x.size(0)
        decoder_input = torch.zeros(batch_size, seq_len, self.decoder.input_size, device=x.device)
        out, _ = self.decoder(decoder_input, (h_n, c_n))
        out = self.linear(out)
        return out

class SkeletonExtractor:
    """骨架提取器用于姿态分析"""
    def __init__(self, max_frames=300, num_joints=25, coords=3):
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.coords = coords
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

    def extract(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_roi)
        data = np.zeros((self.max_frames, self.num_joints, self.coords))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            selected_indices = list(range(self.num_joints))
            for joint in range(min(self.num_joints, len(selected_indices))):
                lm = landmarks[selected_indices[joint]]
                data[0, joint, 0] = lm.x
                data[0, joint, 1] = lm.y
                data[0, joint, 2] = lm.z
            mean = np.mean(data[0], axis=0)
            std = np.std(data[0], axis=0) + 1e-8
            data[0] = (data[0] - mean) / std
            for frame_idx in range(1, self.max_frames):
                data[frame_idx] = data[0]
        data = data.reshape(self.max_frames, -1)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    def close(self):
        self.pose.close()

def detect_phone_usage(model, data, threshold, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """检测是否在玩手机"""
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        output = model(data)
        error = torch.mean((data - output) ** 2).item()
    return error > threshold

def fetch_json_from_registry(registry_url):
    """从注册中心API获取JSON数据"""
    try:
        response = requests.get(registry_url)
        response.raise_for_status()
        json_data = response.json()
        print(f"Successfully fetched JSON from {registry_url}")
        return json_data
    except Exception as e:
        print(f"Error fetching JSON from registry: {e}")
        return []

def parse_json_data(json_data):
    """解析 JSON 数据并返回结构化信息"""
    courses = []
    try:
        for course in json_data:
            course_info = {
                "timeAdd": course.get("timeAdd"),
                "courseName": course.get("courseName"),
                "room_id": course.get("room_id", "unknown_room"),
                "devices": []
            }
            for device in course.get("device", []):
                device_info = {
                    "name": device.get("name"),
                    "type": device.get("type"),
                    "liveUrl": device.get("liveUrl"),
                    "model_number": device.get("model_number", ""),
                    "host": device.get("host", ""),
                    "serial_number": device.get("serial_number", ""),
                    "token": device.get("token", ""),
                    "admin": device.get("admin", ""),
                    "password": device.get("password", ""),
                    "create_by": device.get("create_by", ""),
                    "create_date": device.get("create_date", ""),
                    "update_by": device.get("update_by", ""),
                    "update_date": device.get("update_date", ""),
                    "status": device.get("status", ""),
                    "remarks": device.get("remarks", ""),
                    "del_flag": device.get("del_flag", "0")
                }
                course_info["devices"].append(device_info)
            courses.append(course_info)
        return courses
    except Exception as e:
        print(f"Error parsing JSON data: {e}")
        return []

def generate_output_filename(course_info, device_info):
    """生成输出文件名，格式：时间_课程_摄像头位置.mp4"""
    time_str = course_info["timeAdd"].replace(":", "-")  # 替换非法字符
    course_name = course_info["courseName"]
    camera_name = device_info["name"].replace("/", "_")  # 替换非法字符
    return f"{time_str}_{course_name}_{camera_name}.mp4"

class VideoProcessor:
    def __init__(self):
        # 检测控制
        self.detection_enabled = True  # 默认启用检测
        
        # 检测结果（仅用于日志）
        self.detection_counts = {
            'Person': 0,
            'Raise Hand': 0,
            'Lie Down': 0,
            'Phone Usage': 0
        }
        
        # 时间相关
        self.last_update = time.time()
        self.update_interval = DETECTION_CONFIG['update_interval']  # 检测间隔
        self.last_detection_time = 0
        
        # 模型相关
        self.model = None
        self.is_processing = False
        
        # 手机检测相关
        self.phone_autoencoder = None
        self.skeleton_extractor = None
        self.phone_threshold = None
        self.phone_detection_enabled = True
        
        # 线程控制
        self.detection_thread = None
        self.should_stop_detection = False
        
        # 录制相关
        self.output_video = None
        self.output_filename = None
        self.is_recording = False
        
        # 初始化模型
        self.init_model()
        
        # 初始化手机检测模型
        self.init_phone_detection()
    
    def init_model(self):
        """初始化YOLO模型"""
        try:
            torch.set_float32_matmul_precision('medium')
            if check_model_exists('yolo_best'):
                self.model = YOLO(get_model_path('yolo_best'))
            elif check_model_exists('yolov12n'):
                self.model = YOLO(get_model_path('yolov12n'))
            elif check_model_exists('yolo11n'):
                self.model = YOLO(get_model_path('yolo11n'))
            else:
                self.model = YOLO('yolo11n.pt')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model.to(device=device, dtype=dtype)
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"Model loading failed: {e}")
    
    def init_phone_detection(self):
        """初始化手机检测模型"""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if check_model_exists('phone_autoencoder'):
                self.phone_autoencoder = Autoencoder(25*3).to(device)
                self.phone_autoencoder.load_state_dict(torch.load(get_model_path('phone_autoencoder'), map_location=device))
            else:
                print("Phone detection model not found, disabling")
                return
            if check_model_exists('threshold'):
                self.phone_threshold = np.load(get_model_path('threshold'))
            else:
                self.phone_threshold = 0.1
            self.skeleton_extractor = SkeletonExtractor()
            print("Phone detection initialized")
        except Exception as e:
            print(f"Phone detection init failed: {e}")
            self.phone_detection_enabled = False
    
    def load_video(self, video_source):
        """加载RTSP流或视频"""
        self.cap = None
        if video_source.startswith("rtsp://"):
            print(f"Loading RTSP: {video_source}")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            self.cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
        else:
            if not os.path.exists(video_source):
                print(f"File not found: {video_source}")
                return None
            print(f"Loading file: {video_source}")
            self.cap = cv2.VideoCapture(video_source)
        if self.cap.isOpened():
            self.fps = max(1, int(self.cap.get(cv2.CAP_PROP_FPS)))
            return self.cap
        else:
            print(f"Failed to load: {video_source}")
            return None
    
    def start_detection_thread(self):
        """启动检测线程"""
        def detection_worker():
            while not self.should_stop_detection:
                current_time = time.time()
                if self.detection_enabled and self.model and not self.is_processing and self.current_frame is not None and current_time - self.last_detection_time >= self.update_interval:
                    self.is_processing = True
                    try:
                        self.process_current_frame()
                        self.last_detection_time = current_time
                    except Exception as e:
                        print(f"Detection error: {e}")
                    finally:
                        self.is_processing = False
                time.sleep(0.05)
        self.detection_thread = threading.Thread(target=detection_worker, daemon=True)
        self.detection_thread.start()
    
    def stop_detection_thread(self):
        self.should_stop_detection = True
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
    
    def process_current_frame(self):
        """处理帧并标注四种检测"""
        if not self.model or self.current_frame is None:
            return
        height, width = self.current_frame.shape[:2]
        results = self.model(self.current_frame, conf=DETECTION_CONFIG['conf_threshold'], verbose=False, max_det=DETECTION_CONFIG['max_det'])
        counts = {'Person': 0, 'Raise Hand': 0, 'Lie Down': 0, 'Phone Usage': 0}
        annotated_frame = self.current_frame.copy()
        if results and results[0].boxes:
            for box in results[0].boxes:
                cls = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_height = y2 - y1
                confidence = float(box.conf)
                class_names = ['Raise Hand', 'Person', 'Lie Down']
                class_name = class_names[cls] if cls < len(class_names) else f'Class_{cls}'
                color = {
                    'Raise Hand': (0, 255, 0),
                    'Person': (0, 100, 255),
                    'Lie Down': (255, 0, 0),
                    'Phone Usage': (255, 165, 0)
                }.get(class_name, (128, 128, 128))
                if cls == 0 and confidence >= 0.1:
                    counts['Raise Hand'] += 1
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif cls == 1 and box_height < height * 0.6:
                    counts['Person'] += 1
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif cls == 2 and confidence >= 0.1:
                    counts['Lie Down'] += 1
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if cls == 1 and self.phone_detection_enabled and self.phone_autoencoder and self.skeleton_extractor:
                    skeleton_data = self.skeleton_extractor.extract(self.current_frame, (x1, y1, x2, y2))
                    if skeleton_data is not None:
                        is_using_phone = detect_phone_usage(self.phone_autoencoder, skeleton_data, self.phone_threshold)
                        if is_using_phone:
                            counts['Phone Usage'] += 1
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                            cv2.putText(annotated_frame, f"Phone Usage: {confidence:.2f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        self.detection_counts = counts
        self.last_update = time.time()
        if self.is_recording:
            self.save_frame(annotated_frame)
        print(f"Frame processed: {counts}")
    
    def start_recording(self, output_filename, width, height, fps):
        os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        self.is_recording = True
        print(f"Recording to: {output_filename}")
    
    def stop_recording(self):
        if self.output_video:
            self.output_video.release()
            self.is_recording = False
            print("Recording stopped")
    
    def save_frame(self, frame):
        if self.is_recording and self.output_video:
            self.output_video.write(frame)
    
    def process_video(self, video_source, output_filename, max_frames=None):
        cap = self.load_video(video_source)
        if not cap:
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.fps
        self.start_recording(output_filename, width, height, fps)
        self.start_detection_thread()
        frame_idx = 0
        while True:
            ret, self.current_frame = cap.read()
            if not ret:
                print("End of stream")
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                print(f"Reached max frames: {max_frames}")
                break
            time.sleep(1.0 / fps)  # 模拟实时处理
        self.stop_recording()
        self.stop_detection_thread()
        cap.release()
        if self.skeleton_extractor:
            self.skeleton_extractor.close()

def main():
    parser = argparse.ArgumentParser(description="视频处理脚本")
    parser.add_argument("--registry_url", required=True, help="注册中心URL，用于获取JSON")
    parser.add_argument("--max_frames", type=int, default=None, help="最大处理帧数（默认无限）")
    args = parser.parse_args()
    
    json_data = fetch_json_from_registry(args.registry_url)
    courses = parse_json_data(json_data)
    processor = VideoProcessor()
    
    for course in courses:
        for device in course["devices"]:
            output_filename = generate_output_filename(course, device)
            processor.process_video(device["liveUrl"], output_filename, args.max_frames)

if __name__ == "__main__":
    main()