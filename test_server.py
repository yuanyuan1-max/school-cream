#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
School Cream 模拟API服务器
用于测试主程序的API调用功能
"""

import sys
import codecs
from flask import Flask, jsonify, request

# 设置编码以支持中文输出
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 模拟课程数据
COURSES = [
    {
        "id": 1,
        "name": "高等数学",
        "teacher": "张教授",
        "start_time": "14:30:00",
        "end_time": "16:00:00",
        "devices": [
            {
                "id": 1,
                "name": "前摄像头",
                "type": "camera",
                "stream_url": "test (1).mp4",
                "position": "front"
            },
            {
                "id": 2,
                "name": "后摄像头", 
                "type": "camera",
                "stream_url": "test (2).mp4",
                "position": "back"
            }
        ]
    },
    {
        "id": 2,
        "name": "线性代数",
        "teacher": "李教授",
        "start_time": "16:00:00",
        "end_time": "17:30:00",
        "devices": [
            {
                "id": 3,
                "name": "主摄像头",
                "type": "camera",
                "stream_url": "test (1).mp4",
                "position": "main"
            }
        ]
    }
]

@app.route('/')
def index():
    """API根路径"""
    return jsonify({
        "service": "School Cream Mock API Server",
        "version": "1.0.0",
        "endpoints": [
            "GET /api/courses - 获取课程列表",
            "GET /api/courses/<id> - 获取单个课程",
            "GET /api/devices - 获取所有设备",
            "GET /api/health - 健康检查",
            "POST /api/test - 测试端点"
        ]
    })

@app.route('/api/courses')
def get_courses():
    """获取课程列表"""
    return jsonify(COURSES)

@app.route('/api/courses/<int:course_id>')
def get_course(course_id):
    """获取单个课程"""
    course = next((c for c in COURSES if c['id'] == course_id), None)
    if course:
        return jsonify(course)
    return jsonify({"error": "课程未找到"}), 404

@app.route('/api/devices')
def get_devices():
    """获取所有设备"""
    devices = []
    for course in COURSES:
        devices.extend(course['devices'])
    return jsonify(devices)

@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "School Cream Mock API Server",
        "timestamp": "2025-01-01T00:00:00Z"
    })

@app.route('/api/test', methods=['POST'])
def test_endpoint():
    """测试端点"""
    data = request.get_json()
    return jsonify({
        "message": "测试成功",
        "received_data": data
    })

if __name__ == '__main__':
    print("启动School Cream模拟API服务器...")
    print("API文档: http://127.0.0.1:5000/")
    print("课程列表: http://127.0.0.1:5000/api/courses")
    app.run(host='127.0.0.1', port=5000, debug=True)
