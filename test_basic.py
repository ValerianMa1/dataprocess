#!/usr/bin/env python3
"""
基础测试脚本 - 测试视频处理管道的基本功能
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """测试基础导入"""
    print("=" * 50)
    print("测试基础导入...")
    
    try:
        import cv2
        print(f"✓ OpenCV 版本: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy 版本: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy 导入失败: {e}")
        return False
    
    try:
        import yaml
        print(f"✓ PyYAML 导入成功")
    except ImportError as e:
        print(f"✗ PyYAML 导入失败: {e}")
        return False
    
    try:
        from loguru import logger
        print(f"✓ Loguru 导入成功")
    except ImportError as e:
        print(f"✗ Loguru 导入失败: {e}")
        return False
    
    return True

def test_opencv_cascade():
    """测试OpenCV级联分类器"""
    print("=" * 50)
    print("测试OpenCV级联分类器...")
    
    try:
        # 尝试加载面部检测器
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if not face_cascade.empty():
                print(f"✓ 面部级联分类器加载成功: {cascade_path}")
                return True
            else:
                print(f"✗ 面部级联分类器为空")
                return False
        else:
            print(f"✗ 级联分类器文件不存在: {cascade_path}")
            return False
    except Exception as e:
        print(f"✗ 级联分类器测试失败: {e}")
        return False

def create_test_video():
    """创建一个简单的测试视频"""
    print("=" * 50)
    print("创建测试视频...")
    
    try:
        # 确保rawdata目录存在
        os.makedirs("rawdata/test", exist_ok=True)
        
        video_path = "rawdata/test/test_video.mp4"
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (1920, 1080))
        
        if not out.isOpened():
            print("✗ 无法创建视频文件")
            return None
        
        # 生成30帧测试视频
        for i in range(30):
            # 创建彩色帧
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            # 渐变背景
            frame[:, :, 0] = (i * 8) % 256  # 蓝色通道
            frame[:, :, 1] = (i * 4) % 256  # 绿色通道
            frame[:, :, 2] = (i * 2) % 256  # 红色通道
            
            # 添加一个移动的白色矩形（模拟面部）
            x = 400 + i * 20
            y = 300 + int(50 * np.sin(i * 0.2))
            cv2.rectangle(frame, (x, y), (x + 320, y + 320), (255, 255, 255), -1)
            
            # 添加文本
            cv2.putText(frame, f"Frame {i+1}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            out.write(frame)
        
        out.release()
        
        # 验证视频文件
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            print(f"✓ 测试视频创建成功: {video_path} ({file_size} bytes)")
            return video_path
        else:
            print("✗ 测试视频文件未创建")
            return None
            
    except Exception as e:
        print(f"✗ 创建测试视频失败: {e}")
        return None

def test_video_processing_modules():
    """测试视频处理模块导入"""
    print("=" * 50)
    print("测试视频处理模块导入...")
    
    try:
        from video_processing.filters.resolution_filter import ResolutionFilter
        print("✓ 分辨率过滤器导入成功")
    except ImportError as e:
        print(f"✗ 分辨率过滤器导入失败: {e}")
        return False
    
    try:
        from video_processing.filters.face_filter import FaceFilter
        print("✓ 面部过滤器导入成功")
    except ImportError as e:
        print(f"✗ 面部过滤器导入失败: {e}")
        return False
    
    try:
        from video_processing.utils.config import config
        print("✓ 配置模块导入成功")
    except ImportError as e:
        print(f"✗ 配置模块导入失败: {e}")
        return False
    
    return True

def test_resolution_filter(video_path):
    """测试分辨率过滤器"""
    print("=" * 50)
    print("测试分辨率过滤器...")
    
    try:
        from video_processing.filters.resolution_filter import ResolutionFilter
        
        # 创建过滤器
        filter_obj = ResolutionFilter(min_width=1920, min_height=1080)
        
        # 测试分辨率检测
        resolution = filter_obj.get_video_resolution(video_path)
        print(f"检测到的分辨率: {resolution}")
        
        # 测试过滤
        result = filter_obj.check_resolution(video_path)
        print(f"分辨率过滤结果: {result}")
        
        return result
        
    except Exception as e:
        print(f"✗ 分辨率过滤器测试失败: {e}")
        return False

def test_face_filter(video_path):
    """测试面部过滤器"""
    print("=" * 50)
    print("测试面部过滤器...")
    
    try:
        from video_processing.filters.face_filter import FaceFilter
        
        # 创建过滤器
        face_filter = FaceFilter(min_face_width=256, min_face_height=256)
        
        # 测试面部分析
        result = face_filter.analyze_video_faces(video_path, sample_frames=5)
        print(f"面部分析结果: {result}")
        
        return result.get('valid', False)
        
    except Exception as e:
        print(f"✗ 面部过滤器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始视频处理管道基础测试")
    print("=" * 50)
    
    # 测试基础导入
    if not test_imports():
        print("基础导入测试失败，退出")
        return False
    
    # 测试OpenCV级联分类器
    if not test_opencv_cascade():
        print("OpenCV级联分类器测试失败，但继续测试其他功能")
    
    # 创建测试视频
    video_path = create_test_video()
    if not video_path:
        print("测试视频创建失败，退出")
        return False
    
    # 测试模块导入
    if not test_video_processing_modules():
        print("视频处理模块导入失败，退出")
        return False
    
    # 测试分辨率过滤器
    if not test_resolution_filter(video_path):
        print("分辨率过滤器测试失败")
    else:
        print("✓ 分辨率过滤器测试通过")
    
    # 测试面部过滤器
    if not test_face_filter(video_path):
        print("面部过滤器测试失败")
    else:
        print("✓ 面部过滤器测试通过")
    
    print("=" * 50)
    print("基础测试完成！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
