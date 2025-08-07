#!/usr/bin/env python3
"""
单个视频文件可视化测试脚本
用于详细查看视频过滤的每个步骤
"""

import cv2
import numpy as np
import os
import sys
from simple_video_filter import SimpleVideoFilter

def print_separator(title="", char="=", length=60):
    """打印分隔线"""
    if title:
        print(f"\n{char * length}")
        print(f"{title:^{length}}")
        print(f"{char * length}")
    else:
        print(f"{char * length}")

def get_video_info(video_path):
    """获取视频基本信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    cap.release()
    return info

def extract_sample_frames(video_path, num_frames=5):
    """提取采样帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    
    frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append((frame_idx, frame))
    
    cap.release()
    return frames

def detect_faces_in_frame(frame, face_cascade):
    """在单帧中检测人脸"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    valid_faces = 0
    face_details = []
    
    for (x, y, w, h) in faces:
        is_valid = w >= 256 and h >= 256
        if is_valid:
            valid_faces += 1
        
        face_details.append({
            'bbox': (x, y, w, h),
            'size': f"{w}x{h}",
            'valid': is_valid
        })
    
    return len(faces), valid_faces, face_details

def test_single_video(video_path):
    """测试单个视频文件"""
    
    print_separator("视频过滤器可视化测试")
    print(f"测试视频: {video_path}")
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 错误: 视频文件不存在: {video_path}")
        return
    
    # 1. 获取视频基本信息
    print_separator("1. 视频基本信息", "-")
    video_info = get_video_info(video_path)
    
    if not video_info:
        print("❌ 无法读取视频文件")
        return
    
    print(f"📹 视频信息:")
    print(f"  分辨率: {video_info['width']}x{video_info['height']}")
    print(f"  帧率: {video_info['fps']:.2f} FPS")
    print(f"  总帧数: {video_info['frame_count']}")
    print(f"  时长: {video_info['duration']:.2f} 秒")
    
    # 分辨率检测
    resolution_passed = video_info['width'] >= 720 and video_info['height'] >= 1080
    print(f"\n🔍 分辨率检测: {'✓ 通过' if resolution_passed else '✗ 不通过'} (要求: ≥720x1080)")
    
    # 2. 初始化过滤器
    print_separator("2. 初始化FaceXFormer过滤器", "-")
    try:
        filter_tool = SimpleVideoFilter(use_facexformer=True)
        print("✓ FaceXFormer过滤器初始化成功")
    except Exception as e:
        print(f"❌ 过滤器初始化失败: {e}")
        return
    
    # 3. 人脸检测详细分析
    print_separator("3. 人脸检测详细分析", "-")
    sample_frames = extract_sample_frames(video_path, 5)
    print(f"📸 提取了 {len(sample_frames)} 个采样帧")
    
    if sample_frames and filter_tool.face_cascade:
        total_faces = 0
        total_valid_faces = 0
        
        for i, (frame_idx, frame) in enumerate(sample_frames):
            face_count, valid_face_count, face_details = detect_faces_in_frame(frame, filter_tool.face_cascade)
            total_faces += face_count
            total_valid_faces += valid_face_count
            
            print(f"\n  Frame {frame_idx}:")
            print(f"    检测到人脸: {face_count}")
            print(f"    有效人脸: {valid_face_count} (≥256x256)")
            
            for j, face in enumerate(face_details):
                status = "✓ 有效" if face['valid'] else "✗ 太小"
                print(f"      人脸{j+1}: {face['size']} {status}")
        
        face_passed = total_valid_faces > 0
        print(f"\n👥 人脸检测总结:")
        print(f"  总检测人脸数: {total_faces}")
        print(f"  有效人脸数: {total_valid_faces}")
        print(f"  人脸检测结果: {'✓ 通过' if face_passed else '✗ 不通过'}")
    else:
        print("❌ 无法进行人脸检测")
        face_passed = False
    
    # 4. FaceXFormer年龄估计
    print_separator("4. FaceXFormer年龄估计", "-")
    try:
        age_passed, age_info = filter_tool.check_age(video_path, sample_frames=5)
        
        print(f"🧠 年龄估计结果:")
        print(f"  估计年龄范围: {age_info.get('estimated_age', 'N/A')}")
        print(f"  年龄类别: {age_info.get('age_class', 'N/A')}")
        print(f"  检测方法: {age_info.get('method', 'N/A')}")
        print(f"  置信度: {age_info.get('confidence', 'N/A')}")
        print(f"  处理人脸数: {age_info.get('face_count', 'N/A')}")
        
        if 'age_distribution' in age_info:
            print(f"  年龄分布: {age_info['age_distribution']}")
        
        print(f"  年龄检测结果: {'✓ 通过' if age_passed else '✗ 不通过'} (要求: 0-30岁)")
        
    except Exception as e:
        print(f"❌ 年龄估计失败: {e}")
        age_passed = False
        age_info = {}
    
    # 5. 年龄分类系统说明
    print_separator("5. 年龄分类系统", "-")
    age_classes = [
        ("0", "0-10岁", "✓ 通过"),
        ("1", "11-20岁", "✓ 通过"),
        ("2", "21-30岁", "✓ 通过"),
        ("3", "31-40岁", "✗ 不通过"),
        ("4", "41-50岁", "✗ 不通过"),
        ("5", "51-60岁", "✗ 不通过"),
        ("6", "61-70岁", "✗ 不通过"),
        ("7", "70+岁", "✗ 不通过")
    ]
    
    print("📊 FaceXFormer年龄分类系统:")
    current_class = age_info.get('age_class', -1)
    
    for class_id, age_range, status in age_classes:
        marker = " ← 当前检测" if int(class_id) == current_class else ""
        print(f"  类别{class_id}: {age_range:8} {status}{marker}")
    
    # 6. 最终结果
    print_separator("6. 最终过滤结果")
    
    final_passed = resolution_passed and face_passed and age_passed
    
    print(f"📋 过滤结果总览:")
    print(f"  ✓ 分辨率检测: {'通过' if resolution_passed else '不通过'} ({video_info['width']}x{video_info['height']})")
    print(f"  ✓ 人脸检测: {'通过' if face_passed else '不通过'}")
    print(f"  ✓ 年龄检测: {'通过' if age_passed else '不通过'}")
    print(f"  🎯 最终结果: {'🎉 通过所有过滤条件' if final_passed else '❌ 未通过过滤'}")
    
    # 7. 完整测试验证
    print_separator("7. 完整测试验证", "-")
    print("🔄 执行完整的过滤器测试...")
    
    try:
        result = filter_tool.process_video(video_path)
        print(f"✓ 完整测试完成")
        print(f"  最终通过状态: {result['final_passed']}")
        print(f"  详细信息: {result['age_info']}")
    except Exception as e:
        print(f"❌ 完整测试失败: {e}")
    
    print_separator()
    print("测试完成！")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # 默认测试视频
        video_path = "rawdata/test/111.mp4"
        print(f"使用默认测试视频: {video_path}")
        print("提示: 可以通过命令行参数指定其他视频文件")
        print("用法: python test_single_video.py <video_path>")
    
    test_single_video(video_path)

if __name__ == "__main__":
    main()
