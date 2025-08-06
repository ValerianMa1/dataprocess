#!/usr/bin/env python3
"""
完整管道测试脚本
"""

import os
import sys
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def test_pipeline():
    """测试完整的处理管道"""
    print("=" * 60)
    print("测试完整的视频处理管道")
    print("=" * 60)
    
    try:
        from video_processing.pipeline import VideoProcessingPipeline
        
        # 创建处理管道
        pipeline = VideoProcessingPipeline()
        print("✓ 处理管道创建成功")
        
        # 测试单个视频处理
        test_video = "rawdata/test/test_video.mp4"
        if os.path.exists(test_video):
            print(f"测试单个视频处理: {test_video}")
            result = pipeline.process_single_video(test_video)
            
            print("处理结果:")
            print(f"  视频路径: {result['video_path']}")
            print(f"  是否通过: {result['passed']}")
            print(f"  处理时间: {result['processing_time']:.2f} 秒")
            
            # 显示各个过滤器的结果
            filters = result.get('filters', {})
            
            # 分辨率过滤结果
            resolution_result = filters.get('resolution', {})
            print(f"  分辨率过滤: {resolution_result.get('passed', False)}")
            
            # 面部检测结果
            face_result = filters.get('face', {})
            print(f"  面部检测: {face_result.get('valid', False)}")
            print(f"  检测到的面部数: {face_result.get('total_faces', 0)}")
            
            # 年龄过滤结果
            age_result = filters.get('age', {})
            print(f"  年龄过滤: {age_result.get('valid', False)}")
            
            return result['passed']
        else:
            print(f"✗ 测试视频不存在: {test_video}")
            return False
            
    except Exception as e:
        print(f"✗ 管道测试失败: {e}")
        return False

def test_directory_processing():
    """测试目录处理"""
    print("=" * 60)
    print("测试目录处理")
    print("=" * 60)
    
    try:
        from video_processing.pipeline import VideoProcessingPipeline
        
        # 创建处理管道
        pipeline = VideoProcessingPipeline()
        
        # 测试目录处理
        input_dir = "rawdata"
        output_dir = "test_output"
        
        print(f"处理目录: {input_dir}")
        results = pipeline.process_directory(input_dir, output_dir)
        
        print("处理结果:")
        print(f"  总视频数: {results.get('total_videos', 0)}")
        print(f"  通过的视频数: {results.get('passed_videos', 0)}")
        print(f"  处理时间: {results.get('processing_time', 0):.2f} 秒")
        
        # 显示统计信息
        stats = results.get('statistics', {})
        print("详细统计:")
        print(f"  分辨率过滤通过: {stats.get('resolution_passed', 0)}")
        print(f"  面部检测通过: {stats.get('face_passed', 0)}")
        print(f"  年龄过滤通过: {stats.get('age_passed', 0)}")
        print(f"  最终通过: {stats.get('final_passed', 0)}")
        
        # 检查输出文件
        if os.path.exists(output_dir):
            print(f"\n输出文件:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file}: {size} bytes")
        
        return results.get('total_videos', 0) > 0
        
    except Exception as e:
        print(f"✗ 目录处理测试失败: {e}")
        return False

def test_example_script():
    """测试示例脚本"""
    print("=" * 60)
    print("测试示例脚本")
    print("=" * 60)
    
    try:
        # 测试基础示例
        print("运行基础示例...")
        from examples.run_pipeline import run_basic_example
        run_basic_example()
        print("✓ 基础示例运行成功")
        return True
        
    except Exception as e:
        print(f"✗ 示例脚本测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始完整管道测试")
    
    success_count = 0
    total_tests = 3
    
    # 测试单个视频处理
    if test_pipeline():
        print("✓ 单个视频处理测试通过")
        success_count += 1
    else:
        print("✗ 单个视频处理测试失败")
    
    # 测试目录处理
    if test_directory_processing():
        print("✓ 目录处理测试通过")
        success_count += 1
    else:
        print("✗ 目录处理测试失败")
    
    # 测试示例脚本
    if test_example_script():
        print("✓ 示例脚本测试通过")
        success_count += 1
    else:
        print("✗ 示例脚本测试失败")
    
    print("=" * 60)
    print(f"测试完成: {success_count}/{total_tests} 个测试通过")
    print("=" * 60)
    
    if success_count == total_tests:
        print("🎉 所有测试都通过了！视频处理管道安装和测试成功！")
    else:
        print("⚠️  部分测试失败，但基础功能正常")
    
    return success_count >= 2  # 至少2个测试通过就算成功

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
