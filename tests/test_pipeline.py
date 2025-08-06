"""
视频处理管道测试
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_processing.pipeline import VideoProcessingPipeline
from video_processing.filters.resolution_filter import ResolutionFilter
from video_processing.filters.face_filter import FaceFilter


class TestVideoProcessingPipeline:
    """视频处理管道测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_video(self, temp_dir):
        """创建测试视频"""
        video_path = os.path.join(temp_dir, "test_video.mp4")
        
        # 创建一个简单的测试视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (1920, 1080))
        
        # 生成几帧测试图像
        for i in range(30):
            # 创建一个简单的彩色帧
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:, :] = [i * 8 % 255, (i * 16) % 255, (i * 24) % 255]
            
            # 添加一个简单的矩形作为"面部"
            cv2.rectangle(frame, (800, 400), (1120, 720), (255, 255, 255), -1)
            
            out.write(frame)
        
        out.release()
        return video_path
    
    def test_resolution_filter(self, sample_video):
        """测试分辨率过滤器"""
        filter_obj = ResolutionFilter(min_width=1920, min_height=1080)
        
        # 测试分辨率检查
        resolution = filter_obj.get_video_resolution(sample_video)
        assert resolution == (1920, 1080)
        
        # 测试过滤结果
        result = filter_obj.check_resolution(sample_video)
        assert result is True
    
    def test_face_filter(self, sample_video):
        """测试面部过滤器"""
        try:
            face_filter = FaceFilter(min_face_width=256, min_face_height=256)
            
            # 测试视频面部分析
            result = face_filter.analyze_video_faces(sample_video, sample_frames=5)
            
            # 检查结果结构
            assert "valid" in result
            assert "total_faces" in result
            assert "sample_frames" in result
            
        except FileNotFoundError as e:
            # 如果OpenCV级联分类器不可用，跳过测试
            pytest.skip(f"OpenCV级联分类器不可用: {str(e)}")
    
    def test_pipeline_initialization(self):
        """测试管道初始化"""
        pipeline = VideoProcessingPipeline()
        
        # 检查过滤器是否正确初始化
        assert pipeline.resolution_filter is not None
        assert pipeline.face_filter is not None
        assert pipeline.age_filter is not None
        
        # 检查统计信息初始化
        assert "total_videos" in pipeline.stats
        assert pipeline.stats["total_videos"] == 0
    
    def test_single_video_processing(self, sample_video):
        """测试单个视频处理"""
        pipeline = VideoProcessingPipeline()
        
        result = pipeline.process_single_video(sample_video)
        
        # 检查结果结构
        assert "video_path" in result
        assert "passed" in result
        assert "filters" in result
        assert "processing_time" in result
        
        # 检查视频路径
        assert result["video_path"] == sample_video
        
        # 检查过滤器结果
        assert "resolution" in result["filters"]
        assert "face" in result["filters"]
        assert "age" in result["filters"]
    
    def test_batch_processing(self, temp_dir):
        """测试批量处理"""
        # 创建多个测试视频
        video_paths = []
        for i in range(3):
            video_path = os.path.join(temp_dir, f"test_video_{i}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (1920, 1080))
            
            for j in range(10):
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                frame[:, :] = [j * 25 % 255, (j * 50) % 255, (j * 75) % 255]
                out.write(frame)
            
            out.release()
            video_paths.append(video_path)
        
        # 测试批量处理
        pipeline = VideoProcessingPipeline()
        results = pipeline.process_videos(video_paths, max_workers=2)
        
        # 检查结果
        assert "total_videos" in results
        assert "passed_videos" in results
        assert "detailed_results" in results
        assert results["total_videos"] == 3
        assert len(results["detailed_results"]) == 3
    
    def test_directory_processing(self, temp_dir):
        """测试目录处理"""
        # 创建测试视频
        video_path = os.path.join(temp_dir, "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (1920, 1080))
        
        for i in range(10):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:, :] = [i * 25 % 255, 0, 0]
            out.write(frame)
        
        out.release()
        
        # 创建输出目录
        output_dir = os.path.join(temp_dir, "output")
        
        # 测试目录处理
        pipeline = VideoProcessingPipeline()
        results = pipeline.process_directory(temp_dir, output_dir)
        
        # 检查结果
        assert "total_videos" in results
        assert results["total_videos"] >= 1
        
        # 检查输出文件是否创建
        assert os.path.exists(os.path.join(output_dir, "processing_results.json"))
        assert os.path.exists(os.path.join(output_dir, "passed_videos.txt"))
        assert os.path.exists(os.path.join(output_dir, "statistics.txt"))


def test_config_loading():
    """测试配置加载"""
    from video_processing.utils.config import Config
    
    # 测试默认配置
    config = Config("nonexistent_config.yaml")
    
    # 检查默认配置是否正确加载
    assert config.get("paths.input_dir") == "rawdata"
    assert config.get("video_filters.min_age") == 22
    assert config.get("processing.max_workers") == 4


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
