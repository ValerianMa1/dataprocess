"""
分辨率过滤模块
使用OpenCV检测视频分辨率，过滤掉低于1080p的视频
"""

import cv2
import os
from typing import Tuple, Optional, List
from pathlib import Path
from loguru import logger


class ResolutionFilter:
    """视频分辨率过滤器"""
    
    def __init__(self, min_width: int = 1080, min_height: int = 1080):
        """
        初始化分辨率过滤器
        
        Args:
            min_width: 最小宽度（默认1080，对应1080p）
            min_height: 最小高度（默认1080，对应1080p）
        """
        self.min_width = min_width
        self.min_height = min_height
        logger.info(f"分辨率过滤器初始化: 最小分辨率 {min_width}x{min_height}")
    
    def get_video_resolution(self, video_path: str) -> Optional[Tuple[int, int]]:
        """
        获取视频分辨率
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            (width, height) 或 None（如果无法读取）
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return None
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            logger.debug(f"视频 {video_path} 分辨率: {width}x{height}")
            return (width, height)
            
        except Exception as e:
            logger.error(f"获取视频分辨率时出错 {video_path}: {str(e)}")
            return None
    
    def check_resolution(self, video_path: str) -> bool:
        """
        检查视频分辨率是否满足要求
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            True如果分辨率满足要求，False否则
        """
        resolution = self.get_video_resolution(video_path)
        if resolution is None:
            return False
        
        width, height = resolution
        meets_requirement = width >= self.min_width and height >= self.min_height
        
        if meets_requirement:
            logger.info(f"✓ 视频 {video_path} 分辨率合格: {width}x{height}")
        else:
            logger.warning(f"✗ 视频 {video_path} 分辨率不合格: {width}x{height} < {self.min_width}x{self.min_height}")
        
        return meets_requirement
    
    def filter_videos_by_resolution(self, video_paths: List[str]) -> List[str]:
        """
        批量过滤视频，返回分辨率合格的视频列表
        
        Args:
            video_paths: 视频文件路径列表
            
        Returns:
            分辨率合格的视频路径列表
        """
        valid_videos = []
        total_videos = len(video_paths)
        
        logger.info(f"开始分辨率过滤，共 {total_videos} 个视频")
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"处理进度: {i}/{total_videos} - {video_path}")
            
            if self.check_resolution(video_path):
                valid_videos.append(video_path)
        
        logger.info(f"分辨率过滤完成: {len(valid_videos)}/{total_videos} 个视频通过")
        return valid_videos
    
    def scan_directory(self, directory: str, extensions: List[str] = None) -> List[str]:
        """
        扫描目录中的视频文件并进行分辨率过滤
        
        Args:
            directory: 目录路径
            extensions: 支持的视频文件扩展名列表
            
        Returns:
            分辨率合格的视频路径列表
        """
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
        video_files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"目录不存在: {directory}")
            return []
        
        # 递归扫描视频文件
        for ext in extensions:
            video_files.extend(directory_path.rglob(f"*{ext}"))
            video_files.extend(directory_path.rglob(f"*{ext.upper()}"))
        
        video_paths = [str(path) for path in video_files]
        logger.info(f"在目录 {directory} 中找到 {len(video_paths)} 个视频文件")
        
        return self.filter_videos_by_resolution(video_paths)


def main():
    """测试函数"""
    # 创建分辨率过滤器
    filter_obj = ResolutionFilter()
    
    # 测试单个视频
    test_video = "rawdata/test/sample.mp4"
    if os.path.exists(test_video):
        result = filter_obj.check_resolution(test_video)
        print(f"测试视频 {test_video} 分辨率检查结果: {result}")
    
    # 测试目录扫描
    valid_videos = filter_obj.scan_directory("rawdata")
    print(f"找到 {len(valid_videos)} 个分辨率合格的视频")


if __name__ == "__main__":
    main()
