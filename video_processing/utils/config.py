"""
配置管理模块
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path
from loguru import logger


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"成功加载配置文件: {self.config_path}")
                return config
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "paths": {
                "input_dir": "rawdata",
                "output_dir": "processed_data",
                "temp_dir": "temp",
                "log_dir": "logs"
            },
            "video_filters": {
                "min_resolution": {"width": 1920, "height": 1080},
                "min_face_size": {"width": 256, "height": 256},
                "min_age": 22
            },
            "facexformer": {
                "model_path": "models/facexformer",
                "device": "cuda",
                "batch_size": 8,
                "confidence_threshold": 0.8
            },
            "opencv": {
                "scale_factor": 1.1,
                "min_neighbors": 5
            },
            "processing": {
                "max_workers": 4,
                "chunk_size": 10,
                "enable_gpu": True
            },
            "logging": {
                "level": "INFO",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                "rotation": "100 MB",
                "retention": "30 days"
            }
        }
    
    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_paths(self) -> Dict[str, str]:
        """获取路径配置"""
        return self.get("paths", {})
    
    def get_video_filters(self) -> Dict[str, Any]:
        """获取视频过滤配置"""
        return self.get("video_filters", {})
    
    def get_facexformer_config(self) -> Dict[str, Any]:
        """获取FaceXFormer配置"""
        return self.get("facexformer", {})
    
    def get_opencv_config(self) -> Dict[str, Any]:
        """获取OpenCV配置"""
        return self.get("opencv", {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """获取处理配置"""
        return self.get("processing", {})
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        paths = self.get_paths()
        for path_key, path_value in paths.items():
            if path_value:
                Path(path_value).mkdir(parents=True, exist_ok=True)
                logger.debug(f"确保目录存在: {path_value}")


# 全局配置实例
config = Config()
