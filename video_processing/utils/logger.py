"""
日志配置模块
"""

import sys
from loguru import logger
from pathlib import Path
from .config import config


def setup_logger():
    """设置日志配置"""
    # 移除默认处理器
    logger.remove()
    
    # 获取日志配置
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_format = log_config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")
    rotation = log_config.get("rotation", "100 MB")
    retention = log_config.get("retention", "30 days")
    
    # 确保日志目录存在
    log_dir = config.get("paths.log_dir", "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 控制台输出
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # 文件输出
    logger.add(
        f"{log_dir}/video_processing.log",
        format=log_format,
        level=log_level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8"
    )
    
    # 错误日志单独记录
    logger.add(
        f"{log_dir}/error.log",
        format=log_format,
        level="ERROR",
        rotation=rotation,
        retention=retention,
        encoding="utf-8"
    )
    
    logger.info("日志系统初始化完成")


# 初始化日志
setup_logger()
