"""
视频处理管道示例脚本
演示如何使用视频处理管道处理视频数据
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_processing.pipeline import VideoProcessingPipeline
from video_processing.utils.logger import setup_logger
from loguru import logger


def create_sample_config():
    """创建示例配置文件"""
    config_content = """# 视频数据处理配置文件

# 输入输出路径
paths:
  input_dir: "rawdata"
  output_dir: "processed_data"
  temp_dir: "temp"
  log_dir: "logs"

# 视频过滤参数
video_filters:
  # 分辨率过滤 - 超过1080p
  min_resolution:
    width: 1920
    height: 1080
  
  # 面部区域最小尺寸 - 256x256像素
  min_face_size:
    width: 256
    height: 256
  
  # 年龄过滤 - 过滤掉22岁以下
  min_age: 22

# FaceXFormer模型配置
facexformer:
  model_path: "models/facexformer"
  device: "cuda"  # 或 "cpu"
  batch_size: 8
  confidence_threshold: 0.8

# OpenCV配置
opencv:
  scale_factor: 1.1
  min_neighbors: 5

# 处理参数
processing:
  max_workers: 4
  chunk_size: 10
  enable_gpu: true

# 日志配置
logging:
  level: "INFO"
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
  rotation: "100 MB"
  retention: "30 days"
"""
    
    with open("example_config.yaml", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    logger.info("示例配置文件已创建: example_config.yaml")


def run_basic_example():
    """运行基础示例"""
    logger.info("=" * 60)
    logger.info("运行基础视频处理示例")
    logger.info("=" * 60)
    
    # 创建处理管道
    pipeline = VideoProcessingPipeline()
    
    # 检查rawdata目录是否存在
    input_dir = "rawdata"
    if not os.path.exists(input_dir):
        logger.warning(f"输入目录不存在: {input_dir}")
        logger.info("创建示例目录结构...")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs("rawdata/test", exist_ok=True)
        logger.info("请将视频文件放入 rawdata 目录中")
        return
    
    # 处理目录中的视频
    results = pipeline.process_directory(input_dir)
    
    # 显示结果
    logger.info("处理结果:")
    logger.info(f"总视频数: {results.get('total_videos', 0)}")
    logger.info(f"通过的视频数: {results.get('passed_videos', 0)}")
    logger.info(f"处理时间: {results.get('processing_time', 0):.2f} 秒")
    
    # 显示统计信息
    stats = results.get('statistics', {})
    logger.info("\n详细统计:")
    logger.info(f"  分辨率过滤通过: {stats.get('resolution_passed', 0)}")
    logger.info(f"  面部检测通过: {stats.get('face_passed', 0)}")
    logger.info(f"  年龄过滤通过: {stats.get('age_passed', 0)}")
    logger.info(f"  最终通过: {stats.get('final_passed', 0)}")


def run_single_video_example(video_path: str):
    """运行单个视频处理示例"""
    logger.info("=" * 60)
    logger.info(f"处理单个视频: {video_path}")
    logger.info("=" * 60)
    
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        return
    
    # 创建处理管道
    pipeline = VideoProcessingPipeline()
    
    # 处理单个视频
    result = pipeline.process_single_video(video_path)
    
    # 显示结果
    logger.info("处理结果:")
    logger.info(f"视频路径: {result['video_path']}")
    logger.info(f"是否通过: {result['passed']}")
    logger.info(f"处理时间: {result['processing_time']:.2f} 秒")
    
    # 显示各个过滤器的结果
    filters = result.get('filters', {})
    
    # 分辨率过滤结果
    resolution_result = filters.get('resolution', {})
    logger.info(f"\n分辨率过滤:")
    logger.info(f"  通过: {resolution_result.get('passed', False)}")
    logger.info(f"  分辨率: {resolution_result.get('resolution', 'N/A')}")
    
    # 面部检测结果
    face_result = filters.get('face', {})
    logger.info(f"\n面部检测:")
    logger.info(f"  有效: {face_result.get('valid', False)}")
    logger.info(f"  检测到的面部数: {face_result.get('total_faces', 0)}")
    logger.info(f"  有效面部数: {face_result.get('valid_faces', 0)}")
    
    # 年龄过滤结果
    age_result = filters.get('age', {})
    logger.info(f"\n年龄过滤:")
    logger.info(f"  有效: {age_result.get('valid', False)}")
    
    if result.get('error'):
        logger.error(f"处理错误: {result['error']}")


def run_custom_config_example(config_path: str):
    """使用自定义配置运行示例"""
    logger.info("=" * 60)
    logger.info(f"使用自定义配置: {config_path}")
    logger.info("=" * 60)
    
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    # 使用自定义配置创建管道
    pipeline = VideoProcessingPipeline(config_path)
    
    # 处理默认输入目录
    results = pipeline.process_directory("rawdata")
    
    logger.info("处理完成!")
    logger.info(f"结果已保存到配置文件指定的输出目录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="视频处理管道示例")
    parser.add_argument("--mode", choices=["basic", "single", "config", "create-config"], 
                       default="basic", help="运行模式")
    parser.add_argument("--video", help="单个视频文件路径（single模式使用）")
    parser.add_argument("--config", help="配置文件路径（config模式使用）")
    
    args = parser.parse_args()
    
    # 确保日志系统初始化
    setup_logger()
    
    try:
        if args.mode == "create-config":
            create_sample_config()
        elif args.mode == "basic":
            run_basic_example()
        elif args.mode == "single":
            if not args.video:
                logger.error("single模式需要指定--video参数")
                return
            run_single_video_example(args.video)
        elif args.mode == "config":
            if not args.config:
                logger.error("config模式需要指定--config参数")
                return
            run_custom_config_example(args.config)
    
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"运行示例时出错: {str(e)}")


if __name__ == "__main__":
    main()
