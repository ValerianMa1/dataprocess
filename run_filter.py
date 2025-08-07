#!/usr/bin/env python3
"""
运行视频过滤的简单脚本
"""

import os
import sys
from simple_video_filter import SimpleVideoFilter

def main():
    # 配置
    input_directory = "rawdata"  # 输入视频目录
    output_csv = "filtered_videos.csv"  # 输出CSV文件
    
    print("=" * 60)
    print("简单视频数据过滤工具")
    print("=" * 60)
    print(f"输入目录: {input_directory}")
    print(f"输出文件: {output_csv}")
    print()
    
    # 检查输入目录
    if not os.path.exists(input_directory):
        print(f"错误: 输入目录不存在: {input_directory}")
        print("请确保 rawdata 目录存在并包含视频文件")
        return
    
    # 创建过滤器
    print("初始化视频过滤器...")
    filter_tool = SimpleVideoFilter(use_facexformer=True)  # 启用FaceXFormer
    
    # 处理视频
    filter_tool.process_directory(input_directory, output_csv)
    
    print(f"\n处理完成！请查看结果文件: {output_csv}")

if __name__ == "__main__":
    main()
