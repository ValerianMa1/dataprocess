# 视频数据处理流 (Video Data Processing Pipeline)

基于Dataset Collection.md流程实现的视频数据处理管道，用于自动过滤和处理视频数据。

## 功能特性

- **分辨率过滤**: 使用OpenCV检测视频分辨率，过滤掉低于1080p的视频
- **面部检测**: 使用OpenCV检测面部区域，确保面部区域至少为256×256像素
- **年龄过滤**: 集成FaceXFormer模型，自动过滤掉22岁以下个人的面部视频
- **批量处理**: 支持多线程并行处理大量视频文件
- **详细日志**: 完整的处理日志和统计报告
- **灵活配置**: 支持YAML配置文件自定义处理参数

## 项目结构

```
dataprocess/
├── video_processing/           # 主要处理模块
│   ├── filters/               # 过滤器模块
│   │   ├── resolution_filter.py  # 分辨率过滤
│   │   └── face_filter.py        # 面部检测过滤
│   ├── models/                # 模型模块
│   │   └── facexformer_age_filter.py  # FaceXFormer年龄过滤
│   ├── utils/                 # 工具模块
│   │   ├── config.py          # 配置管理
│   │   └── logger.py          # 日志配置
│   └── pipeline.py            # 主处理管道
├── tests/                     # 测试文件
├── examples/                  # 示例脚本
├── rawdata/                   # 原始视频数据目录
├── processed_data/            # 处理结果输出目录
├── config.yaml               # 配置文件
└── requirements.txt          # 依赖包列表
```

## 安装和设置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

复制并编辑配置文件：

```bash
cp config.yaml my_config.yaml
# 根据需要修改配置参数
```

### 3. 准备数据

将原始视频文件放入 `rawdata` 目录：

```bash
mkdir -p rawdata
# 将视频文件复制到 rawdata 目录
```

## 使用方法

### 基础使用

```bash
# 处理rawdata目录中的所有视频
python -m video_processing.pipeline --input rawdata --output processed_data

# 处理单个视频文件
python -m video_processing.pipeline --input path/to/video.mp4

# 使用自定义配置
python -m video_processing.pipeline --input rawdata --config my_config.yaml
```

### 使用示例脚本

```bash
# 运行基础示例
python examples/run_pipeline.py --mode basic

# 处理单个视频
python examples/run_pipeline.py --mode single --video path/to/video.mp4

# 使用自定义配置
python examples/run_pipeline.py --mode config --config my_config.yaml

# 创建示例配置文件
python examples/run_pipeline.py --mode create-config
```

### Python API使用

```python
from video_processing.pipeline import VideoProcessingPipeline

# 创建处理管道
pipeline = VideoProcessingPipeline()

# 处理单个视频
result = pipeline.process_single_video("path/to/video.mp4")
print(f"处理结果: {result['passed']}")

# 批量处理视频
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = pipeline.process_videos(video_paths)
print(f"通过的视频: {results['passed_videos']}/{results['total_videos']}")

# 处理整个目录
results = pipeline.process_directory("rawdata", "output")
```

## 配置说明

主要配置参数：

```yaml
# 视频过滤参数
video_filters:
  min_resolution:
    width: 1920      # 最小宽度
    height: 1080     # 最小高度
  min_face_size:
    width: 256       # 最小面部宽度
    height: 256      # 最小面部高度
  min_age: 22        # 最小年龄

# FaceXFormer配置
facexformer:
  model_path: "models/facexformer"
  device: "cuda"     # 或 "cpu"
  confidence_threshold: 0.8

# 处理参数
processing:
  max_workers: 4     # 并行工作线程数
  enable_gpu: true   # 是否启用GPU加速
```

## 处理流程

根据Dataset Collection.md，处理流程包括：

1. **分辨率过滤**: 检查视频分辨率是否超过1080p
2. **面部检测**: 使用OpenCV检测面部区域
3. **面部尺寸验证**: 确保面部区域至少为256×256像素
4. **年龄过滤**: 使用FaceXFormer估计年龄，过滤掉22岁以下的面部
5. **结果输出**: 生成处理报告和通过过滤的视频列表

## 输出结果

处理完成后，会在输出目录生成：

- `processing_results.json`: 详细的处理结果
- `passed_videos.txt`: 通过所有过滤器的视频列表
- `statistics.txt`: 处理统计报告
- `logs/`: 详细的处理日志

## 测试

运行测试套件：

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_pipeline.py -v

# 生成覆盖率报告
pytest tests/ --cov=video_processing --cov-report=html
```

## 注意事项

1. **FaceXFormer模型**: 需要单独下载和配置FaceXFormer模型文件
2. **GPU支持**: 建议使用GPU加速处理大量视频
3. **内存使用**: 处理高分辨率视频时注意内存使用情况
4. **OpenCV依赖**: 确保OpenCV正确安装并包含级联分类器文件

## 故障排除

### 常见问题

1. **OpenCV级联分类器未找到**
   ```bash
   # 检查OpenCV安装
   python -c "import cv2; print(cv2.data.haarcascades)"
   ```

2. **CUDA不可用**
   ```bash
   # 检查PyTorch CUDA支持
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **内存不足**
   - 减少 `max_workers` 参数
   - 降低 `batch_size` 参数
   - 使用CPU而非GPU处理

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目采用MIT许可证。