"""
视频数据处理流安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="video-processing-pipeline",
    version="1.0.0",
    author="Video Processing Team",
    author_email="team@example.com",
    description="视频数据处理流 - 基于Dataset Collection.md流程的视频过滤和处理管道",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/video-processing-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "torchvision[cuda]>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "video-process=video_processing.pipeline:main",
            "video-process-example=examples.run_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "video_processing": ["*.yaml", "*.yml"],
    },
    keywords="video processing, face detection, age filtering, computer vision, opencv, deep learning",
    project_urls={
        "Bug Reports": "https://github.com/example/video-processing-pipeline/issues",
        "Source": "https://github.com/example/video-processing-pipeline",
        "Documentation": "https://github.com/example/video-processing-pipeline/wiki",
    },
)
