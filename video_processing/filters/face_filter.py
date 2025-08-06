"""
面部检测和尺寸验证模块
使用OpenCV检测视频中的面部区域，确保面部区域至少为256x256像素
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from loguru import logger
import os
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


class FaceFilter:
    """面部检测和尺寸验证过滤器"""

    def __init__(self, min_face_width: int = 256, min_face_height: int = 256,
                 scale_factor: float = 1.1, min_neighbors: int = 5, min_faces_required: int = 2):
        """
        初始化面部过滤器

        Args:
            min_face_width: 最小面部宽度
            min_face_height: 最小面部高度
            scale_factor: 检测时的缩放因子
            min_neighbors: 最小邻居数
            min_faces_required: 要求的最少面部数量
        """
        self.min_face_width = min_face_width
        self.min_face_height = min_face_height
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_faces_required = min_faces_required

        # 初始化面部检测器
        self.face_cascade = self._load_face_cascade()
        logger.info(f"面部过滤器初始化: 最小面部尺寸 {min_face_width}x{min_face_height}, 要求最少面部数: {min_faces_required}")
    
    def _load_face_cascade(self) -> cv2.CascadeClassifier:
        """加载面部检测级联分类器"""
        # 尝试多个可能的路径
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    logger.info(f"成功加载面部检测器: {path}")
                    return cascade
        
        logger.error("无法加载面部检测器，请确保OpenCV正确安装")
        raise FileNotFoundError("面部检测器文件未找到")
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        在单帧中检测面部
        
        Args:
            frame: 输入帧
            
        Returns:
            面部区域列表 [(x, y, w, h), ...]
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测面部
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(self.min_face_width, self.min_face_height)
            )
            
            # 确保faces是numpy数组并转换为列表
            if len(faces) > 0:
                if hasattr(faces, 'tolist'):
                    return faces.tolist()
                else:
                    return list(faces)
            else:
                return []
            
        except Exception as e:
            logger.error(f"面部检测时出错: {str(e)}")
            return []
    
    def check_face_size(self, face_rect: Tuple[int, int, int, int]) -> bool:
        """
        检查面部尺寸是否满足要求

        Args:
            face_rect: 面部矩形 (x, y, w, h)

        Returns:
            True如果尺寸满足要求
        """
        x, y, w, h = face_rect
        return w >= self.min_face_width and h >= self.min_face_height

    def extract_face_features(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取面部特征向量（简化版本）

        Args:
            face_image: 面部图像

        Returns:
            特征向量或None
        """
        try:
            # 将面部图像调整为固定尺寸
            face_resized = cv2.resize(face_image, (64, 64))

            # 转换为灰度图
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized

            # 使用LBP（局部二值模式）提取特征
            # 这是一个简化的特征提取方法
            features = []

            # 计算直方图特征
            hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
            features.extend(hist.flatten())

            # 计算梯度特征
            grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)

            # 添加梯度统计特征
            features.extend([
                np.mean(grad_x), np.std(grad_x),
                np.mean(grad_y), np.std(grad_y),
                np.mean(face_gray), np.std(face_gray)
            ])

            # 归一化特征
            features = np.array(features)
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)

            return features

        except Exception as e:
            logger.error(f"提取面部特征时出错: {str(e)}")
            return None

    def cluster_faces(self, face_features: List[np.ndarray], similarity_threshold: float = 0.7) -> int:
        """
        对面部特征进行聚类，识别不同的人

        Args:
            face_features: 面部特征向量列表
            similarity_threshold: 相似度阈值

        Returns:
            识别出的不同人数
        """
        if len(face_features) < 2:
            return len(face_features)

        try:
            # 计算特征向量之间的相似度矩阵
            features_matrix = np.array(face_features)
            similarity_matrix = cosine_similarity(features_matrix)

            # 转换为距离矩阵（1 - 相似度）
            distance_matrix = 1 - similarity_matrix

            # 使用DBSCAN聚类
            # eps参数控制聚类的紧密程度
            eps = 1 - similarity_threshold
            clustering = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
            cluster_labels = clustering.fit_predict(distance_matrix)

            # 计算不同的聚类数量（排除噪声点-1）
            unique_clusters = set(cluster_labels)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)  # 移除噪声点

            num_unique_faces = len(unique_clusters)

            logger.debug(f"面部聚类结果: {len(face_features)}个面部 -> {num_unique_faces}个不同的人")
            return num_unique_faces

        except Exception as e:
            logger.error(f"面部聚类时出错: {str(e)}")
            # 如果聚类失败，保守估计返回原始面部数量
            return len(face_features)
    
    def analyze_video_faces(self, video_path: str, sample_frames: int = 10) -> Dict:
        """
        分析视频中的面部信息
        
        Args:
            video_path: 视频文件路径
            sample_frames: 采样帧数
            
        Returns:
            分析结果字典
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return {"valid": False, "error": "无法打开视频"}
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 计算采样间隔
            if total_frames <= sample_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
            
            valid_faces_count = 0
            total_faces_detected = 0
            face_sizes = []
            all_face_features = []  # 存储所有有效面部的特征

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                faces = self.detect_faces_in_frame(frame)
                total_faces_detected += len(faces)

                for face in faces:
                    x, y, w, h = face
                    face_sizes.append((w, h))

                    if self.check_face_size(face):
                        valid_faces_count += 1

                        # 提取面部区域并计算特征
                        face_region = frame[y:y+h, x:x+w]
                        face_features = self.extract_face_features(face_region)

                        if face_features is not None:
                            all_face_features.append(face_features)
            
            cap.release()
            
            # 对面部特征进行聚类，识别不同的人
            unique_people_count = 0
            if all_face_features:
                unique_people_count = self.cluster_faces(all_face_features)

            # 计算统计信息 - 现在基于不同人数而不是面部数量
            has_valid_faces = unique_people_count >= self.min_faces_required
            avg_face_size = np.mean(face_sizes, axis=0) if face_sizes else (0, 0)

            # 安全地转换avg_face_size为列表
            if hasattr(avg_face_size, 'tolist'):
                avg_face_size_list = avg_face_size.tolist()
            else:
                avg_face_size_list = list(avg_face_size) if isinstance(avg_face_size, (tuple, list)) else [0, 0]

            result = {
                "valid": has_valid_faces,
                "total_faces": total_faces_detected,
                "valid_faces": valid_faces_count,
                "unique_people": unique_people_count,  # 新增：不同人数
                "min_faces_required": self.min_faces_required,
                "avg_face_size": avg_face_size_list,
                "face_sizes": face_sizes,
                "sample_frames": len(frame_indices),
                "total_frames": total_frames,
                "fps": fps
            }
            
            if has_valid_faces:
                logger.info(f"✓ 视频 {video_path} 面部检测通过: 识别出{unique_people_count}个不同的人 (检测到{valid_faces_count}/{total_faces_detected}个有效面部，要求≥{self.min_faces_required}个不同的人)")
            else:
                if unique_people_count == 0:
                    logger.warning(f"✗ 视频 {video_path} 面部检测未通过: 未识别出不同的人 (要求≥{self.min_faces_required}个不同的人)")
                else:
                    logger.warning(f"✗ 视频 {video_path} 面部检测未通过: 只识别出{unique_people_count}个不同的人，要求≥{self.min_faces_required}个不同的人")
            
            return result
            
        except Exception as e:
            logger.error(f"分析视频面部时出错 {video_path}: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def filter_videos_by_face_size(self, video_paths: List[str], sample_frames: int = 10) -> List[str]:
        """
        批量过滤视频，返回包含有效面部的视频列表
        
        Args:
            video_paths: 视频文件路径列表
            sample_frames: 每个视频的采样帧数
            
        Returns:
            包含有效面部的视频路径列表
        """
        valid_videos = []
        total_videos = len(video_paths)
        
        logger.info(f"开始面部尺寸过滤，共 {total_videos} 个视频")
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"处理进度: {i}/{total_videos} - {video_path}")
            
            result = self.analyze_video_faces(video_path, sample_frames)
            if result.get("valid", False):
                valid_videos.append(video_path)
        
        logger.info(f"面部尺寸过滤完成: {len(valid_videos)}/{total_videos} 个视频通过")
        return valid_videos
    
    def visualize_faces(self, video_path: str, output_path: str = None, max_frames: int = 5):
        """
        可视化检测到的面部区域（用于调试）
        
        Args:
            video_path: 输入视频路径
            output_path: 输出图片保存路径
            max_frames: 最大处理帧数
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            
            if output_path is None:
                output_path = f"face_detection_results_{Path(video_path).stem}"
            
            os.makedirs(output_path, exist_ok=True)
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                faces = self.detect_faces_in_frame(frame)
                
                # 绘制面部矩形
                for face in faces:
                    x, y, w, h = face
                    color = (0, 255, 0) if self.check_face_size(face) else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{w}x{h}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 保存结果
                output_file = os.path.join(output_path, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(output_file, frame)
            
            cap.release()
            logger.info(f"面部检测可视化结果保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"可视化面部检测时出错: {str(e)}")


def main():
    """测试函数"""
    # 创建面部过滤器
    face_filter = FaceFilter()
    
    # 测试单个视频
    test_video = "rawdata/test/sample.mp4"
    if os.path.exists(test_video):
        result = face_filter.analyze_video_faces(test_video)
        print(f"测试视频 {test_video} 面部分析结果: {result}")
        
        # 可视化结果
        face_filter.visualize_faces(test_video)
    else:
        print(f"测试视频不存在: {test_video}")


if __name__ == "__main__":
    main()
