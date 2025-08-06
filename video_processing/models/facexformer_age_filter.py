"""
FaceXFormer年龄过滤模块
集成FaceXFormer模型来自动过滤掉22岁以下个人的面部视频
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger
import os
from PIL import Image
import torchvision.transforms as transforms


class FaceXFormerAgeFilter:
    """FaceXFormer年龄过滤器"""
    
    def __init__(self, model_path: str = None, device: str = "cuda", 
                 min_age: int = 22, confidence_threshold: float = 0.8):
        """
        初始化FaceXFormer年龄过滤器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备 ("cuda" 或 "cpu")
            min_age: 最小年龄阈值
            confidence_threshold: 置信度阈值
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.min_age = min_age
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        self.model = self._load_model()
        logger.info(f"FaceXFormer年龄过滤器初始化: 最小年龄 {min_age}, 设备 {self.device}")
    
    def _load_model(self) -> Optional[nn.Module]:
        """
        加载FaceXFormer模型
        注意: 这里提供模型加载的框架，实际需要根据FaceXFormer的具体实现调整
        """
        try:
            if self.model_path and os.path.exists(self.model_path):
                # 实际的FaceXFormer模型加载逻辑
                # 这里需要根据FaceXFormer的具体API进行调整
                logger.info(f"尝试加载FaceXFormer模型: {self.model_path}")
                
                # 示例代码 - 需要替换为实际的FaceXFormer加载逻辑
                # model = FaceXFormer.load_pretrained(self.model_path)
                # model.to(self.device)
                # model.eval()
                # return model
                
                logger.warning("FaceXFormer模型加载功能待实现")
                return None
            else:
                logger.warning(f"FaceXFormer模型文件不存在: {self.model_path}")
                return None
                
        except Exception as e:
            logger.error(f"加载FaceXFormer模型时出错: {str(e)}")
            return None
    
    def _preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        预处理面部图像
        
        Args:
            face_image: 面部图像 (BGR格式)
            
        Returns:
            预处理后的张量
        """
        try:
            # 转换为RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(face_rgb)
            
            # 应用变换
            tensor = self.transform(pil_image)
            
            # 添加批次维度
            tensor = tensor.unsqueeze(0).to(self.device)
            
            return tensor
            
        except Exception as e:
            logger.error(f"预处理面部图像时出错: {str(e)}")
            return None
    
    def estimate_age(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        估计面部年龄
        
        Args:
            face_image: 面部图像
            
        Returns:
            年龄估计结果字典 {"age": float, "confidence": float}
        """
        if self.model is None:
            # 如果模型未加载，使用替代方案或返回默认值
            logger.warning("FaceXFormer模型未加载，使用替代年龄估计方法")
            return self._fallback_age_estimation(face_image)
        
        try:
            # 预处理图像
            input_tensor = self._preprocess_face(face_image)
            if input_tensor is None:
                return None
            
            # 模型推理
            with torch.no_grad():
                # 这里需要根据FaceXFormer的实际API调整
                # output = self.model(input_tensor)
                # age_pred = output['age']  # 假设模型输出包含年龄预测
                # confidence = output['confidence']  # 假设模型输出包含置信度
                
                # 临时示例 - 需要替换为实际的FaceXFormer推理逻辑
                logger.warning("FaceXFormer推理功能待实现")
                return self._fallback_age_estimation(face_image)
            
        except Exception as e:
            logger.error(f"年龄估计时出错: {str(e)}")
            return None
    
    def _fallback_age_estimation(self, face_image: np.ndarray) -> Dict:
        """
        替代年龄估计方法（当FaceXFormer不可用时）
        这里可以集成其他年龄估计模型或方法
        """
        # 简单的基于面部特征的年龄估计（示例）
        # 实际应用中可以集成其他预训练的年龄估计模型
        
        height, width = face_image.shape[:2]
        
        # 基于面部尺寸的简单启发式估计（仅作示例）
        if height < 100 or width < 100:
            estimated_age = 25.0  # 小面部可能是年轻人
        else:
            estimated_age = 30.0  # 大面部可能是成年人
        
        return {
            "age": estimated_age,
            "confidence": 0.5,  # 低置信度，因为这是简单估计
            "method": "fallback"
        }
    
    def check_age_requirement(self, face_image: np.ndarray) -> bool:
        """
        检查面部是否满足年龄要求
        
        Args:
            face_image: 面部图像
            
        Returns:
            True如果年龄满足要求（>=min_age）
        """
        age_result = self.estimate_age(face_image)
        if age_result is None:
            logger.warning("无法估计年龄，默认不通过")
            return False
        
        age = age_result.get("age", 0)
        confidence = age_result.get("confidence", 0)
        
        # 检查置信度和年龄
        if confidence < self.confidence_threshold:
            logger.warning(f"年龄估计置信度过低: {confidence:.2f} < {self.confidence_threshold}")
            return False
        
        meets_requirement = age >= self.min_age
        
        if meets_requirement:
            logger.debug(f"✓ 年龄检查通过: {age:.1f}岁 >= {self.min_age}岁")
        else:
            logger.info(f"✗ 年龄检查未通过: {age:.1f}岁 < {self.min_age}岁")
        
        return meets_requirement
    
    def analyze_video_ages(self, video_path: str, face_regions: List[Tuple], 
                          sample_frames: int = 5) -> Dict:
        """
        分析视频中的年龄信息
        
        Args:
            video_path: 视频文件路径
            face_regions: 面部区域列表 [(frame_idx, x, y, w, h), ...]
            sample_frames: 采样帧数
            
        Returns:
            年龄分析结果
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"valid": False, "error": "无法打开视频"}
            
            valid_faces = 0
            total_faces = 0
            age_estimates = []
            
            # 按帧分组面部区域
            frame_faces = {}
            for face_data in face_regions:
                frame_idx = face_data[0]
                if frame_idx not in frame_faces:
                    frame_faces[frame_idx] = []
                frame_faces[frame_idx].append(face_data[1:])  # (x, y, w, h)
            
            # 采样帧进行分析
            frame_indices = sorted(frame_faces.keys())
            if len(frame_indices) > sample_frames:
                step = len(frame_indices) // sample_frames
                frame_indices = frame_indices[::step][:sample_frames]
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                for x, y, w, h in frame_faces[frame_idx]:
                    total_faces += 1
                    
                    # 提取面部区域
                    face_image = frame[y:y+h, x:x+w]
                    
                    if self.check_age_requirement(face_image):
                        valid_faces += 1
                    
                    # 记录年龄估计
                    age_result = self.estimate_age(face_image)
                    if age_result:
                        age_estimates.append(age_result["age"])
            
            cap.release()
            
            # 计算统计信息
            has_valid_faces = valid_faces > 0
            avg_age = np.mean(age_estimates) if age_estimates else 0
            
            result = {
                "valid": has_valid_faces,
                "total_faces": total_faces,
                "valid_faces": valid_faces,
                "avg_age": float(avg_age),
                "age_estimates": age_estimates,
                "min_age_requirement": self.min_age
            }
            
            if has_valid_faces:
                logger.info(f"✓ 视频 {video_path} 年龄过滤通过: {valid_faces}/{total_faces} 个有效面部")
            else:
                logger.warning(f"✗ 视频 {video_path} 年龄过滤未通过: 无符合年龄要求的面部")
            
            return result
            
        except Exception as e:
            logger.error(f"分析视频年龄时出错 {video_path}: {str(e)}")
            return {"valid": False, "error": str(e)}


def main():
    """测试函数"""
    # 创建年龄过滤器
    age_filter = FaceXFormerAgeFilter()
    
    # 测试图像
    test_image_path = "test_face.jpg"
    if os.path.exists(test_image_path):
        test_image = cv2.imread(test_image_path)
        result = age_filter.check_age_requirement(test_image)
        print(f"测试图像年龄检查结果: {result}")
    else:
        print(f"测试图像不存在: {test_image_path}")


if __name__ == "__main__":
    main()
