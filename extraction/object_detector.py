"""
YOLO 기반 객체 검출기

기능:
- YOLOv8 통합
- 관심 객체 필터링
- Bounding box 추출
- 클래스별 신뢰도 관리
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from core.logging_config import logger


@dataclass
class DetectedObject:
    """검출된 객체"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    
    # 중심점
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # 면적
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    # 정규화된 bbox (0~1)
    def normalized_bbox(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return (
            x1 / img_width,
            y1 / img_height,
            x2 / img_width,
            y2 / img_height,
        )


@dataclass
class FrameDetections:
    """프레임별 검출 결과"""
    frame_idx: int
    timestamp: float
    objects: List[DetectedObject] = field(default_factory=list)
    
    def filter_by_class(self, class_names: List[str]) -> List[DetectedObject]:
        """클래스별 필터링"""
        return [obj for obj in self.objects if obj.class_name in class_names]
    
    def filter_by_confidence(self, min_conf: float) -> List[DetectedObject]:
        """신뢰도 필터링"""
        return [obj for obj in self.objects if obj.confidence >= min_conf]
    
    def get_largest_object(self, class_name: Optional[str] = None) -> Optional[DetectedObject]:
        """가장 큰 객체"""
        candidates = self.objects
        
        if class_name:
            candidates = self.filter_by_class([class_name])
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda obj: obj.area)


class ObjectDetector:
    """YOLO 기반 객체 검출기"""
    
    # COCO 데이터셋 클래스 (80개)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # 로봇 작업 관련 객체
    ROBOT_RELEVANT_CLASSES = [
        'bottle', 'cup', 'bowl', 'knife', 'spoon', 'fork',  # 식기
        'book', 'laptop', 'mouse', 'keyboard', 'cell phone',  # 전자기기
        'scissors', 'toothbrush', 'hair drier',  # 도구
        'chair', 'dining table', 'couch',  # 가구
        'apple', 'banana', 'orange', 'sandwich',  # 음식
    ]
    
    def __init__(
        self,
        model_name: str = 'yolov8n.pt',  # n, s, m, l, x (크기 순)
        device: str = 'cuda:0',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Args:
            model_name: YOLO 모델 (yolov8n.pt, yolov8s.pt 등)
            device: 디바이스 ('cuda:0', 'cpu')
            conf_threshold: 최소 신뢰도
            iou_threshold: NMS IoU 임계값
            target_classes: 검출 대상 클래스 (None = 모든 클래스)
        """
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 타겟 클래스 설정
        if target_classes is None:
            self.target_classes = self.ROBOT_RELEVANT_CLASSES
        else:
            self.target_classes = target_classes
        
        # YOLO 모델 로드 (lazy loading)
        self.model = None
        self.class_names = {}
        
        logger.info(f"ObjectDetector initialized (model will be loaded on first use)")
        logger.info(f"Target classes: {len(self.target_classes)}")
    
    def _load_model(self):
        """YOLO 모델 지연 로딩 (다양한 ultralytics 버전 대비 장치 설정 시도)"""
        if self.model is not None:
            return

        try:
            from ultralytics import YOLO
            import torch

            logger.info(f"Loading YOLO model: {self.model_name}")

            # Instantiate model (avoid passing device arg which may not be supported)
            self.model = YOLO(self.model_name)

            # 여러 방법으로 모델을 요청한 디바이스로 이동 시도
            try:
                # ultralytics YOLO 객체에 .to()가 구현된 경우 사용
                self.model.to(self.device)
            except Exception:
                try:
                    # 내부에 torch.nn.Module 형태의 .model 어트리뷰트가 있으면 직접 이동
                    internal_model = getattr(self.model, "model", None)
                    if internal_model is not None:
                        dev = torch.device(self.device if isinstance(self.device, str) else str(self.device))
                        internal_model.to(dev)
                except Exception as e:
                    logger.warning(f"Could not move internal model to device: {e}")

            # 클래스 이름 매핑 (없을 수 있으므로 안전하게 가져옴)
            self.class_names = getattr(self.model, "names", {}) or {}

            logger.info(f"YOLO model loaded (requested device: {self.device})")
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> FrameDetections:
        """
        단일 프레임 검출
        
        Args:
            frame: BGR 이미지
            frame_idx: 프레임 번호
            timestamp: 타임스탬프
        
        Returns:
            FrameDetections
        """
        self._load_model()
        
        # YOLO 추론
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        
        detections = FrameDetections(
            frame_idx=frame_idx,
            timestamp=timestamp,
        )
        
        # 결과 파싱
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                # 타겟 클래스 필터링
                if class_name not in self.target_classes:
                    continue
                
                confidence = float(box.conf[0])
                
                # Bounding box (xyxy 형식)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                obj = DetectedObject(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                )
                
                detections.objects.append(obj)
        
        return detections
    
    def detect_video(
        self,
        video_path: str,
        output_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        visualize: bool = False,
        output_video_path: Optional[str] = None,
    ) -> List[FrameDetections]:
        """
        비디오 검출
        
        Args:
            video_path: 비디오 경로
            output_fps: 출력 FPS (None = 원본)
            max_frames: 최대 프레임 수
            visualize: 시각화 여부
            output_video_path: 시각화 비디오 저장 경로
        
        Returns:
            FrameDetections 리스트
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_fps is None:
            output_fps = original_fps
        
        frame_interval = int(original_fps / output_fps) if output_fps < original_fps else 1
        
        # 비디오 라이터 (시각화용)
        video_writer = None
        if visualize and output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                output_fps,
                (width, height),
            )
        
        all_detections = []
        frame_idx = 0
        processed_count = 0
        
        logger.info(f"Detecting objects in {video_path}")
        logger.info(f"Total frames: {total_frames}, Output FPS: {output_fps}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 프레임 샘플링
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            if max_frames and processed_count >= max_frames:
                break
            
            timestamp = frame_idx / original_fps
            
            # 검출
            detections = self.detect_frame(frame, frame_idx, timestamp)
            all_detections.append(detections)
            
            # 시각화
            if visualize:
                vis_frame = self.visualize_detections(frame, detections)
                
                if video_writer:
                    video_writer.write(vis_frame)
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
            
            frame_idx += 1
            processed_count += 1
        
        cap.release()
        if video_writer:
            video_writer.release()
        
        logger.info(f"Detection completed: {len(all_detections)} frames")
        
        return all_detections
    
    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
    ) -> np.ndarray:
        """검출 결과 시각화"""
        vis_frame = frame.copy()
        
        for obj in detections.objects:
            x1, y1, x2, y2 = obj.bbox
            
            # Bounding box
            color = self._get_color_for_class(obj.class_id)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 레이블
            label = f"{obj.class_name} {obj.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 레이블 배경
            cv2.rectangle(
                vis_frame,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            
            # 레이블 텍스트
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        
        # 프레임 정보
        info_text = f"Frame {detections.frame_idx} | Objects: {len(detections.objects)}"
        cv2.putText(
            vis_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        
        return vis_frame
    
    @staticmethod
    def _get_color_for_class(class_id: int) -> Tuple[int, int, int]:
        """클래스별 색상"""
        # 고정된 색상 (클래스 ID 기반)
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
    
    def save_detections(self, detections: List[FrameDetections], output_path: str):
        """검출 결과 저장 (JSON)"""
        import json
        
        data = []
        
        for frame_det in detections:
            frame_data = {
                'frame_idx': frame_det.frame_idx,
                'timestamp': frame_det.timestamp,
                'objects': [
                    {
                        'class_id': obj.class_id,
                        'class_name': obj.class_name,
                        'confidence': obj.confidence,
                        'bbox': obj.bbox,
                        'center': obj.center,
                        'area': obj.area,
                    }
                    for obj in frame_det.objects
                ],
            }
            data.append(frame_data)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved detections to {output_path}")
    
    @staticmethod
    def load_detections(filepath: str) -> List[FrameDetections]:
        """저장된 검출 결과 로드"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        detections = []
        
        for frame_data in data:
            frame_det = FrameDetections(
                frame_idx=frame_data['frame_idx'],
                timestamp=frame_data['timestamp'],
            )
            
            for obj_data in frame_data['objects']:
                obj = DetectedObject(
                    class_id=obj_data['class_id'],
                    class_name=obj_data['class_name'],
                    confidence=obj_data['confidence'],
                    bbox=tuple(obj_data['bbox']),
                )
                frame_det.objects.append(obj)
            
            detections.append(frame_det)
        
        return detections
