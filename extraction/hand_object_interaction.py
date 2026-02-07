"""
손-객체 상호작용 분석

기능:
- 손과 객체 간 거리 계산
- Grasping 상태 추정
- 접촉 감지
- 상호작용 이벤트 추적
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from extraction.pose_estimator import FramePose, PoseLandmark
from extraction.object_detector import DetectedObject, FrameDetections
from core.logging_config import logger


class InteractionType(Enum):
    """상호작용 타입"""
    NO_INTERACTION = "no_interaction"
    APPROACHING = "approaching"
    TOUCHING = "touching"
    GRASPING = "grasping"
    RELEASING = "releasing"


@dataclass
class HandObjectInteraction:
    """손-객체 상호작용"""
    frame_idx: int
    timestamp: float
    
    # 손 정보
    hand_type: str  # 'left' or 'right'
    hand_center: Tuple[float, float, float]  # 3D 중심점
    
    # 객체 정보
    object_id: int
    object_name: str
    object_center: Tuple[float, float]  # 2D 중심점
    object_bbox: Tuple[int, int, int, int]
    
    # 상호작용 메트릭
    distance_2d: float  # 2D 거리 (픽셀)
    distance_3d: Optional[float] = None  # 3D 거리 (미터)
    
    interaction_type: InteractionType = InteractionType.NO_INTERACTION
    interaction_confidence: float = 0.0
    
    # 손 상태
    is_open: bool = True  # 손이 펼쳐져 있는지
    grasp_strength: float = 0.0  # 파지 강도 (0~1)


@dataclass
class InteractionEvent:
    """상호작용 이벤트"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    
    hand_type: str
    object_name: str
    
    event_type: InteractionType
    peak_confidence: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


class HandObjectAnalyzer:
    """손-객체 상호작용 분석기"""
    
    # 거리 임계값 (픽셀)
    APPROACHING_THRESHOLD_PX = 100
    TOUCHING_THRESHOLD_PX = 30
    GRASPING_THRESHOLD_PX = 20
    
    # 3D 거리 임계값 (미터)
    TOUCHING_THRESHOLD_3D = 0.05  # 5cm
    GRASPING_THRESHOLD_3D = 0.02  # 2cm
    
    # 손 상태 판단 임계값
    HAND_OPEN_FINGER_SPREAD = 0.15  # 손가락 펼침 정도
    
    def __init__(self, image_width: int = 1920, image_height: int = 1080):
        self.image_width = image_width
        self.image_height = image_height
    
    def analyze_frame(
        self,
        pose: FramePose,
        detections: FrameDetections,
    ) -> List[HandObjectInteraction]:
        """
        프레임 분석
        
        Args:
            pose: 포즈 데이터
            detections: 객체 검출 결과
        
        Returns:
            HandObjectInteraction 리스트
        """
        interactions = []
        
        # 양손 처리
        for hand_type in ['left', 'right']:
            hand_landmarks = (
                pose.left_hand_landmarks if hand_type == 'left'
                else pose.right_hand_landmarks
            )
            
            if not hand_landmarks:
                continue
            
            # 손 중심점 (손목 기준)
            hand_center_2d = self._get_hand_center_2d(hand_landmarks)
            hand_center_3d = self._get_hand_center_3d(hand_landmarks)
            
            # 손 상태 추정
            is_open = self._is_hand_open(hand_landmarks)
            grasp_strength = self._estimate_grasp_strength(hand_landmarks)
            
            # 각 객체와의 상호작용 체크
            for i, obj in enumerate(detections.objects):
                # 거리 계산
                distance_2d = self._compute_distance_2d(
                    hand_center_2d,
                    obj.center,
                )
                
                distance_3d = self._compute_distance_3d(
                    hand_center_3d,
                    obj.center,
                    obj.bbox,
                )
                
                # 상호작용 타입 판단
                interaction_type, confidence = self._classify_interaction(
                    distance_2d,
                    distance_3d,
                    is_open,
                    grasp_strength,
                )
                
                # 상호작용 객체 생성
                interaction = HandObjectInteraction(
                    frame_idx=pose.frame_idx,
                    timestamp=pose.timestamp,
                    hand_type=hand_type,
                    hand_center=hand_center_3d,
                    object_id=i,
                    object_name=obj.class_name,
                    object_center=obj.center,
                    object_bbox=obj.bbox,
                    distance_2d=distance_2d,
                    distance_3d=distance_3d,
                    interaction_type=interaction_type,
                    interaction_confidence=confidence,
                    is_open=is_open,
                    grasp_strength=grasp_strength,
                )
                
                interactions.append(interaction)
        
        return interactions
    
    def _get_hand_center_2d(self, hand_landmarks: List[PoseLandmark]) -> Tuple[float, float]:
        """손 중심점 (2D)"""
        # 손목 (인덱스 0)
        wrist = hand_landmarks[0]
        
        x = wrist.x * self.image_width
        y = wrist.y * self.image_height
        
        return (x, y)
    
    def _get_hand_center_3d(self, hand_landmarks: List[PoseLandmark]) -> Tuple[float, float, float]:
        """손 중심점 (3D)"""
        wrist = hand_landmarks[0]
        return (wrist.x, wrist.y, wrist.z)
    
    @staticmethod
    def _is_hand_open(hand_landmarks: List[PoseLandmark]) -> bool:
        """손이 펼쳐져 있는지 판단"""
        # 손가락 끝과 손바닥 중심 간 거리로 판단
        # MediaPipe Hand: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
        
        palm_center = hand_landmarks[0]  # 손목을 손바닥 중심으로 근사
        
        finger_tips = [
            hand_landmarks[4],   # 엄지
            hand_landmarks[8],   # 검지
            hand_landmarks[12],  # 중지
            hand_landmarks[16],  # 약지
            hand_landmarks[20],  # 새끼
        ]
        
        # 평균 거리
        distances = []
        for tip in finger_tips:
            dist = np.sqrt(
                (tip.x - palm_center.x)**2 +
                (tip.y - palm_center.y)**2 +
                (tip.z - palm_center.z)**2
            )
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # 임계값 이상이면 열린 손
        return avg_distance > HandObjectAnalyzer.HAND_OPEN_FINGER_SPREAD
    
    @staticmethod
    def _estimate_grasp_strength(hand_landmarks: List[PoseLandmark]) -> float:
        """파지 강도 추정 (0~1)"""
        # 손가락 끝 간 거리로 추정
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        
        # 엄지-검지 거리
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        
        # 정규화 (0.2 = 완전히 닫힘, 0.5 = 완전히 열림)
        max_dist = 0.5
        min_dist = 0.05
        
        normalized = (max_dist - thumb_index_dist) / (max_dist - min_dist)
        strength = np.clip(normalized, 0.0, 1.0)
        
        return strength
    
    @staticmethod
    def _compute_distance_2d(
        point1: Tuple[float, float],
        point2: Tuple[float, float],
    ) -> float:
        """2D 거리"""
        return np.sqrt(
            (point1[0] - point2[0])**2 +
            (point1[1] - point2[1])**2
        )
    
    def _compute_distance_3d(
        self,
        hand_center_3d: Tuple[float, float, float],
        obj_center_2d: Tuple[float, float],
        obj_bbox: Tuple[int, int, int, int],
    ) -> Optional[float]:
        """
        3D 거리 추정
        
        객체의 깊이 정보가 없으므로 근사 추정
        """
        # 간단한 근사: 객체 크기로 깊이 추정
        x1, y1, x2, y2 = obj_bbox
        obj_width = x2 - x1
        obj_height = y2 - y1
        obj_size = np.sqrt(obj_width**2 + obj_height**2)
        
        # 크기가 클수록 가까움 (휴리스틱)
        estimated_depth = 1000 / max(obj_size, 1)  # 임의의 스케일
        
        # 3D 거리 계산
        hand_x, hand_y, hand_z = hand_center_3d
        obj_x = obj_center_2d[0] / self.image_width
        obj_y = obj_center_2d[1] / self.image_height
        obj_z = estimated_depth
        
        distance = np.sqrt(
            (hand_x - obj_x)**2 +
            (hand_y - obj_y)**2 +
            (hand_z - obj_z)**2
        )
        
        return distance
    
    def _classify_interaction(
        self,
        distance_2d: float,
        distance_3d: Optional[float],
        is_open: bool,
        grasp_strength: float,
    ) -> Tuple[InteractionType, float]:
        """
        상호작용 타입 분류
        
        Returns:
            (InteractionType, confidence)
        """
        # Grasping: 거리 가깝고 + 손 닫혀있음
        if distance_2d < self.GRASPING_THRESHOLD_PX and grasp_strength > 0.5:
            return InteractionType.GRASPING, grasp_strength
        
        # Touching: 거리 가까움
        if distance_2d < self.TOUCHING_THRESHOLD_PX:
            confidence = 1.0 - (distance_2d / self.TOUCHING_THRESHOLD_PX)
            return InteractionType.TOUCHING, confidence
        
        # Approaching: 중간 거리
        if distance_2d < self.APPROACHING_THRESHOLD_PX:
            confidence = 1.0 - (distance_2d / self.APPROACHING_THRESHOLD_PX)
            return InteractionType.APPROACHING, confidence
        
        # No interaction
        return InteractionType.NO_INTERACTION, 0.0
    
    def detect_events(
        self,
        interactions: List[List[HandObjectInteraction]],
        min_duration: float = 0.5,  # 최소 지속 시간 (초)
    ) -> List[InteractionEvent]:
        """
        상호작용 이벤트 감지
        
        Args:
            interactions: 프레임별 상호작용 리스트
            min_duration: 최소 이벤트 지속 시간
        
        Returns:
            InteractionEvent 리스트
        """
        events = []
        
        # 손-객체 쌍별로 추적
        active_interactions = {}  # (hand_type, object_name) -> InteractionEvent
        
        for frame_interactions in interactions:
            for interaction in frame_interactions:
                key = (interaction.hand_type, interaction.object_name)
                
                # 의미있는 상호작용만
                if interaction.interaction_type == InteractionType.NO_INTERACTION:
                    # 진행 중인 이벤트 종료
                    if key in active_interactions:
                        event = active_interactions[key]
                        event.end_frame = interaction.frame_idx - 1
                        event.end_time = interaction.timestamp
                        
                        # 최소 지속 시간 체크
                        if event.duration >= min_duration:
                            events.append(event)
                        
                        del active_interactions[key]
                    
                    continue
                
                # 새 이벤트 시작 또는 계속
                if key not in active_interactions:
                    # 새 이벤트
                    event = InteractionEvent(
                        start_frame=interaction.frame_idx,
                        end_frame=interaction.frame_idx,
                        start_time=interaction.timestamp,
                        end_time=interaction.timestamp,
                        hand_type=interaction.hand_type,
                        object_name=interaction.object_name,
                        event_type=interaction.interaction_type,
                        peak_confidence=interaction.interaction_confidence,
                    )
                    active_interactions[key] = event
                else:
                    # 기존 이벤트 업데이트
                    event = active_interactions[key]
                    event.end_frame = interaction.frame_idx
                    event.end_time = interaction.timestamp
                    event.peak_confidence = max(
                        event.peak_confidence,
                        interaction.interaction_confidence
                    )
                    
                    # 타입 업그레이드 (approaching → touching → grasping)
                    current_priority = ['no_interaction', 'approaching', 'touching', 'grasping'].index(
                        interaction.interaction_type.value
                    )
                    event_priority = ['no_interaction', 'approaching', 'touching', 'grasping'].index(
                        event.event_type.value
                    )
                    
                    if current_priority > event_priority:
                        event.event_type = interaction.interaction_type
        
        # 종료되지 않은 이벤트 처리
        for event in active_interactions.values():
            if event.duration >= min_duration:
                events.append(event)
        
        # 시간순 정렬
        events.sort(key=lambda e: e.start_time)
        
        return events
    
    def save_interactions(
        self,
        interactions: List[List[HandObjectInteraction]],
        output_path: str,
    ):
        """상호작용 데이터 저장"""
        import json
        
        data = []
        
        for frame_interactions in interactions:
            frame_data = []
            
            for inter in frame_interactions:
                inter_data = {
                    'frame_idx': inter.frame_idx,
                    'timestamp': inter.timestamp,
                    'hand_type': inter.hand_type,
                    'object_name': inter.object_name,
                    'distance_2d': inter.distance_2d,
                    'distance_3d': inter.distance_3d,
                    'interaction_type': inter.interaction_type.value,
                    'interaction_confidence': inter.interaction_confidence,
                    'is_open': inter.is_open,
                    'grasp_strength': inter.grasp_strength,
                }
                frame_data.append(inter_data)
            
            data.append(frame_data)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved interactions to {output_path}")
    
    def save_events(self, events: List[InteractionEvent], output_path: str):
        """이벤트 저장"""
        import json
        
        data = [
            {
                'start_frame': e.start_frame,
                'end_frame': e.end_frame,
                'start_time': e.start_time,
                'end_time': e.end_time,
                'duration': e.duration,
                'hand_type': e.hand_type,
                'object_name': e.object_name,
                'event_type': e.event_type.value,
                'peak_confidence': e.peak_confidence,
            }
            for e in events
        ]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(events)} events to {output_path}")
