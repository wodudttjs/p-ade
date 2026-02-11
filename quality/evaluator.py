"""
로봇팔 영상 품질 평가 시스템

4-DOF 관절 검출, 파지 동작 감지 등을 통해 데이터 품질을 평가합니다.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class Grade(Enum):
    """품질 등급"""
    A = "A"  # 90-100점: 완벽
    B = "B"  # 80-89점: 우수
    C = "C"  # 70-79점: 양호
    D = "D"  # 60-69점: 통과
    F = "F"  # 60점 미만: 불합격


@dataclass
class EvaluationResult:
    """평가 결과"""
    video_id: str
    total_score: float = 0.0
    grade: Grade = Grade.F
    passed: bool = False
    
    # 상세 점수
    joint_score: float = 0.0       # 관절 검출 (30점)
    motion_score: float = 0.0      # 동작 품질 (25점)
    grasping_score: float = 0.0    # 파지 동작 (20점)
    stability_score: float = 0.0   # 안정성 (15점)
    coverage_score: float = 0.0    # 프레임 커버리지 (10점)
    
    # 상세 정보
    detected_joints: Dict[str, bool] = field(default_factory=dict)
    has_grasping: bool = False
    frame_coverage: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class QualityConfig:
    """평가 설정"""
    min_joint_confidence: float = 0.5  # 관절 검출 최소 신뢰도
    min_motion_threshold: float = 0.05  # 최소 동작 크기
    grasping_threshold: float = 0.15    # 파지 감지 임계값
    min_frame_coverage: float = 0.7     # 최소 프레임 커버리지
    pass_threshold: float = 60.0        # 통과 점수


class RobotArmQualityEvaluator:
    """
    로봇팔 영상 품질 평가자
    
    평가 기준:
      - 4-DOF 관절 검출 (어깨, 팔꿈치, 손목, 그리퍼): 30점
      - 동작 품질 (움직임 크기, 연속성): 25점
      - 파지 동작 (pick & place): 20점
      - 안정성 (떨림, 오검출): 15점
      - 프레임 커버리지: 10점
    """
    
    # MediaPipe 랜드마크 인덱스
    JOINT_INDICES = {
        "shoulder": [11, 12],   # 양쪽 어깨
        "elbow": [13, 14],      # 양쪽 팔꿈치
        "wrist": [15, 16],      # 양쪽 손목
        "gripper": [19, 20],    # 양쪽 손끝 (검지)
    }
    
    # 손가락 인덱스 (MediaPipe Hand)
    HAND_INDICES = {
        "thumb_tip": 4,
        "index_tip": 8,
        "middle_tip": 12,
        "ring_tip": 16,
        "pinky_tip": 20,
    }
    
    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
    
    def evaluate(
        self,
        sequence: Dict[str, Any],
        video_id: str = "",
    ) -> EvaluationResult:
        """
        시퀀스 품질 평가
        
        Args:
            sequence: 포즈 시퀀스 데이터 (body, left_hand, right_hand)
            video_id: 비디오 ID
            
        Returns:
            EvaluationResult
        """
        result = EvaluationResult(video_id=video_id)
        
        # 데이터 검증
        if not sequence or "body" not in sequence:
            result.issues.append("포즈 데이터 없음")
            return result
        
        body_frames = sequence.get("body", [])
        left_hand = sequence.get("left_hand", [])
        right_hand = sequence.get("right_hand", [])
        
        if len(body_frames) < 10:
            result.issues.append("프레임 수 부족 (<10)")
            return result
        
        # 1. 관절 검출 점수 (30점)
        result.joint_score, result.detected_joints = self._evaluate_joints(body_frames)
        
        # 2. 동작 품질 점수 (25점)
        result.motion_score = self._evaluate_motion(body_frames)
        
        # 3. 파지 동작 점수 (20점)
        result.grasping_score, result.has_grasping = self._evaluate_grasping(
            left_hand, right_hand
        )
        
        # 4. 안정성 점수 (15점)
        result.stability_score = self._evaluate_stability(body_frames)
        
        # 5. 커버리지 점수 (10점)
        result.coverage_score, result.frame_coverage = self._evaluate_coverage(body_frames)
        
        # 총점 계산
        result.total_score = (
            result.joint_score +
            result.motion_score +
            result.grasping_score +
            result.stability_score +
            result.coverage_score
        )
        
        # 등급 결정
        result.grade = self._determine_grade(result.total_score)
        result.passed = result.total_score >= self.config.pass_threshold
        
        return result
    
    def _evaluate_joints(self, body_frames: List[np.ndarray]) -> Tuple[float, Dict[str, bool]]:
        """4-DOF 관절 검출 평가 (30점 만점)"""
        detected = {joint: False for joint in self.JOINT_INDICES}
        joint_confidences = {joint: [] for joint in self.JOINT_INDICES}
        
        for frame in body_frames:
            if frame is None or len(frame) < 21:
                continue
            
            for joint_name, indices in self.JOINT_INDICES.items():
                # 양쪽 중 높은 confidence 사용
                confidences = []
                for idx in indices:
                    if idx < len(frame) and len(frame[idx]) >= 4:
                        # [x, y, z, visibility]
                        conf = frame[idx][3] if len(frame[idx]) > 3 else frame[idx][2]
                        confidences.append(conf)
                
                if confidences:
                    joint_confidences[joint_name].append(max(confidences))
        
        # 관절별 검출 여부 판정
        score = 0
        points_per_joint = 30 / 4  # 관절당 7.5점
        
        for joint_name, confs in joint_confidences.items():
            if confs:
                avg_conf = np.mean(confs)
                if avg_conf >= self.config.min_joint_confidence:
                    detected[joint_name] = True
                    score += points_per_joint
                else:
                    # 부분 점수
                    score += points_per_joint * (avg_conf / self.config.min_joint_confidence) * 0.5
        
        return min(30, score), detected
    
    def _evaluate_motion(self, body_frames: List[np.ndarray]) -> float:
        """동작 품질 평가 (25점 만점)"""
        if len(body_frames) < 2:
            return 0
        
        # 손목 위치 추적 (메인 동작 지표)
        wrist_positions = []
        
        for frame in body_frames:
            if frame is None or len(frame) < 17:
                continue
            
            # 오른쪽 손목 (16) 또는 왼쪽 손목 (15)
            for idx in [16, 15]:
                if idx < len(frame) and frame[idx] is not None:
                    wrist_positions.append(frame[idx][:3])
                    break
        
        if len(wrist_positions) < 2:
            return 0
        
        wrist_positions = np.array(wrist_positions)
        
        # 전체 이동 거리
        total_distance = np.sum(np.linalg.norm(np.diff(wrist_positions, axis=0), axis=1))
        
        # 이동 범위
        range_x = np.max(wrist_positions[:, 0]) - np.min(wrist_positions[:, 0])
        range_y = np.max(wrist_positions[:, 1]) - np.min(wrist_positions[:, 1])
        range_z = np.max(wrist_positions[:, 2]) - np.min(wrist_positions[:, 2]) if wrist_positions.shape[1] > 2 else 0
        
        motion_range = np.sqrt(range_x**2 + range_y**2 + range_z**2)
        
        # 점수 계산
        score = 0
        
        # 충분한 동작이 있는지 (15점)
        if motion_range > self.config.min_motion_threshold:
            score += min(15, motion_range * 100)
        
        # 동작 연속성 (10점) - 급격한 점프 없음
        velocity = np.linalg.norm(np.diff(wrist_positions, axis=0), axis=1)
        velocity_std = np.std(velocity)
        velocity_mean = np.mean(velocity)
        
        if velocity_mean > 0:
            smoothness = 1 - min(1, velocity_std / velocity_mean)
            score += smoothness * 10
        
        return min(25, score)
    
    def _evaluate_grasping(
        self,
        left_hand: List[np.ndarray],
        right_hand: List[np.ndarray],
    ) -> Tuple[float, bool]:
        """파지 동작 평가 (20점 만점)"""
        hand_frames = right_hand if right_hand else left_hand
        
        if not hand_frames or len(hand_frames) < 5:
            return 0, False
        
        # 엄지-검지 거리 변화 추적
        finger_distances = []
        
        for frame in hand_frames:
            if frame is None or len(frame) < 21:
                continue
            
            thumb = frame[4] if len(frame) > 4 else None
            index = frame[8] if len(frame) > 8 else None
            
            if thumb is not None and index is not None:
                dist = np.sqrt(
                    (thumb[0] - index[0])**2 +
                    (thumb[1] - index[1])**2
                )
                finger_distances.append(dist)
        
        if len(finger_distances) < 2:
            return 0, False
        
        # 거리 변화량
        distance_variation = max(finger_distances) - min(finger_distances)
        has_grasping = distance_variation > self.config.grasping_threshold
        
        score = 0
        
        if has_grasping:
            # 파지 동작 감지됨 (15점)
            score = 15
            
            # 파지-해제 사이클 (추가 5점)
            distances = np.array(finger_distances)
            threshold = np.mean(distances)
            
            # 상태 변화 횟수
            states = distances > threshold
            changes = np.sum(np.diff(states.astype(int)) != 0)
            
            if changes >= 2:  # 최소 1회 사이클
                score += min(5, changes * 1.5)
        else:
            # 손 움직임만 있는 경우 (부분 점수)
            score = min(8, distance_variation * 50)
        
        return min(20, score), has_grasping
    
    def _evaluate_stability(self, body_frames: List[np.ndarray]) -> float:
        """안정성 평가 (15점 만점)"""
        if len(body_frames) < 5:
            return 0
        
        # 어깨 위치 (고정점)로 안정성 측정
        shoulder_positions = []
        
        for frame in body_frames:
            if frame is None or len(frame) < 13:
                continue
            
            # 양쪽 어깨 중점
            left = frame[11][:2] if len(frame) > 11 else None
            right = frame[12][:2] if len(frame) > 12 else None
            
            if left is not None and right is not None:
                center = (np.array(left) + np.array(right)) / 2
                shoulder_positions.append(center)
        
        if len(shoulder_positions) < 5:
            return 7.5  # 데이터 부족 시 중간 점수
        
        positions = np.array(shoulder_positions)
        
        # 표준편차 계산 (낮을수록 안정)
        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])
        total_std = np.sqrt(std_x**2 + std_y**2)
        
        # 안정성 점수 (std < 0.01 이면 만점)
        stability = max(0, 1 - total_std / 0.05)
        
        return stability * 15
    
    def _evaluate_coverage(self, body_frames: List[np.ndarray]) -> Tuple[float, float]:
        """프레임 커버리지 평가 (10점 만점)"""
        valid_frames = sum(1 for f in body_frames if f is not None and len(f) > 0)
        total_frames = len(body_frames)
        
        coverage = valid_frames / total_frames if total_frames > 0 else 0
        
        # 커버리지 70% 이상이면 만점
        if coverage >= self.config.min_frame_coverage:
            score = 10
        else:
            score = (coverage / self.config.min_frame_coverage) * 10
        
        return score, coverage
    
    def _determine_grade(self, score: float) -> Grade:
        """점수로 등급 결정"""
        if score >= 90:
            return Grade.A
        elif score >= 80:
            return Grade.B
        elif score >= 70:
            return Grade.C
        elif score >= 60:
            return Grade.D
        else:
            return Grade.F


@dataclass
class QualityStats:
    """품질 통계"""
    grades: Dict[str, int] = field(default_factory=lambda: {
        "A": 0, "B": 0, "C": 0, "D": 0, "F": 0
    })
    total: int = 0
    passed: int = 0
    
    def record(self, result: EvaluationResult):
        """결과 기록"""
        self.grades[result.grade.value] += 1
        self.total += 1
        if result.passed:
            self.passed += 1
    
    @property
    def pass_rate(self) -> float:
        """통과율"""
        return self.passed / self.total * 100 if self.total > 0 else 0
    
    def print_report(self):
        """보고서 출력"""
        print(f"""
{'='*60}
📊 품질 평가 보고서
{'='*60}
""")
        for grade in ["A", "B", "C", "D", "F"]:
            count = self.grades[grade]
            pct = count / self.total * 100 if self.total > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  Grade {grade}: {count:4d} ({pct:5.1f}%) {bar}")
        
        print(f"""
{'─'*60}
  총 평가: {self.total}개
  통과 (≥60점): {self.passed}개 ({self.pass_rate:.1f}%)
""")


if __name__ == "__main__":
    # 테스트
    evaluator = RobotArmQualityEvaluator()
    stats = QualityStats()
    
    # 더미 데이터 테스트
    import random
    
    for i in range(20):
        # 랜덤 시퀀스 생성
        num_frames = random.randint(30, 100)
        body_frames = [
            np.random.rand(33, 4) for _ in range(num_frames)
        ]
        right_hand = [
            np.random.rand(21, 3) for _ in range(num_frames)
        ]
        
        sequence = {
            "body": body_frames,
            "right_hand": right_hand,
        }
        
        result = evaluator.evaluate(sequence, f"test_video_{i}")
        stats.record(result)
        
        print(f"Video {i}: {result.total_score:.1f}점 ({result.grade.value})")
    
    stats.print_report()
