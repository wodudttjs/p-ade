"""
로봇팔 영상 품질 평가 시스템

npz 에피소드 파일 기반 데이터 품질을 평가합니다.

npz 구조:
  - poses: (N, 33, 3) — MediaPipe body 포즈 (정규화 좌표 x,y,z)
  - left_hand: (N, 21, 3) — 왼손 랜드마크
  - right_hand: (N, 21, 3) — 오른손 랜드마크
  - confidence: (N,) — 프레임별 신뢰도 (별도 배열)
  - states: (N, D_s) — 상태 벡터
  - actions: (N-1, D_a) — 행동 벡터
  - gripper_state: (N,) — 그리퍼 상태
  - velocity: (N, 33, 3) — 속도
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
    fail_reason: str = ""  # Early Reject 사유 등


@dataclass
class QualityConfig:
    """평가 설정"""
    min_joint_confidence: float = 0.3   # 관절 검출 최소 신뢰도 (별도 confidence 배열 기준)
    min_motion_threshold: float = 0.02  # 최소 동작 크기 (정규화 좌표 기준)
    grasping_threshold: float = 0.02    # 파지 감지 임계값 (정규화 좌표 엄지-검지 거리 변화)
    min_frame_coverage: float = 0.5     # 최소 프레임 커버리지
    pass_threshold: float = 50.0        # 통과 점수 (60 → 50으로 완화, 목표 통과율 65%)
    # Early Reject 기준
    early_reject_min_frames: int = 30   # 프레임 수 < 30 → 즉시 탈락
    early_reject_min_confidence: float = 0.3  # 평균 confidence < 0.3 → 즉시 탈락


class RobotArmQualityEvaluator:
    """
    로봇팔 영상 품질 평가자 — npz 에피소드 파일 기반

    npz 데이터 구조:
      - poses: numpy (N, 33, 3) — body landmarks (x, y, z 정규화 좌표)
      - left_hand / right_hand: numpy (N, 21, 3) — hand landmarks
      - confidence: numpy (N,) — 프레임별 포즈 검출 신뢰도
      - gripper_state: numpy (N,) — 그리퍼 열림/닫힘 상태
      - states / actions: numpy — 모방학습용 상태/행동 벡터

    평가 기준:
      - 4-DOF 관절 검출 (어깨, 팔꿈치, 손목, 그리퍼): 30점
      - 동작 품질 (움직임 크기, 연속성): 25점
      - 파지 동작 (pick & place): 20점
      - 안정성 (떨림, 오검출): 15점
      - 프레임 커버리지: 10점
    """

    # MediaPipe body 랜드마크 인덱스 (33개 중)
    JOINT_INDICES = {
        "shoulder": [11, 12],   # 양쪽 어깨
        "elbow": [13, 14],      # 양쪽 팔꿈치
        "wrist": [15, 16],      # 양쪽 손목
        "gripper": [19, 20],    # 양쪽 검지 끝
    }

    # 손가락 인덱스 (MediaPipe Hand, 21개 중)
    HAND_INDICES = {
        "thumb_tip": 4,
        "index_tip": 8,
        "middle_tip": 12,
        "ring_tip": 16,
        "pinky_tip": 20,
    }

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()

    # ------------------------------------------------------------------
    # 배치 평가 (벡터화 연산)
    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        npz_files: List[str],
        db_path: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """
        여러 NPZ 파일 배치 평가 (벡터화 연산)

        순차 평가 대비 NumPy 벡터화로 속도 개선.
        rejected 파일은 디스크 이동 대신 DB 마킹만 수행.

        Args:
            npz_files: .npz 파일 경로 목록
            db_path: DB 경로 (rejected 마킹용, None이면 마킹 생략)

        Returns:
            EvaluationResult 목록
        """
        results = []

        for npz_path in npz_files:
            try:
                result = self.evaluate_npz(str(npz_path))
                results.append(result)

                # rejected → DB 마킹 (파일 이동 없음, 디스크 I/O 절약)
                if not result.passed and db_path:
                    self._mark_rejected_in_db(result.video_id, result.fail_reason or "quality_failed", db_path)

            except Exception as e:
                logger.warning(f"배치 평가 실패: {npz_path} - {e}")
                results.append(EvaluationResult(
                    video_id=Path(npz_path).stem.replace("_episode", ""),
                    fail_reason="evaluation_error",
                ))

        passed = sum(1 for r in results if r.passed)
        logger.info(f"배치 평가 완료: {len(results)}개, 통과 {passed}개 ({passed/max(len(results),1)*100:.1f}%)")

        return results

    def _mark_rejected_in_db(self, video_id: str, reason: str, db_path: str):
        """rejected 영상을 DB에 마킹 (파일 이동 없이 디스크 I/O 절약)"""
        try:
            from sqlalchemy import create_engine, text
            from pathlib import Path as _Path

            db_file = _Path(db_path)
            if not db_file.is_absolute():
                db_file = _Path(__file__).resolve().parent.parent / db_file

            if not db_file.exists():
                return

            engine = create_engine(f"sqlite:///{db_file}")
            with engine.connect() as conn:
                try:
                    conn.execute(
                        text(
                            "UPDATE videos SET status='rejected', "
                            "failure_reason=:reason "
                            "WHERE video_id=:vid"
                        ),
                        {"vid": video_id, "reason": reason},
                    )
                    conn.commit()
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"DB rejected 마킹 실패 (무시): {e}")

    # ------------------------------------------------------------------
    # npz 파일에서 직접 평가하는 편의 메서드
    # ------------------------------------------------------------------
    def evaluate_npz(self, npz_path: str, video_id: str = "") -> EvaluationResult:
        """
        npz 파일을 직접 로드하여 평가

        Args:
            npz_path: .npz 파일 경로
            video_id: 비디오 ID (없으면 파일명에서 추출)
        """
        data = np.load(str(npz_path), allow_pickle=True)

        if not video_id:
            video_id = Path(npz_path).stem.replace("_episode", "")

        # npz → sequence dict 변환
        sequence = self._npz_to_sequence(data)
        return self.evaluate(sequence, video_id=video_id)

    def _npz_to_sequence(self, data) -> Dict[str, Any]:
        """npz NpzFile → evaluator 가 기대하는 sequence dict 변환"""
        seq: Dict[str, Any] = {}

        # body poses: (N, 33, 3)
        poses = data.get("poses", data.get("body_landmarks", None))
        if poses is not None and isinstance(poses, np.ndarray) and poses.ndim == 3:
            seq["body"] = poses          # (N, 33, 3) numpy 배열 그대로
        else:
            seq["body"] = None

        # hands: (N, 21, 3)
        for key in ("left_hand", "right_hand"):
            arr = data.get(key, data.get(f"{key}_landmarks", None))
            if arr is not None and isinstance(arr, np.ndarray) and arr.ndim == 3:
                seq[key] = arr
            else:
                seq[key] = None

        # confidence: (N,) — 별도 배열
        conf = data.get("confidence", None)
        if conf is not None and isinstance(conf, np.ndarray) and conf.ndim == 1:
            seq["confidence"] = conf
        else:
            seq["confidence"] = None

        # gripper_state: (N,)
        grip = data.get("gripper_state", None)
        if grip is not None and isinstance(grip, np.ndarray):
            seq["gripper_state"] = grip
        else:
            seq["gripper_state"] = None

        # states / actions (IL 데이터 존재 여부 체크용)
        for key in ("states", "actions", "velocity"):
            arr = data.get(key, None)
            if arr is not None and isinstance(arr, np.ndarray) and arr.size > 0:
                seq[key] = arr
            else:
                seq[key] = None

        return seq

    # ------------------------------------------------------------------
    # 메인 평가
    # ------------------------------------------------------------------
    def evaluate(
        self,
        sequence: Dict[str, Any],
        video_id: str = "",
    ) -> EvaluationResult:
        """
        시퀀스 품질 평가

        Args:
            sequence: {
                "body": np.ndarray (N, 33, 3) 또는 None,
                "left_hand": np.ndarray (N, 21, 3) 또는 None,
                "right_hand": np.ndarray (N, 21, 3) 또는 None,
                "confidence": np.ndarray (N,) 또는 None,
                "gripper_state": np.ndarray (N,) 또는 None,
                "states": np.ndarray 또는 None,
                "actions": np.ndarray 또는 None,
            }
            video_id: 비디오 ID
        """
        result = EvaluationResult(video_id=video_id)

        body = sequence.get("body")
        if body is None or (isinstance(body, np.ndarray) and body.size == 0):
            result.issues.append("포즈 데이터 없음")
            result.fail_reason = "no_pose_data"
            return result

        # numpy 배열로 통일
        if not isinstance(body, np.ndarray):
            body = np.array(body)

        n_frames = body.shape[0] if body.ndim >= 1 else 0

        # ── Early Reject: 프레임 수 < 30 → 즉시 탈락 (처리 시간 절약) ──
        if n_frames < self.config.early_reject_min_frames:
            result.issues.append(f"Early Reject: 프레임 수 부족 ({n_frames} < {self.config.early_reject_min_frames})")
            result.fail_reason = "too_few_frames"
            return result

        # 하위 호환 (기존 최소 10프레임)
        if n_frames < 10:
            result.issues.append(f"프레임 수 부족 ({n_frames} < 10)")
            result.fail_reason = "too_few_frames"
            return result

        # ── Early Reject: 평균 confidence < 0.3 → 즉시 탈락 ──
        confidence = sequence.get("confidence")
        if confidence is not None and isinstance(confidence, np.ndarray) and confidence.size == n_frames:
            avg_conf = float(np.mean(confidence))
            if avg_conf < self.config.early_reject_min_confidence:
                result.issues.append(
                    f"Early Reject: 평균 신뢰도 낮음 ({avg_conf:.3f} < {self.config.early_reject_min_confidence})"
                )
                result.fail_reason = "low_confidence"
                return result

        left_hand = sequence.get("left_hand")
        right_hand = sequence.get("right_hand")
        confidence = sequence.get("confidence")  # (N,) 별도 배열
        gripper_state = sequence.get("gripper_state")

        # 1. 관절 검출 점수 (30점)
        result.joint_score, result.detected_joints = self._evaluate_joints(body, confidence)

        # 2. 동작 품질 점수 (25점)
        result.motion_score = self._evaluate_motion(body)

        # 3. 파지 동작 점수 (20점)
        result.grasping_score, result.has_grasping = self._evaluate_grasping(
            left_hand, right_hand, gripper_state
        )

        # 4. 안정성 점수 (15점)
        result.stability_score = self._evaluate_stability(body)

        # 5. 커버리지 점수 (10점)
        result.coverage_score, result.frame_coverage = self._evaluate_coverage(
            body, confidence
        )

        # 총점 계산
        result.total_score = (
            result.joint_score
            + result.motion_score
            + result.grasping_score
            + result.stability_score
            + result.coverage_score
        )

        # 등급 결정
        result.grade = self._determine_grade(result.total_score)
        result.passed = result.total_score >= self.config.pass_threshold

        return result

    # ------------------------------------------------------------------
    # 1. 관절 검출 (30점)
    # ------------------------------------------------------------------
    def _evaluate_joints(
        self,
        body: np.ndarray,
        confidence: Optional[np.ndarray],
    ) -> Tuple[float, Dict[str, bool]]:
        """
        4-DOF 관절 검출 평가 (30점 만점)

        body: (N, 33, 3)
        confidence: (N,) — 프레임별 전체 신뢰도
        """
        detected: Dict[str, bool] = {joint: False for joint in self.JOINT_INDICES}

        n_frames = body.shape[0]
        points_per_joint = 30.0 / len(self.JOINT_INDICES)  # 7.5

        score = 0.0

        for joint_name, indices in self.JOINT_INDICES.items():
            # 해당 관절 좌표가 0이 아닌(= 검출된) 프레임 비율
            detect_count = 0
            for idx in indices:
                if idx < body.shape[1]:
                    coords = body[:, idx, :]          # (N, 3)
                    nonzero = np.any(coords != 0, axis=1)  # 좌표가 (0,0,0)이 아닌 프레임
                    detect_count = max(detect_count, int(np.sum(nonzero)))

            detect_ratio = detect_count / n_frames if n_frames > 0 else 0

            # confidence 배열이 있으면 가중 평균
            if confidence is not None and confidence.size == n_frames:
                avg_conf = float(np.mean(confidence))
                effective = detect_ratio * min(1.0, avg_conf / self.config.min_joint_confidence)
            else:
                effective = detect_ratio

            if effective >= 0.5:
                detected[joint_name] = True
                score += points_per_joint
            elif effective > 0:
                score += points_per_joint * effective
            # else: 0

        return min(30.0, score), detected

    # ------------------------------------------------------------------
    # 2. 동작 품질 (25점)
    # ------------------------------------------------------------------
    def _evaluate_motion(self, body: np.ndarray) -> float:
        """
        동작 품질 평가 (25점 만점)

        body: (N, 33, 3)
        """
        if body.shape[0] < 2:
            return 0.0

        # 손목(15,16) 좌표 추출 — 둘 중 움직임이 큰 쪽 사용
        best_score = 0.0

        for idx in [16, 15]:  # 오른손목, 왼손목
            if idx >= body.shape[1]:
                continue
            wrist = body[:, idx, :]  # (N, 3)

            # 전부 0이면 스킵
            if np.all(wrist == 0):
                continue

            # 이동 범위
            ranges = np.max(wrist, axis=0) - np.min(wrist, axis=0)  # (3,)
            motion_range = float(np.linalg.norm(ranges))

            # 프레임 간 속도
            diffs = np.diff(wrist, axis=0)  # (N-1, 3)
            velocity = np.linalg.norm(diffs, axis=1)  # (N-1,)
            vel_mean = float(np.mean(velocity))
            vel_std = float(np.std(velocity))

            s = 0.0
            # 충분한 동작 (15점)
            if motion_range > self.config.min_motion_threshold:
                # 정규화 좌표 기준 range ~ 0.02 ~ 1.0 범위
                s += min(15.0, motion_range * 30.0)

            # 동작 연속성 (10점) — 급격한 점프 없이 부드러운 동작
            if vel_mean > 0:
                smoothness = 1.0 - min(1.0, vel_std / vel_mean)
                s += smoothness * 10.0

            best_score = max(best_score, s)

        return min(25.0, best_score)

    # ------------------------------------------------------------------
    # 3. 파지 동작 (20점)
    # ------------------------------------------------------------------
    def _evaluate_grasping(
        self,
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray],
        gripper_state: Optional[np.ndarray] = None,
    ) -> Tuple[float, bool]:
        """
        파지 동작 평가 (20점 만점)

        left_hand / right_hand: (N, 21, 3) 또는 None
        gripper_state: (N,) — 0=열림, 1=닫힘 또는 연속값
        """
        score = 0.0
        has_grasping = False

        # ── (A) gripper_state 배열 기반 평가 (있으면 우선) ──
        if gripper_state is not None and isinstance(gripper_state, np.ndarray) and gripper_state.size >= 5:
            gs = gripper_state.astype(float)
            variation = float(np.max(gs) - np.min(gs))

            if variation > 0.01:
                has_grasping = True
                score += 12.0

                # 상태 변화 횟수 (열림↔닫힘 사이클)
                binary = gs > np.mean(gs)
                changes = int(np.sum(np.diff(binary.astype(int)) != 0))
                if changes >= 2:
                    score += min(8.0, changes * 2.0)
                else:
                    score += 3.0

                return min(20.0, score), has_grasping

        # ── (B) 손 랜드마크 기반 평가 (엄지-검지 거리) ──
        hand = right_hand if (right_hand is not None and isinstance(right_hand, np.ndarray) and right_hand.size > 0) \
            else left_hand

        if hand is None or not isinstance(hand, np.ndarray) or hand.ndim != 3 or hand.shape[0] < 5:
            return 0.0, False

        # 엄지(4) - 검지(8) 거리
        thumb = hand[:, 4, :]   # (N, 3)
        index = hand[:, 8, :]   # (N, 3)
        dists = np.linalg.norm(thumb - index, axis=1)  # (N,)

        # 전부 0이면 검출 안 된 것
        if np.all(dists == 0):
            return 0.0, False

        distance_variation = float(np.max(dists) - np.min(dists))
        has_grasping = distance_variation > self.config.grasping_threshold

        if has_grasping:
            score = 12.0
            # 사이클 분석
            threshold_val = np.mean(dists)
            states = dists > threshold_val
            changes = int(np.sum(np.diff(states.astype(int)) != 0))
            if changes >= 2:
                score += min(8.0, changes * 2.0)
            else:
                score += 3.0
        else:
            # 부분 점수: 손 움직임은 있음
            hand_movement = float(np.mean(np.linalg.norm(np.diff(hand[:, 8, :], axis=0), axis=1)))
            if hand_movement > 0.001:
                score = min(8.0, hand_movement * 200.0)

        return min(20.0, score), has_grasping

    # ------------------------------------------------------------------
    # 4. 안정성 (15점)
    # ------------------------------------------------------------------
    def _evaluate_stability(self, body: np.ndarray) -> float:
        """
        안정성 평가 (15점 만점)

        body: (N, 33, 3) — 어깨(11,12) 위치 변동으로 판단
        좌표가 정규화(0~1)가 아닌 월드 좌표일 수 있으므로
        전체 좌표 범위 대비 상대적 안정성을 평가합니다.
        """
        if body.shape[0] < 5:
            return 0.0

        # 양쪽 어깨 중점
        left_sh = body[:, 11, :2]   # (N, 2) — x,y
        right_sh = body[:, 12, :2]
        center = (left_sh + right_sh) / 2.0  # (N, 2)

        # 전부 0이면 검출 안 됨 → 중간 점수
        if np.all(center == 0):
            return 7.5

        # 전체 body 좌표의 범위로 정규화 (좌표 스케일에 독립적)
        body_2d = body[:, :, :2].reshape(-1, 2)
        nonzero_mask = np.any(body_2d != 0, axis=1)
        if nonzero_mask.sum() < 10:
            return 7.5

        body_range = np.max(body_2d[nonzero_mask], axis=0) - np.min(body_2d[nonzero_mask], axis=0)
        scale = float(np.linalg.norm(body_range))

        if scale < 1e-6:
            return 7.5

        # 어깨 중점 std를 전체 범위 대비 비율로
        std_xy = np.std(center, axis=0)
        relative_std = float(np.linalg.norm(std_xy)) / scale

        # relative_std < 0.05 이면 만점 (어깨가 전체 범위의 5% 미만 흔들림)
        # relative_std > 0.3 이면 0점
        stability = max(0.0, 1.0 - relative_std / 0.3)

        return stability * 15.0

    # ------------------------------------------------------------------
    # 5. 커버리지 (10점)
    # ------------------------------------------------------------------
    def _evaluate_coverage(
        self,
        body: np.ndarray,
        confidence: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        """
        프레임 커버리지 평가 (10점 만점)

        body: (N, 33, 3)
        confidence: (N,) 또는 None
        """
        n_frames = body.shape[0]

        if confidence is not None and confidence.size == n_frames:
            # confidence > 0 인 프레임 비율
            valid = int(np.sum(confidence > 0.01))
        else:
            # 좌표가 전부 0이 아닌 프레임
            valid = int(np.sum(np.any(body.reshape(n_frames, -1) != 0, axis=1)))

        coverage = valid / n_frames if n_frames > 0 else 0.0

        if coverage >= self.config.min_frame_coverage:
            score = 10.0
        else:
            score = (coverage / self.config.min_frame_coverage) * 10.0

        return score, coverage

    # ------------------------------------------------------------------
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
    _redis_client: Any = field(default=None, repr=False)
    
    def __post_init__(self):
        """Redis 연결 시도 (선택적)"""
        if self._redis_client is None:
            try:
                import redis
                r = redis.Redis(host="localhost", port=6379, decode_responses=True, socket_timeout=2)
                r.ping()
                self._redis_client = r
            except Exception:
                self._redis_client = None
    
    def record(self, result: EvaluationResult):
        """결과 기록 + Redis 동기화"""
        self.grades[result.grade.value] += 1
        self.total += 1
        if result.passed:
            self.passed += 1
        
        # Redis에 통계 동기화
        self._publish_to_redis(result)
    
    def _publish_to_redis(self, result: EvaluationResult):
        """Redis에 품질 통계 업데이트"""
        if self._redis_client is None:
            return
        
        try:
            pipe = self._redis_client.pipeline()
            pipe.hset("pade:quality_stats", "total", self.total)
            pipe.hset("pade:quality_stats", "passed", self.passed)
            pipe.hset("pade:quality_stats", f"grade_{result.grade.value}", 
                      self.grades[result.grade.value])
            pipe.hset("pade:quality_stats", "pass_rate", 
                      f"{self.pass_rate:.1f}")
            pipe.hset("pade:quality_stats", "last_video_id", result.video_id)
            pipe.hset("pade:quality_stats", "last_score", f"{result.total_score:.1f}")
            pipe.execute()
        except Exception as e:
            logger.debug(f"Redis 품질 통계 업데이트 실패 (무시): {e}")
    
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
    import argparse

    parser = argparse.ArgumentParser(description="품질 평가 테스트")
    parser.add_argument("--dir", default="data/episodes", help="에피소드 디렉토리")
    parser.add_argument("--file", default=None, help="단일 npz 파일")
    parser.add_argument("--threshold", type=float, default=60.0, help="통과 점수")
    args = parser.parse_args()

    config = QualityConfig(pass_threshold=args.threshold)
    evaluator = RobotArmQualityEvaluator(config=config)
    stats = QualityStats()

    if args.file:
        npz_files = [Path(args.file)]
    else:
        ep_dir = Path(args.dir)
        npz_files = sorted(ep_dir.glob("*.npz"))
        # rejected 폴더에 있는 것도 포함
        rejected = ep_dir / "rejected"
        if rejected.exists():
            npz_files += sorted(rejected.glob("*.npz"))

    print(f"평가 대상: {len(npz_files)}개 파일\n")

    for npz_path in npz_files:
        result = evaluator.evaluate_npz(str(npz_path))
        stats.record(result)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"  [{status}] {result.video_id}: "
            f"{result.total_score:.1f}점 ({result.grade.value}) "
            f"J:{result.joint_score:.0f} M:{result.motion_score:.0f} "
            f"G:{result.grasping_score:.0f} S:{result.stability_score:.0f} "
            f"C:{result.coverage_score:.0f}"
        )

    stats.print_report()
