"""
FR-4.2.3: Temporal Alignment

시간축 정렬 및 스무딩
- 고정 타임스텝 리샘플링 (dt = 1/target_fps)
- 프레임 보간 (누락 프레임)
- Temporal smoothing (Savitzky-Golay)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from core.logging_config import setup_logger
from transformation.spec import TransformConfig

logger = setup_logger(__name__)


@dataclass
class ResampleResult:
    """리샘플링 결과"""
    timestamps: np.ndarray  # [T'] 새 타임스탬프
    data: np.ndarray  # [T', ...] 리샘플링된 데이터
    mask_valid: np.ndarray  # [T'] 원본 존재 여부
    mask_interpolated: np.ndarray  # [T'] 보간 여부
    gap_indices: List[Tuple[int, int]]  # 큰 gap 구간들


class TemporalResampler:
    """
    고정 FPS로 리샘플링
    """
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
        self.target_fps = self.config.target_fps
        self.dt = 1.0 / self.target_fps
        self.gap_max = self.config.gap_max_frames
        
    def resample(
        self,
        timestamps: np.ndarray,
        data: np.ndarray,
        conf: Optional[np.ndarray] = None,
    ) -> ResampleResult:
        """
        균일 간격으로 리샘플링
        
        Args:
            timestamps: [T] 원본 타임스탬프 (초 단위)
            data: [T, ...] 원본 데이터
            conf: [T, ...] 신뢰도 (선택)
            
        Returns:
            ResampleResult: 리샘플링 결과
        """
        T = len(timestamps)
        original_shape = data.shape[1:]
        
        # 새 타임라인 생성
        t_start = timestamps[0]
        t_end = timestamps[-1]
        new_timestamps = np.arange(t_start, t_end, self.dt)
        T_new = len(new_timestamps)
        
        # 결과 배열 초기화
        new_data = np.zeros((T_new,) + original_shape)
        mask_valid = np.zeros(T_new, dtype=bool)
        mask_interpolated = np.zeros(T_new, dtype=bool)
        gap_indices = []
        
        # 데이터 평탄화
        data_flat = data.reshape(T, -1)
        D = data_flat.shape[1]
        new_data_flat = np.zeros((T_new, D))
        
        # 보간 수행
        for d in range(D):
            # 유효 데이터만 사용
            if conf is not None:
                if conf.ndim > 1:
                    valid = conf.mean(axis=tuple(range(1, conf.ndim))) >= self.config.conf_min
                else:
                    valid = conf >= self.config.conf_min
            else:
                valid = np.ones(T, dtype=bool)
                
            valid_times = timestamps[valid]
            valid_data = data_flat[valid, d]
            
            if len(valid_times) < 2:
                continue
                
            # 선형 보간
            interp_func = interp1d(
                valid_times,
                valid_data,
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )
            new_data_flat[:, d] = interp_func(new_timestamps)
            
        # 마스크 생성
        for i, t in enumerate(new_timestamps):
            # 가장 가까운 원본 프레임 찾기
            closest_idx = np.argmin(np.abs(timestamps - t))
            time_diff = np.abs(timestamps[closest_idx] - t)
            
            if time_diff < self.dt * 0.5:
                mask_valid[i] = True
            else:
                mask_interpolated[i] = True
                
        # Gap 탐지
        gap_indices = self._detect_gaps(timestamps, new_timestamps)
        
        # 결과 reshape
        new_data = new_data_flat.reshape((T_new,) + original_shape)
        
        logger.info(
            f"Resampled: {T} frames @ variable → {T_new} frames @ {self.target_fps}fps, "
            f"gaps: {len(gap_indices)}"
        )
        
        return ResampleResult(
            timestamps=new_timestamps,
            data=new_data,
            mask_valid=mask_valid,
            mask_interpolated=mask_interpolated,
            gap_indices=gap_indices,
        )
    
    def _detect_gaps(
        self,
        orig_timestamps: np.ndarray,
        new_timestamps: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """큰 gap 구간 탐지"""
        gaps = []
        gap_threshold = self.dt * self.gap_max
        
        # 원본 타임스탬프 간격 확인
        time_diffs = np.diff(orig_timestamps)
        gap_mask = time_diffs > gap_threshold
        
        if not np.any(gap_mask):
            return gaps
            
        gap_starts = np.where(gap_mask)[0]
        
        for start_idx in gap_starts:
            gap_start_time = orig_timestamps[start_idx]
            gap_end_time = orig_timestamps[start_idx + 1]
            
            # 새 타임라인에서 해당 구간 찾기
            new_start = np.searchsorted(new_timestamps, gap_start_time)
            new_end = np.searchsorted(new_timestamps, gap_end_time)
            
            if new_end > new_start:
                gaps.append((new_start, new_end))
                
        return gaps
    
    def split_by_gaps(
        self,
        result: ResampleResult,
    ) -> List[Dict[str, np.ndarray]]:
        """Gap을 기준으로 에피소드 분할"""
        segments = []
        
        if not result.gap_indices:
            segments.append({
                "timestamps": result.timestamps,
                "data": result.data,
                "mask_valid": result.mask_valid,
                "mask_interpolated": result.mask_interpolated,
            })
            return segments
            
        # Gap 경계에서 분할
        boundaries = [0]
        for start, end in result.gap_indices:
            boundaries.extend([start, end])
        boundaries.append(len(result.timestamps))
        
        for i in range(0, len(boundaries) - 1, 2):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(result.timestamps)
            
            if end - start < self.gap_max:
                continue
                
            segments.append({
                "timestamps": result.timestamps[start:end],
                "data": result.data[start:end],
                "mask_valid": result.mask_valid[start:end],
                "mask_interpolated": result.mask_interpolated[start:end],
            })
            
        return segments


class SavgolSmoother:
    """
    Savitzky-Golay 필터 기반 스무딩
    """
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
        self.window = self.config.smoothing.window
        self.polyorder = self.config.smoothing.polyorder
        
    def smooth(
        self,
        data: np.ndarray,
        window: Optional[int] = None,
        polyorder: Optional[int] = None,
        axis: int = 0,
    ) -> np.ndarray:
        """
        Savitzky-Golay 스무딩 적용
        
        Args:
            data: [T, ...] 입력 데이터
            window: 윈도우 크기 (홀수)
            polyorder: 다항식 차수
            axis: 시간축
            
        Returns:
            [T, ...] 스무딩된 데이터
        """
        if window is None:
            window = self.window
        if polyorder is None:
            polyorder = self.polyorder
            
        T = data.shape[axis]
        
        # 윈도우 크기 조정 (데이터보다 작아야 함)
        if window > T:
            window = T if T % 2 == 1 else T - 1
        if window < 3:
            return data.copy()
            
        # polyorder < window 확인
        if polyorder >= window:
            polyorder = window - 1
            
        try:
            smoothed = savgol_filter(data, window, polyorder, axis=axis)
            return smoothed
        except Exception as e:
            logger.warning(f"Savgol smoothing failed: {e}, returning original")
            return data.copy()
    
    def smooth_poses(
        self,
        poses: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        포즈와 속도 스무딩
        
        Args:
            poses: [T, J, 3] 포즈
            velocities: [T, J, 3] 속도 (선택)
            
        Returns:
            스무딩된 (poses, velocities)
        """
        smoothed_poses = self.smooth(poses, axis=0)
        
        if velocities is not None:
            smoothed_velocities = self.smooth(velocities, axis=0)
        else:
            smoothed_velocities = None
            
        return smoothed_poses, smoothed_velocities


class TemporalAligner:
    """
    시간축 정렬 통합 파이프라인
    
    리샘플링 + 스무딩 + 마스크 생성
    """
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
        self.resampler = TemporalResampler(config)
        self.smoother = SavgolSmoother(config)
        
    def align(
        self,
        timestamps: np.ndarray,
        pose: np.ndarray,
        pose_conf: Optional[np.ndarray] = None,
        additional_arrays: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        시간축 정렬 수행
        
        Args:
            timestamps: [T] 원본 타임스탬프
            pose: [T, J, 3] 포즈 데이터
            pose_conf: [T, J] 신뢰도
            additional_arrays: 추가 데이터 (같이 리샘플링)
            
        Returns:
            정렬된 데이터 딕셔너리
        """
        result = {}
        
        # 1. 리샘플링
        resample_result = self.resampler.resample(timestamps, pose, pose_conf)
        result["timestamps"] = resample_result.timestamps
        result["pose"] = resample_result.data
        result["mask_valid"] = resample_result.mask_valid
        result["mask_interpolated"] = resample_result.mask_interpolated
        result["gap_indices"] = resample_result.gap_indices
        
        # 2. 신뢰도도 리샘플링
        if pose_conf is not None:
            conf_result = self.resampler.resample(timestamps, pose_conf)
            result["pose_conf"] = conf_result.data
            
        # 3. 추가 배열 리샘플링
        if additional_arrays:
            for name, arr in additional_arrays.items():
                arr_result = self.resampler.resample(timestamps, arr)
                result[name] = arr_result.data
                
        # 4. 스무딩 적용
        if self.config.smoothing.method != "none":
            result["pose_smoothed"] = self.smoother.smooth(result["pose"])
        else:
            result["pose_smoothed"] = result["pose"]
            
        # 5. dt 정보
        result["dt"] = 1.0 / self.config.target_fps
        result["fps"] = self.config.target_fps
        
        # 6. 저신뢰도 마스크
        if pose_conf is not None:
            result["mask_lowconf"] = result.get("pose_conf", np.ones_like(result["mask_valid"])) < self.config.conf_min
        else:
            result["mask_lowconf"] = np.zeros_like(result["mask_valid"])
            
        logger.info(
            f"Temporal alignment: {len(timestamps)} → {len(result['timestamps'])} frames, "
            f"dt={result['dt']:.4f}s"
        )
        
        return result
    
    def compute_velocity(
        self,
        pose: np.ndarray,
        dt: float,
        smooth: bool = True,
    ) -> np.ndarray:
        """
        속도 계산 (시간 미분)
        
        Args:
            pose: [T, J, 3] 포즈
            dt: 시간 간격
            smooth: 스무딩 적용 여부
            
        Returns:
            [T, J, 3] 속도
        """
        T = pose.shape[0]
        velocity = np.zeros_like(pose)
        
        # 중앙 차분
        velocity[1:-1] = (pose[2:] - pose[:-2]) / (2 * dt)
        
        # 경계 처리
        velocity[0] = (pose[1] - pose[0]) / dt
        velocity[-1] = (pose[-1] - pose[-2]) / dt
        
        # 스무딩
        if smooth and self.config.smoothing.method != "none":
            velocity = self.smoother.smooth(velocity)
            
        return velocity
    
    def compute_acceleration(
        self,
        velocity: np.ndarray,
        dt: float,
        smooth: bool = True,
    ) -> np.ndarray:
        """
        가속도 계산 (속도의 시간 미분)
        
        Args:
            velocity: [T, J, 3] 속도
            dt: 시간 간격
            smooth: 스무딩 적용 여부
            
        Returns:
            [T, J, 3] 가속도
        """
        return self.compute_velocity(velocity, dt, smooth)
