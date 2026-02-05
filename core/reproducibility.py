"""
재현성(Reproducibility) 관리

Feedback #3: 수집→다운로드→추출의 재현성 키
- 모델 버전 (mediapipe, yolo 등)
- 도구 버전 (yt-dlp, ffmpeg, opencv)
- 코드 버전 (git commit hash)
- 처리 파라미터 (fps, threshold 등)
"""

import subprocess
import sys
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime

from core.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class VersionInfo:
    """버전 정보"""
    name: str
    version: str
    hash: Optional[str] = None  # 바이너리/weights 해시
    path: Optional[str] = None


@dataclass
class ReproducibilityContext:
    """
    재현성 컨텍스트
    
    처리 파이프라인의 모든 버전 정보 수집
    """
    # 코드 버전
    git_commit_hash: str = ""
    git_branch: str = ""
    git_dirty: bool = False  # uncommitted changes
    
    # Python 환경
    python_version: str = ""
    
    # 핵심 도구 버전
    yt_dlp_version: str = ""
    ffmpeg_version: str = ""
    opencv_version: str = ""
    numpy_version: str = ""
    
    # AI 모델 버전
    mediapipe_version: str = ""
    mediapipe_pose_model: str = ""
    yolo_version: str = ""
    yolo_weights_hash: str = ""
    
    # 처리 파라미터
    processing_params: Dict[str, Any] = field(default_factory=dict)
    
    # 타임스탬프
    captured_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def get_fingerprint(self) -> str:
        """재현성 지문 생성 (짧은 해시)"""
        content = json.dumps({
            "git": self.git_commit_hash,
            "mediapipe": self.mediapipe_version,
            "yolo_weights": self.yolo_weights_hash,
            "params": self.processing_params,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class VersionCollector:
    """버전 정보 수집기"""
    
    @staticmethod
    def get_git_info() -> Dict[str, Any]:
        """Git 정보 수집"""
        info = {
            "commit_hash": "",
            "branch": "",
            "dirty": False,
        }
        
        try:
            # Commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["commit_hash"] = result.stdout.strip()
            
            # Short hash
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["short_hash"] = result.stdout.strip()
            
            # Branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()
            
            # Dirty check
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["dirty"] = len(result.stdout.strip()) > 0
                
        except Exception as e:
            logger.warning(f"Git 정보 수집 실패: {e}")
        
        return info
    
    @staticmethod
    def get_python_version() -> str:
        """Python 버전"""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    @staticmethod
    def get_package_version(package_name: str) -> str:
        """패키지 버전 조회"""
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except Exception:
            try:
                # 직접 import 시도
                module = __import__(package_name)
                if hasattr(module, "__version__"):
                    return module.__version__
                if hasattr(module, "VERSION"):
                    return str(module.VERSION)
            except Exception:
                pass
        return "unknown"
    
    @staticmethod
    def get_ffmpeg_version() -> str:
        """FFmpeg 버전"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                first_line = result.stdout.split('\n')[0]
                # "ffmpeg version 5.1.2 ..." -> "5.1.2"
                parts = first_line.split()
                for i, p in enumerate(parts):
                    if p == "version" and i + 1 < len(parts):
                        return parts[i + 1]
        except Exception as e:
            logger.warning(f"FFmpeg 버전 조회 실패: {e}")
        return "unknown"
    
    @staticmethod
    def get_yt_dlp_version() -> str:
        """yt-dlp 버전"""
        try:
            result = subprocess.run(
                ["yt-dlp", "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        # Python 모듈로 시도
        return VersionCollector.get_package_version("yt_dlp")
    
    @staticmethod
    def get_file_hash(file_path: str, algorithm: str = "sha256") -> str:
        """파일 해시 계산"""
        path = Path(file_path)
        if not path.exists():
            return ""
        
        hasher = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    @classmethod
    def collect_all(cls, processing_params: Optional[Dict[str, Any]] = None) -> ReproducibilityContext:
        """모든 버전 정보 수집"""
        git_info = cls.get_git_info()
        
        context = ReproducibilityContext(
            # Git
            git_commit_hash=git_info.get("commit_hash", ""),
            git_branch=git_info.get("branch", ""),
            git_dirty=git_info.get("dirty", False),
            
            # Python
            python_version=cls.get_python_version(),
            
            # 도구
            yt_dlp_version=cls.get_yt_dlp_version(),
            ffmpeg_version=cls.get_ffmpeg_version(),
            opencv_version=cls.get_package_version("cv2"),
            numpy_version=cls.get_package_version("numpy"),
            
            # AI 모델
            mediapipe_version=cls.get_package_version("mediapipe"),
            yolo_version=cls.get_package_version("ultralytics"),
            
            # 파라미터
            processing_params=processing_params or {},
        )
        
        logger.info(f"재현성 컨텍스트 수집 완료: {context.get_fingerprint()}")
        return context


class ReproducibilityManager:
    """
    재현성 관리자
    
    처리 결과와 함께 재현성 정보 저장/조회
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
    
    def save_context(
        self,
        context: ReproducibilityContext,
        platform: str,
        video_id: str,
    ) -> Path:
        """재현성 컨텍스트 저장"""
        output_dir = self.base_dir / "episodes" / platform / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "reproducibility.json"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(context.to_json())
        
        logger.info(f"재현성 정보 저장: {output_path}")
        return output_path
    
    def load_context(self, platform: str, video_id: str) -> Optional[ReproducibilityContext]:
        """재현성 컨텍스트 로드"""
        context_path = self.base_dir / "episodes" / platform / video_id / "reproducibility.json"
        
        if not context_path.exists():
            return None
        
        try:
            with open(context_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ReproducibilityContext(**{
                k: v for k, v in data.items() 
                if k in ReproducibilityContext.__dataclass_fields__
            })
        except Exception as e:
            logger.error(f"재현성 정보 로드 실패: {e}")
            return None
    
    def compare_contexts(
        self,
        ctx1: ReproducibilityContext,
        ctx2: ReproducibilityContext
    ) -> Dict[str, Any]:
        """두 컨텍스트 비교"""
        differences = {}
        
        dict1 = ctx1.to_dict()
        dict2 = ctx2.to_dict()
        
        for key in dict1:
            if key in ["captured_at", "processing_params"]:
                continue
            
            if dict1[key] != dict2[key]:
                differences[key] = {
                    "old": dict1[key],
                    "new": dict2[key],
                }
        
        # 파라미터 비교
        params1 = dict1.get("processing_params", {})
        params2 = dict2.get("processing_params", {})
        
        param_diffs = {}
        all_keys = set(params1.keys()) | set(params2.keys())
        
        for key in all_keys:
            v1 = params1.get(key)
            v2 = params2.get(key)
            if v1 != v2:
                param_diffs[key] = {"old": v1, "new": v2}
        
        if param_diffs:
            differences["processing_params"] = param_diffs
        
        return differences
    
    def verify_reproducibility(
        self,
        platform: str,
        video_id: str,
        current_context: Optional[ReproducibilityContext] = None
    ) -> Dict[str, Any]:
        """재현성 검증"""
        saved_context = self.load_context(platform, video_id)
        
        if saved_context is None:
            return {
                "status": "no_saved_context",
                "reproducible": None,
            }
        
        current = current_context or VersionCollector.collect_all()
        differences = self.compare_contexts(saved_context, current)
        
        # 주요 버전 차이 확인
        critical_fields = ["git_commit_hash", "mediapipe_version", "yolo_version"]
        critical_diffs = {k: v for k, v in differences.items() if k in critical_fields}
        
        return {
            "status": "compared",
            "reproducible": len(critical_diffs) == 0,
            "critical_differences": critical_diffs,
            "all_differences": differences,
            "saved_fingerprint": saved_context.get_fingerprint(),
            "current_fingerprint": current.get_fingerprint(),
        }


# 기본 처리 파라미터 (Feedback #3)
DEFAULT_PROCESSING_PARAMS = {
    "sampling_fps": 30,
    "pose_model_complexity": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "conf_threshold": 0.5,
    "nms_threshold": 0.4,
    "max_episode_gap_frames": 30,
    "min_episode_frames": 30,
    "normalize_to_hip": True,
    "smooth_window_size": 5,
}


def get_current_reproducibility_context(
    custom_params: Optional[Dict[str, Any]] = None
) -> ReproducibilityContext:
    """현재 환경의 재현성 컨텍스트 가져오기"""
    params = {**DEFAULT_PROCESSING_PARAMS}
    if custom_params:
        params.update(custom_params)
    
    return VersionCollector.collect_all(processing_params=params)
