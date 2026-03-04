"""
DiskPolicy — 파이프라인 디스크 라이프사이클 관리

정책:
1. 다운로드 전:   여유 공간 체크 (최소 100GB)
2. 처리 완료 후:  원본 MP4 삭제 대기 (data/raw)
3. S3 업로드 확인 후: 로컬 NPZ 삭제
4. 품질 탈락:    NPZ 즉시 삭제
5. 파이프라인 시작 시: 이전 실행 잔여파일 정리
"""

import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 최소 여유 공간 기본값
_DEFAULT_MIN_FREE_GB = 100.0


class DiskPolicy:
    """
    파이프라인 디스크 라이프사이클 정책

    사용법:
        policy = DiskPolicy(
            raw_dir="data/raw",
            episodes_dir="data/episodes",
        )
        if not policy.ensure_space(required_gb=50):
            raise RuntimeError("디스크 공간 부족")
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        episodes_dir: str = "data/episodes",
        min_free_gb: float = _DEFAULT_MIN_FREE_GB,
    ):
        """
        Args:
            raw_dir: 원본 MP4 저장 디렉토리
            episodes_dir: 처리된 NPZ 저장 디렉토리
            min_free_gb: 최소 여유 공간 (GB)
        """
        self.raw_dir = Path(raw_dir)
        self.episodes_dir = Path(episodes_dir)
        self.min_free_gb = min_free_gb

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    # ─── 공간 체크 ────────────────────────────────────────────────────────────

    def get_disk_usage(self, path: Optional[str] = None) -> Dict[str, float]:
        """
        디스크 사용량 조회

        Args:
            path: 조회 경로 (기본: raw_dir 기준 디스크)

        Returns:
            {total_gb, used_gb, free_gb, usage_percent}
        """
        check_path = Path(path) if path else self.raw_dir
        usage = shutil.disk_usage(check_path)
        gb = 1024 ** 3
        return {
            "total_gb": usage.total / gb,
            "used_gb": usage.used / gb,
            "free_gb": usage.free / gb,
            "usage_percent": usage.used / usage.total * 100,
        }

    def ensure_space(self, required_gb: float = 0.0) -> bool:
        """
        다운로드 전 여유 공간 확인

        최소 여유 공간(min_free_gb) + required_gb 이상인지 확인합니다.
        부족하면 cleanup_rejected()를 자동 실행 후 재확인합니다.

        Args:
            required_gb: 추가로 필요한 공간 (GB)

        Returns:
            공간이 충분하면 True
        """
        needed = self.min_free_gb + required_gb
        usage = self.get_disk_usage()
        free = usage["free_gb"]

        if free >= needed:
            return True

        logger.warning(
            f"디스크 여유 공간 부족: {free:.1f}GB / 필요 {needed:.1f}GB — "
            f"품질 탈락 파일 정리 시작"
        )
        self.cleanup_rejected()

        usage = self.get_disk_usage()
        free = usage["free_gb"]

        if free >= needed:
            logger.info(f"정리 후 여유 공간: {free:.1f}GB — 충분")
            return True

        logger.error(
            f"정리 후에도 여유 공간 부족: {free:.1f}GB < {needed:.1f}GB"
        )
        return False

    # ─── 정리 메서드 ──────────────────────────────────────────────────────────

    def cleanup_after_upload(self, run_id: Optional[str] = None) -> Tuple[int, float]:
        """
        S3 업로드 확인 후 로컬 NPZ 삭제

        Args:
            run_id: 특정 run_id 파일만 삭제 (None이면 전체)

        Returns:
            (삭제 파일 수, 확보 GB)
        """
        pattern = f"*{run_id}*_episode.npz" if run_id else "*_episode.npz"
        return self._delete_files(self.episodes_dir, pattern, label="업로드 후 NPZ")

    def cleanup_rejected(self) -> Tuple[int, float]:
        """
        품질 탈락 NPZ 즉시 삭제

        품질 평가기가 '*_rejected.npz' 또는 일반 '_episode.npz' 중
        DB에서 'rejected' 상태인 파일을 삭제합니다.
        여기서는 episodes_dir에서 rejected 표시가 있는 파일을 삭제합니다.

        Returns:
            (삭제 파일 수, 확보 GB)
        """
        return self._delete_files(self.episodes_dir, "*_rejected.npz", label="품질탈락 NPZ")

    def cleanup_old_runs(self, days: int = 7) -> Tuple[int, float]:
        """
        N일 이전 실행의 잔여 파일 정리

        raw_dir의 오래된 MP4와 episodes_dir의 오래된 NPZ를 삭제합니다.

        Args:
            days: 기준 일수 (기본 7일)

        Returns:
            (삭제 파일 수, 확보 GB)
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_ts = cutoff.timestamp()

        total_count = 0
        total_freed_gb = 0.0

        for directory, pattern, label in [
            (self.raw_dir, "*.mp4", f"{days}일 이전 MP4"),
            (self.episodes_dir, "*.npz", f"{days}일 이전 NPZ"),
        ]:
            count, freed = self._delete_files(
                directory, pattern, label=label, older_than_ts=cutoff_ts
            )
            total_count += count
            total_freed_gb += freed

        return total_count, total_freed_gb

    def cleanup_raw_videos(
        self,
        video_ids: Optional[List[str]] = None,
    ) -> Tuple[int, float]:
        """
        처리 완료된 원본 MP4 삭제

        Args:
            video_ids: 삭제할 video_id 목록 (None이면 전체)

        Returns:
            (삭제 파일 수, 확보 GB)
        """
        if video_ids is not None:
            count, freed_gb = 0, 0.0
            for vid in video_ids:
                for mp4 in self.raw_dir.glob(f"{vid}*.mp4"):
                    freed_gb += self._safe_delete(mp4)
                    count += 1
            logger.info(f"원본 MP4 {count}개 삭제, {freed_gb:.2f}GB 확보")
            return count, freed_gb
        return self._delete_files(self.raw_dir, "*.mp4", label="원본 MP4")

    # ─── 내부 유틸리티 ────────────────────────────────────────────────────────

    def _safe_delete(self, path: Path) -> float:
        """안전하게 파일 삭제, 확보된 GB 반환"""
        try:
            size_bytes = path.stat().st_size
            path.unlink()
            return size_bytes / (1024 ** 3)
        except FileNotFoundError:
            return 0.0
        except Exception as e:
            logger.error(f"삭제 실패: {path} — {e}")
            return 0.0

    def _delete_files(
        self,
        directory: Path,
        pattern: str,
        label: str = "파일",
        older_than_ts: Optional[float] = None,
    ) -> Tuple[int, float]:
        """
        디렉토리에서 패턴에 맞는 파일 삭제

        Args:
            directory: 대상 디렉토리
            pattern: glob 패턴
            label: 로깅용 레이블
            older_than_ts: 이 timestamp 이전 파일만 삭제 (None이면 전체)

        Returns:
            (삭제 파일 수, 확보 GB)
        """
        count = 0
        freed_gb = 0.0

        for f in directory.glob(pattern):
            if not f.is_file():
                continue
            if older_than_ts is not None:
                try:
                    if f.stat().st_mtime > older_than_ts:
                        continue
                except Exception:
                    continue
            freed_gb += self._safe_delete(f)
            count += 1

        if count:
            logger.info(f"{label} {count}개 삭제, {freed_gb:.2f}GB 확보")
        return count, freed_gb

    def get_summary(self) -> Dict:
        """디스크 상태 요약"""
        usage = self.get_disk_usage()
        raw_size = sum(f.stat().st_size for f in self.raw_dir.glob("*") if f.is_file())
        ep_size = sum(f.stat().st_size for f in self.episodes_dir.glob("*") if f.is_file())
        gb = 1024 ** 3
        return {
            "disk": usage,
            "raw_dir_gb": raw_size / gb,
            "episodes_dir_gb": ep_size / gb,
            "raw_file_count": sum(1 for _ in self.raw_dir.glob("*.mp4")),
            "episode_file_count": sum(1 for _ in self.episodes_dir.glob("*.npz")),
            "min_free_gb": self.min_free_gb,
        }
