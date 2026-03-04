"""
스토리지 관리자

임시 비디오 파일 저장소 관리 및 디스크 공간 최적화
"""

import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from core.logging_config import logger
from storage.disk_policy import DiskPolicy


@dataclass
class StorageStats:
    """스토리지 통계"""
    total_bytes: int
    used_bytes: int
    free_bytes: int
    usage_percent: float


@dataclass
class FileInfo:
    """파일 정보"""
    filepath: Path
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    age_seconds: float


class StorageManager:
    """스토리지 관리 클래스"""
    
    def __init__(
        self,
        temp_dir: Path,
        max_usage_percent: float = 80.0,
        min_free_gb: float = 10.0,
        cleanup_age_hours: int = 24,
        raw_dir: Optional[str] = None,
        episodes_dir: Optional[str] = None,
    ):
        """
        Args:
            temp_dir: 임시 파일 저장 디렉토리
            max_usage_percent: 최대 디스크 사용률 (%)
            min_free_gb: 최소 여유 공간 (GB)
            cleanup_age_hours: 자동 정리 기준 시간
            raw_dir: 원본 MP4 디렉토리 (DiskPolicy 연동)
            episodes_dir: 에피소드 NPZ 디렉토리 (DiskPolicy 연동)
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.max_usage_percent = max_usage_percent
        self.min_free_gb = min_free_gb
        self.cleanup_age_hours = cleanup_age_hours

        # DiskPolicy 연동
        _raw = raw_dir or str(self.temp_dir.parent / "raw")
        _eps = episodes_dir or str(self.temp_dir.parent / "episodes")
        self.disk_policy = DiskPolicy(
            raw_dir=_raw,
            episodes_dir=_eps,
            min_free_gb=min_free_gb,
        )

        # 메타데이터 파일
        self.metadata_file = self.temp_dir / ".storage_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        
        return {'files': {}, 'last_cleanup': None}
    
    def _save_metadata(self):
        """메타데이터 저장"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_disk_usage(self) -> StorageStats:
        """
        디스크 사용량 조회
        
        Returns:
            StorageStats 객체
        """
        usage = shutil.disk_usage(self.temp_dir)
        
        return StorageStats(
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
            usage_percent=(usage.used / usage.total) * 100
        )
    
    def has_enough_space(self, required_bytes: int) -> bool:
        """
        필요한 공간 확인
        
        Args:
            required_bytes: 필요한 바이트 수
            
        Returns:
            공간 충분 여부
        """
        stats = self.get_disk_usage()
        
        # 여유 공간 체크
        free_after_download = stats.free_bytes - required_bytes
        min_free_bytes = self.min_free_gb * 1024 * 1024 * 1024
        
        if free_after_download < min_free_bytes:
            logger.warning(
                f"Insufficient space: need {required_bytes} bytes, "
                f"but only {stats.free_bytes} bytes available"
            )
            return False
        
        # 사용률 체크
        future_used = stats.used_bytes + required_bytes
        future_usage_percent = (future_used / stats.total_bytes) * 100
        
        if future_usage_percent > self.max_usage_percent:
            logger.warning(
                f"Usage would exceed limit: {future_usage_percent:.1f}% "
                f"(max: {self.max_usage_percent}%)"
            )
            return False
        
        return True
    
    def register_file(
        self,
        filepath: Path,
        video_id: str,
        metadata: Optional[Dict] = None
    ):
        """
        파일 등록
        
        Args:
            filepath: 파일 경로
            video_id: 비디오 ID
            metadata: 추가 메타데이터
        """
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return
        
        file_info = {
            'video_id': video_id,
            'filepath': str(filepath),
            'size_bytes': filepath.stat().st_size,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {},
        }
        
        self.metadata['files'][str(filepath)] = file_info
        self._save_metadata()
        
        logger.debug(f"Registered file: {filepath}")
    
    def unregister_file(self, filepath: Path):
        """
        파일 등록 해제
        
        Args:
            filepath: 파일 경로
        """
        filepath_str = str(filepath)
        
        if filepath_str in self.metadata['files']:
            del self.metadata['files'][filepath_str]
            self._save_metadata()
            logger.debug(f"Unregistered file: {filepath}")
    
    def list_files(self, pattern: str = "*") -> List[FileInfo]:
        """
        파일 목록 조회
        
        Args:
            pattern: 파일 패턴 (glob)
            
        Returns:
            FileInfo 리스트
        """
        files = []
        now = datetime.now()
        
        for filepath in self.temp_dir.glob(pattern):
            if filepath.is_file() and filepath.name != ".storage_metadata.json":
                try:
                    stat = filepath.stat()
                    created = datetime.fromtimestamp(stat.st_ctime)
                    modified = datetime.fromtimestamp(stat.st_mtime)
                    age = (now - created).total_seconds()
                    
                    files.append(FileInfo(
                        filepath=filepath,
                        size_bytes=stat.st_size,
                        created_at=created,
                        modified_at=modified,
                        age_seconds=age,
                    ))
                except Exception as e:
                    logger.error(f"Failed to get file info: {filepath} - {e}")
        
        return files
    
    def get_old_files(
        self,
        older_than_hours: Optional[int] = None
    ) -> List[FileInfo]:
        """
        오래된 파일 목록
        
        Args:
            older_than_hours: 기준 시간 (None이면 cleanup_age_hours 사용)
            
        Returns:
            FileInfo 리스트
        """
        threshold_hours = older_than_hours or self.cleanup_age_hours
        threshold_seconds = threshold_hours * 3600
        
        all_files = self.list_files()
        old_files = [f for f in all_files if f.age_seconds > threshold_seconds]
        
        return sorted(old_files, key=lambda f: f.age_seconds, reverse=True)
    
    def cleanup_old_files(
        self,
        older_than_hours: Optional[int] = None,
        dry_run: bool = False
    ) -> Tuple[int, int]:
        """
        오래된 파일 정리
        
        Args:
            older_than_hours: 기준 시간
            dry_run: 테스트 모드 (실제 삭제 안 함)
            
        Returns:
            (삭제된 파일 수, 확보된 바이트)
        """
        old_files = self.get_old_files(older_than_hours)
        
        deleted_count = 0
        freed_bytes = 0
        
        for file_info in old_files:
            try:
                if not dry_run:
                    file_info.filepath.unlink()
                    self.unregister_file(file_info.filepath)
                
                deleted_count += 1
                freed_bytes += file_info.size_bytes
                
                logger.info(
                    f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: "
                    f"{file_info.filepath} ({file_info.size_bytes} bytes, "
                    f"{file_info.age_seconds / 3600:.1f} hours old)"
                )
                
            except Exception as e:
                logger.error(f"Failed to delete file: {file_info.filepath} - {e}")
        
        if not dry_run:
            self.metadata['last_cleanup'] = datetime.now().isoformat()
            self._save_metadata()
        
        logger.info(
            f"Cleanup complete: {deleted_count} files, "
            f"{freed_bytes / 1024 / 1024:.2f} MB freed"
        )
        
        return deleted_count, freed_bytes
    
    def cleanup_by_size(
        self,
        target_free_gb: float,
        dry_run: bool = False
    ) -> Tuple[int, int]:
        """
        목표 여유 공간 확보를 위한 정리
        
        Args:
            target_free_gb: 목표 여유 공간 (GB)
            dry_run: 테스트 모드
            
        Returns:
            (삭제된 파일 수, 확보된 바이트)
        """
        stats = self.get_disk_usage()
        target_free_bytes = target_free_gb * 1024 * 1024 * 1024
        
        if stats.free_bytes >= target_free_bytes:
            logger.info("Already have enough free space")
            return 0, 0
        
        needed_bytes = target_free_bytes - stats.free_bytes
        logger.info(f"Need to free {needed_bytes / 1024 / 1024:.2f} MB")
        
        # 오래된 파일부터 삭제
        files = self.get_old_files(older_than_hours=0)  # 모든 파일
        
        deleted_count = 0
        freed_bytes = 0
        
        for file_info in files:
            if freed_bytes >= needed_bytes:
                break
            
            try:
                if not dry_run:
                    file_info.filepath.unlink()
                    self.unregister_file(file_info.filepath)
                
                deleted_count += 1
                freed_bytes += file_info.size_bytes
                
                logger.info(
                    f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: "
                    f"{file_info.filepath} ({file_info.size_bytes} bytes)"
                )
                
            except Exception as e:
                logger.error(f"Failed to delete file: {file_info.filepath} - {e}")
        
        if not dry_run:
            self.metadata['last_cleanup'] = datetime.now().isoformat()
            self._save_metadata()
        
        logger.info(
            f"Size-based cleanup: {deleted_count} files, "
            f"{freed_bytes / 1024 / 1024:.2f} MB freed"
        )
        
        return deleted_count, freed_bytes
    
    def ensure_space(self, required_bytes: int) -> bool:
        """
        필요한 공간 확보
        
        Args:
            required_bytes: 필요한 바이트 수
            
        Returns:
            확보 성공 여부
        """
        if self.has_enough_space(required_bytes):
            return True
        
        logger.info(f"Attempting to free space for {required_bytes} bytes")
        
        # 먼저 오래된 파일 정리
        self.cleanup_old_files()
        
        if self.has_enough_space(required_bytes):
            return True
        
        # 그래도 부족하면 크기 기반 정리
        stats = self.get_disk_usage()
        additional_needed_gb = (required_bytes - stats.free_bytes) / 1024 / 1024 / 1024
        target_free_gb = self.min_free_gb + additional_needed_gb
        
        self.cleanup_by_size(target_free_gb)
        
        return self.has_enough_space(required_bytes)
    
    def get_total_size(self) -> int:
        """
        임시 디렉토리 전체 크기
        
        Returns:
            전체 크기 (bytes)
        """
        total = 0
        for file_info in self.list_files():
            total += file_info.size_bytes
        return total
    
    def get_file_count(self) -> int:
        """
        파일 개수
        
        Returns:
            파일 수
        """
        return len(self.list_files())
    
    def clear_all(self, confirm: bool = False) -> Tuple[int, int]:
        """
        모든 파일 삭제
        
        Args:
            confirm: 확인 플래그 (안전장치)
            
        Returns:
            (삭제된 파일 수, 확보된 바이트)
        """
        if not confirm:
            logger.warning("clear_all() called without confirmation")
            return 0, 0
        
        files = self.list_files()
        deleted_count = 0
        freed_bytes = 0
        
        for file_info in files:
            try:
                file_info.filepath.unlink()
                deleted_count += 1
                freed_bytes += file_info.size_bytes
            except Exception as e:
                logger.error(f"Failed to delete: {file_info.filepath} - {e}")
        
        # 메타데이터 초기화
        self.metadata = {'files': {}, 'last_cleanup': datetime.now().isoformat()}
        self._save_metadata()
        
        logger.warning(
            f"Cleared all files: {deleted_count} files, "
            f"{freed_bytes / 1024 / 1024:.2f} MB"
        )
        
        return deleted_count, freed_bytes
    
    # =========================================================================
    # DiskPolicy 연동 메서드
    # =========================================================================

    def cleanup_after_upload(
        self,
        video_id: Optional[str] = None,
        run_id: Optional[str] = None,
        s3_paths: Optional[List[str]] = None,
    ) -> Tuple[int, float]:
        """
        S3 업로드 성공 확인 후 로컬 NPZ + 원본 MP4 삭제

        Args:
            video_id: 특정 video_id만 삭제 (None이면 run_id 기준)
            run_id: run_id 기준 삭제 (None이면 전체)
            s3_paths: 업로드된 S3 경로 목록 (검증용, 현재 미사용)

        Returns:
            (삭제 파일 수, 확보 GB)
        """
        total_count = 0
        total_freed = 0.0

        if video_id:
            # 특정 video_id 파일만 삭제
            for pattern in [f"{video_id}*.npz", f"{video_id}*.mp4"]:
                for target_dir in [self.disk_policy.episodes_dir, self.disk_policy.raw_dir]:
                    for f in target_dir.glob(pattern):
                        try:
                            size = f.stat().st_size
                            f.unlink()
                            total_freed += size / (1024 ** 3)
                            total_count += 1
                            logger.info(f"업로드 후 삭제: {f.name}")
                        except Exception as e:
                            logger.error(f"삭제 실패: {f} — {e}")
        else:
            # run_id 기준 또는 전체 NPZ 삭제
            count, freed = self.disk_policy.cleanup_after_upload(run_id=run_id)
            total_count += count
            total_freed += freed

        logger.info(
            f"cleanup_after_upload 완료: {total_count}개 삭제, "
            f"{total_freed:.2f}GB 확보"
        )
        return total_count, total_freed

    def cleanup_rejected(self) -> Tuple[int, float]:
        """
        품질 탈락 NPZ 즉시 삭제

        Returns:
            (삭제 파일 수, 확보 GB)
        """
        count, freed = self.disk_policy.cleanup_rejected()
        logger.info(f"cleanup_rejected 완료: {count}개 삭제, {freed:.2f}GB 확보")
        return count, freed

    def cleanup_old_runs(self, days: int = 7) -> Tuple[int, float]:
        """
        N일 이전 실행 잔여파일 정리 (MP4 + NPZ)

        Args:
            days: 기준 일수 (기본 7일)

        Returns:
            (삭제 파일 수, 확보 GB)
        """
        count, freed = self.disk_policy.cleanup_old_runs(days=days)
        logger.info(
            f"cleanup_old_runs({days}일) 완료: {count}개 삭제, {freed:.2f}GB 확보"
        )
        return count, freed

    def get_stats_summary(self) -> Dict:
        """
        통계 요약
        
        Returns:
            통계 딕셔너리
        """
        disk_stats = self.get_disk_usage()
        temp_size = self.get_total_size()
        file_count = self.get_file_count()
        
        return {
            'disk': {
                'total_gb': disk_stats.total_bytes / 1024 / 1024 / 1024,
                'used_gb': disk_stats.used_bytes / 1024 / 1024 / 1024,
                'free_gb': disk_stats.free_bytes / 1024 / 1024 / 1024,
                'usage_percent': disk_stats.usage_percent,
            },
            'temp_dir': {
                'path': str(self.temp_dir),
                'size_mb': temp_size / 1024 / 1024,
                'file_count': file_count,
            },
            'config': {
                'max_usage_percent': self.max_usage_percent,
                'min_free_gb': self.min_free_gb,
                'cleanup_age_hours': self.cleanup_age_hours,
            },
            'last_cleanup': self.metadata.get('last_cleanup'),
        }
