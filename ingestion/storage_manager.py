"""
저장소 최적화 관리자

기능:
- 디스크 공간 모니터링
- 자동 정리 (처리 완료 파일)
- 압축 및 아카이빙
- 임시 파일 관리
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
import psutil

from core.logging_config import logger


@dataclass
class StorageStats:
    """저장소 통계"""
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent_used: float
    
    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)
    
    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)
    
    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)
    
    def __repr__(self):
        return (f"StorageStats(total={self.total_gb:.1f}GB, "
                f"used={self.used_gb:.1f}GB ({self.percent_used:.1f}%), "
                f"free={self.free_gb:.1f}GB)")


@dataclass
class FileInfo:
    """파일 정보"""
    path: Path
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    accessed_at: datetime
    
    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 ** 2)
    
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.modified_at).total_seconds() / 3600


class StorageManager:
    """저장소 관리자"""
    
    # 임계값
    WARNING_THRESHOLD_PERCENT = 80  # 경고
    CRITICAL_THRESHOLD_PERCENT = 90  # 위험
    
    # 정리 정책
    AUTO_CLEANUP_AGE_HOURS = 24  # 24시간 이상 된 파일 삭제
    MIN_FREE_SPACE_GB = 10  # 최소 여유 공간
    
    def __init__(
        self,
        storage_paths: List[str],
    ):
        """
        Args:
            storage_paths: 관리할 디렉토리 목록
        """
        self.storage_paths = [Path(p) for p in storage_paths]
        
        # 디렉토리 생성
        for path in self.storage_paths:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_storage_stats(self, path: Path) -> StorageStats:
        """저장소 통계 조회"""
        usage = psutil.disk_usage(str(path))
        
        return StorageStats(
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
            percent_used=usage.percent,
        )
    
    def check_all_storage(self) -> Dict[str, StorageStats]:
        """모든 저장소 체크"""
        stats = {}
        
        for path in self.storage_paths:
            stats[str(path)] = self.get_storage_stats(path)
        
        return stats
    
    def is_storage_critical(self, path: Path) -> bool:
        """저장소 위험 여부"""
        stats = self.get_storage_stats(path)
        return stats.percent_used >= self.CRITICAL_THRESHOLD_PERCENT
    
    def get_files_info(self, directory: Path, pattern: str = "*") -> List[FileInfo]:
        """디렉토리 내 파일 정보 수집"""
        files = []
        
        for filepath in directory.glob(pattern):
            if not filepath.is_file():
                continue
            
            stat = filepath.stat()
            
            file_info = FileInfo(
                path=filepath,
                size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                accessed_at=datetime.fromtimestamp(stat.st_atime),
            )
            files.append(file_info)
        
        return files
    
    def cleanup_old_files(
        self,
        directory: Path,
        max_age_hours: float = AUTO_CLEANUP_AGE_HOURS,
        dry_run: bool = False,
    ) -> List[Path]:
        """오래된 파일 삭제"""
        files = self.get_files_info(directory)
        deleted_files = []
        total_freed_bytes = 0
        
        for file_info in files:
            if file_info.age_hours >= max_age_hours:
                if dry_run:
                    logger.info(f"[DRY RUN] Would delete: {file_info.path} "
                              f"(age: {file_info.age_hours:.1f}h, size: {file_info.size_mb:.1f}MB)")
                else:
                    try:
                        file_info.path.unlink()
                        deleted_files.append(file_info.path)
                        total_freed_bytes += file_info.size_bytes
                        logger.info(f"Deleted: {file_info.path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_info.path}: {e}")
        
        if not dry_run and deleted_files:
            logger.info(f"Cleanup completed: {len(deleted_files)} files deleted, "
                      f"{total_freed_bytes / (1024**3):.2f}GB freed")
        
        return deleted_files
    
    def cleanup_by_size(
        self,
        directory: Path,
        target_free_gb: float = MIN_FREE_SPACE_GB,
    ) -> List[Path]:
        """크기 기준 정리 (오래된 것부터)"""
        stats = self.get_storage_stats(directory)
        
        if stats.free_gb >= target_free_gb:
            logger.info(f"Sufficient free space: {stats.free_gb:.1f}GB")
            return []
        
        # 필요한 공간
        bytes_to_free = int((target_free_gb - stats.free_gb) * (1024 ** 3))
        logger.info(f"Need to free {bytes_to_free / (1024**3):.2f}GB")
        
        # 파일들을 오래된 순으로 정렬
        files = self.get_files_info(directory)
        files.sort(key=lambda f: f.modified_at)
        
        deleted_files = []
        freed_bytes = 0
        
        for file_info in files:
            if freed_bytes >= bytes_to_free:
                break
            
            try:
                file_info.path.unlink()
                deleted_files.append(file_info.path)
                freed_bytes += file_info.size_bytes
                logger.info(f"Deleted: {file_info.path} ({file_info.size_mb:.1f}MB)")
            except Exception as e:
                logger.error(f"Failed to delete {file_info.path}: {e}")
        
        logger.info(f"Freed {freed_bytes / (1024**3):.2f}GB by deleting {len(deleted_files)} files")
        return deleted_files
    
    def move_to_archive(
        self,
        source_dir: Path,
        archive_dir: Path,
        max_age_hours: float = 72,
    ) -> List[Path]:
        """오래된 파일을 아카이브로 이동"""
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        files = self.get_files_info(source_dir)
        moved_files = []
        
        for file_info in files:
            if file_info.age_hours >= max_age_hours:
                dest_path = archive_dir / file_info.path.name
                
                try:
                    shutil.move(str(file_info.path), str(dest_path))
                    moved_files.append(file_info.path)
                    logger.info(f"Archived: {file_info.path} -> {dest_path}")
                except Exception as e:
                    logger.error(f"Failed to archive {file_info.path}: {e}")
        
        return moved_files
    
    def compress_directory(
        self,
        directory: Path,
        output_archive: Path,
        format: str = 'gztar',  # 'zip', 'tar', 'gztar', 'bztar', 'xztar'
    ):
        """디렉토리 압축"""
        try:
            shutil.make_archive(
                str(output_archive.with_suffix('')),
                format,
                str(directory),
            )
            logger.info(f"Compressed {directory} -> {output_archive}")
        except Exception as e:
            logger.error(f"Compression failed: {e}")
    
    def get_largest_files(self, directory: Path, top_n: int = 10) -> List[FileInfo]:
        """가장 큰 파일들 찾기"""
        files = self.get_files_info(directory)
        files.sort(key=lambda f: f.size_bytes, reverse=True)
        return files[:top_n]
    
    def auto_cleanup(self):
        """자동 정리 실행"""
        logger.info("=== Auto Cleanup Started ===")
        
        for path in self.storage_paths:
            logger.info(f"Checking: {path}")
            
            # 저장소 상태 확인
            stats = self.get_storage_stats(path)
            logger.info(f"Storage: {stats}")
            
            # 위험 수준 체크
            if stats.percent_used >= self.CRITICAL_THRESHOLD_PERCENT:
                logger.warning(f"⚠ CRITICAL: {stats.percent_used:.1f}% used")
                # 긴급 정리
                self.cleanup_by_size(path, target_free_gb=self.MIN_FREE_SPACE_GB * 2)
            
            elif stats.percent_used >= self.WARNING_THRESHOLD_PERCENT:
                logger.warning(f"⚠ WARNING: {stats.percent_used:.1f}% used")
                # 오래된 파일 정리
                self.cleanup_old_files(path, max_age_hours=self.AUTO_CLEANUP_AGE_HOURS)
            
            else:
                logger.info(f"✓ Storage healthy: {stats.percent_used:.1f}% used")
        
        logger.info("=== Auto Cleanup Completed ===")
    
    def periodic_cleanup(self, interval_minutes: int = 30):
        """주기적 정리 (백그라운드 스레드)"""
        import threading
        import time
        
        def _cleanup_loop():
            while True:
                self.auto_cleanup()
                time.sleep(interval_minutes * 60)
        
        thread = threading.Thread(target=_cleanup_loop, daemon=True)
        thread.start()
        logger.info(f"Started periodic cleanup (every {interval_minutes} min)")
    
    def get_total_size(self, directory: Path) -> int:
        """디렉토리 전체 크기"""
        total = 0
        for file_info in self.get_files_info(directory):
            total += file_info.size_bytes
        return total
    
    def get_file_count(self, directory: Path, pattern: str = "*") -> int:
        """파일 개수"""
        return len(self.get_files_info(directory, pattern))
    
    def clear_all(self, directory: Path, confirm: bool = False):
        """전체 삭제 (위험)"""
        if not confirm:
            logger.warning("clear_all requires confirm=True")
            return
        
        files = self.get_files_info(directory)
        
        for file_info in files:
            try:
                file_info.path.unlink()
                logger.info(f"Deleted: {file_info.path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_info.path}: {e}")
        
        logger.info(f"Cleared {len(files)} files from {directory}")
    
    def get_stats_summary(self, directory: Path) -> Dict:
        """통계 요약"""
        files = self.get_files_info(directory)
        
        if not files:
            return {
                'file_count': 0,
                'total_size_mb': 0.0,
                'avg_size_mb': 0.0,
                'oldest_file': None,
                'newest_file': None,
            }
        
        total_size = sum(f.size_bytes for f in files)
        oldest = min(files, key=lambda f: f.modified_at)
        newest = max(files, key=lambda f: f.modified_at)
        
        return {
            'file_count': len(files),
            'total_size_mb': total_size / (1024 ** 2),
            'avg_size_mb': (total_size / len(files)) / (1024 ** 2),
            'oldest_file': str(oldest.path),
            'oldest_age_hours': oldest.age_hours,
            'newest_file': str(newest.path),
            'newest_age_hours': newest.age_hours,
        }
