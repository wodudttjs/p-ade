"""
Version Manager

데이터셋 시맨틱 버전닝 및 매니페스트 관리
- v{major}.{minor}.{patch} 형식
- 매니페스트 생성/파싱
- 버전 비교 및 diff
"""

import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class VersionBump(Enum):
    """버전 증가 유형"""
    MAJOR = "major"  # 호환되지 않는 변경
    MINOR = "minor"  # 새로운 기능 추가 (하위 호환)
    PATCH = "patch"  # 버그 수정


@dataclass
class SemanticVersion:
    """시맨틱 버전"""
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """문자열에서 버전 파싱"""
        # v1.0.0 또는 1.0.0 형식
        match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
        )
    
    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"
    
    def bump(self, bump_type: VersionBump) -> "SemanticVersion":
        """버전 증가"""
        if bump_type == VersionBump.MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        elif bump_type == VersionBump.MINOR:
            return SemanticVersion(self.major, self.minor + 1, 0)
        else:
            return SemanticVersion(self.major, self.minor, self.patch + 1)
    
    def __lt__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))


@dataclass
class FileEntry:
    """매니페스트 파일 엔트리"""
    key: str
    sha256: str
    size_bytes: int
    file_type: str
    storage_class: Optional[str] = None
    compression: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetManifest:
    """
    데이터셋 매니페스트
    
    버전별 파일 목록 및 메타데이터
    """
    version: str
    created_at: str
    description: str = ""
    
    # 통계
    total_files: int = 0
    total_bytes: int = 0
    total_videos: int = 0
    total_episodes: int = 0
    
    # 파일 목록
    files: List[FileEntry] = field(default_factory=list)
    
    # 클라우드 정보
    provider: str = "s3"
    bucket: str = ""
    base_prefix: str = ""
    
    # 체크섬
    manifest_sha256: str = ""
    
    # 이전 버전
    parent_version: Optional[str] = None
    
    # 태그
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_file(self, entry: FileEntry):
        """파일 추가"""
        self.files.append(entry)
        self.total_files = len(self.files)
        self.total_bytes = sum(f.size_bytes for f in self.files)
        
    def compute_checksum(self) -> str:
        """매니페스트 체크섬 계산"""
        # 파일 해시 정렬하여 결합
        file_hashes = sorted([f.sha256 for f in self.files])
        combined = "".join(file_hashes)
        self.manifest_sha256 = hashlib.sha256(combined.encode()).hexdigest()
        return self.manifest_sha256
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "description": self.description,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "total_videos": self.total_videos,
            "total_episodes": self.total_episodes,
            "files": [asdict(f) for f in self.files],
            "provider": self.provider,
            "bucket": self.bucket,
            "base_prefix": self.base_prefix,
            "manifest_sha256": self.manifest_sha256,
            "parent_version": self.parent_version,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        """딕셔너리에서 생성"""
        files = [FileEntry(**f) for f in data.get("files", [])]
        
        return cls(
            version=data["version"],
            created_at=data["created_at"],
            description=data.get("description", ""),
            total_files=data.get("total_files", len(files)),
            total_bytes=data.get("total_bytes", 0),
            total_videos=data.get("total_videos", 0),
            total_episodes=data.get("total_episodes", 0),
            files=files,
            provider=data.get("provider", "s3"),
            bucket=data.get("bucket", ""),
            base_prefix=data.get("base_prefix", ""),
            manifest_sha256=data.get("manifest_sha256", ""),
            parent_version=data.get("parent_version"),
            tags=data.get("tags", {}),
        )
    
    def save(self, path: str):
        """파일로 저장"""
        self.compute_checksum()
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved manifest {self.version} to {path}")
    
    @classmethod
    def load(cls, path: str) -> "DatasetManifest":
        """파일에서 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return cls.from_dict(data)


@dataclass
class VersionDiff:
    """
    버전 간 차이점
    """
    from_version: str
    to_version: str
    
    # 변경 사항
    added_files: List[FileEntry] = field(default_factory=list)
    removed_files: List[FileEntry] = field(default_factory=list)
    modified_files: List[Tuple[FileEntry, FileEntry]] = field(default_factory=list)  # (old, new)
    
    # 통계
    added_bytes: int = 0
    removed_bytes: int = 0
    net_bytes: int = 0
    
    def summary(self) -> Dict[str, Any]:
        """차이점 요약"""
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "added_count": len(self.added_files),
            "removed_count": len(self.removed_files),
            "modified_count": len(self.modified_files),
            "added_bytes": self.added_bytes,
            "removed_bytes": self.removed_bytes,
            "net_bytes": self.net_bytes,
        }


class VersionManager:
    """
    데이터셋 버전 관리자
    
    FR-5.3: Version Control
    """
    
    def __init__(
        self,
        manifests_dir: str = "manifests",
        provider: str = "s3",
        bucket: str = "",
    ):
        """
        VersionManager 초기화
        
        Args:
            manifests_dir: 매니페스트 저장 디렉토리
            provider: 클라우드 프로바이더
            bucket: 기본 버킷
        """
        self.manifests_dir = Path(manifests_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        
        self.provider = provider
        self.bucket = bucket
        
        self._manifests: Dict[str, DatasetManifest] = {}
        self._load_manifests()
        
    def _load_manifests(self):
        """로컬 매니페스트 로드"""
        for path in self.manifests_dir.glob("manifest_v*.json"):
            try:
                manifest = DatasetManifest.load(str(path))
                self._manifests[manifest.version] = manifest
            except Exception as e:
                logger.warning(f"Failed to load manifest {path}: {e}")
                
        logger.info(f"Loaded {len(self._manifests)} manifests")
    
    def get_latest_version(self) -> Optional[SemanticVersion]:
        """최신 버전 조회"""
        if not self._manifests:
            return None
            
        versions = [SemanticVersion.parse(v) for v in self._manifests.keys()]
        return max(versions)
    
    def create_new_version(
        self,
        bump_type: VersionBump = VersionBump.PATCH,
        description: str = "",
        base_prefix: str = "",
    ) -> DatasetManifest:
        """새 버전 생성"""
        latest = self.get_latest_version()
        
        if latest:
            new_version = latest.bump(bump_type)
            parent_version = str(latest)
        else:
            new_version = SemanticVersion(1, 0, 0)
            parent_version = None
            
        manifest = DatasetManifest(
            version=str(new_version),
            created_at=datetime.utcnow().isoformat(),
            description=description,
            provider=self.provider,
            bucket=self.bucket,
            base_prefix=base_prefix or f"datasets/{new_version}",
            parent_version=parent_version,
        )
        
        self._manifests[str(new_version)] = manifest
        logger.info(f"Created new version: {new_version}")
        
        return manifest
    
    def get_manifest(self, version: str) -> Optional[DatasetManifest]:
        """버전별 매니페스트 조회"""
        return self._manifests.get(version)
    
    def save_manifest(self, manifest: DatasetManifest):
        """매니페스트 저장"""
        path = self.manifests_dir / f"manifest_{manifest.version}.json"
        manifest.save(str(path))
        self._manifests[manifest.version] = manifest
    
    def compare_versions(
        self,
        from_version: str,
        to_version: str,
    ) -> Optional[VersionDiff]:
        """버전 비교"""
        from_manifest = self.get_manifest(from_version)
        to_manifest = self.get_manifest(to_version)
        
        if not from_manifest or not to_manifest:
            return None
            
        diff = VersionDiff(
            from_version=from_version,
            to_version=to_version,
        )
        
        # 키 기반 비교
        from_keys = {f.key: f for f in from_manifest.files}
        to_keys = {f.key: f for f in to_manifest.files}
        
        # 추가된 파일
        for key in to_keys:
            if key not in from_keys:
                diff.added_files.append(to_keys[key])
                diff.added_bytes += to_keys[key].size_bytes
                
        # 삭제된 파일
        for key in from_keys:
            if key not in to_keys:
                diff.removed_files.append(from_keys[key])
                diff.removed_bytes += from_keys[key].size_bytes
                
        # 수정된 파일
        for key in from_keys:
            if key in to_keys:
                old = from_keys[key]
                new = to_keys[key]
                
                if old.sha256 != new.sha256:
                    diff.modified_files.append((old, new))
                    diff.added_bytes += new.size_bytes
                    diff.removed_bytes += old.size_bytes
                    
        diff.net_bytes = diff.added_bytes - diff.removed_bytes
        
        return diff
    
    def list_versions(self) -> List[str]:
        """모든 버전 목록"""
        versions = list(self._manifests.keys())
        versions.sort(key=lambda v: SemanticVersion.parse(v))
        return versions
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """버전 히스토리 조회"""
        history = []
        
        for version in self.list_versions():
            manifest = self._manifests[version]
            history.append({
                "version": version,
                "created_at": manifest.created_at,
                "description": manifest.description,
                "total_files": manifest.total_files,
                "total_bytes": manifest.total_bytes,
                "total_episodes": manifest.total_episodes,
                "parent_version": manifest.parent_version,
            })
            
        return history
    
    def suggest_bump_type(
        self,
        added_files: int,
        removed_files: int,
        modified_files: int,
    ) -> VersionBump:
        """
        변경 사항 기반 버전 증가 유형 제안
        
        - MAJOR: 파일 삭제가 있는 경우 (하위 호환성 깨짐)
        - MINOR: 새 파일 추가만 있는 경우
        - PATCH: 수정만 있는 경우
        """
        if removed_files > 0:
            return VersionBump.MAJOR
        elif added_files > 0:
            return VersionBump.MINOR
        else:
            return VersionBump.PATCH


class ManifestBuilder:
    """
    매니페스트 빌더
    
    파일 시스템 또는 클라우드에서 매니페스트 생성
    """
    
    def __init__(self, version_manager: VersionManager):
        self.version_manager = version_manager
        
    def build_from_files(
        self,
        files: List[Dict[str, Any]],
        bump_type: VersionBump = VersionBump.PATCH,
        description: str = "",
    ) -> DatasetManifest:
        """
        파일 목록에서 매니페스트 생성
        
        Args:
            files: 파일 정보 리스트
                [{"key": ..., "sha256": ..., "size_bytes": ..., "file_type": ...}, ...]
            bump_type: 버전 증가 유형
            description: 버전 설명
        """
        manifest = self.version_manager.create_new_version(bump_type, description)
        
        for file_info in files:
            entry = FileEntry(
                key=file_info["key"],
                sha256=file_info["sha256"],
                size_bytes=file_info["size_bytes"],
                file_type=file_info.get("file_type", "unknown"),
                storage_class=file_info.get("storage_class"),
                compression=file_info.get("compression"),
                metadata=file_info.get("metadata", {}),
            )
            manifest.add_file(entry)
            
        # 통계 업데이트
        manifest.total_episodes = sum(
            1 for f in manifest.files if f.file_type == "episode_npz"
        )
        manifest.total_videos = sum(
            1 for f in manifest.files if f.file_type == "video_mp4"
        )
        
        # 체크섬 계산 및 저장
        self.version_manager.save_manifest(manifest)
        
        return manifest
    
    def build_from_cloud(
        self,
        provider_name: str,
        bucket: str,
        prefix: str,
        bump_type: VersionBump = VersionBump.PATCH,
        description: str = "",
    ) -> DatasetManifest:
        """
        클라우드 스토리지에서 매니페스트 생성
        """
        from storage.upload_manager import UploadManager
        
        manager = UploadManager(default_provider=provider_name)
        provider = manager.get_provider(provider_name)
        
        # 파일 목록 조회
        objects = provider.list_objects(bucket, prefix, max_keys=10000)
        
        files = []
        for obj in objects:
            # 메타데이터 조회
            details = provider.head_object(obj.key, bucket)
            
            files.append({
                "key": obj.key,
                "sha256": details.metadata.get("sha256", "") if details else "",
                "size_bytes": obj.size_bytes,
                "file_type": self._infer_file_type(obj.key),
                "storage_class": obj.storage_class,
            })
            
        return self.build_from_files(files, bump_type, description)
    
    def _infer_file_type(self, key: str) -> str:
        """파일 타입 추론"""
        key_lower = key.lower()
        
        if key_lower.endswith(".npz"):
            return "episode_npz"
        elif key_lower.endswith(".mp4"):
            return "video_mp4"
        elif key_lower.endswith(".json"):
            return "manifest_json"
        elif key_lower.endswith(".parquet"):
            return "parquet"
        elif key_lower.endswith(".h5") or key_lower.endswith(".hdf5"):
            return "hdf5"
        else:
            return "unknown"
