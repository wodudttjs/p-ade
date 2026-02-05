"""
Storage & Cloud Sync Tests

MODULE 5 테스트
- FR-5.1: Cloud Upload Manager
- FR-5.2: Metadata Database  
- FR-5.3: Version Control
- FR-5.4: Cost Optimization
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FR-5.1: Cloud Upload Manager Tests
# =============================================================================

class TestStorageProviderBase:
    """StorageProvider 베이스 클래스 테스트"""
    
    def test_storage_class_enum(self):
        """StorageClass 열거형 테스트"""
        from storage.providers.base import StorageClass
        
        assert StorageClass.STANDARD.value == "STANDARD"
        assert StorageClass.GLACIER.value == "GLACIER"
        assert StorageClass.NEARLINE.value == "NEARLINE"
        
    def test_upload_status_enum(self):
        """UploadStatus 열거형 테스트"""
        from storage.providers.base import UploadStatus
        
        assert UploadStatus.PENDING.value == "pending"
        assert UploadStatus.COMPLETED.value == "completed"
        assert UploadStatus.FAILED.value == "failed"
        
    def test_multipart_config(self):
        """MultipartConfig 기본값 테스트"""
        from storage.providers.base import MultipartConfig
        
        config = MultipartConfig()
        assert config.threshold_bytes == 5 * 1024 * 1024  # 5MB
        assert config.part_size_bytes == 8 * 1024 * 1024  # 8MB
        assert config.max_concurrency == 10
        
    def test_upload_result_dataclass(self):
        """UploadResult 데이터클래스 테스트"""
        from storage.providers.base import UploadResult, UploadStatus
        
        result = UploadResult(
            provider="s3",
            bucket="test-bucket",
            key="test/file.npz",
            uri="s3://test-bucket/test/file.npz",
            status=UploadStatus.COMPLETED,
        )
        
        assert result.provider == "s3"
        assert result.bucket == "test-bucket"
        assert result.status == UploadStatus.COMPLETED
        
    def test_object_info_dataclass(self):
        """ObjectInfo 데이터클래스 테스트"""
        from storage.providers.base import ObjectInfo
        
        info = ObjectInfo(
            key="test/file.npz",
            bucket="test-bucket",
            size_bytes=1024,
        )
        
        assert info.key == "test/file.npz"
        assert info.size_bytes == 1024
        assert info.metadata == {}  # 기본값


class TestS3Provider:
    """S3 Provider 테스트"""
    
    def test_provider_init(self):
        """S3Provider 초기화 테스트"""
        from storage.providers.s3_provider import S3Provider
        
        provider = S3Provider(
            region="ap-northeast-2",
            endpoint_url="http://localhost:4566",  # LocalStack
        )
        
        assert provider.region == "ap-northeast-2"
        assert provider.provider_name == "s3"
        
    def test_get_uri(self):
        """URI 생성 테스트"""
        from storage.providers.s3_provider import S3Provider
        
        provider = S3Provider()
        uri = provider.get_uri("my-bucket", "path/to/file.npz")
        
        assert uri == "s3://my-bucket/path/to/file.npz"
        
    @patch("storage.providers.s3_provider.S3Provider.client")
    def test_upload_file_not_found(self, mock_client):
        """존재하지 않는 파일 업로드 테스트"""
        from storage.providers.s3_provider import S3Provider
        from storage.providers.base import UploadStatus, ErrorType
        
        provider = S3Provider()
        
        result = provider.upload_file(
            local_path="/nonexistent/file.npz",
            remote_key="test/file.npz",
            bucket="test-bucket",
        )
        
        assert result.status == UploadStatus.FAILED
        assert result.error_type == ErrorType.NOT_FOUND


class TestGCSProvider:
    """GCS Provider 테스트"""
    
    def test_provider_init(self):
        """GCSProvider 초기화 테스트"""
        from storage.providers.gcs_provider import GCSProvider
        
        provider = GCSProvider(
            project_id="test-project",
            location="asia-northeast3",
        )
        
        assert provider.location == "asia-northeast3"
        assert provider.provider_name == "gcs"
        
    def test_get_uri(self):
        """URI 생성 테스트"""
        from storage.providers.gcs_provider import GCSProvider
        
        provider = GCSProvider()
        uri = provider.get_uri("my-bucket", "path/to/file.npz")
        
        assert uri == "gcs://my-bucket/path/to/file.npz"


class TestUploadManager:
    """UploadManager 테스트"""
    
    def test_manager_init(self):
        """UploadManager 초기화 테스트"""
        from storage.upload_manager import UploadManager
        
        manager = UploadManager(
            default_provider="s3",
            max_workers=4,
        )
        
        assert manager.default_provider == "s3"
        assert manager.max_workers == 4
        
    def test_get_provider(self):
        """프로바이더 가져오기 테스트"""
        from storage.upload_manager import UploadManager
        from storage.providers.s3_provider import S3Provider
        
        manager = UploadManager()
        provider = manager.get_provider("s3")
        
        assert isinstance(provider, S3Provider)
        
    def test_queue_priority(self):
        """큐 우선순위 테스트"""
        from storage.upload_manager import QueuePriority
        
        assert QueuePriority.HIGH.value < QueuePriority.NORMAL.value
        assert QueuePriority.NORMAL.value < QueuePriority.LOW.value
        
    def test_upload_job_comparison(self):
        """UploadJob 비교 테스트"""
        from storage.upload_manager import UploadJob, QueuePriority
        
        high_job = UploadJob(
            job_id="1",
            local_path="/test.npz",
            remote_key="test.npz",
            bucket="bucket",
            priority=QueuePriority.HIGH,
        )
        
        low_job = UploadJob(
            job_id="2",
            local_path="/test2.npz",
            remote_key="test2.npz",
            bucket="bucket",
            priority=QueuePriority.LOW,
        )
        
        assert high_job < low_job
        
    def test_batch_progress(self):
        """BatchProgress 테스트"""
        from storage.upload_manager import BatchProgress
        
        batch = BatchProgress(
            batch_id="test-batch",
            total_files=10,
            completed=5,
            failed=1,
        )
        
        assert batch.success_rate == 50.0
        assert not batch.is_complete


# =============================================================================
# FR-5.2: Metadata Database Tests
# =============================================================================

class TestDatabaseManager:
    """DatabaseManager 테스트"""
    
    def test_manager_init(self):
        """DatabaseManager 초기화 테스트"""
        from db.crud import DatabaseManager
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
            
        try:
            manager = DatabaseManager(f"sqlite:///{db_path}")
            manager.create_tables()
            
            assert manager.engine is not None
            
        finally:
            manager.close()
            os.unlink(db_path)
            
    def test_session_creation(self):
        """세션 생성 테스트"""
        from db.crud import DatabaseManager
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
            
        try:
            manager = DatabaseManager(f"sqlite:///{db_path}")
            manager.create_tables()
            
            session = manager.get_session()
            assert session is not None
            session.close()
            
        finally:
            manager.close()
            os.unlink(db_path)


class TestCloudFileCRUD:
    """CloudFile CRUD 테스트"""
    
    @pytest.fixture
    def db_session(self):
        """테스트용 DB 세션"""
        from db.crud import DatabaseManager
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
            
        manager = DatabaseManager(f"sqlite:///{db_path}")
        manager.create_tables()
        
        session = manager.get_session()
        yield session
        
        session.close()
        manager.close()
        os.unlink(db_path)
        
    def test_create_cloud_file(self, db_session):
        """CloudFile 생성 테스트"""
        from db.crud import CloudFileCRUD
        
        crud = CloudFileCRUD(db_session)
        
        cloud_file = crud.create(
            file_name="episode_001.npz",
            file_type="episode_npz",
            file_size_bytes=1024 * 1024,
            sha256="abc123" * 10 + "abcd",
            provider="s3",
            bucket="test-bucket",
            key="episodes/episode_001.npz",
            uri="s3://test-bucket/episodes/episode_001.npz",
        )
        
        assert cloud_file.id is not None
        assert cloud_file.file_name == "episode_001.npz"
        assert cloud_file.status == "uploaded"
        
    def test_get_by_sha256(self, db_session):
        """SHA256으로 조회 테스트 (중복 체크)"""
        from db.crud import CloudFileCRUD
        
        crud = CloudFileCRUD(db_session)
        sha256 = "def456" * 10 + "defg"
        
        # 생성
        crud.create(
            file_name="test.npz",
            file_type="episode_npz",
            file_size_bytes=1024,
            sha256=sha256,
            provider="s3",
            bucket="bucket",
            key="test.npz",
            uri="s3://bucket/test.npz",
        )
        
        # 조회
        result = crud.get_by_sha256(sha256)
        assert result is not None
        assert result.sha256 == sha256
        
    def test_get_stats(self, db_session):
        """통계 조회 테스트"""
        from db.crud import CloudFileCRUD
        
        crud = CloudFileCRUD(db_session)
        
        # 파일 2개 생성
        crud.create(
            file_name="file1.npz",
            file_type="episode_npz",
            file_size_bytes=1000,
            sha256="a" * 64,
            provider="s3",
            bucket="bucket",
            key="file1.npz",
            uri="s3://bucket/file1.npz",
        )
        
        crud.create(
            file_name="file2.npz",
            file_type="episode_npz",
            file_size_bytes=2000,
            sha256="b" * 64,
            provider="s3",
            bucket="bucket",
            key="file2.npz",
            uri="s3://bucket/file2.npz",
        )
        
        stats = crud.get_stats()
        
        assert stats["total_files"] == 2
        assert stats["total_bytes"] == 3000


class TestUploadTaskCRUD:
    """UploadTask CRUD 테스트"""
    
    @pytest.fixture
    def db_session(self):
        """테스트용 DB 세션"""
        from db.crud import DatabaseManager
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
            
        manager = DatabaseManager(f"sqlite:///{db_path}")
        manager.create_tables()
        
        session = manager.get_session()
        yield session
        
        session.close()
        manager.close()
        os.unlink(db_path)
        
    def test_create_upload_task(self, db_session):
        """UploadTask 생성 테스트"""
        from db.crud import UploadTaskCRUD
        
        crud = UploadTaskCRUD(db_session)
        
        task = crud.create(
            task_id="task-123",
            task_type="upload_file",
            local_path="/local/file.npz",
            remote_key="remote/file.npz",
            bucket="test-bucket",
            provider="s3",
        )
        
        assert task.task_id == "task-123"
        assert task.status == "pending"
        
    def test_get_pending_tasks(self, db_session):
        """대기 중인 태스크 조회 테스트"""
        from db.crud import UploadTaskCRUD
        
        crud = UploadTaskCRUD(db_session)
        
        # 태스크 생성
        crud.create(
            task_id="task-1",
            task_type="upload_file",
            local_path="/file1.npz",
            remote_key="file1.npz",
            bucket="bucket",
            provider="s3",
            priority=1,  # HIGH
        )
        
        crud.create(
            task_id="task-2",
            task_type="upload_file",
            local_path="/file2.npz",
            remote_key="file2.npz",
            bucket="bucket",
            provider="s3",
            priority=3,  # LOW
        )
        
        pending = crud.get_pending()
        
        assert len(pending) == 2
        assert pending[0].task_id == "task-1"  # HIGH 우선순위 먼저


# =============================================================================
# FR-5.3: Version Control Tests
# =============================================================================

class TestSemanticVersion:
    """SemanticVersion 테스트"""
    
    def test_parse_version(self):
        """버전 파싱 테스트"""
        from versioning.version_manager import SemanticVersion
        
        v1 = SemanticVersion.parse("v1.2.3")
        assert v1.major == 1
        assert v1.minor == 2
        assert v1.patch == 3
        
        v2 = SemanticVersion.parse("2.0.0")
        assert v2.major == 2
        
    def test_version_string(self):
        """버전 문자열 변환 테스트"""
        from versioning.version_manager import SemanticVersion
        
        v = SemanticVersion(1, 2, 3)
        assert str(v) == "v1.2.3"
        
    def test_version_bump(self):
        """버전 증가 테스트"""
        from versioning.version_manager import SemanticVersion, VersionBump
        
        v = SemanticVersion(1, 2, 3)
        
        # Patch bump
        v_patch = v.bump(VersionBump.PATCH)
        assert str(v_patch) == "v1.2.4"
        
        # Minor bump
        v_minor = v.bump(VersionBump.MINOR)
        assert str(v_minor) == "v1.3.0"
        
        # Major bump
        v_major = v.bump(VersionBump.MAJOR)
        assert str(v_major) == "v2.0.0"
        
    def test_version_comparison(self):
        """버전 비교 테스트"""
        from versioning.version_manager import SemanticVersion
        
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)
        v3 = SemanticVersion(2, 0, 0)
        
        assert v1 < v2
        assert v2 < v3
        assert v1 == SemanticVersion(1, 0, 0)


class TestDatasetManifest:
    """DatasetManifest 테스트"""
    
    def test_manifest_creation(self):
        """매니페스트 생성 테스트"""
        from versioning.version_manager import DatasetManifest
        
        manifest = DatasetManifest(
            version="v1.0.0",
            created_at=datetime.utcnow().isoformat(),
            description="Test dataset",
        )
        
        assert manifest.version == "v1.0.0"
        assert manifest.total_files == 0
        
    def test_add_file(self):
        """파일 추가 테스트"""
        from versioning.version_manager import DatasetManifest, FileEntry
        
        manifest = DatasetManifest(
            version="v1.0.0",
            created_at=datetime.utcnow().isoformat(),
        )
        
        entry = FileEntry(
            key="episodes/ep001.npz",
            sha256="abc" * 21 + "a",
            size_bytes=1024,
            file_type="episode_npz",
        )
        
        manifest.add_file(entry)
        
        assert manifest.total_files == 1
        assert manifest.total_bytes == 1024
        
    def test_manifest_serialization(self):
        """매니페스트 직렬화 테스트"""
        from versioning.version_manager import DatasetManifest, FileEntry
        
        manifest = DatasetManifest(
            version="v1.0.0",
            created_at="2024-01-01T00:00:00",
            provider="s3",
            bucket="test-bucket",
        )
        
        manifest.add_file(FileEntry(
            key="test.npz",
            sha256="x" * 64,
            size_bytes=100,
            file_type="episode_npz",
        ))
        
        # 딕셔너리 변환
        data = manifest.to_dict()
        
        assert data["version"] == "v1.0.0"
        assert len(data["files"]) == 1
        
        # 역직렬화
        restored = DatasetManifest.from_dict(data)
        
        assert restored.version == manifest.version
        assert len(restored.files) == 1
        
    def test_manifest_checksum(self):
        """매니페스트 체크섬 테스트"""
        from versioning.version_manager import DatasetManifest, FileEntry
        
        manifest = DatasetManifest(
            version="v1.0.0",
            created_at="2024-01-01T00:00:00",
        )
        
        manifest.add_file(FileEntry(
            key="test.npz",
            sha256="abc" * 21 + "a",
            size_bytes=100,
            file_type="episode_npz",
        ))
        
        checksum = manifest.compute_checksum()
        
        assert checksum is not None
        assert len(checksum) == 64  # SHA256


class TestVersionManager:
    """VersionManager 테스트"""
    
    @pytest.fixture
    def version_manager(self):
        """테스트용 VersionManager"""
        from versioning.version_manager import VersionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = VersionManager(
                manifests_dir=tmpdir,
                provider="s3",
                bucket="test-bucket",
            )
            yield manager
            
    def test_create_new_version(self, version_manager):
        """새 버전 생성 테스트"""
        from versioning.version_manager import VersionBump
        
        manifest = version_manager.create_new_version(
            bump_type=VersionBump.MAJOR,
            description="Initial version",
        )
        
        assert manifest.version == "v1.0.0"
        assert manifest.parent_version is None
        
    def test_version_bump_chain(self, version_manager):
        """연속 버전 증가 테스트"""
        from versioning.version_manager import VersionBump
        
        # v1.0.0
        v1 = version_manager.create_new_version(VersionBump.MAJOR)
        version_manager.save_manifest(v1)
        
        # v1.1.0
        v2 = version_manager.create_new_version(VersionBump.MINOR)
        version_manager.save_manifest(v2)
        
        # v1.1.1
        v3 = version_manager.create_new_version(VersionBump.PATCH)
        
        assert v3.version == "v1.1.1"
        assert v3.parent_version == "v1.1.0"
        
    def test_list_versions(self, version_manager):
        """버전 목록 테스트"""
        from versioning.version_manager import VersionBump
        
        version_manager.create_new_version(VersionBump.MAJOR)
        version_manager.create_new_version(VersionBump.MINOR)
        
        versions = version_manager.list_versions()
        
        assert len(versions) == 2
        assert "v1.0.0" in versions
        assert "v1.1.0" in versions


class TestVersionDiff:
    """버전 비교 테스트"""
    
    @pytest.fixture
    def version_manager(self):
        """테스트용 VersionManager"""
        from versioning.version_manager import VersionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = VersionManager(manifests_dir=tmpdir)
            yield manager
            
    def test_compare_versions(self, version_manager):
        """버전 비교 테스트"""
        from versioning.version_manager import VersionBump, FileEntry
        
        # v1.0.0 - 파일 2개
        v1 = version_manager.create_new_version(VersionBump.MAJOR)
        v1.add_file(FileEntry(key="file1.npz", sha256="a"*64, size_bytes=100, file_type="npz"))
        v1.add_file(FileEntry(key="file2.npz", sha256="b"*64, size_bytes=200, file_type="npz"))
        version_manager.save_manifest(v1)
        
        # v1.1.0 - 파일 추가, 수정
        v2 = version_manager.create_new_version(VersionBump.MINOR)
        v2.add_file(FileEntry(key="file1.npz", sha256="c"*64, size_bytes=150, file_type="npz"))  # 수정
        v2.add_file(FileEntry(key="file3.npz", sha256="d"*64, size_bytes=300, file_type="npz"))  # 추가
        version_manager.save_manifest(v2)
        
        # 비교
        diff = version_manager.compare_versions("v1.0.0", "v1.1.0")
        
        assert diff is not None
        assert len(diff.added_files) == 1  # file3
        assert len(diff.removed_files) == 1  # file2
        assert len(diff.modified_files) == 1  # file1


# =============================================================================
# FR-5.4: Cost Optimization Tests
# =============================================================================

class TestCostManager:
    """CostManager 테스트"""
    
    def test_estimate_cost(self):
        """비용 추정 테스트"""
        from cost.cost_manager import CostManager
        
        manager = CostManager(provider="s3")
        
        estimate = manager.estimate_cost(
            storage_bytes=10 * 1024 ** 3,  # 10GB
            storage_class="STANDARD",
            put_requests=1000,
            get_requests=10000,
        )
        
        assert estimate.storage_gb == 10.0
        assert estimate.storage_cost > 0
        assert estimate.total_cost > 0
        
    def test_recommend_storage_class(self):
        """스토리지 클래스 추천 테스트"""
        from cost.cost_manager import CostManager, AccessPattern
        
        manager = CostManager(provider="s3")
        
        hot = manager.recommend_storage_class(AccessPattern.HOT)
        assert hot == "STANDARD"
        
        cold = manager.recommend_storage_class(AccessPattern.COLD, retention_days=365)
        assert cold == "GLACIER"
        
        archive = manager.recommend_storage_class(AccessPattern.ARCHIVE)
        assert archive == "DEEP_ARCHIVE"
        
    def test_generate_lifecycle_rules(self):
        """라이프사이클 규칙 생성 테스트"""
        from cost.cost_manager import CostManager
        
        manager = CostManager(provider="s3")
        
        rules = manager.generate_lifecycle_rules(
            prefix="datasets/",
            hot_days=30,
            warm_days=90,
            cold_days=365,
        )
        
        assert len(rules) >= 3
        assert any("glacier" in r.id for r in rules)
        
    def test_analyze_optimization(self):
        """최적화 분석 테스트"""
        from cost.cost_manager import CostManager, AccessPattern
        
        manager = CostManager(provider="s3")
        
        files = [
            {"size_bytes": 1024 ** 3, "storage_class": "STANDARD"},
            {"size_bytes": 2 * 1024 ** 3, "storage_class": "STANDARD"},
        ]
        
        recommendation = manager.analyze_optimization(
            files=files,
            access_pattern=AccessPattern.COLD,
        )
        
        assert recommendation.current_cost > 0
        assert recommendation.savings >= 0
        assert len(recommendation.recommendations) > 0


class TestCompressionManager:
    """CompressionManager 테스트"""
    
    def test_should_compress(self):
        """압축 필요 여부 판단 테스트"""
        from cost.cost_manager import CompressionManager
        
        manager = CompressionManager()
        
        # 이미 압축된 파일
        assert not manager.should_compress("file.gz")
        assert not manager.should_compress("file.zip")
        
        # 미디어 파일
        assert not manager.should_compress("video.mp4")
        assert not manager.should_compress("image.jpg")
        
    def test_compress_gzip(self):
        """gzip 압축 테스트"""
        from cost.cost_manager import CompressionManager, CompressionType
        
        manager = CompressionManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 테스트 파일 생성
            input_path = os.path.join(tmpdir, "test.txt")
            with open(input_path, "w") as f:
                f.write("Hello World! " * 1000)
                
            # 압축
            output_path, original, compressed, ratio = manager.compress(
                input_path,
                compression_type=CompressionType.GZIP,
            )
            
            assert output_path.endswith(".gz")
            assert compressed < original
            assert ratio > 1.0
            
    def test_decompress_gzip(self):
        """gzip 압축 해제 테스트"""
        from cost.cost_manager import CompressionManager, CompressionType
        
        manager = CompressionManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 테스트 파일 생성
            input_path = os.path.join(tmpdir, "test.txt")
            original_content = "Hello World! " * 100
            with open(input_path, "w") as f:
                f.write(original_content)
                
            # 압축
            compressed_path, _, _, _ = manager.compress(input_path)
            
            # 압축 해제
            decompressed_path = manager.decompress(compressed_path)
            
            with open(decompressed_path, "r") as f:
                content = f.read()
                
            assert content == original_content


# =============================================================================
# Integration Tests
# =============================================================================

class TestStorageIntegration:
    """스토리지 통합 테스트"""
    
    def test_upload_with_versioning(self):
        """업로드와 버전 관리 통합 테스트"""
        from storage.upload_manager import UploadManager
        from versioning.version_manager import VersionManager, VersionBump, FileEntry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # VersionManager 생성
            vm = VersionManager(manifests_dir=tmpdir)
            
            # 새 버전 생성
            manifest = vm.create_new_version(VersionBump.MAJOR, "Test version")
            
            # 파일 엔트리 추가 (실제 업로드 시뮬레이션)
            manifest.add_file(FileEntry(
                key="episodes/ep001.npz",
                sha256="test_hash" * 8,
                size_bytes=1024 * 1024,
                file_type="episode_npz",
            ))
            
            # 저장
            vm.save_manifest(manifest)
            
            # 검증
            loaded = vm.get_manifest("v1.0.0")
            assert loaded is not None
            assert loaded.total_files == 1
            
    def test_cost_aware_storage_class(self):
        """비용 인식 스토리지 클래스 선택 테스트"""
        from cost.cost_manager import CostManager, AccessPattern
        from storage.providers.base import LifecycleRule
        
        manager = CostManager(provider="s3")
        
        # 콜드 데이터는 GLACIER 추천
        recommended = manager.recommend_storage_class(AccessPattern.COLD)
        
        # 라이프사이클 규칙 생성
        rules = manager.generate_lifecycle_rules(
            prefix="old-data/",
            hot_days=30,
            cold_days=90,
        )
        
        assert recommended in ["GLACIER", "STANDARD_IA"]
        assert len(rules) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
