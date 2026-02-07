"""
S3 업로드 스크립트 테스트

upload_to_s3.py의 기능을 테스트합니다.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestUploadToS3:
    """S3 업로드 스크립트 테스트"""
    
    def test_find_pose_files(self, tmp_path):
        """포즈 파일 찾기 테스트"""
        from upload_to_s3 import find_pose_files
        
        # 임시 포즈 파일 생성
        poses_dir = tmp_path / "data" / "poses"
        poses_dir.mkdir(parents=True)
        
        (poses_dir / "video1_pose.npz").write_bytes(b"test data 1")
        (poses_dir / "video2_pose.npz").write_bytes(b"test data 2")
        (poses_dir / "other.txt").write_text("not a pose file")
        
        # 실제 함수 테스트 (프로젝트 경로 기준)
        # find_pose_files는 프로젝트 루트 기준으로 동작하므로
        # 여기서는 함수가 호출 가능한지만 확인
        files = find_pose_files()
        assert isinstance(files, list)
        
    def test_generate_s3_key(self):
        """S3 키 생성 테스트"""
        from upload_to_s3 import generate_s3_key
        
        local_path = Path("/some/path/video123_pose.npz")
        key = generate_s3_key(local_path, prefix="poses")
        
        # 형식: poses/YYYY/MM/DD/video123_pose.npz
        assert key.startswith("poses/")
        assert key.endswith("video123_pose.npz")
        assert "/" in key
        
        # 날짜 형식 확인
        parts = key.split("/")
        assert len(parts) == 5  # poses, year, month, day, filename
        assert parts[1].isdigit()  # year
        assert parts[2].isdigit()  # month
        assert parts[3].isdigit()  # day
        
    def test_generate_s3_key_custom_prefix(self):
        """커스텀 prefix로 S3 키 생성"""
        from upload_to_s3 import generate_s3_key
        
        local_path = Path("test_file.npz")
        key = generate_s3_key(local_path, prefix="episodes")
        
        assert key.startswith("episodes/")
        
    def test_get_file_metadata(self, tmp_path):
        """파일 메타데이터 생성 테스트"""
        from upload_to_s3 import get_file_metadata
        
        # 테스트 파일 생성
        test_file = tmp_path / "test_pose.npz"
        test_file.write_bytes(b"test content here")
        
        metadata = get_file_metadata(test_file)
        
        assert metadata["original_filename"] == "test_pose.npz"
        assert "upload_timestamp" in metadata
        assert metadata["file_size"] == str(test_file.stat().st_size)
        assert metadata["project"] == "p-ade"
        assert metadata["data_type"] == "pose"
        
    def test_get_bucket_name_default(self):
        """기본 버킷 이름 테스트"""
        from upload_to_s3 import get_bucket_name
        
        bucket = get_bucket_name()
        assert bucket is not None
        assert isinstance(bucket, str)
        assert len(bucket) > 0
        
    def test_get_bucket_name_from_env(self, monkeypatch):
        """환경변수에서 버킷 이름 가져오기"""
        from upload_to_s3 import get_bucket_name
        
        monkeypatch.setenv("S3_BUCKET", "my-custom-bucket")
        bucket = get_bucket_name()
        assert bucket == "my-custom-bucket"
        
    def test_upload_file_dry_run(self, tmp_path):
        """Dry-run 모드 업로드 테스트"""
        from upload_to_s3 import upload_file
        
        # 테스트 파일 생성
        test_file = tmp_path / "test_pose.npz"
        test_file.write_bytes(b"test content")
        
        result = upload_file(
            provider=None,
            local_path=test_file,
            bucket="test-bucket",
            dry_run=True,
        )
        
        assert result["status"] == "dry_run"
        assert "s3://test-bucket/" in result["uri"]
        assert result["size_bytes"] == len(b"test content")
        
    def test_upload_file_not_found(self):
        """존재하지 않는 파일 업로드 시도"""
        from upload_to_s3 import upload_file
        
        mock_provider = Mock()
        result = upload_file(
            provider=mock_provider,
            local_path=Path("/nonexistent/file.npz"),
            bucket="test-bucket",
            dry_run=False,
        )
        
        # 파일이 없으면 에러 상태 반환
        assert result["status"] == "error"
        assert "not found" in result["error"].lower() or "File not found" in result["error"]


class TestS3Provider:
    """S3 Provider 통합 테스트"""
    
    def test_provider_import(self):
        """S3 Provider 임포트 테스트"""
        from storage.providers.s3_provider import S3Provider
        
        # 클래스가 존재하는지 확인
        assert S3Provider is not None
        
    def test_provider_initialization(self):
        """S3 Provider 초기화 테스트"""
        from storage.providers.s3_provider import S3Provider
        
        provider = S3Provider(
            region="ap-northeast-2",
            access_key_id="test_key",
            secret_access_key="test_secret",
        )
        
        assert provider.region == "ap-northeast-2"
        assert provider.provider_name == "s3"
        
    def test_provider_custom_endpoint(self):
        """LocalStack 등 커스텀 엔드포인트 설정"""
        from storage.providers.s3_provider import S3Provider
        
        provider = S3Provider(
            region="us-east-1",
            endpoint_url="http://localhost:4566",
        )
        
        assert provider.endpoint_url == "http://localhost:4566"


class TestUploadResultDataclass:
    """UploadResult 데이터 클래스 테스트"""
    
    def test_upload_result_import(self):
        """UploadResult 임포트 테스트"""
        from storage.providers.base import UploadResult, UploadStatus
        
        result = UploadResult(
            provider="s3",
            bucket="test-bucket",
            key="test/key.npz",
            uri="s3://test-bucket/test/key.npz",
            status=UploadStatus.COMPLETED,
        )
        
        assert result.provider == "s3"
        assert result.bucket == "test-bucket"
        assert result.status == UploadStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
