"""
Cost Manager

스토리지 비용 최적화
- 스토리지 클래스 선택
- 라이프사이클 정책
- 압축 전략
- 비용 추정
"""

import os
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from core.logging_config import setup_logger
from storage.providers.base import (
    StorageClass,
    LifecycleRule,
)

logger = setup_logger(__name__)


class CompressionType(Enum):
    """압축 유형"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


class AccessPattern(Enum):
    """접근 패턴"""
    HOT = "hot"          # 자주 접근 (일간)
    WARM = "warm"        # 가끔 접근 (주간)
    COLD = "cold"        # 드물게 접근 (월간)
    ARCHIVE = "archive"  # 거의 안함 (년간)


@dataclass
class StoragePricing:
    """스토리지 가격 정보 (USD per GB per month)"""
    # S3 ap-northeast-2 기준
    standard: float = 0.025
    intelligent_tiering: float = 0.023
    standard_ia: float = 0.0138
    glacier: float = 0.004
    glacier_deep_archive: float = 0.00099
    
    # GCS asia-northeast3 기준
    gcs_standard: float = 0.023
    gcs_nearline: float = 0.013
    gcs_coldline: float = 0.006
    gcs_archive: float = 0.0025
    
    # API 요청 비용 (per 1000 requests)
    put_request: float = 0.005
    get_request: float = 0.0004
    
    # 데이터 전송 (per GB)
    transfer_out: float = 0.09  # 인터넷으로


@dataclass
class CostEstimate:
    """비용 추정"""
    storage_cost: float = 0.0
    request_cost: float = 0.0
    transfer_cost: float = 0.0
    total_cost: float = 0.0
    
    # 상세
    storage_class: str = "STANDARD"
    storage_gb: float = 0.0
    put_requests: int = 0
    get_requests: int = 0
    transfer_out_gb: float = 0.0
    
    # 기간
    period_months: int = 1


@dataclass
class OptimizationRecommendation:
    """최적화 권장 사항"""
    current_cost: float
    optimized_cost: float
    savings: float
    savings_percent: float
    
    recommendations: List[Dict[str, Any]] = field(default_factory=list)


class CostManager:
    """
    스토리지 비용 관리자
    
    FR-5.4: Cost Optimization
    """
    
    def __init__(
        self,
        provider: str = "s3",
        region: str = "ap-northeast-2",
        pricing: Optional[StoragePricing] = None,
    ):
        """
        CostManager 초기화
        
        Args:
            provider: 클라우드 프로바이더
            region: 리전
            pricing: 가격 정보 (None이면 기본값)
        """
        self.provider = provider
        self.region = region
        self.pricing = pricing or StoragePricing()
        
    def estimate_cost(
        self,
        storage_bytes: int,
        storage_class: str = "STANDARD",
        put_requests: int = 0,
        get_requests: int = 0,
        transfer_out_bytes: int = 0,
        period_months: int = 1,
    ) -> CostEstimate:
        """월간 비용 추정"""
        storage_gb = storage_bytes / (1024 ** 3)
        transfer_out_gb = transfer_out_bytes / (1024 ** 3)
        
        # 스토리지 비용
        storage_rate = self._get_storage_rate(storage_class)
        storage_cost = storage_gb * storage_rate * period_months
        
        # API 요청 비용
        put_cost = (put_requests / 1000) * self.pricing.put_request
        get_cost = (get_requests / 1000) * self.pricing.get_request
        request_cost = put_cost + get_cost
        
        # 데이터 전송 비용
        transfer_cost = transfer_out_gb * self.pricing.transfer_out
        
        return CostEstimate(
            storage_cost=round(storage_cost, 4),
            request_cost=round(request_cost, 4),
            transfer_cost=round(transfer_cost, 4),
            total_cost=round(storage_cost + request_cost + transfer_cost, 4),
            storage_class=storage_class,
            storage_gb=round(storage_gb, 2),
            put_requests=put_requests,
            get_requests=get_requests,
            transfer_out_gb=round(transfer_out_gb, 2),
            period_months=period_months,
        )
    
    def _get_storage_rate(self, storage_class: str) -> float:
        """스토리지 클래스별 요금"""
        if self.provider == "s3":
            rates = {
                "STANDARD": self.pricing.standard,
                "INTELLIGENT_TIERING": self.pricing.intelligent_tiering,
                "STANDARD_IA": self.pricing.standard_ia,
                "GLACIER": self.pricing.glacier,
                "DEEP_ARCHIVE": self.pricing.glacier_deep_archive,
            }
        else:  # gcs
            rates = {
                "STANDARD": self.pricing.gcs_standard,
                "NEARLINE": self.pricing.gcs_nearline,
                "COLDLINE": self.pricing.gcs_coldline,
                "ARCHIVE": self.pricing.gcs_archive,
            }
            
        return rates.get(storage_class, self.pricing.standard)
    
    def recommend_storage_class(
        self,
        access_pattern: AccessPattern,
        retention_days: int = 365,
    ) -> str:
        """접근 패턴 기반 스토리지 클래스 추천"""
        if self.provider == "s3":
            if access_pattern == AccessPattern.HOT:
                return "STANDARD"
            elif access_pattern == AccessPattern.WARM:
                return "INTELLIGENT_TIERING"
            elif access_pattern == AccessPattern.COLD:
                return "STANDARD_IA" if retention_days < 180 else "GLACIER"
            else:
                return "DEEP_ARCHIVE"
        else:  # gcs
            if access_pattern == AccessPattern.HOT:
                return "STANDARD"
            elif access_pattern == AccessPattern.WARM:
                return "NEARLINE"
            elif access_pattern == AccessPattern.COLD:
                return "COLDLINE"
            else:
                return "ARCHIVE"
    
    def generate_lifecycle_rules(
        self,
        prefix: str = "",
        hot_days: int = 30,
        warm_days: int = 90,
        cold_days: int = 365,
        delete_days: Optional[int] = None,
    ) -> List[LifecycleRule]:
        """라이프사이클 규칙 생성"""
        rules = []
        
        if self.provider == "s3":
            # HOT -> INTELLIGENT_TIERING
            rules.append(LifecycleRule(
                id=f"transition-to-it-{hot_days}d",
                prefix=prefix,
                transition_days=hot_days,
                transition_storage_class="INTELLIGENT_TIERING",
                enabled=True,
            ))
            
            # WARM -> STANDARD_IA
            rules.append(LifecycleRule(
                id=f"transition-to-ia-{warm_days}d",
                prefix=prefix,
                transition_days=warm_days,
                transition_storage_class="STANDARD_IA",
                enabled=True,
            ))
            
            # COLD -> GLACIER
            rules.append(LifecycleRule(
                id=f"transition-to-glacier-{cold_days}d",
                prefix=prefix,
                transition_days=cold_days,
                transition_storage_class="GLACIER",
                enabled=True,
            ))
            
        else:  # gcs
            # HOT -> NEARLINE
            rules.append(LifecycleRule(
                id=f"transition-to-nearline-{hot_days}d",
                prefix=prefix,
                transition_days=hot_days,
                transition_storage_class="NEARLINE",
                enabled=True,
            ))
            
            # WARM -> COLDLINE
            rules.append(LifecycleRule(
                id=f"transition-to-coldline-{warm_days}d",
                prefix=prefix,
                transition_days=warm_days,
                transition_storage_class="COLDLINE",
                enabled=True,
            ))
            
            # COLD -> ARCHIVE
            rules.append(LifecycleRule(
                id=f"transition-to-archive-{cold_days}d",
                prefix=prefix,
                transition_days=cold_days,
                transition_storage_class="ARCHIVE",
                enabled=True,
            ))
            
        # 만료 규칙
        if delete_days:
            rules.append(LifecycleRule(
                id=f"delete-after-{delete_days}d",
                prefix=prefix,
                expiration_days=delete_days,
                enabled=True,
            ))
            
        # 비현재 버전 정리
        rules.append(LifecycleRule(
            id="cleanup-noncurrent-versions",
            prefix=prefix,
            noncurrent_version_expiration_days=30,
            enabled=True,
        ))
        
        return rules
    
    def analyze_optimization(
        self,
        files: List[Dict[str, Any]],
        access_pattern: AccessPattern = AccessPattern.COLD,
    ) -> OptimizationRecommendation:
        """
        최적화 분석
        
        Args:
            files: 파일 정보 리스트
                [{"size_bytes": ..., "storage_class": ..., "last_accessed": ...}, ...]
            access_pattern: 예상 접근 패턴
        """
        total_bytes = sum(f["size_bytes"] for f in files)
        
        # 현재 비용 계산
        current_costs = {}
        for f in files:
            storage_class = f.get("storage_class", "STANDARD")
            if storage_class not in current_costs:
                current_costs[storage_class] = 0
            current_costs[storage_class] += f["size_bytes"]
            
        current_total = sum(
            self.estimate_cost(bytes_size, storage_class).storage_cost
            for storage_class, bytes_size in current_costs.items()
        )
        
        # 최적화된 비용 계산
        recommended_class = self.recommend_storage_class(access_pattern)
        optimized_estimate = self.estimate_cost(total_bytes, recommended_class)
        
        savings = current_total - optimized_estimate.storage_cost
        savings_percent = (savings / current_total * 100) if current_total > 0 else 0
        
        recommendations = []
        
        # 압축 권장
        uncompressed_bytes = sum(
            f["size_bytes"] for f in files
            if not f.get("compression")
        )
        if uncompressed_bytes > 0:
            estimated_compressed = uncompressed_bytes * 0.3  # 예상 70% 압축
            compression_savings = self.estimate_cost(
                int(uncompressed_bytes - estimated_compressed),
                recommended_class,
            ).storage_cost
            
            recommendations.append({
                "type": "compression",
                "description": f"압축되지 않은 {uncompressed_bytes / (1024**3):.2f}GB 파일을 gzip 압축하면 약 ${compression_savings:.2f}/월 절약 가능",
                "savings_monthly": compression_savings,
            })
            
        # 스토리지 클래스 변경 권장
        for storage_class, bytes_size in current_costs.items():
            if storage_class == "STANDARD" and access_pattern != AccessPattern.HOT:
                savings_per_class = (
                    self.estimate_cost(bytes_size, "STANDARD").storage_cost -
                    self.estimate_cost(bytes_size, recommended_class).storage_cost
                )
                
                recommendations.append({
                    "type": "storage_class",
                    "description": f"STANDARD 클래스의 {bytes_size / (1024**3):.2f}GB를 {recommended_class}로 변경하면 약 ${savings_per_class:.2f}/월 절약 가능",
                    "from_class": "STANDARD",
                    "to_class": recommended_class,
                    "savings_monthly": savings_per_class,
                })
                
        # 라이프사이클 정책 권장
        recommendations.append({
            "type": "lifecycle",
            "description": "라이프사이클 정책을 적용하여 오래된 데이터를 자동으로 저비용 스토리지로 이동",
            "rules": [r.id for r in self.generate_lifecycle_rules()],
        })
        
        return OptimizationRecommendation(
            current_cost=round(current_total, 2),
            optimized_cost=round(optimized_estimate.storage_cost, 2),
            savings=round(savings, 2),
            savings_percent=round(savings_percent, 1),
            recommendations=recommendations,
        )


class CompressionManager:
    """
    파일 압축 관리자
    
    업로드 전 파일 압축
    """
    
    def __init__(
        self,
        default_type: CompressionType = CompressionType.GZIP,
        compression_level: int = 6,
        min_size_bytes: int = 1024,  # 1KB 이하는 압축 안함
    ):
        self.default_type = default_type
        self.compression_level = compression_level
        self.min_size_bytes = min_size_bytes
        
    def should_compress(
        self,
        file_path: str,
        file_type: Optional[str] = None,
    ) -> bool:
        """압축 필요 여부 판단"""
        path = Path(file_path)
        
        # 이미 압축된 파일
        if path.suffix.lower() in [".gz", ".zip", ".7z", ".lz4", ".zst"]:
            return False
            
        # 미디어 파일 (이미 압축됨)
        if path.suffix.lower() in [".mp4", ".mp3", ".jpg", ".jpeg", ".png", ".webp"]:
            return False
            
        # 크기가 너무 작음
        if path.stat().st_size < self.min_size_bytes:
            return False
            
        return True
    
    def compress(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        compression_type: Optional[CompressionType] = None,
    ) -> Tuple[str, int, int, float]:
        """
        파일 압축
        
        Returns:
            (output_path, original_size, compressed_size, ratio)
        """
        compression_type = compression_type or self.default_type
        input_path = Path(input_path)
        
        if not output_path:
            if compression_type == CompressionType.GZIP:
                output_path = str(input_path) + ".gz"
            elif compression_type == CompressionType.LZ4:
                output_path = str(input_path) + ".lz4"
            elif compression_type == CompressionType.ZSTD:
                output_path = str(input_path) + ".zst"
            else:
                return str(input_path), input_path.stat().st_size, input_path.stat().st_size, 1.0
                
        original_size = input_path.stat().st_size
        
        if compression_type == CompressionType.GZIP:
            self._compress_gzip(str(input_path), output_path)
        elif compression_type == CompressionType.LZ4:
            self._compress_lz4(str(input_path), output_path)
        elif compression_type == CompressionType.ZSTD:
            self._compress_zstd(str(input_path), output_path)
        else:
            shutil.copy(str(input_path), output_path)
            
        compressed_size = Path(output_path).stat().st_size
        ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        logger.info(
            f"Compressed {input_path.name}: "
            f"{original_size} -> {compressed_size} bytes "
            f"(ratio: {ratio:.2f}x)"
        )
        
        return output_path, original_size, compressed_size, ratio
    
    def _compress_gzip(self, input_path: str, output_path: str):
        """gzip 압축"""
        with open(input_path, "rb") as f_in:
            with gzip.open(output_path, "wb", compresslevel=self.compression_level) as f_out:
                shutil.copyfileobj(f_in, f_out)
                
    def _compress_lz4(self, input_path: str, output_path: str):
        """lz4 압축"""
        try:
            import lz4.frame
            
            with open(input_path, "rb") as f_in:
                with lz4.frame.open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except ImportError:
            logger.warning("lz4 not installed, falling back to gzip")
            self._compress_gzip(input_path, output_path.replace(".lz4", ".gz"))
            
    def _compress_zstd(self, input_path: str, output_path: str):
        """zstd 압축"""
        try:
            import zstandard as zstd
            
            cctx = zstd.ZstdCompressor(level=self.compression_level)
            
            with open(input_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    cctx.copy_stream(f_in, f_out)
        except ImportError:
            logger.warning("zstandard not installed, falling back to gzip")
            self._compress_gzip(input_path, output_path.replace(".zst", ".gz"))
    
    def decompress(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """파일 압축 해제"""
        input_path = Path(input_path)
        
        if input_path.suffix.lower() == ".gz":
            if not output_path:
                output_path = str(input_path)[:-3]
            self._decompress_gzip(str(input_path), output_path)
        elif input_path.suffix.lower() == ".lz4":
            if not output_path:
                output_path = str(input_path)[:-4]
            self._decompress_lz4(str(input_path), output_path)
        elif input_path.suffix.lower() == ".zst":
            if not output_path:
                output_path = str(input_path)[:-4]
            self._decompress_zstd(str(input_path), output_path)
        else:
            output_path = str(input_path)
            
        return output_path
    
    def _decompress_gzip(self, input_path: str, output_path: str):
        """gzip 압축 해제"""
        with gzip.open(input_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
                
    def _decompress_lz4(self, input_path: str, output_path: str):
        """lz4 압축 해제"""
        try:
            import lz4.frame
            
            with lz4.frame.open(input_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except ImportError:
            raise ImportError("lz4 not installed")
            
    def _decompress_zstd(self, input_path: str, output_path: str):
        """zstd 압축 해제"""
        try:
            import zstandard as zstd
            
            dctx = zstd.ZstdDecompressor()
            
            with open(input_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    dctx.copy_stream(f_in, f_out)
        except ImportError:
            raise ImportError("zstandard not installed")
