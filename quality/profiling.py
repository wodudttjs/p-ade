"""
Data Profiling

통계적 데이터 프로파일링
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import math

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class ColumnType(str, Enum):
    """컬럼 유형"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass
class NumericStats:
    """수치형 통계"""
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    zeros: int = 0
    negatives: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "zeros": self.zeros,
            "negatives": self.negatives,
        }


@dataclass
class CategoricalStats:
    """범주형 통계"""
    count: int = 0
    unique_count: int = 0
    top_value: Optional[str] = None
    top_frequency: int = 0
    value_counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "unique_count": self.unique_count,
            "top_value": self.top_value,
            "top_frequency": self.top_frequency,
            "value_counts": dict(list(self.value_counts.items())[:10]),  # top 10
        }


@dataclass
class TextStats:
    """텍스트 통계"""
    count: int = 0
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    empty_count: int = 0
    unique_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "avg_length": self.avg_length,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "empty_count": self.empty_count,
            "unique_count": self.unique_count,
        }


@dataclass
class ColumnProfile:
    """컬럼 프로파일"""
    name: str
    column_type: ColumnType
    dtype: str
    total_count: int = 0
    null_count: int = 0
    null_ratio: float = 0.0
    numeric_stats: Optional[NumericStats] = None
    categorical_stats: Optional[CategoricalStats] = None
    text_stats: Optional[TextStats] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "column_type": self.column_type.value,
            "dtype": self.dtype,
            "total_count": self.total_count,
            "null_count": self.null_count,
            "null_ratio": self.null_ratio,
        }
        
        if self.numeric_stats:
            result["numeric_stats"] = self.numeric_stats.to_dict()
        if self.categorical_stats:
            result["categorical_stats"] = self.categorical_stats.to_dict()
        if self.text_stats:
            result["text_stats"] = self.text_stats.to_dict()
        
        return result


@dataclass
class DatasetProfile:
    """데이터셋 프로파일"""
    name: str
    row_count: int = 0
    column_count: int = 0
    columns: List[ColumnProfile] = field(default_factory=list)
    memory_usage_bytes: int = 0
    generated_at: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_usage_bytes": self.memory_usage_bytes,
            "generated_at": self.generated_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "columns": [c.to_dict() for c in self.columns],
        }
    
    def get_column(self, name: str) -> Optional[ColumnProfile]:
        """컬럼 프로파일 조회"""
        for col in self.columns:
            if col.name == name:
                return col
        return None


class DataProfiler:
    """
    데이터 프로파일러
    
    Task 6.3.2: 통계적 데이터 프로파일링
    """
    
    def __init__(
        self,
        max_unique_categories: int = 50,
        sample_size: Optional[int] = None,
    ):
        self.max_unique_categories = max_unique_categories
        self.sample_size = sample_size
    
    def profile(self, data: Any, name: str = "dataset") -> DatasetProfile:
        """데이터 프로파일링"""
        import time
        start = time.time()
        
        try:
            import pandas as pd
            
            if not isinstance(data, pd.DataFrame):
                logger.warning("Data is not a DataFrame, creating empty profile")
                return DatasetProfile(name=name)
            
            df = data
            
            # 샘플링
            if self.sample_size and len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)
            
            profile = DatasetProfile(
                name=name,
                row_count=len(data),  # 원본 크기
                column_count=len(data.columns),
                memory_usage_bytes=data.memory_usage(deep=True).sum(),
            )
            
            for col_name in df.columns:
                col_profile = self._profile_column(df, col_name)
                profile.columns.append(col_profile)
            
            profile.duration_seconds = time.time() - start
            
            logger.info(
                f"Profiled '{name}': {profile.row_count} rows, "
                f"{profile.column_count} columns in {profile.duration_seconds:.2f}s"
            )
            
            return profile
            
        except ImportError:
            logger.warning("pandas not installed")
            return DatasetProfile(name=name)
        except Exception as e:
            logger.error(f"Profiling error: {e}")
            return DatasetProfile(name=name)
    
    def _profile_column(self, df, col_name: str) -> ColumnProfile:
        """컬럼 프로파일링"""
        import pandas as pd
        import numpy as np
        
        col = df[col_name]
        dtype = str(col.dtype)
        total = len(col)
        null_count = col.isna().sum()
        null_ratio = null_count / total if total > 0 else 0.0
        
        # 타입 판정
        column_type = self._detect_column_type(col)
        
        profile = ColumnProfile(
            name=col_name,
            column_type=column_type,
            dtype=dtype,
            total_count=total,
            null_count=int(null_count),
            null_ratio=null_ratio,
        )
        
        # 유효 데이터
        valid = col.dropna()
        
        if len(valid) == 0:
            return profile
        
        # 타입별 통계
        if column_type == ColumnType.NUMERIC:
            profile.numeric_stats = self._numeric_stats(valid)
        elif column_type == ColumnType.CATEGORICAL:
            profile.categorical_stats = self._categorical_stats(valid)
        elif column_type == ColumnType.TEXT:
            profile.text_stats = self._text_stats(valid)
        elif column_type == ColumnType.BOOLEAN:
            profile.categorical_stats = self._categorical_stats(valid.astype(str))
        
        return profile
    
    def _detect_column_type(self, col) -> ColumnType:
        """컬럼 타입 자동 감지"""
        import pandas as pd
        import numpy as np
        
        dtype = col.dtype
        
        # 수치형
        if pd.api.types.is_numeric_dtype(dtype):
            return ColumnType.NUMERIC
        
        # 날짜형
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnType.DATETIME
        
        # 불린
        if pd.api.types.is_bool_dtype(dtype):
            return ColumnType.BOOLEAN
        
        # 문자열/객체
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            # 고유값 비율로 categorical vs text 판단
            valid = col.dropna()
            if len(valid) > 0:
                unique_ratio = valid.nunique() / len(valid)
                if unique_ratio < 0.5 and valid.nunique() <= self.max_unique_categories:
                    return ColumnType.CATEGORICAL
                else:
                    return ColumnType.TEXT
        
        return ColumnType.UNKNOWN
    
    def _numeric_stats(self, col) -> NumericStats:
        """수치형 통계 계산"""
        import numpy as np
        
        try:
            from scipy import stats as scipy_stats
            skew = float(scipy_stats.skew(col))
            kurt = float(scipy_stats.kurtosis(col))
        except ImportError:
            skew = 0.0
            kurt = 0.0
        except Exception:
            skew = 0.0
            kurt = 0.0
        
        q1 = float(col.quantile(0.25))
        q3 = float(col.quantile(0.75))
        
        return NumericStats(
            count=int(len(col)),
            mean=float(col.mean()),
            std=float(col.std()),
            min=float(col.min()),
            max=float(col.max()),
            median=float(col.median()),
            q1=q1,
            q3=q3,
            iqr=q3 - q1,
            skewness=skew,
            kurtosis=kurt,
            zeros=int((col == 0).sum()),
            negatives=int((col < 0).sum()),
        )
    
    def _categorical_stats(self, col) -> CategoricalStats:
        """범주형 통계 계산"""
        value_counts = col.value_counts()
        
        return CategoricalStats(
            count=int(len(col)),
            unique_count=int(col.nunique()),
            top_value=str(value_counts.index[0]) if len(value_counts) > 0 else None,
            top_frequency=int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            value_counts={str(k): int(v) for k, v in value_counts.head(20).items()},
        )
    
    def _text_stats(self, col) -> TextStats:
        """텍스트 통계 계산"""
        lengths = col.astype(str).str.len()
        
        return TextStats(
            count=int(len(col)),
            avg_length=float(lengths.mean()),
            min_length=int(lengths.min()),
            max_length=int(lengths.max()),
            empty_count=int((col.astype(str).str.strip() == "").sum()),
            unique_count=int(col.nunique()),
        )


class ProfileComparator:
    """프로파일 비교기"""
    
    def compare(
        self,
        baseline: DatasetProfile,
        current: DatasetProfile,
    ) -> Dict[str, Any]:
        """두 프로파일 비교"""
        comparison = {
            "baseline": baseline.name,
            "current": current.name,
            "row_count_diff": current.row_count - baseline.row_count,
            "column_count_diff": current.column_count - baseline.column_count,
            "column_changes": [],
        }
        
        baseline_cols = {c.name: c for c in baseline.columns}
        current_cols = {c.name: c for c in current.columns}
        
        # 추가된 컬럼
        added = set(current_cols.keys()) - set(baseline_cols.keys())
        for col in added:
            comparison["column_changes"].append({
                "column": col,
                "change": "added",
            })
        
        # 삭제된 컬럼
        removed = set(baseline_cols.keys()) - set(current_cols.keys())
        for col in removed:
            comparison["column_changes"].append({
                "column": col,
                "change": "removed",
            })
        
        # 공통 컬럼 비교
        common = set(baseline_cols.keys()) & set(current_cols.keys())
        for col in common:
            b = baseline_cols[col]
            c = current_cols[col]
            
            changes = []
            
            # 타입 변경
            if b.column_type != c.column_type:
                changes.append(f"type: {b.column_type.value} -> {c.column_type.value}")
            
            # null 비율 변화
            null_diff = abs(c.null_ratio - b.null_ratio)
            if null_diff > 0.05:  # 5% 이상 변화
                changes.append(f"null_ratio: {b.null_ratio:.2%} -> {c.null_ratio:.2%}")
            
            # 수치형 통계 비교
            if b.numeric_stats and c.numeric_stats:
                b_stats = b.numeric_stats
                c_stats = c.numeric_stats
                
                # 평균 변화
                if b_stats.mean != 0:
                    mean_change = abs((c_stats.mean - b_stats.mean) / b_stats.mean)
                    if mean_change > 0.1:  # 10% 이상 변화
                        changes.append(f"mean: {b_stats.mean:.2f} -> {c_stats.mean:.2f}")
                
                # 표준편차 변화
                if b_stats.std != 0:
                    std_change = abs((c_stats.std - b_stats.std) / b_stats.std)
                    if std_change > 0.2:  # 20% 이상 변화
                        changes.append(f"std: {b_stats.std:.2f} -> {c_stats.std:.2f}")
            
            if changes:
                comparison["column_changes"].append({
                    "column": col,
                    "change": "modified",
                    "details": changes,
                })
        
        return comparison


# 싱글톤 인스턴스
_profiler: Optional[DataProfiler] = None


def get_profiler() -> DataProfiler:
    """프로파일러 싱글톤 반환"""
    global _profiler
    if _profiler is None:
        _profiler = DataProfiler()
    return _profiler


# 편의 함수
def profile_data(data: Any, name: str = "dataset") -> DatasetProfile:
    """데이터 프로파일링 편의 함수"""
    return get_profiler().profile(data, name)
