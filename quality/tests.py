"""
Data Quality Tests

데이터 품질 자동 테스트
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import math

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class TestResult(str, Enum):
    """테스트 결과"""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"
    ERROR = "error"


class TestSeverity(str, Enum):
    """테스트 심각도"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class QualityTestResult:
    """품질 테스트 결과"""
    test_name: str
    result: TestResult
    severity: TestSeverity
    message: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "result": self.result.value,
            "severity": self.severity.value,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class QualityReport:
    """품질 테스트 보고서"""
    dataset_name: str
    results: List[QualityTestResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.result == TestResult.PASS)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.result == TestResult.FAIL)
    
    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.result == TestResult.WARN)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 100.0
        return (self.passed / self.total) * 100
    
    @property
    def overall_result(self) -> TestResult:
        if any(r.result == TestResult.FAIL and r.severity == TestSeverity.CRITICAL 
               for r in self.results):
            return TestResult.FAIL
        if any(r.result == TestResult.FAIL for r in self.results):
            return TestResult.WARN
        if all(r.result == TestResult.PASS for r in self.results):
            return TestResult.PASS
        return TestResult.WARN
    
    def add_result(self, result: QualityTestResult):
        self.results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "warnings": self.warnings,
                "success_rate": self.success_rate,
                "overall_result": self.overall_result.value,
            },
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }


class QualityTest:
    """품질 테스트 기본 클래스"""
    
    name: str = "base_test"
    severity: TestSeverity = TestSeverity.MEDIUM
    
    def run(self, data: Any, **kwargs) -> QualityTestResult:
        raise NotImplementedError


class NaNCheckTest(QualityTest):
    """NaN 값 검사 테스트"""
    
    name = "nan_check"
    severity = TestSeverity.HIGH
    
    def __init__(
        self,
        max_nan_ratio: float = 0.0,
        columns: Optional[List[str]] = None,
    ):
        self.max_nan_ratio = max_nan_ratio
        self.columns = columns
    
    def run(self, data: Any, **kwargs) -> QualityTestResult:
        try:
            import pandas as pd
            
            if not isinstance(data, pd.DataFrame):
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.SKIP,
                    severity=self.severity,
                    message="Data is not a DataFrame",
                )
            
            columns = self.columns or data.columns.tolist()
            total_cells = len(data) * len(columns)
            
            if total_cells == 0:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.PASS,
                    severity=self.severity,
                    message="Empty dataset",
                )
            
            nan_count = data[columns].isna().sum().sum()
            nan_ratio = nan_count / total_cells
            
            if nan_ratio > self.max_nan_ratio:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.FAIL,
                    severity=self.severity,
                    message=f"NaN ratio {nan_ratio:.2%} exceeds threshold {self.max_nan_ratio:.2%}",
                    expected=self.max_nan_ratio,
                    actual=nan_ratio,
                    metadata={"nan_count": int(nan_count), "total_cells": total_cells},
                )
            
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.PASS,
                severity=self.severity,
                message=f"NaN ratio {nan_ratio:.2%} within threshold",
                expected=self.max_nan_ratio,
                actual=nan_ratio,
            )
            
        except ImportError:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.SKIP,
                severity=self.severity,
                message="pandas not installed",
            )
        except Exception as e:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.ERROR,
                severity=self.severity,
                message=f"Error: {str(e)}",
            )


class ShapeCheckTest(QualityTest):
    """데이터 형태 검사 테스트"""
    
    name = "shape_check"
    severity = TestSeverity.CRITICAL
    
    def __init__(
        self,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        expected_columns: Optional[int] = None,
        required_columns: Optional[List[str]] = None,
    ):
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.expected_columns = expected_columns
        self.required_columns = required_columns or []
    
    def run(self, data: Any, **kwargs) -> QualityTestResult:
        try:
            import pandas as pd
            
            if not isinstance(data, pd.DataFrame):
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.SKIP,
                    severity=self.severity,
                    message="Data is not a DataFrame",
                )
            
            rows, cols = data.shape
            errors = []
            
            # 행 수 검사
            if self.min_rows is not None and rows < self.min_rows:
                errors.append(f"Row count {rows} < min {self.min_rows}")
            if self.max_rows is not None and rows > self.max_rows:
                errors.append(f"Row count {rows} > max {self.max_rows}")
            
            # 열 수 검사
            if self.expected_columns is not None and cols != self.expected_columns:
                errors.append(f"Column count {cols} != expected {self.expected_columns}")
            
            # 필수 컬럼 검사
            missing_cols = set(self.required_columns) - set(data.columns)
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")
            
            if errors:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.FAIL,
                    severity=self.severity,
                    message="; ".join(errors),
                    actual={"rows": rows, "cols": cols},
                    metadata={"errors": errors},
                )
            
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.PASS,
                severity=self.severity,
                message=f"Shape ({rows}, {cols}) is valid",
                actual={"rows": rows, "cols": cols},
            )
            
        except ImportError:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.SKIP,
                severity=self.severity,
                message="pandas not installed",
            )
        except Exception as e:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.ERROR,
                severity=self.severity,
                message=f"Error: {str(e)}",
            )


class RangeCheckTest(QualityTest):
    """값 범위 검사 테스트"""
    
    name = "range_check"
    severity = TestSeverity.MEDIUM
    
    def __init__(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = True,
    ):
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
    
    def run(self, data: Any, **kwargs) -> QualityTestResult:
        try:
            import pandas as pd
            
            if not isinstance(data, pd.DataFrame):
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.SKIP,
                    severity=self.severity,
                    message="Data is not a DataFrame",
                )
            
            if self.column not in data.columns:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.FAIL,
                    severity=self.severity,
                    message=f"Column '{self.column}' not found",
                )
            
            col_data = data[self.column]
            
            if not self.allow_nan and col_data.isna().any():
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.FAIL,
                    severity=self.severity,
                    message=f"Column '{self.column}' contains NaN values",
                )
            
            # NaN 제외하고 검사
            valid_data = col_data.dropna()
            
            if len(valid_data) == 0:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.WARN,
                    severity=self.severity,
                    message=f"Column '{self.column}' has no valid values",
                )
            
            actual_min = valid_data.min()
            actual_max = valid_data.max()
            errors = []
            
            if self.min_value is not None and actual_min < self.min_value:
                errors.append(f"Min value {actual_min} < {self.min_value}")
            if self.max_value is not None and actual_max > self.max_value:
                errors.append(f"Max value {actual_max} > {self.max_value}")
            
            if errors:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.FAIL,
                    severity=self.severity,
                    message=f"Column '{self.column}': " + "; ".join(errors),
                    expected={"min": self.min_value, "max": self.max_value},
                    actual={"min": actual_min, "max": actual_max},
                )
            
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.PASS,
                severity=self.severity,
                message=f"Column '{self.column}' values in range [{actual_min}, {actual_max}]",
                expected={"min": self.min_value, "max": self.max_value},
                actual={"min": actual_min, "max": actual_max},
            )
            
        except ImportError:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.SKIP,
                severity=self.severity,
                message="pandas not installed",
            )
        except Exception as e:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.ERROR,
                severity=self.severity,
                message=f"Error: {str(e)}",
            )


class UniqueCheckTest(QualityTest):
    """고유성 검사 테스트"""
    
    name = "unique_check"
    severity = TestSeverity.HIGH
    
    def __init__(
        self,
        columns: List[str],
        allow_duplicates: bool = False,
        max_duplicate_ratio: float = 0.0,
    ):
        self.columns = columns
        self.allow_duplicates = allow_duplicates
        self.max_duplicate_ratio = max_duplicate_ratio
    
    def run(self, data: Any, **kwargs) -> QualityTestResult:
        try:
            import pandas as pd
            
            if not isinstance(data, pd.DataFrame):
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.SKIP,
                    severity=self.severity,
                    message="Data is not a DataFrame",
                )
            
            missing_cols = set(self.columns) - set(data.columns)
            if missing_cols:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.FAIL,
                    severity=self.severity,
                    message=f"Missing columns: {missing_cols}",
                )
            
            total_rows = len(data)
            if total_rows == 0:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.PASS,
                    severity=self.severity,
                    message="Empty dataset",
                )
            
            duplicates = data.duplicated(subset=self.columns, keep=False)
            duplicate_count = duplicates.sum()
            duplicate_ratio = duplicate_count / total_rows
            
            if not self.allow_duplicates and duplicate_count > 0:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.FAIL,
                    severity=self.severity,
                    message=f"Found {duplicate_count} duplicate rows ({duplicate_ratio:.2%})",
                    expected=0,
                    actual=duplicate_count,
                    metadata={"duplicate_ratio": duplicate_ratio},
                )
            
            if duplicate_ratio > self.max_duplicate_ratio:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.FAIL,
                    severity=self.severity,
                    message=f"Duplicate ratio {duplicate_ratio:.2%} exceeds {self.max_duplicate_ratio:.2%}",
                    expected=self.max_duplicate_ratio,
                    actual=duplicate_ratio,
                )
            
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.PASS,
                severity=self.severity,
                message=f"Uniqueness check passed ({duplicate_count} duplicates)",
                actual=duplicate_count,
            )
            
        except ImportError:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.SKIP,
                severity=self.severity,
                message="pandas not installed",
            )
        except Exception as e:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.ERROR,
                severity=self.severity,
                message=f"Error: {str(e)}",
            )


class TypeCheckTest(QualityTest):
    """데이터 타입 검사 테스트"""
    
    name = "type_check"
    severity = TestSeverity.MEDIUM
    
    def __init__(
        self,
        column_types: Dict[str, str],
    ):
        self.column_types = column_types
    
    def run(self, data: Any, **kwargs) -> QualityTestResult:
        try:
            import pandas as pd
            
            if not isinstance(data, pd.DataFrame):
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.SKIP,
                    severity=self.severity,
                    message="Data is not a DataFrame",
                )
            
            errors = []
            for col, expected_type in self.column_types.items():
                if col not in data.columns:
                    errors.append(f"Column '{col}' not found")
                    continue
                
                actual_type = str(data[col].dtype)
                if expected_type not in actual_type:
                    errors.append(f"Column '{col}': expected {expected_type}, got {actual_type}")
            
            if errors:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.FAIL,
                    severity=self.severity,
                    message="; ".join(errors),
                    metadata={"errors": errors},
                )
            
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.PASS,
                severity=self.severity,
                message="All column types match",
                expected=self.column_types,
            )
            
        except ImportError:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.SKIP,
                severity=self.severity,
                message="pandas not installed",
            )
        except Exception as e:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.ERROR,
                severity=self.severity,
                message=f"Error: {str(e)}",
            )


class CustomTest(QualityTest):
    """커스텀 테스트"""
    
    def __init__(
        self,
        name: str,
        test_func: Callable[[Any], bool],
        severity: TestSeverity = TestSeverity.MEDIUM,
        description: str = "",
    ):
        self.name = name
        self.test_func = test_func
        self.severity = severity
        self.description = description
    
    def run(self, data: Any, **kwargs) -> QualityTestResult:
        try:
            result = self.test_func(data)
            
            if result:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.PASS,
                    severity=self.severity,
                    message=self.description or f"Custom test '{self.name}' passed",
                )
            else:
                return QualityTestResult(
                    test_name=self.name,
                    result=TestResult.FAIL,
                    severity=self.severity,
                    message=self.description or f"Custom test '{self.name}' failed",
                )
                
        except Exception as e:
            return QualityTestResult(
                test_name=self.name,
                result=TestResult.ERROR,
                severity=self.severity,
                message=f"Error: {str(e)}",
            )


class QualityTestRunner:
    """
    품질 테스트 실행기
    
    Task 6.3.1: 데이터 품질 자동 테스트
    """
    
    def __init__(self):
        self.tests: List[QualityTest] = []
    
    def add_test(self, test: QualityTest):
        """테스트 추가"""
        self.tests.append(test)
        return self
    
    def add_nan_check(self, max_nan_ratio: float = 0.0, columns: Optional[List[str]] = None):
        """NaN 검사 추가"""
        return self.add_test(NaNCheckTest(max_nan_ratio, columns))
    
    def add_shape_check(
        self,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        expected_columns: Optional[int] = None,
        required_columns: Optional[List[str]] = None,
    ):
        """형태 검사 추가"""
        return self.add_test(ShapeCheckTest(min_rows, max_rows, expected_columns, required_columns))
    
    def add_range_check(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """범위 검사 추가"""
        return self.add_test(RangeCheckTest(column, min_value, max_value))
    
    def add_unique_check(
        self,
        columns: List[str],
        allow_duplicates: bool = False,
    ):
        """고유성 검사 추가"""
        return self.add_test(UniqueCheckTest(columns, allow_duplicates))
    
    def add_type_check(self, column_types: Dict[str, str]):
        """타입 검사 추가"""
        return self.add_test(TypeCheckTest(column_types))
    
    def add_custom_check(
        self,
        name: str,
        test_func: Callable[[Any], bool],
        severity: TestSeverity = TestSeverity.MEDIUM,
    ):
        """커스텀 검사 추가"""
        return self.add_test(CustomTest(name, test_func, severity))
    
    def run(self, data: Any, dataset_name: str = "dataset") -> QualityReport:
        """테스트 실행"""
        import time
        
        report = QualityReport(dataset_name=dataset_name)
        
        for test in self.tests:
            start = time.time()
            result = test.run(data)
            result.duration_ms = (time.time() - start) * 1000
            report.add_result(result)
            
            logger.debug(
                f"Quality test '{test.name}': {result.result.value} - {result.message}"
            )
        
        report.completed_at = datetime.utcnow()
        
        logger.info(
            f"Quality tests completed: {report.passed}/{report.total} passed "
            f"({report.success_rate:.1f}%)"
        )
        
        return report
    
    def clear(self):
        """테스트 목록 초기화"""
        self.tests.clear()


# 편의 함수
def run_quality_tests(
    data: Any,
    dataset_name: str = "dataset",
    tests: Optional[List[QualityTest]] = None,
) -> QualityReport:
    """품질 테스트 실행 편의 함수"""
    runner = QualityTestRunner()
    
    if tests:
        for test in tests:
            runner.add_test(test)
    else:
        # 기본 테스트
        runner.add_nan_check(max_nan_ratio=0.1)
        runner.add_shape_check(min_rows=1)
    
    return runner.run(data, dataset_name)
