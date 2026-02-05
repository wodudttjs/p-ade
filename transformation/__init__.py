"""
MODULE 4: Data Transformation

좌표 정규화, Action 인코딩, 포맷 변환, 데이터 증강 모듈
"""

from transformation.normalization import (
    ReferenceFrameAligner,
    RelativeCoordinateComputer,
    DimensionalityReducer,
    CoordinateNormalizer,
)
from transformation.temporal import (
    TemporalResampler,
    SavgolSmoother,
    TemporalAligner,
)
from transformation.encoding import (
    StateBuilder,
    ActionComputer,
    StateActionEncoder,
)
from transformation.augment import (
    SpatialAugmenter,
    TemporalAugmenter,
    ViewpointAugmenter,
    DataAugmentationPipeline,
)
from transformation.export import (
    NpzExporter,
    ParquetExporter,
    HDF5Exporter,
    FormatConverter,
)
from transformation.spec import (
    StateSpec,
    ActionSpec,
    TransformConfig,
    VERSION,
)

__all__ = [
    # Normalization
    "ReferenceFrameAligner",
    "RelativeCoordinateComputer",
    "DimensionalityReducer",
    "CoordinateNormalizer",
    # Temporal
    "TemporalResampler",
    "SavgolSmoother",
    "TemporalAligner",
    # Encoding
    "StateBuilder",
    "ActionComputer",
    "StateActionEncoder",
    # Augmentation
    "SpatialAugmenter",
    "TemporalAugmenter",
    "ViewpointAugmenter",
    "DataAugmentationPipeline",
    # Export
    "NpzExporter",
    "ParquetExporter",
    "HDF5Exporter",
    "FormatConverter",
    # Spec
    "StateSpec",
    "ActionSpec",
    "TransformConfig",
    "VERSION",
]
