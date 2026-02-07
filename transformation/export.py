"""
FR-4.3: Format Conversion

학습 프레임워크 호환 포맷 변환
- NumPy Archive (.npz)
- Parquet Export
- HDF5 (선택)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import numpy as np

from core.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class ExportMetadata:
    """Export 메타데이터"""
    video_id: str
    episode_id: str
    start_t: float
    end_t: float
    fps: float
    dt: float
    transform_version: str
    state_spec: str
    action_spec: str
    quality_metrics: Dict[str, float]
    params_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "episode_id": self.episode_id,
            "start_t": self.start_t,
            "end_t": self.end_t,
            "fps": self.fps,
            "dt": self.dt,
            "transform_version": self.transform_version,
            "state_spec": self.state_spec,
            "action_spec": self.action_spec,
            "quality_metrics": self.quality_metrics,
            "params_hash": self.params_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportMetadata":
        return cls(**data)


class NpzExporter:
    """
    Task 4.3.1: NumPy Archive (.npz)
    
    빠른 로컬 학습/실험용 저장
    """
    
    def __init__(self, base_dir: str = "data/episodes"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
        metadata: Union[Dict[str, Any], ExportMetadata],
        masks: Optional[np.ndarray] = None,
        video_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        compress: bool = True,
    ) -> Path:
        """
        NPZ 파일로 저장
        
        Args:
            states: [T, S] 상태 배열
            actions: [T-1, A] 액션 배열
            timestamps: [T] 타임스탬프
            metadata: 메타데이터 (dict 또는 ExportMetadata)
            masks: [T] 마스크 (선택)
            video_id: 비디오 ID
            episode_id: 에피소드 ID
            compress: 압축 여부
            
        Returns:
            저장된 파일 경로
        """
        # 경로 결정
        if video_id is None:
            video_id = "unknown"
        if episode_id is None:
            episode_id = f"ep_{np.random.randint(10000):05d}"
            
        output_dir = self.base_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{episode_id}.npz"
        
        # 메타데이터 처리
        if isinstance(metadata, ExportMetadata):
            metadata_dict = metadata.to_dict()
        else:
            metadata_dict = metadata
            
        # 메타데이터를 JSON 문자열로 변환 (dtype object 방지)
        metadata_json = json.dumps(metadata_dict)
        
        # 마스크 처리
        if masks is None:
            masks = np.ones(len(timestamps), dtype=np.uint8)
        else:
            masks = masks.astype(np.uint8)
            
        # Contiguous 배열 보장
        states = np.ascontiguousarray(states)
        actions = np.ascontiguousarray(actions)
        timestamps = np.ascontiguousarray(timestamps)
        
        # 저장
        save_func = np.savez_compressed if compress else np.savez
        save_func(
            output_path,
            states=states,
            actions=actions,
            timestamps=timestamps,
            masks=masks,
            metadata=metadata_json,
        )
        
        logger.info(f"Saved NPZ: {output_path} (states={states.shape}, actions={actions.shape})")
        
        return output_path
    
    def load(
        self,
        path: Union[str, Path],
        dtype: str = "float32",
    ) -> Dict[str, Any]:
        """
        NPZ 파일 로드
        
        Args:
            path: 파일 경로
            dtype: 로드 시 dtype (float16 → float32 캐스팅 등)
            
        Returns:
            로드된 데이터 딕셔너리
        """
        path = Path(path)
        
        with np.load(path, allow_pickle=True) as data:
            result = {
                "states": data["states"].astype(dtype),
                "actions": data["actions"].astype(dtype),
                "timestamps": data["timestamps"].astype(np.float32),
                "masks": data["masks"],
            }
            
            # 메타데이터 파싱
            metadata_str = str(data["metadata"])
            try:
                result["metadata"] = json.loads(metadata_str)
            except json.JSONDecodeError:
                result["metadata"] = {}
                
        return result
    
    def save_shard(
        self,
        episodes: List[Dict[str, Any]],
        shard_name: str,
        compress: bool = True,
    ) -> Path:
        """
        여러 에피소드를 하나의 샤드로 저장
        
        Args:
            episodes: 에피소드 리스트
            shard_name: 샤드 이름 (예: "train_0001")
            compress: 압축 여부
            
        Returns:
            저장된 파일 경로
        """
        shard_dir = self.base_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        output_path = shard_dir / f"{shard_name}.npz"
        
        # 에피소드별 데이터 수집
        all_states = []
        all_actions = []
        all_timestamps = []
        all_masks = []
        episode_indices = []
        all_metadata = []
        
        current_idx = 0
        for ep in episodes:
            states = ep["states"]
            T = len(states)
            
            all_states.append(states)
            all_actions.append(ep["actions"])
            all_timestamps.append(ep["timestamps"])
            all_masks.append(ep.get("masks", np.ones(T, dtype=np.uint8)))
            
            episode_indices.append({
                "start": current_idx,
                "end": current_idx + T,
                "episode_id": ep.get("episode_id", f"ep_{len(episode_indices)}"),
            })
            current_idx += T
            
            all_metadata.append(ep.get("metadata", {}))
            
        # 연결
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        timestamps = np.concatenate(all_timestamps, axis=0)
        masks = np.concatenate(all_masks, axis=0)
        
        # 저장
        save_func = np.savez_compressed if compress else np.savez
        save_func(
            output_path,
            states=states,
            actions=actions,
            timestamps=timestamps,
            masks=masks,
            episode_indices=json.dumps(episode_indices),
            metadata=json.dumps(all_metadata),
        )
        
        logger.info(f"Saved shard: {output_path} ({len(episodes)} episodes)")
        
        return output_path


class ParquetExporter:
    """
    Task 4.3.2: Parquet Export
    
    대규모 분석/필터링/분산처리용
    """
    
    def __init__(self, base_dir: str = "data/parquet"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_frame_level(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
        metadata: Dict[str, Any],
        partition_keys: Optional[Dict[str, str]] = None,
        compression: str = "snappy",
    ) -> Path:
        """
        프레임 단위 Parquet 저장
        
        Args:
            states: [T, S] 상태
            actions: [T-1, A] 액션
            timestamps: [T] 타임스탬프
            metadata: 메타데이터
            partition_keys: 파티션 키 (date, category 등)
            compression: 압축 방식
            
        Returns:
            저장된 파일 경로
        """
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("pyarrow/pandas required for Parquet export")
            raise
            
        T = len(timestamps)
        
        # DataFrame 구성
        records = []
        for t in range(T):
            record = {
                "video_id": metadata.get("video_id", "unknown"),
                "episode_id": metadata.get("episode_id", "unknown"),
                "t": t,
                "timestamp": timestamps[t],
                "state": states[t].tolist(),
                "action": actions[t].tolist() if t < T - 1 else None,
                "conf_mean": metadata.get("quality_metrics", {}).get("conf_mean", 1.0),
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        
        # 파티션 경로 구성
        if partition_keys:
            partition_path = "/".join(
                f"{k}={v}" for k, v in partition_keys.items()
            )
            output_dir = self.base_dir / partition_path
        else:
            output_dir = self.base_dir
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명
        video_id = metadata.get("video_id", "unknown")
        episode_id = metadata.get("episode_id", "unknown")
        output_path = output_dir / f"{video_id}_{episode_id}.parquet"
        
        # 저장
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            output_path,
            compression=compression,
        )
        
        logger.info(f"Saved Parquet: {output_path} ({T} rows)")
        
        return output_path
    
    def save_episode_level(
        self,
        episodes: List[Dict[str, Any]],
        output_name: str,
        partition_keys: Optional[Dict[str, str]] = None,
        compression: str = "snappy",
    ) -> Path:
        """
        에피소드 단위 Parquet 저장 (인덱스 역할)
        
        Args:
            episodes: 에피소드 리스트
            output_name: 출력 파일명
            partition_keys: 파티션 키
            compression: 압축 방식
            
        Returns:
            저장된 파일 경로
        """
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("pyarrow/pandas required for Parquet export")
            raise
            
        # DataFrame 구성
        records = []
        for ep in episodes:
            metadata = ep.get("metadata", {})
            timestamps = ep.get("timestamps", [])
            
            record = {
                "video_id": metadata.get("video_id", "unknown"),
                "episode_id": metadata.get("episode_id", "unknown"),
                "start_t": timestamps[0] if len(timestamps) > 0 else 0,
                "end_t": timestamps[-1] if len(timestamps) > 0 else 0,
                "duration": len(timestamps),
                "fps": metadata.get("fps", 30),
                "state_dim": ep["states"].shape[1] if "states" in ep else 0,
                "action_dim": ep["actions"].shape[1] if "actions" in ep else 0,
                "data_path": metadata.get("data_path", ""),
                "quality_score": metadata.get("quality_metrics", {}).get("valid_ratio", 1.0),
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        
        # 파티션 경로
        if partition_keys:
            partition_path = "/".join(f"{k}={v}" for k, v in partition_keys.items())
            output_dir = self.base_dir / partition_path
        else:
            output_dir = self.base_dir
            
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{output_name}.parquet"
        
        # 저장
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression=compression)
        
        logger.info(f"Saved episode index: {output_path} ({len(records)} episodes)")
        
        return output_path
    
    def load(self, path: Union[str, Path]) -> "pd.DataFrame":
        """Parquet 파일 로드"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for Parquet loading")
            
        return pd.read_parquet(path)


class HDF5Exporter:
    """
    Task 4.3.3: HDF5 (선택)
    
    대용량 시계열 계층적 저장
    """
    
    def __init__(self, base_dir: str = "data/hdf5"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save(
        self,
        episodes: List[Dict[str, Any]],
        output_name: str,
        chunk_size: int = 256,
        compression: str = "gzip",
        compression_level: int = 4,
    ) -> Path:
        """
        HDF5 파일로 저장
        
        Args:
            episodes: 에피소드 리스트
            output_name: 출력 파일명
            chunk_size: 청크 크기
            compression: 압축 방식
            compression_level: 압축 레벨
            
        Returns:
            저장된 파일 경로
        """
        try:
            import h5py
        except ImportError:
            logger.error("h5py required for HDF5 export")
            raise
            
        output_path = self.base_dir / f"{output_name}.h5"
        
        with h5py.File(output_path, "w") as f:
            # 에피소드 그룹 생성
            episodes_group = f.create_group("episodes")
            
            for i, ep in enumerate(episodes):
                metadata = ep.get("metadata", {})
                episode_id = metadata.get("episode_id", f"ep_{i:05d}")
                
                ep_group = episodes_group.create_group(episode_id)
                
                # 데이터셋 저장
                states = ep["states"]
                actions = ep["actions"]
                timestamps = ep.get("timestamps", np.arange(len(states)))
                
                # 청크 크기 조정
                state_chunks = (min(chunk_size, len(states)),) + states.shape[1:]
                action_chunks = (min(chunk_size, len(actions)),) + actions.shape[1:]
                
                ep_group.create_dataset(
                    "states",
                    data=states,
                    chunks=state_chunks,
                    compression=compression,
                    compression_opts=compression_level,
                )
                
                ep_group.create_dataset(
                    "actions",
                    data=actions,
                    chunks=action_chunks,
                    compression=compression,
                    compression_opts=compression_level,
                )
                
                ep_group.create_dataset(
                    "timestamps",
                    data=timestamps,
                    compression=compression,
                    compression_opts=compression_level,
                )
                
                # 메타데이터를 속성으로 저장
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        ep_group.attrs[key] = value
                    elif isinstance(value, dict):
                        ep_group.attrs[key] = json.dumps(value)
                        
            # 전역 메타데이터
            f.attrs["num_episodes"] = len(episodes)
            f.attrs["version"] = "4.0.0"
            
        logger.info(f"Saved HDF5: {output_path} ({len(episodes)} episodes)")
        
        return output_path
    
    def load(
        self,
        path: Union[str, Path],
        episode_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        HDF5 파일 로드
        
        Args:
            path: 파일 경로
            episode_ids: 로드할 에피소드 ID (None이면 전체)
            
        Returns:
            에피소드 리스트
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 loading")
            
        episodes = []
        
        with h5py.File(path, "r") as f:
            episodes_group = f["episodes"]
            
            if episode_ids is None:
                episode_ids = list(episodes_group.keys())
                
            for ep_id in episode_ids:
                if ep_id not in episodes_group:
                    continue
                    
                ep_group = episodes_group[ep_id]
                
                episode = {
                    "states": ep_group["states"][:],
                    "actions": ep_group["actions"][:],
                    "timestamps": ep_group["timestamps"][:],
                    "metadata": dict(ep_group.attrs),
                }
                
                # JSON 메타데이터 파싱
                for key, value in episode["metadata"].items():
                    if isinstance(value, str) and value.startswith("{"):
                        try:
                            episode["metadata"][key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                            
                episodes.append(episode)
                
        return episodes
    
    def get_episode_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """HDF5 파일 정보 조회"""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required")
            
        info = {"episodes": []}
        
        with h5py.File(path, "r") as f:
            info["num_episodes"] = f.attrs.get("num_episodes", 0)
            info["version"] = f.attrs.get("version", "unknown")
            
            for ep_id in f["episodes"].keys():
                ep_group = f["episodes"][ep_id]
                info["episodes"].append({
                    "episode_id": ep_id,
                    "states_shape": ep_group["states"].shape,
                    "actions_shape": ep_group["actions"].shape,
                })
                
        return info


class FormatConverter:
    """
    포맷 변환 통합 클래스
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.npz_exporter = NpzExporter(str(self.base_dir / "episodes"))
        self.parquet_exporter = ParquetExporter(str(self.base_dir / "parquet"))
        self.hdf5_exporter = HDF5Exporter(str(self.base_dir / "hdf5"))
        
    def convert(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
        metadata: Dict[str, Any],
        formats: List[str] = ["npz"],
        **kwargs,
    ) -> Dict[str, Path]:
        """
        여러 포맷으로 변환 저장
        
        Args:
            states: 상태 배열
            actions: 액션 배열
            timestamps: 타임스탬프
            metadata: 메타데이터
            formats: 저장 포맷 리스트 ("npz", "parquet", "hdf5")
            **kwargs: 추가 인자
            
        Returns:
            포맷별 저장 경로
        """
        paths = {}
        
        video_id = metadata.get("video_id", "unknown")
        episode_id = metadata.get("episode_id", "unknown")
        
        if "npz" in formats:
            paths["npz"] = self.npz_exporter.save(
                states=states,
                actions=actions,
                timestamps=timestamps,
                metadata=metadata,
                video_id=video_id,
                episode_id=episode_id,
                masks=kwargs.get("masks"),
            )
            
        if "parquet" in formats:
            paths["parquet"] = self.parquet_exporter.save_frame_level(
                states=states,
                actions=actions,
                timestamps=timestamps,
                metadata=metadata,
                partition_keys=kwargs.get("partition_keys"),
            )
            
        if "hdf5" in formats:
            episode = {
                "states": states,
                "actions": actions,
                "timestamps": timestamps,
                "metadata": metadata,
            }
            paths["hdf5"] = self.hdf5_exporter.save(
                episodes=[episode],
                output_name=f"{video_id}_{episode_id}",
            )
            
        return paths
    
    def batch_convert(
        self,
        episodes: List[Dict[str, Any]],
        formats: List[str] = ["npz"],
        batch_name: str = "batch",
    ) -> Dict[str, List[Path]]:
        """
        여러 에피소드 일괄 변환
        """
        paths = {fmt: [] for fmt in formats}
        
        for i, ep in enumerate(episodes):
            result = self.convert(
                states=ep["states"],
                actions=ep["actions"],
                timestamps=ep.get("timestamps", np.arange(len(ep["states"]))),
                metadata=ep.get("metadata", {"episode_id": f"ep_{i:05d}"}),
                formats=formats,
            )
            
            for fmt, path in result.items():
                paths[fmt].append(path)
                
        logger.info(f"Batch converted {len(episodes)} episodes to {formats}")
        
        return paths
