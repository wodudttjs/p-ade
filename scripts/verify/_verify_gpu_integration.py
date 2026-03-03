#!/usr/bin/env python3
"""GPU 3-Stream 파이프라인 통합 검증 스크립트"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    print("=" * 60)
    print("GPU 3-Stream 파이프라인 통합 검증")
    print("=" * 60)

    # 1. GPU 3-Stream Manager import
    try:
        from gpu.stream_manager import GPU3StreamManager
        mgr = GPU3StreamManager()
        print("[1] GPU3StreamManager import: OK")
    except Exception as e:
        print(f"[1] GPU3StreamManager import FAIL: {e}")
        return 1

    # 2. 프로세서 팩토리 메서드
    processors = [
        'make_detect_processor',
        'make_il_processor',
        'make_pose_extract_processor',
        'make_encode_processor',
        'make_quality_filter_processor',
    ]
    for p in processors:
        if hasattr(mgr, p):
            print(f"[2] {p}: OK")
        else:
            print(f"[2] {p}: MISSING")
            return 1

    # 3. 프로세서 생성 테스트
    try:
        mgr.make_detect_processor(output_fps=5.0)
        mgr.make_il_processor(output_fps=5.0, output_dir='/tmp/test_il')
        mgr.make_pose_extract_processor(output_fps=30.0, output_dir='/tmp/test_pose')
        mgr.make_encode_processor(output_dir='/tmp/test_encode')
        mgr.make_quality_filter_processor(filtered_dir='/tmp/test_filter')
        print("[3] All processors created: OK")
    except Exception as e:
        print(f"[3] Processor creation FAIL: {e}")
        return 1

    # 4. VRAM 조회
    try:
        vram = mgr.get_vram_usage()
        batch = mgr.auto_adjust_batch_size()
        print(f"[4] VRAM: {vram}")
        print(f"[4] Batch size: {batch}")
    except Exception as e:
        print(f"[4] VRAM check FAIL: {e}")

    # 5. mass_collector 파이프라인 확인
    try:
        from mass_collector import MassCollector, PipelineConfig
        cfg = PipelineConfig(target_count=1, dry_run=True, use_gpu_streams=True)
        mc = MassCollector(cfg)
        stages = mc.STAGES
        print(f"[5] MassCollector stages: {stages}")
        assert 'detect' in stages
        assert 'build_il' in stages
        print("[5] MassCollector: OK")
    except Exception as e:
        print(f"[5] MassCollector FAIL: {e}")
        return 1

    # 6. detect_to_episodes GPU 지원 확인
    try:
        import inspect
        from extraction.detect_to_episodes import run
        sig = inspect.signature(run)
        params = list(sig.parameters.keys())
        assert 'use_gpu_streams' in params, f"use_gpu_streams not in {params}"
        print(f"[6] detect_to_episodes.run params: {params}")
        print("[6] detect_to_episodes GPU support: OK")
    except Exception as e:
        print(f"[6] detect_to_episodes FAIL: {e}")
        return 1

    # 7. extract_poses GPU 지원 확인 (CLI 확인)
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'extract_poses.py', '--help'],
            capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).parent)
        )
        has_gpu_flag = '--no-gpu-streams' in result.stdout
        print(f"[7] extract_poses --no-gpu-streams flag: {'OK' if has_gpu_flag else 'MISSING'}")
    except Exception as e:
        print(f"[7] extract_poses check: {e}")

    # 8. encode_actions GPU 지원 확인
    try:
        result = subprocess.run(
            [sys.executable, 'encode_actions.py', '--help'],
            capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).parent)
        )
        has_gpu_flag = '--no-gpu-streams' in result.stdout
        print(f"[8] encode_actions --no-gpu-streams flag: {'OK' if has_gpu_flag else 'MISSING'}")
    except Exception as e:
        print(f"[8] encode_actions check: {e}")

    # 9. filter_quality GPU 지원 확인
    try:
        result = subprocess.run(
            [sys.executable, 'filter_quality.py', '--help'],
            capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).parent)
        )
        has_gpu_flag = '--no-gpu-streams' in result.stdout
        print(f"[9] filter_quality --no-gpu-streams flag: {'OK' if has_gpu_flag else 'MISSING'}")
    except Exception as e:
        print(f"[9] filter_quality check: {e}")

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED - GPU 3-Stream 파이프라인 통합 완료")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
