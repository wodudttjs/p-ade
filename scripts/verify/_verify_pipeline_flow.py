#!/usr/bin/env python3
"""
P-ADE 파이프라인 전체 흐름 검증 스크립트

서버 시작 → 파이프라인 자동 실행 → 대시보드 모니터링 흐름을 검증합니다.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

results = []

def check(name, condition, detail=""):
    status = "✅" if condition else "❌"
    results.append((name, condition, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))

print("=" * 70)
print("🔍 P-ADE 파이프라인 전체 흐름 검증")
print("=" * 70)

# ============================================================
# 1. 핵심 모듈 존재 확인
# ============================================================
print("\n📦 1. 핵심 모듈 존재 확인")

modules = {
    "main.py": PROJECT_ROOT / "main.py",
    "mass_collector.py": PROJECT_ROOT / "mass_collector.py",
    "dashboard/web_app.py": PROJECT_ROOT / "dashboard" / "web_app.py",
    "gpu/stream_manager.py": PROJECT_ROOT / "gpu" / "stream_manager.py",
    "scripts/pipeline/build_imitation_data.py": PROJECT_ROOT / "scripts" / "pipeline" / "build_imitation_data.py",
    "quality/evaluator.py": PROJECT_ROOT / "quality" / "evaluator.py",
    "ingestion/keyword_generator.py": PROJECT_ROOT / "ingestion" / "keyword_generator.py",
    "ingestion/multi_source_crawler.py": PROJECT_ROOT / "ingestion" / "multi_source_crawler.py",
    "scripts/pipeline/parallel_download.py": PROJECT_ROOT / "scripts" / "pipeline" / "parallel_download.py",
    "extraction/detect_to_episodes.py": PROJECT_ROOT / "extraction" / "detect_to_episodes.py",
    "scripts/pipeline/upload_to_s3.py": PROJECT_ROOT / "scripts" / "pipeline" / "upload_to_s3.py",
}
for name, path in modules.items():
    check(name, path.exists())

# ============================================================
# 2. MassCollector 파이프라인 스테이지 검증
# ============================================================
print("\n🔄 2. MassCollector 파이프라인 스테이지 검증")

from mass_collector import MassCollector, PipelineConfig

expected_stages = ["crawl", "download", "detect", "build_il", "quality", "upload"]
check("STAGES 정의", MassCollector.STAGES == expected_stages,
      f"실제: {MassCollector.STAGES}")

config = PipelineConfig(target_count=10, dry_run=True)
collector = MassCollector(config)

for stage in expected_stages:
    handler = getattr(collector, f"_stage_{stage}", None)
    check(f"_stage_{stage} 핸들러", handler is not None and callable(handler))

# ============================================================
# 3. 콜백 시스템 검증
# ============================================================
print("\n🔗 3. MassCollector 콜백 시스템 검증")

log_messages = []
stage_starts = []
stage_completes = []

def on_start(stage):
    stage_starts.append(stage)

def on_complete(stage, result):
    stage_completes.append((stage, result.success))

def on_log(msg):
    log_messages.append(msg)

collector_cb = MassCollector(
    config,
    on_stage_start=on_start,
    on_stage_complete=on_complete,
    on_log=on_log,
)

check("콜백 등록", 
      collector_cb._on_stage_start is not None and 
      collector_cb._on_stage_complete is not None and
      collector_cb._on_log is not None)

# ============================================================
# 4. CollectionPipeline 대시보드 동기화 검증
# ============================================================
print("\n📊 4. CollectionPipeline 대시보드 동기화 검증")

from main import CollectionPipeline

pipeline_sync = CollectionPipeline(target_count=10, dashboard_sync=True)
check("dashboard_sync 플래그", pipeline_sync.dashboard_sync is True)

pipeline_nosync = CollectionPipeline(target_count=10, dashboard_sync=False)
check("dashboard_sync=False 기본 동작", pipeline_nosync.dashboard_sync is False)

# 대시보드 콜백 생성 테스트
callbacks, p_state = pipeline_sync._get_dashboard_callbacks()
check("대시보드 콜백 생성", 
      "on_stage_start" in callbacks and 
      "on_stage_complete" in callbacks and 
      "on_log" in callbacks,
      f"콜백 키: {list(callbacks.keys())}")

# ============================================================
# 5. 대시보드 pipeline_state 검증
# ============================================================
print("\n🖥️  5. 대시보드 pipeline_state 검증")

from dashboard.web_app import pipeline_state

check("pipeline_state 존재", pipeline_state is not None)
check("is_running 키", "is_running" in pipeline_state)
check("current_stage 키", "current_stage" in pipeline_state)
check("progress 키", "progress" in pipeline_state)
check("logs 키", "logs" in pipeline_state)

# 6개 스테이지 진행률
progress = pipeline_state["progress"]
for stage in expected_stages:
    check(f"progress[{stage}]", stage in progress, f"값: {progress.get(stage, 'MISSING')}")

# ============================================================
# 6. main.py serve 모드 검증
# ============================================================
print("\n🚀 6. main.py serve 모드 검증")

from main import run_serve, build_parser

parser = build_parser()

# serve 명령어 파싱 테스트
args = parser.parse_args(["serve", "--target", "100", "--port", "5000"])
check("serve 명령 파싱", args.command == "serve")
check("serve --target", args.target == 100)
check("serve --port", args.port == 5000)

# 기본 모드 (명령어 없을 때 serve)
args_default = parser.parse_args([])
check("기본 모드 = serve", args_default.command is None, "None → serve 모드로 실행")

# ============================================================
# 7. GPU 3-Stream 통합 검증
# ============================================================
print("\n🎮 7. GPU 3-Stream 통합 검증")

check("use_gpu_streams 설정", config.use_gpu_streams is True, "기본값: True")

try:
    from gpu.stream_manager import GPU3StreamManager
    check("GPU3StreamManager import", True)
    mgr = GPU3StreamManager()
    check("GPU3StreamManager 인스턴스 생성", True)
    vram = mgr.get_vram_usage()
    check("VRAM 사용량 조회", isinstance(vram, dict) and "allocated" in vram, f"VRAM: {vram}")
except Exception as e:
    check("GPU3StreamManager", False, f"에러: {e}")

# ============================================================
# 8. 품질 평가 등급 검증 (A~E)
# ============================================================
print("\n📊 8. 품질 평가 등급 검증")

try:
    from quality.evaluator import RobotArmQualityEvaluator, QualityConfig
    evaluator = RobotArmQualityEvaluator()
    check("QualityEvaluator import", True)
    
    # 등급 기준 확인
    q_config = QualityConfig()
    check("품질 기본 임계값", q_config.pass_threshold == 60.0, f"임계값: {q_config.pass_threshold}")
except Exception as e:
    check("QualityEvaluator", False, f"에러: {e}")

# ============================================================
# 9. 대시보드 API 엔드포인트 검증
# ============================================================
print("\n🌐 9. 대시보드 API 엔드포인트 검증")

from dashboard.web_app import app

with app.test_client() as client:
    # API 상태
    resp = client.get("/api/pipeline/status")
    check("/api/pipeline/status", resp.status_code == 200)
    data = resp.get_json()
    check("status 응답 구조", 
          all(k in data for k in ["is_running", "current_stage", "progress", "logs"]),
          f"키: {list(data.keys())}")
    check("progress에 6개 스테이지",
          all(s in data["progress"] for s in expected_stages),
          f"키: {list(data['progress'].keys())}")
    
    # 헬스체크
    resp = client.get("/api/health")
    check("/api/health", resp.status_code == 200)
    
    # 통계
    resp = client.get("/api/stats")
    check("/api/stats", resp.status_code == 200)

# ============================================================
# 10. 파이프라인 흐름 시뮬레이션
# ============================================================
print("\n🎬 10. 파이프라인 흐름 시뮬레이션 (dry-run)")

sim_log = []
sim_starts = []
sim_completes = []

def sim_on_start(s):
    sim_starts.append(s)

def sim_on_complete(s, r):
    sim_completes.append((s, r.success))

def sim_on_log(m):
    sim_log.append(m)

sim_config = PipelineConfig(target_count=5, dry_run=True)
sim_collector = MassCollector(
    sim_config,
    on_stage_start=sim_on_start,
    on_stage_complete=sim_on_complete,
    on_log=sim_on_log,
)

try:
    sim_collector.run()
    check("dry-run 실행 성공", True)
    check("모든 스테이지 시작 콜백", 
          sim_starts == expected_stages,
          f"실제: {sim_starts}")
    check("모든 스테이지 완료 콜백",
          len(sim_completes) == len(expected_stages),
          f"완료: {len(sim_completes)}개")
    check("로그 콜백 호출", len(sim_log) > 0, f"로그 수: {len(sim_log)}")
except Exception as e:
    check("dry-run 실행", False, f"에러: {e}")

# ============================================================
# 결과 요약
# ============================================================
print("\n" + "=" * 70)
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)

print(f"📋 검증 결과: {passed}/{total} 통과, {failed} 실패")

if failed > 0:
    print("\n❌ 실패 항목:")
    for name, ok, detail in results:
        if not ok:
            print(f"   • {name}: {detail}")
else:
    print("\n🎉 모든 검증 통과! 파이프라인이 정상적으로 구성되어 있습니다.")
    print("   python main.py serve 로 서버를 시작하면 파이프라인이 자동 실행됩니다.")

print("=" * 70)
