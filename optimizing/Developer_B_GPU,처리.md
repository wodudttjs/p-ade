🟢 Developer B: GPU/처리 담당
📋 작업 목표

GPU 처리: 1,012개/6,254초 → 5,000개/4시간 (2배 효율)
핵심: Detect + IL 1-Pass 통합 처리로 40% 시간 절감


Phase 0: 사전 준비 (4시간)
✅ Task B0-1: Merge Conflict 해결
우선순위: P0
시간: 2시간
파일: extraction/detect_to_episodes.py
작업 내용:

Git 충돌 해결
YOLO 모델 설정 병합
Episode 생성 로직 확인
샘플 10개 영상 테스트


📊 Task B0-2: GPU 성능 프로파일링
우선순위: P1
시간: 2시간
측정 항목:

현재 성능:

1,012개 처리 시간: 6,254초
평균 영상당 처리 시간: 6.2초
GPU 사용률, VRAM 사용량


병목 식별:

YOLO 추론 시간
MediaPipe 처리 시간
비디오 디코딩 시간
프레임 전송(CPU→GPU) 시간



산출물:

gpu_profiling_report.md
3-Stream vs 6-Stream 성능 예측
통합 처리 시간 절감 예측


Phase 1: Detect+IL 통합 처리기 (Day 1-3)
⭐⭐⭐⭐⭐ Task B1-1: UnifiedVideoProcessor 구현
우선순위: P0
시간: 2일
파일: gpu/unified_processor.py (신규)
핵심 개념:
AS-IS (2-Pass):
  비디오 → YOLO → NPZ 저장
  비디오 → MediaPipe → NPZ 저장
  = 비디오 2회 디코딩

TO-BE (1-Pass):
  비디오 → (YOLO + MediaPipe) → NPZ 1회 저장
  = 비디오 1회 디코딩

예상 효과: 처리 시간 40% 단축
구현 단계:
1. 통합 프로세서 클래스 설계
클래스: UnifiedVideoProcessor

입력: video_path
출력: unified_npz (Detection + Pose + State-Action 포함)

핵심 로직:
1. cv2로 비디오 열기
2. 각 프레임 순회:
   - YOLO 추론 (객체 검출)
   - MediaPipe 추론 (포즈 추출)
   - State-Action 계산
3. 통합 NPZ 저장
2. YOLO + MediaPipe 동시 추론
주의사항:
- 두 모델 모두 GPU 사용
- VRAM 관리 필수
- 배치 추론 고려

최적화:
- 프레임 단위 스트리밍 (메모리 절약)
- GPU에서 직접 처리 (CPU ↔ GPU 전송 최소화)
3. NPZ 포맷 통합
unified_npz 구조:
{
  'detections': [...],      # YOLO 결과
  'poses': [...],           # MediaPipe 결과
  'states': [...],          # State 인코딩
  'actions': [...],         # Action 인코딩
  'metadata': {...}
}
검증:

샘플 100개 영상:

기존 2-Pass: 620초
통합 1-Pass: 370초 목표 (40% 단축)


NPZ 파일 무결성 확인
품질 평가 점수 동일 유지


🔗 Task B1-2: MassCollector 스테이지 통합
우선순위: P0
시간: 0.5일
파일: mass_collector.py
작업 내용:
1. STAGES 배열 변경
위치: mass_collector.py 상단

AS-IS:
STAGES = ["crawl", "download", "detect", "build_il", "quality", "upload"]

TO-BE:
STAGES = ["crawl", "download", "process", "quality", "upload", "cleanup"]
2. _stage_detect + _stage_build_il 통합
작업:
1. _stage_detect() 삭제
2. _stage_build_il() 삭제
3. _stage_process() 신규 생성

_stage_process() 내용:
- UnifiedVideoProcessor 호출
- GPU 3-Stream (또는 6-Stream) 처리
- 통합 NPZ 저장
3. PipelineConfig 업데이트
추가 옵션:
- unified_processing: bool = True  # 1-Pass 통합 처리
- num_gpu_streams: int = 6         # 6-stream dual-GPU

Phase 2: GPU 6-Stream 확장 (Day 4-5)
🚀 Task B2-1: Dual-GPU 지원
우선순위: P0
시간: 1.5일
파일: gpu/stream_manager.py
작업 내용:
1. Multi-GPU 감지 및 초기화
작업:
1. torch.cuda.device_count() 확인
2. GPU 0, GPU 1 각각 초기화
3. 각 GPU에 3개 Stream 할당 (총 6 Stream)
2. StreamConfig 확장
파일: gpu/stream_manager.py:36-41

변경:
- num_streams: 3 → 6
- gpu_ids: [0] → [0, 1]
- streams_per_gpu: 3
3. VRAM 동적 관리
작업:
1. GPU별 VRAM 독립 체크
2. 총 VRAM의 85% 사용
3. auto_adjust_batch_size() GPU별 적용
4. 영상 길이별 배칭
로직:
1. 영상 길이로 분류:
   - 짧은 영상 (< 30초): batch 4
   - 중간 영상 (30-60초): batch 2
   - 긴 영상 (> 60초): CPU 워커로 위임
   
2. 같은 길이 영상끼리 배치 구성
3. GPU 효율 극대화
검증:

6-Stream 안정성 테스트 (1시간 연속 실행)
OOM 에러 발생 여부 확인
GPU 0, GPU 1 사용률 균등 확인
처리 속도: 90개/시간 → 150개/시간 목표


⚙️ Task B2-2: CPU Worker Pool (긴 영상 전용)
우선순위: P1
시간: 1일
파일: gpu/cpu_worker_pool.py (신규)
목적:

60초 이상 긴 영상이 GPU 병목 방지
CPU 멀티프로세스로 별도 처리

구현 내용:
1. CPUWorkerPool 클래스
구조:
- ProcessPoolExecutor(max_workers=CPU_COUNT // 2)
- MediaPipe CPU 모드
- 15fps 다운샘플링
2. StreamManager 연동
로직:
1. process_batch()에서 영상 길이 체크
2. 60초 이상 영상 → CPUWorkerPool에 위임
3. 60초 미만 영상 → GPU 6-Stream 처리
3. 결과 병합
작업:
- CPU 처리 결과 + GPU 처리 결과 병합
- 순서 유지
- 통계 분리 집계
검증:

긴 영상 10개 (60-120초) CPU 처리 확인
GPU와 CPU 동시 처리 안정성 확인
전체 처리량 개선 확인


Phase 3: 품질 평가 최적화 (Day 6-7)
📊 Task B3-1: 품질 평가 벡터화
우선순위: P1
시간: 1일
파일: quality/evaluator.py
작업 내용:
1. 배치 평가 메서드 추가
메서드: evaluate_batch(npz_files)

작업:
1. 여러 NPZ 파일 동시 로드
2. NumPy 배열로 변환
3. 벡터화 연산으로 점수 계산
4. 배치 결과 반환
2. 통과 기준 완화
변경:
- min_passing_score: 60 → 50
- 목표: 통과율 48% → 65%
3. Early Reject 추가
조건:
- 프레임 수 < 30 → 즉시 탈락
- 평균 Confidence < 0.3 → 즉시 탈락
- 처리 시간 절약
4. rejected 파일 처리 최적화
변경:
- 파일 이동(mv) → DB 마킹만
- 디스크 I/O 절약
- 필요 시 나중에 일괄 삭제
검증:

1,000개 NPZ 파일 배치 평가
순차 평가 vs 배치 평가 시간 비교
통과율 65% 달성 확인


🔗 Task B3-2: MassCollector 품질 스테이지 수정
우선순위: P1
시간: 0.5일
파일: mass_collector.py:798-883
작업 내용:

_stage_quality() 메서드 수정
evaluate_batch() 호출
rejected 영상 Registry 등록
통과/탈락 통계 로깅


📊 Developer B 최종 체크리스트
필수 완료 항목

 UnifiedVideoProcessor 구현 및 테스트
 Detect+IL 통합 1-Pass 처리
 Dual-GPU 6-Stream 지원
 CPU Worker Pool 구현
 품질 평가 배치 벡터화
 통과 기준 50점으로 완화

성능 목표

 5,000개 처리 / 4시간
 처리 속도: 150개/시간 (기존 90개/시간)
 통과율: 65% 이상