"""DB 상태 확인 스크립트"""
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent.parent.parent / "data" / "pade.db"
print(f"📁 DB 경로: {db_path}")
print(f"📁 존재 여부: {db_path.exists()}")

if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 테이블 목록
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\n=== 테이블 목록 ({len(tables)}개) ===")
    for t in tables:
        print(f"  - {t[0]}")
    
    # 각 테이블 데이터 수
    print("\n=== 데이터 현황 ===")
    for t in tables:
        table_name = t[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  {table_name}: {count}개")
    
    # videos 테이블 내용
    if any(t[0] == 'videos' for t in tables):
        cursor.execute("SELECT id, video_id, title, duration_sec, status FROM videos ORDER BY id DESC LIMIT 10")
        videos = cursor.fetchall()
        print("\n=== 최근 영상 10개 ===")
        for v in videos:
            title = v[2][:30] + "..." if v[2] and len(v[2]) > 30 else v[2]
            print(f"  {v[0]}: {v[1]} - {title} ({v[3]}초) [{v[4]}]")
    
    # episodes 테이블 내용
    if any(t[0] == 'episodes' for t in tables):
        cursor.execute("SELECT id, video_id, start_frame, end_frame, quality_score, jittering_score FROM episodes ORDER BY id DESC LIMIT 15")
        episodes = cursor.fetchall()
        print(f"\n=== 최근 에피소드 15개 ===")
        for e in episodes:
            frames = e[3] - e[2]
            jitter = e[5] if e[5] else 0
            print(f"  에피소드 {e[0]}: video_id={e[1]}, {frames}프레임 ({e[2]}-{e[3]}), 품질={e[4]:.3f}, 지터={jitter:.3f}")
    
    conn.close()
