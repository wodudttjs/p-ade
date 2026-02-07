"""목 데이터와 실제 데이터 구분 확인"""
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "data" / "pade.db"
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# 목 데이터 vs 실제 데이터 구분
cursor.execute("SELECT video_id, title, status FROM videos WHERE video_id LIKE 'vid_%' LIMIT 5")
mock = cursor.fetchall()
print('=== 목(Mock) 데이터 샘플 ===')
for v in mock:
    print(f'  {v[0]}: {v[1]} [{v[2]}]')

cursor.execute("SELECT COUNT(*) FROM videos WHERE video_id LIKE 'vid_%'")
mock_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM videos WHERE video_id NOT LIKE 'vid_%'")
real_count = cursor.fetchone()[0]

print()
print('=== 영상 분류 ===')
print(f'  목 데이터 (vid_xxxx): {mock_count}개')
print(f'  실제 수집 영상: {real_count}개')

print()
print('=== 실제 수집된 영상 ===')
cursor.execute("SELECT id, video_id, title, duration_sec, status FROM videos WHERE video_id NOT LIKE 'vid_%'")
real = cursor.fetchall()
for v in real:
    title = v[2][:40] + '...' if v[2] and len(v[2]) > 40 else v[2]
    print(f'  {v[0]}: {v[1]} - {title} ({v[3]}초) [{v[4]}]')

# 에피소드도 확인
print()
print('=== 에피소드 분류 ===')
cursor.execute("SELECT COUNT(*) FROM episodes WHERE video_id IN (SELECT id FROM videos WHERE video_id LIKE 'vid_%')")
mock_ep = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM episodes WHERE video_id IN (SELECT id FROM videos WHERE video_id NOT LIKE 'vid_%')")
real_ep = cursor.fetchone()[0]
print(f'  목 데이터 에피소드: {mock_ep}개')
print(f'  실제 에피소드: {real_ep}개')

# 처리 작업 확인
print()
print('=== 처리 작업 분류 ===')
cursor.execute("SELECT COUNT(*) FROM processing_jobs WHERE job_key LIKE 'job_%'")
mock_jobs = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM processing_jobs WHERE job_key NOT LIKE 'job_%'")
real_jobs = cursor.fetchone()[0]
print(f'  목 데이터 작업: {mock_jobs}개')
print(f'  실제 작업: {real_jobs}개')

conn.close()
