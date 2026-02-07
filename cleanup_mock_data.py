"""목 데이터 정리 스크립트"""
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "data" / "pade.db"
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print('=== 삭제 전 현황 ===')
cursor.execute('SELECT COUNT(*) FROM videos')
print(f'  videos: {cursor.fetchone()[0]}개')
cursor.execute('SELECT COUNT(*) FROM episodes')
print(f'  episodes: {cursor.fetchone()[0]}개')
cursor.execute('SELECT COUNT(*) FROM processing_jobs')
print(f'  processing_jobs: {cursor.fetchone()[0]}개')

# 목 에피소드 삭제 (vid_로 시작하는 비디오에 연결된)
cursor.execute("DELETE FROM episodes WHERE video_id IN (SELECT id FROM videos WHERE video_id LIKE 'vid_%')")
print(f'\n  삭제된 목 에피소드: {cursor.rowcount}개')

# 목 비디오 삭제
cursor.execute("DELETE FROM videos WHERE video_id LIKE 'vid_%'")
print(f'  삭제된 목 비디오: {cursor.rowcount}개')

# 목 처리 작업 삭제
cursor.execute("DELETE FROM processing_jobs WHERE job_key LIKE 'job_%'")
print(f'  삭제된 목 작업: {cursor.rowcount}개')

conn.commit()

print('\n=== 삭제 후 현황 ===')
cursor.execute('SELECT COUNT(*) FROM videos')
print(f'  videos: {cursor.fetchone()[0]}개')
cursor.execute('SELECT COUNT(*) FROM episodes')
print(f'  episodes: {cursor.fetchone()[0]}개')
cursor.execute('SELECT COUNT(*) FROM processing_jobs')
print(f'  processing_jobs: {cursor.fetchone()[0]}개')

conn.close()
print('\n✅ 목 데이터 정리 완료!')
