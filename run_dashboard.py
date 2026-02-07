#!/usr/bin/env python
"""
P-ADE Dashboard Runner

대시보드 애플리케이션 실행 스크립트
"""

import sys
import os

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dashboard import run_dashboard


def main():
    """대시보드 실행"""
    print("🎬 P-ADE Dashboard 시작...")
    print("=" * 40)
    
    try:
        return run_dashboard()
    except ImportError as e:
        print(f"\n❌ 필수 패키지가 설치되지 않았습니다: {e}")
        print("\n📦 다음 명령으로 설치해주세요:")
        print("   pip install PySide6>=6.5.0")
        return 1
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
