FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 기본 설정
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# 의존성 먼저 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 데이터 디렉토리 생성
RUN mkdir -p data/raw data/episodes data/processed logs

# 포트 노출 (대시보드)
EXPOSE 8000

# 기본 실행 커맨드
CMD ["python", "main.py", "run-once", "--target", "5000"]
