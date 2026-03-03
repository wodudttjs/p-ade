## 📊 Phase 4: 운영 안정화 - 지속

### Task 4.1: systemd 서비스 등록 ⭐⭐⭐

**해결 문제**: 24시간 백그라운드 실행 (문제 3)  
**우선순위**: P1 (높음)  
**예상 소요**: 1일  
**난이도**: ★★☆☆☆

**작업 내용**:
```
1. systemd 서비스 파일 생성
   # /etc/systemd/system/robot-collector.service
   
   [Unit]
   Description=Robot Arm Video Collection Service
   After=network.target redis.service postgresql.service
   Wants=redis.service postgresql.service
   
   [Service]
   Type=simple
   User=robotuser
   Group=robotuser
   WorkingDirectory=/home/robotuser/robot-collector
   
   # 환경 변수
   Environment="PATH=/home/robotuser/venv/bin"
   Environment="PYTHONUNBUFFERED=1"
   Environment="CUDA_VISIBLE_DEVICES=0"
   
   # 실행 명령
   ExecStart=/home/robotuser/venv/bin/python main.py run-forever
   
   # 재시작 정책
   Restart=always
   RestartSec=10
   
   # 로그
   StandardOutput=journal
   StandardError=journal
   
   [Install]
   WantedBy=multi-user.target

2. main.py run-forever 모드 구현
   # main.py
   
   import time
   import logging
   from pipeline import CollectionPipeline
   from datetime import datetime
   
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   def run_forever():
       """무한 루프 파이프라인"""
       
       pipeline = CollectionPipeline()
       
       while True:
           try:
               # 1. 목표 달성 체크
               if pipeline.is_daily_target_reached():
                   logger.info("Daily target reached. Waiting for next day...")
                   wait_until_tomorrow()
                   continue
               
               # 2. 파이프라인 1회 실행
               logger.info("Starting pipeline iteration...")
               pipeline.run_once()
               
               # 3. 짧은 휴식
               time.sleep(60)  # 1분 대기
               
           except KeyboardInterrupt:
               logger.info("Shutting down gracefully...")
               pipeline.cleanup()
               break
           
           except Exception as e:
               logger.error(f"Pipeline error: {e}", exc_info=True)
               time.sleep(300)  # 5분 대기 후 재시도
   
   def wait_until_tomorrow():
       """다음 날 6시까지 대기"""
       now = datetime.now()
       tomorrow_6am = now.replace(
           day=now.day + 1,
           hour=6,
           minute=0,
           second=0
       )
       sleep_seconds = (tomorrow_6am - now).total_seconds()
       time.sleep(sleep_seconds)
   
   if __name__ == "__main__":
       import sys
       
       if len(sys.argv) > 1 and sys.argv[1] == "run-forever":
           run_forever()
       else:
           # 단일 실행
           pipeline = CollectionPipeline()
           pipeline.run_once()

3. 서비스 등록 및 시작
   # 서비스 등록
   sudo systemctl daemon-reload
   sudo systemctl enable robot-collector
   
   # 시작
   sudo systemctl start robot-collector
   
   # 상태 확인
   sudo systemctl status robot-collector
   
   # 로그 확인
   sudo journalctl -u robot-collector -f

4. 로그 로테이션 설정
   # /etc/logrotate.d/robot-collector
   
   /var/log/robot-collector/*.log {
       daily
       rotate 7
       compress
       delaycompress
       missingok
       notifempty
   }
필요 기술:

Linux systemd
Bash
프로세스 관리

예상 효과:

부팅 시 자동 시작
크래시 시 자동 재시작
백그라운드 실행
무인 운영 가능

검증 방법:
bash# 서비스 시작 테스트
sudo systemctl start robot-collector
sleep 10
sudo systemctl status robot-collector

# 재시작 테스트
sudo pkill -9 python  # 강제 종료
sleep 15
sudo systemctl status robot-collector  # 자동 재시작 확인

# 부팅 테스트
sudo reboot
# 재부팅 후
sudo systemctl status robot-collector  # 자동 시작 확인
```

**의존성**: main.py run-forever 모드 구현

---

### Task 4.2: 모니터링 알림 시스템 ⭐⭐⭐

**해결 문제**: 이상 상황 조기 감지  
**우선순위**: P2 (중간)  
**예상 소요**: 2-3일  
**난이도**: ★★★☆☆

**작업 내용**:
```
1. 알림 조건 정의
   # monitor/alerts.py
   
   class AlertManager:
       """알림 관리자"""
       
       THRESHOLDS = {
           'gpu_util_low': 30,      # GPU 30% 미만
           'queue_depleted': 100,   # 큐 100개 미만
           'failure_rate_high': 0.4,  # 실패율 40% 초과
           'target_behind': 0.4,    # 목표 대비 40% 이하 (18시 기준)
       }
       
       def check_alerts(self):
           """알림 조건 체크"""
           alerts = []
           
           # GPU 사용률 낮음
           if self.get_gpu_util() < self.THRESHOLDS['gpu_util_low']:
               alerts.append({
                   'level': 'WARNING',
                   'message': f"Low GPU utilization: {self.get_gpu_util()}%"
               })
           
           # 큐 고갈
           if self.get_queue_size() < self.THRESHOLDS['queue_depleted']:
               alerts.append({
                   'level': 'WARNING',
                   'message': f"Queue depleted: {self.get_queue_size()} items"
               })
           
           # 실패율 높음
           failure_rate = self.get_failure_rate()
           if failure_rate > self.THRESHOLDS['failure_rate_high']:
               alerts.append({
                   'level': 'ERROR',
                   'message': f"High failure rate: {failure_rate:.1%}"
               })
           
           # 목표 대비 지연 (18시 체크)
           if datetime.now().hour == 18:
               progress = self.get_daily_progress()
               if progress < self.THRESHOLDS['target_behind']:
                   alerts.append({
                       'level': 'ERROR',
                       'message': f"Behind target: {progress:.1%} at 18:00"
                   })
           
           return alerts

2. 이메일 알림
   # monitor/email_notifier.py
   
   import smtplib
   from email.mime.text import MIMEText
   
   class EmailNotifier:
       """이메일 알림"""
       
       def __init__(self, smtp_server, from_email, password):
           self.smtp_server = smtp_server
           self.from_email = from_email
           self.password = password
       
       def send_alert(self, alert):
           """알림 발송"""
           
           subject = f"[{alert['level']}] Robot Collector Alert"
           body = f"""
           Alert Level: {alert['level']}
           Message: {alert['message']}
           Time: {datetime.now()}
           
           Dashboard: http://your-server:8000
           """
           
           msg = MIMEText(body)
           msg['Subject'] = subject
           msg['From'] = self.from_email
           msg['To'] = 'admin@example.com'
           
           with smtplib.SMTP(self.smtp_server, 587) as server:
               server.starttls()
               server.login(self.from_email, self.password)
               server.send_message(msg)

3. Slack 알림 (선택)
   # monitor/slack_notifier.py
   
   import requests
   
   class SlackNotifier:
       """Slack 웹훅"""
       
       def __init__(self, webhook_url):
           self.webhook_url = webhook_url
       
       def send_alert(self, alert):
           """Slack 메시지 전송"""
           
           color = {
               'INFO': '#36a64f',
               'WARNING': '#ff9800',
               'ERROR': '#f44336'
           }.get(alert['level'], '#808080')
           
           payload = {
               'attachments': [{
                   'color': color,
                   'title': f"{alert['level']} Alert",
                   'text': alert['message'],
                   'footer': 'Robot Collector',
                   'ts': int(time.time())
               }]
           }
           
           requests.post(self.webhook_url, json=payload)

4. 알림 루프
   # monitor/alert_loop.py
   
   def run_alert_monitor():
       """주기적 알림 체크"""
       
       alert_manager = AlertManager()
       email_notifier = EmailNotifier(...)
       slack_notifier = SlackNotifier(...)
       
       while True:
           # 알림 체크
           alerts = alert_manager.check_alerts()
           
           for alert in alerts:
               # 중복 발송 방지 (1시간 내 동일 알림)
               if not is_recently_sent(alert):
                   email_notifier.send_alert(alert)
                   slack_notifier.send_alert(alert)
                   mark_as_sent(alert)
           
           # 5분마다 체크
           time.sleep(300)
필요 기술:

SMTP (이메일)
Slack Webhook
알림 로직

예상 효과:

이상 조기 발견
빠른 대응 가능
무인 운영 안정성 향상

검증 방법:
bash# 테스트 알림 발송
python test_alerts.py

# 실제 조건 시뮬레이션
# - GPU 사용률 낮춤
# - 큐 비움
# - 알림 수신 확인
```

**의존성**: 모니터링 시스템 구축 후