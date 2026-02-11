## ☁️ Phase 3: 클라우드 확장 - 2주

### Task 3.1: AWS Lambda 서버리스 크롤러 구축 ⭐⭐⭐⭐

**해결 문제**: 병렬 웹크롤링 (문제 4)  
**우선순위**: P1 (높음)  
**예상 소요**: 5-7일  
**난이도**: ★★★★☆

**작업 내용**:
```
1. Lambda 함수 개발
   # lambda/crawler_function.py
   
   import json
   import boto3
   from youtube_crawler import YouTubeCrawler
   
   dynamodb = boto3.resource('dynamodb')
   table = dynamodb.Table('robot-videos')
   
   def lambda_handler(event, context):
       """
       입력:
       {
           "keywords": ["robot arm", "pick place"],
           "max_per_keyword": 50
       }
       """
       
       keywords = event['keywords']
       max_results = event.get('max_per_keyword', 50)
       
       crawler = YouTubeCrawler()
       total_found = 0
       
       for keyword in keywords:
           videos = crawler.search(keyword, max_results)
           
           # DynamoDB 저장
           for video in videos:
               table.put_item(Item={
                   'video_id': video['id'],
                   'keyword': keyword,
                   'title': video['title'],
                   'url': video['url'],
                   'metadata': json.dumps(video),
                   'collected': False
               })
           
           total_found += len(videos)
       
       return {
           'statusCode': 200,
           'body': json.dumps({
               'keywords_processed': len(keywords),
               'videos_found': total_found
           })
       }

2. Lambda 배포 패키지 생성
   # 의존성 설치
   mkdir lambda_package
   cd lambda_package
   pip install -t . requests google-api-python-client
   
   # 함수 코드 복사
   cp ../lambda/crawler_function.py .
   cp -r ../crawler .
   
   # ZIP 생성
   zip -r lambda_function.zip .

3. AWS Lambda 함수 생성 (CLI)
   aws lambda create-function \
       --function-name robot-video-crawler \
       --runtime python3.10 \
       --role arn:aws:iam::123456789:role/lambda-execution \
       --handler crawler_function.lambda_handler \
       --zip-file fileb://lambda_function.zip \
       --timeout 300 \
       --memory-size 512

4. DynamoDB 테이블 생성
   aws dynamodb create-table \
       --table-name robot-videos \
       --attribute-definitions \
           AttributeName=video_id,AttributeType=S \
       --key-schema \
           AttributeName=video_id,KeyType=HASH \
       --billing-mode PAY_PER_REQUEST

5. 로컬에서 Lambda 호출 스크립트
   # local/invoke_lambda.py
   
   import boto3
   import json
   import asyncio
   
   lambda_client = boto3.client('lambda', region_name='us-east-1')
   
   def invoke_crawler(keywords):
       """Lambda 함수 비동기 호출"""
       response = lambda_client.invoke(
           FunctionName='robot-video-crawler',
           InvocationType='Event',  # 비동기
           Payload=json.dumps({
               'keywords': keywords,
               'max_per_keyword': 50
           })
       )
       return response
   
   def parallel_crawl(all_keywords, batch_size=10):
       """키워드를 배치로 나누어 Lambda 호출"""
       
       # 10개씩 묶기
       batches = [
           all_keywords[i:i+batch_size]
           for i in range(0, len(all_keywords), batch_size)
       ]
       
       print(f"Launching {len(batches)} Lambda functions...")
       
       # 모든 배치 호출
       for i, batch in enumerate(batches):
           invoke_crawler(batch)
           print(f"Batch {i+1}/{len(batches)} launched")
       
       print("All Lambda functions launched!")
   
   # 실행
   keywords = load_keywords(1000)  # 1000개 키워드
   parallel_crawl(keywords, batch_size=10)
   # → 100개 Lambda 동시 실행

6. DynamoDB에서 로컬로 동기화
   # sync/dynamodb_sync.py
   
   import boto3
   import psycopg2
   
   dynamodb = boto3.resource('dynamodb')
   table = dynamodb.Table('robot-videos')
   
   pg_conn = psycopg2.connect(
       host='localhost',
       database='robot_videos',
       user='postgres',
       password='password'
   )
   
   def sync_from_dynamodb():
       """DynamoDB → PostgreSQL 동기화"""
       
       # 모든 아이템 스캔
       response = table.scan()
       items = response['Items']
       
       # PostgreSQL에 삽입
       cursor = pg_conn.cursor()
       
       for item in items:
           if not item.get('collected'):
               cursor.execute("""
                   INSERT INTO videos (video_id, title, url, metadata)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (video_id) DO NOTHING
               """, (
                   item['video_id'],
                   item['title'],
                   item['url'],
                   item['metadata']
               ))
               
               # DynamoDB에서 collected 마킹
               table.update_item(
                   Key={'video_id': item['video_id']},
                   UpdateExpression='SET collected = :val',
                   ExpressionAttributeValues={':val': True}
               )
       
       pg_conn.commit()
       cursor.close()
필요 기술:

AWS Lambda
AWS DynamoDB
boto3 (AWS SDK)
서버리스 아키텍처

예상 효과:

100개 Lambda 동시 실행
크롤링 속도 50배 향상
1,000 키워드 → 5분 내 완료
비용: $0.003 (1센트 미만)

검증 방법:
bash# Lambda 함수 테스트
aws lambda invoke \
    --function-name robot-video-crawler \
    --payload '{"keywords":["robot arm"],"max_per_keyword":10}' \
    response.json

# 결과 확인
cat response.json

# DynamoDB 확인
aws dynamodb scan --table-name robot-videos --max-items 10
```

**의존성**: AWS 계정 필요

---

### Task 3.2: 키워드 대규모 확장 ⭐⭐⭐⭐⭐

**해결 문제**: 영상 개수 부족 (문제 6)  
**우선순위**: P0 (긴급)  
**예상 소요**: 3-4일  
**난이도**: ★★☆☆☆

**작업 내용**:
```
1. 조합형 키워드 생성기
   # keywords/generator.py
   
   class KeywordGenerator:
       """조합형 키워드 생성"""
       
       ACTIONS = [
           'pick and place', 'grasping', 'manipulation',
           'assembly', 'bin picking', 'sorting', 'palletizing',
           'packaging', 'transfer', 'loading', 'unloading',
           'handling', 'reach and grasp', 'pick place move',
           'object manipulation', 'part handling', 'gripper control',
           'automated picking', 'robotic assembly', 'material handling',
           # ... 총 50개
       ]
       
       ROBOTS = [
           'robot arm', 'robotic arm', 'industrial robot',
           'collaborative robot', 'cobot', '6-axis robot',
           'articulated robot', 'manipulator', 'robotic manipulator',
           'UR robot', 'Universal Robot', 'ABB robot', 'KUKA robot',
           'Fanuc robot', 'Yaskawa robot', 'Kawasaki robot',
           'Doosan robot', 'robotic gripper', 'automated arm',
           'delta robot', 'SCARA robot',
           # ... 총 30개
       ]
       
       OBJECTS = [
           'object', 'part', 'component', 'box', 'bottle',
           'product', 'item', 'piece', 'workpiece', 'block',
           'tool', 'container', 'can', 'jar', 'carton',
           'bag', 'assembly', 'module', 'sensor', 'motor',
           'bearing', 'gear', 'fastener', 'screw', 'nut', 'bolt',
           # ... 총 40개
       ]
       
       CONTEXTS = [
           'factory', 'warehouse', 'manufacturing', 'production line',
           'assembly line', 'industrial', 'automation', 'laboratory',
           'demonstration', 'test', 'tutorial', 'training',
           # ... 총 20개
       ]
       
       def generate_2word(self):
           """2-word 조합: action + robot"""
           keywords = []
           for action in self.ACTIONS:
               for robot in self.ROBOTS:
                   keywords.append(f"{action} {robot}")
           return keywords  # 50 × 30 = 1,500개
       
       def generate_3word(self):
           """3-word 조합: action + robot + object"""
           keywords = []
           for action in self.ACTIONS[:20]:  # 상위 20개만
               for robot in self.ROBOTS[:15]:  # 상위 15개만
                   for obj in self.OBJECTS[:10]:  # 상위 10개만
                       keywords.append(f"{action} {robot} {obj}")
           return keywords  # 20 × 15 × 10 = 3,000개
       
       def generate_with_context(self):
           """context 추가 조합"""
           keywords = []
           for action in self.ACTIONS[:15]:
               for robot in self.ROBOTS[:10]:
                   for context in self.CONTEXTS[:10]:
                       keywords.append(f"{action} {robot} {context}")
           return keywords  # 15 × 10 × 10 = 1,500개
       
       def generate_all(self):
           """모든 조합 생성"""
           all_keywords = []
           all_keywords.extend(self.generate_2word())
           all_keywords.extend(self.generate_3word())
           all_keywords.extend(self.generate_with_context())
           
           # 중복 제거
           unique_keywords = list(set(all_keywords))
           return unique_keywords  # 약 5,000-6,000개

2. 다국어 키워드 확장
   # keywords/multilingual.py
   
   from googletrans import Translator
   
   class MultilingualExpander:
       """다국어 키워드 확장"""
       
       LANGUAGES = {
           'ko': 'Korean',
           'ja': 'Japanese',
           'zh-cn': 'Chinese Simplified',
           'de': 'German',
           'es': 'Spanish'
       }
       
       def __init__(self):
           self.translator = Translator()
       
       def translate_keywords(self, keywords, target_lang):
           """키워드 번역"""
           translated = []
           
           for keyword in keywords:
               try:
                   result = self.translator.translate(
                       keyword,
                       dest=target_lang
                   )
                   translated.append(result.text)
               except:
                   continue
           
           return translated
       
       def expand_all_languages(self, base_keywords):
           """모든 언어로 확장"""
           all_keywords = base_keywords.copy()  # 영어 원본
           
           for lang_code in self.LANGUAGES.keys():
               translated = self.translate_keywords(base_keywords, lang_code)
               all_keywords.extend(translated)
               print(f"{lang_code}: +{len(translated)} keywords")
           
           return all_keywords
   
   # 사용
   generator = KeywordGenerator()
   base_keywords = generator.generate_all()  # 6,000개
   
   expander = MultilingualExpander()
   multilingual = expander.expand_all_languages(base_keywords)
   # 6,000 × 6개 언어 = 36,000개

3. 롱테일 키워드 자동 발견
   # keywords/longtail_discovery.py
   
   import requests
   
   class LongtailDiscovery:
       """YouTube 자동완성 활용"""
       
       def get_youtube_suggestions(self, seed):
           """YouTube 자동완성 API"""
           url = "http://suggestqueries.google.com/complete/search"
           params = {
               'client': 'youtube',
               'ds': 'yt',
               'q': seed
           }
           
           response = requests.get(url, params=params)
           suggestions = response.json()[1]
           return [s[0] for s in suggestions]
       
       def expand_with_alphabet(self, seed):
           """알파벳 추가로 확장"""
           suggestions = []
           
           for char in 'abcdefghijklmnopqrstuvwxyz':
               query = f"{seed} {char}"
               results = self.get_youtube_suggestions(query)
               suggestions.extend(results)
           
           return list(set(suggestions))
       
       def discover_from_seeds(self, seed_keywords):
           """시드 키워드에서 롱테일 발견"""
           all_longtail = []
           
           for seed in seed_keywords:
               longtail = self.expand_with_alphabet(seed)
               all_longtail.extend(longtail)
               print(f"{seed}: +{len(longtail)} longtail")
           
           return list(set(all_longtail))
   
   # 사용
   seeds = ["robot arm pick", "industrial robot", "cobot"]
   discovery = LongtailDiscovery()
   longtail = discovery.discover_from_seeds(seeds)
   # 약 500-1,000개 추가 발견

4. 키워드 DB 저장 및 관리
   # 최종 통합
   all_keywords = []
   
   # 1. 조합형 키워드
   generator = KeywordGenerator()
   all_keywords.extend(generator.generate_all())  # 6,000개
   
   # 2. 다국어 확장
   expander = MultilingualExpander()
   all_keywords.extend(expander.expand_all_languages(all_keywords[:1000]))  # +5,000개
   
   # 3. 롱테일 발견
   discovery = LongtailDiscovery()
   all_keywords.extend(discovery.discover_from_seeds(all_keywords[:100]))  # +1,000개
   
   # 중복 제거
   unique_keywords = list(set(all_keywords))
   print(f"Total unique keywords: {len(unique_keywords)}")
   # 예상: 10,000-15,000개
   
   # DB 저장
   save_to_database(unique_keywords)
필요 기술:

Python 리스트 조작
Google Translate API (선택)
조합 알고리즘

예상 효과:

키워드 수: 100개 → 10,000개 (100배)
검색 커버리지 극대화
중복 최소화
장기간 사용 가능

검증 방법:
bash# 키워드 생성 테스트
python keywords/test_generator.py

# 예상 출력
# 2-word combinations: 1,500
# 3-word combinations: 3,000
# With context: 1,500
# Multilingual: +5,000
# Longtail: +1,000
# Total unique: 12,000
```

**의존성**: 없음 (독립 실행)