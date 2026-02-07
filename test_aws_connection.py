#!/usr/bin/env python
"""
AWS S3 연결 테스트 스크립트

AWS 자격 증명이 올바르게 설정되었는지 확인합니다.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# .env 로드
load_dotenv(project_root / ".env")


def test_credentials():
    """AWS 자격 증명 확인"""
    print("=" * 60)
    print("🔐 AWS 자격 증명 테스트")
    print("=" * 60)
    
    # 환경 변수 확인
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    region = os.getenv("AWS_REGION", "us-east-1")
    bucket = os.getenv("AWS_S3_BUCKET", "p-ade-datasets")
    
    print(f"\n📋 환경 변수:")
    print(f"  AWS_ACCESS_KEY_ID: {access_key[:8]}...{access_key[-4:] if len(access_key) > 12 else '(too short)'}")
    print(f"  AWS_SECRET_ACCESS_KEY: {'*' * 20} (hidden)")
    print(f"  AWS_REGION: {region}")
    print(f"  AWS_S3_BUCKET: {bucket}")
    
    if not access_key or not secret_key:
        print("\n❌ AWS 자격 증명이 설정되지 않았습니다.")
        print("   .env 파일에 AWS_ACCESS_KEY_ID와 AWS_SECRET_ACCESS_KEY를 설정하세요.")
        return False
        
    return True


def test_boto3_connection():
    """boto3 연결 테스트"""
    print("\n" + "=" * 60)
    print("🔌 boto3 연결 테스트")
    print("=" * 60)
    
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        print(f"\n  boto3 버전: {boto3.__version__}")
        
        # STS를 통한 자격 증명 확인
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        
        print(f"\n✅ AWS 연결 성공!")
        print(f"  Account: {identity['Account']}")
        print(f"  ARN: {identity['Arn']}")
        print(f"  UserId: {identity['UserId']}")
        
        return True
        
    except NoCredentialsError:
        print("\n❌ AWS 자격 증명을 찾을 수 없습니다.")
        return False
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        print(f"\n❌ AWS 연결 실패: {error_code}")
        print(f"   {error_msg}")
        return False
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False


def test_s3_access():
    """S3 버킷 접근 테스트"""
    print("\n" + "=" * 60)
    print("🪣 S3 버킷 접근 테스트")
    print("=" * 60)
    
    bucket = os.getenv("AWS_S3_BUCKET", "p-ade-datasets")
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3 = boto3.client("s3")
        
        # 버킷 존재 확인
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"\n✅ 버킷 '{bucket}' 접근 가능!")
            
            # 버킷 위치 확인
            location = s3.get_bucket_location(Bucket=bucket)
            region = location.get("LocationConstraint") or "us-east-1"
            print(f"  리전: {region}")
            
            # 객체 목록 조회 (최대 5개)
            response = s3.list_objects_v2(Bucket=bucket, MaxKeys=5)
            obj_count = response.get("KeyCount", 0)
            print(f"  객체 수 (샘플): {obj_count}개")
            
            if obj_count > 0:
                print("  최근 객체:")
                for obj in response.get("Contents", []):
                    print(f"    - {obj['Key']} ({obj['Size']} bytes)")
                    
            return True
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            
            if error_code == "404":
                print(f"\n⚠️ 버킷 '{bucket}'이 존재하지 않습니다.")
                print("   버킷을 생성하시겠습니까? (upload_to_s3.py 실행 시 자동 생성)")
                return True  # 자격 증명은 유효함
                
            elif error_code == "403":
                print(f"\n❌ 버킷 '{bucket}' 접근 권한이 없습니다.")
                print("   IAM 정책을 확인하세요.")
                return False
                
            else:
                print(f"\n❌ 버킷 접근 실패: {error_code}")
                print(f"   {e.response.get('Error', {}).get('Message', str(e))}")
                return False
                
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False


def test_upload():
    """테스트 파일 업로드"""
    print("\n" + "=" * 60)
    print("📤 테스트 업로드")
    print("=" * 60)
    
    bucket = os.getenv("AWS_S3_BUCKET", "p-ade-datasets")
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        from datetime import datetime
        
        s3 = boto3.client("s3")
        
        # 테스트 데이터
        test_key = f"test/connection-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        test_content = f"P-ADE S3 연결 테스트\n시간: {datetime.now().isoformat()}"
        
        # 업로드
        s3.put_object(
            Bucket=bucket,
            Key=test_key,
            Body=test_content.encode("utf-8"),
            ContentType="text/plain",
        )
        
        print(f"\n✅ 테스트 파일 업로드 성공!")
        print(f"  URI: s3://{bucket}/{test_key}")
        
        # 다운로드 확인
        response = s3.get_object(Bucket=bucket, Key=test_key)
        downloaded = response["Body"].read().decode("utf-8")
        
        if downloaded == test_content:
            print("✅ 다운로드 검증 성공!")
        else:
            print("⚠️ 다운로드된 내용이 다릅니다.")
            
        # 정리 (테스트 파일 삭제)
        s3.delete_object(Bucket=bucket, Key=test_key)
        print("🗑️ 테스트 파일 삭제 완료")
        
        return True
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        
        if error_code == "NoSuchBucket":
            print(f"\n⚠️ 버킷 '{bucket}'이 없습니다. 생성이 필요합니다.")
            create = input("   버킷을 생성하시겠습니까? (y/n): ").strip().lower()
            if create == "y":
                return create_bucket(bucket)
            return False
        else:
            print(f"\n❌ 업로드 실패: {error_code}")
            print(f"   {e.response.get('Error', {}).get('Message', str(e))}")
            return False
            
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False


def create_bucket(bucket_name: str) -> bool:
    """버킷 생성"""
    try:
        import boto3
        
        region = os.getenv("AWS_REGION", "us-east-1")
        s3 = boto3.client("s3")
        
        create_params = {"Bucket": bucket_name}
        if region != "us-east-1":
            create_params["CreateBucketConfiguration"] = {
                "LocationConstraint": region
            }
            
        s3.create_bucket(**create_params)
        print(f"\n✅ 버킷 '{bucket_name}' 생성 완료! (리전: {region})")
        
        # 퍼블릭 액세스 차단
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        print("🔒 퍼블릭 액세스 차단 설정 완료")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 버킷 생성 실패: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("🧪 P-ADE AWS S3 연결 테스트")
    print("=" * 60 + "\n")
    
    # 1. 자격 증명 확인
    if not test_credentials():
        sys.exit(1)
        
    # 2. boto3 연결 테스트
    if not test_boto3_connection():
        sys.exit(1)
        
    # 3. S3 버킷 접근 테스트
    if not test_s3_access():
        sys.exit(1)
        
    # 4. 테스트 업로드
    if not test_upload():
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("🎉 모든 테스트 통과!")
    print("=" * 60)
    print("\n다음 명령으로 포즈 데이터를 업로드할 수 있습니다:")
    print("  python upload_to_s3.py --all")
    print()


if __name__ == "__main__":
    main()
