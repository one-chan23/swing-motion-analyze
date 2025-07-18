#!/usr/bin/env python3
"""
야구 타격자세 분석기 설정 스크립트
전자통신공학과 지능시스템 및 실험 최종 프로젝트
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Python 버전 확인"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        print(f"현재 버전: {sys.version}")
        return False
    print(f"✅ Python 버전 확인: {sys.version}")
    return True

def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        'uploads',
        'static/results',
        'static/processed_videos',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 디렉토리 생성: {directory}")

def install_requirements():
    """requirements.txt에서 패키지 설치"""
    try:
        print("📦 필요한 패키지 설치 중...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ 패키지 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt 파일을 찾을 수 없습니다.")
        return False

def create_template_files():
    """템플릿 파일 생성 확인"""
    template_files = [
        'templates/index.html',
        'templates/analyze.html', 
        'templates/results.html'
    ]
    
    for template_file in template_files:
        if not os.path.exists(template_file):
            print(f"⚠️  템플릿 파일이 없습니다: {template_file}")
            print("   아티팩트에서 해당 파일을 다운로드하여 저장해주세요.")
        else:
            print(f"✅ 템플릿 파일 확인: {template_file}")

def check_dependencies():
    """주요 라이브러리 임포트 테스트"""
    dependencies = [
        ('Flask', 'flask'),
        ('OpenCV', 'cv2'),
        ('MediaPipe', 'mediapipe'),
        ('NumPy', 'numpy'),
        ('Scikit-learn', 'sklearn')
    ]
    
    print("\n🔍 라이브러리 의존성 확인:")
    all_good = True
    
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - 설치 필요")
            all_good = False
    
    return all_good

def create_demo_data():
    """데모 데이터 디렉토리 생성"""
    demo_dir = 'static/demo_videos'
    os.makedirs(demo_dir, exist_ok=True)
    print(f"✅ 데모 비디오 디렉토리 생성: {demo_dir}")
    
    # 데모 파일이 없다는 안내
    demo_files = ['demo_swing_1.mp4', 'demo_swing_2.mp4', 'demo_bunt.mp4']
    for demo_file in demo_files:
        demo_path = os.path.join(demo_dir, demo_file)
        if not os.path.exists(demo_path):
            print(f"ℹ️  데모 파일 위치: {demo_path} (선택사항)")

def print_usage_instructions():
    """사용법 안내"""
    print("\n" + "="*60)
    print("🚀 야구 타격자세 분석기 설정 완료!")
    print("="*60)
    print("\n📋 실행 방법:")
    print("   python app.py")
    print("\n🌐 접속 주소:")
    print("   http://localhost:5500")
    print("\n📁 프로젝트 구조:")
    print("   ├── app.py                 # Flask 서버")
    print("   ├── requirements.txt       # 필요한 패키지")
    print("   ├── templates/            # HTML 템플릿")
    print("   │   ├── index.html")
    print("   │   ├── analyze.html")
    print("   │   └── results.html")
    print("   ├── uploads/              # 업로드된 비디오")
    print("   ├── static/               # 정적 파일")
    print("   │   ├── results/          # 분석 결과")
    print("   │   └── processed_videos/ # 처리된 비디오")
    print("   └── static/demo_videos/   # 데모 비디오 (선택)")
    print("\n💡 문제 해결:")
    print("   - 포트 충돌 시: app.py에서 port=5500을 다른 번호로 변경")
    print("   - 카메라 권한: MediaPipe 사용을 위해 카메라 권한 허용")
    print("   - 메모리 부족: 큰 비디오 파일은 여러 번에 나누어 처리")

def main():
    """메인 설정 함수"""
    print("🏗️  야구 타격자세 분석기 프로젝트 설정")
    print("=" * 50)
    
    # Python 버전 확인
    if not check_python_version():
        sys.exit(1)
    
    # 디렉토리 생성
    create_directories()
    
    # 패키지 설치
    if not install_requirements():
        print("\n⚠️  패키지 설치에 실패했습니다.")
        print("   수동으로 설치해주세요: pip install -r requirements.txt")
    
    # 템플릿 파일 확인
    create_template_files()
    
    # 의존성 확인
    if not check_dependencies():
        print("\n⚠️  일부 라이브러리가 제대로 설치되지 않았습니다.")
        print("   다시 시도해주세요: pip install -r requirements.txt")
    
    # 데모 데이터 준비
    create_demo_data()
    
    # 사용법 안내
    print_usage_instructions()

if __name__ == "__main__":
    main()