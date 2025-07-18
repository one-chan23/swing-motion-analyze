#!/usr/bin/env python3
"""
OpenCV 비디오 코덱 문제 해결 스크립트
"""

import cv2
import os
import urllib.request
import platform

def check_opencv_codecs():
    """사용 가능한 OpenCV 코덱 확인"""
    print("🔍 OpenCV 비디오 코덱 테스트")
    print("=" * 40)
    
    # 테스트할 코덱들
    codecs = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('X264', cv2.VideoWriter_fourcc(*'X264')),
    ]
    
    working_codecs = []
    
    for codec_name, fourcc in codecs:
        try:
            # 임시 비디오 파일로 테스트
            test_file = f'test_{codec_name}.mp4'
            out = cv2.VideoWriter(test_file, fourcc, 30, (640, 480))
            
            if out.isOpened():
                print(f"✅ {codec_name}: 사용 가능")
                working_codecs.append(codec_name)
                out.release()
                
                # 테스트 파일 삭제
                if os.path.exists(test_file):
                    os.remove(test_file)
            else:
                print(f"❌ {codec_name}: 사용 불가")
                out.release()
                
        except Exception as e:
            print(f"❌ {codec_name}: 오류 - {e}")
    
    print(f"\n사용 가능한 코덱: {working_codecs}")
    return working_codecs

def download_openh264():
    """OpenH264 라이브러리 다운로드 (Windows)"""
    if platform.system() != "Windows":
        print("이 스크립트는 Windows용입니다.")
        return False
    
    print("\n📥 OpenH264 라이브러리 다운로드 시도...")
    
    try:
        # OpenH264 라이브러리 URL
        url = "https://github.com/cisco/openh264/releases/download/v2.1.1/openh264-2.1.1-win64.dll.bz2"
        
        # 다운로드 디렉토리
        download_dir = "."
        dll_path = os.path.join(download_dir, "openh264-1.8.0-win64.dll")
        
        print(f"다운로드 중: {url}")
        print("⚠️  수동 다운로드 필요:")
        print("1. https://github.com/cisco/openh264/releases 방문")
        print("2. 최신 Windows 64bit DLL 다운로드")
        print("3. 프로젝트 폴더에 'openh264-1.8.0-win64.dll' 이름으로 저장")
        
        return False
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return False

def fix_opencv_video():
    """OpenCV 비디오 문제 종합 해결"""
    print("🛠️  OpenCV 비디오 코덱 문제 해결")
    print("=" * 50)
    
    # 1. 코덱 테스트
    working_codecs = check_opencv_codecs()
    
    if working_codecs:
        print(f"\n✅ 해결됨! 사용 가능한 코덱: {', '.join(working_codecs)}")
        print("앱이 자동으로 사용 가능한 코덱을 선택합니다.")
        return True
    
    # 2. OpenH264 라이브러리 문제 해결 시도
    print("\n❌ 사용 가능한 코덱이 없습니다.")
    
    if platform.system() == "Windows":
        print("\n💡 해결 방법:")
        print("1. 관리자 권한으로 명령 프롬프트 실행")
        print("2. 다음 명령 실행:")
        print("   pip uninstall opencv-python")
        print("   pip install opencv-python==4.8.1.78")
        print("\n3. 또는 다른 버전 시도:")
        print("   pip install opencv-contrib-python")
        
        download_openh264()
    
    return False

def test_video_creation():
    """비디오 생성 테스트"""
    print("\n🎬 비디오 생성 테스트")
    print("=" * 30)
    
    try:
        import numpy as np
        
        # 테스트 이미지 생성
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (100, 150, 200)  # 배경색
        
        # 텍스트 추가
        cv2.putText(img, 'Test Video', (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # 비디오 생성 테스트
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_output.mp4', fourcc, 30, (640, 480))
        
        if out.isOpened():
            # 30프레임 쓰기
            for i in range(30):
                out.write(img)
            
            out.release()
            
            # 파일 생성 확인
            if os.path.exists('test_output.mp4') and os.path.getsize('test_output.mp4') > 1000:
                print("✅ 비디오 생성 테스트 성공!")
                print(f"   파일 크기: {os.path.getsize('test_output.mp4')} bytes")
                
                # 테스트 파일 삭제
                os.remove('test_output.mp4')
                return True
            else:
                print("❌ 비디오 파일이 제대로 생성되지 않음")
                return False
        else:
            print("❌ VideoWriter 초기화 실패")
            return False
            
    except Exception as e:
        print(f"❌ 비디오 생성 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("🎥 OpenCV 비디오 코덱 문제 진단 및 해결")
    print("=" * 60)
    
    # OpenCV 버전 확인
    print(f"OpenCV 버전: {cv2.__version__}")
    
    # 코덱 문제 해결
    success = fix_opencv_video()
    
    if success:
        # 비디오 생성 테스트
        test_video_creation()
    
    print("\n" + "=" * 60)
    print("완료! 이제 app.py를 다시 실행해보세요.")