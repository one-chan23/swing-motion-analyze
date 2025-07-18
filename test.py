#!/usr/bin/env python3
"""
비디오 저장 문제 진단 및 테스트 스크립트
이 스크립트를 실행해서 어떤 코덱이 작동하는지 확인하세요
"""

import cv2
import numpy as np
import os
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_video_codecs():
    """다양한 비디오 코덱 테스트"""
    
    logger.info("🎬 비디오 코덱 테스트 시작")
    logger.info(f"📱 OpenCV 버전: {cv2.__version__}")
    
    # 테스트할 코덱들
    codecs_to_test = [
        ('mp4v', '.mp4', 'MP4V - 가장 호환성 좋음'),
        ('XVID', '.avi', 'XVID - 일반적으로 잘 작동'),
        ('MJPG', '.avi', 'MJPEG - 압축률 낮지만 안정적'),
        ('X264', '.mp4', 'X264 - 고화질 압축'),
        ('H264', '.mp4', 'H264 - 하드웨어 가속'),
        ('avc1', '.mp4', 'AVC1 - Apple 호환'),
        ('FMP4', '.mp4', 'FMP4 - MPEG-4'),
        ('DIV3', '.avi', 'DivX 3'),
        ('DIVX', '.avi', 'DivX'),
        ('WMV1', '.wmv', 'Windows Media Video'),
        ('WMV2', '.wmv', 'Windows Media Video 2'),
    ]
    
    # 테스트 비디오 파라미터
    width, height = 640, 480
    fps = 30
    duration = 2  # 2초
    total_frames = fps * duration
    
    successful_codecs = []
    failed_codecs = []
    
    # 출력 폴더 생성
    output_folder = "codec_test_output"
    os.makedirs(output_folder, exist_ok=True)
    
    for codec_name, extension, description in codecs_to_test:
        logger.info(f"\n🧪 테스트 중: {codec_name} ({description})")
        
        output_filename = f"test_{codec_name}{extension}"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            # 코덱 설정
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.warning(f"   ❌ VideoWriter 초기화 실패")
                failed_codecs.append((codec_name, extension, "VideoWriter 초기화 실패"))
                continue
            
            # 테스트 프레임 생성 및 저장
            for frame_num in range(total_frames):
                # 무지개색 배경 생성
                hue = int((frame_num / total_frames) * 180)
                hsv = np.full((height, width, 3), [hue, 255, 255], dtype=np.uint8)
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                # 움직이는 원 그리기
                center_x = int((frame_num / total_frames) * (width - 100)) + 50
                center_y = height // 2
                cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
                
                # 프레임 번호 텍스트
                cv2.putText(frame, f'Frame: {frame_num}', (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Codec: {codec_name}', (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 프레임 쓰기
                out.write(frame)
            
            out.release()
            
            # 파일 생성 확인
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                
                if file_size > 0:
                    # 생성된 비디오 검증
                    test_cap = cv2.VideoCapture(output_path)
                    if test_cap.isOpened():
                        saved_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        saved_fps = test_cap.get(cv2.CAP_PROP_FPS)
                        test_cap.release()
                        
                        logger.info(f"   ✅ 성공!")
                        logger.info(f"      📁 파일: {output_path}")
                        logger.info(f"      📏 크기: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                        logger.info(f"      🎬 프레임: {saved_frames}/{total_frames}")
                        logger.info(f"      ⏱️ FPS: {saved_fps}")
                        
                        successful_codecs.append({
                            'codec': codec_name,
                            'extension': extension,
                            'description': description,
                            'file_path': output_path,
                            'file_size': file_size,
                            'frames': saved_frames,
                            'fps': saved_fps
                        })
                    else:
                        logger.warning(f"   ❌ 생성된 파일을 읽을 수 없음")
                        failed_codecs.append((codec_name, extension, "생성된 파일을 읽을 수 없음"))
                        os.remove(output_path)
                else:
                    logger.warning(f"   ❌ 파일 크기가 0 bytes")
                    failed_codecs.append((codec_name, extension, "파일 크기가 0 bytes"))
                    os.remove(output_path)
            else:
                logger.warning(f"   ❌ 파일이 생성되지 않음")
                failed_codecs.append((codec_name, extension, "파일이 생성되지 않음"))
                
        except Exception as e:
            logger.error(f"   ❌ 오류: {e}")
            failed_codecs.append((codec_name, extension, f"오류: {str(e)}"))
    
    # 결과 요약
    logger.info(f"\n🎯 테스트 결과 요약")
    logger.info(f"✅ 성공한 코덱: {len(successful_codecs)}개")
    logger.info(f"❌ 실패한 코덱: {len(failed_codecs)}개")
    
    if successful_codecs:
        logger.info(f"\n🏆 사용 가능한 코덱들:")
        for codec_info in successful_codecs:
            logger.info(f"   - {codec_info['codec']}{codec_info['extension']}: {codec_info['description']}")
            logger.info(f"     크기: {codec_info['file_size']:,} bytes, 프레임: {codec_info['frames']}")
        
        # 권장 코덱
        recommended = successful_codecs[0]
        logger.info(f"\n💡 권장 코덱: {recommended['codec']}{recommended['extension']}")
        logger.info(f"   이유: 첫 번째로 성공한 코덱이므로 가장 안정적")
        
        return successful_codecs
    else:
        logger.error(f"\n❌ 사용 가능한 코덱이 없습니다!")
        logger.error(f"OpenCV 설치에 문제가 있을 수 있습니다.")
        
        # 문제 해결 제안
        logger.info(f"\n🔧 문제 해결 방법:")
        logger.info(f"1. OpenCV 재설치: pip uninstall opencv-python && pip install opencv-python")
        logger.info(f"2. 추가 코덱 설치: pip install opencv-contrib-python")
        logger.info(f"3. 시스템 코덱 확인 (Windows: K-Lite Codec Pack, macOS: 기본 지원)")
        
        return []

def test_mediapipe_integration():
    """MediaPipe와 비디오 저장 통합 테스트"""
    
    logger.info(f"\n🤖 MediaPipe 통합 테스트 시작")
    
    try:
        import mediapipe as mp
        logger.info(f"📱 MediaPipe 버전: {mp.__version__}")
    except ImportError:
        logger.error(f"❌ MediaPipe가 설치되지 않았습니다: pip install mediapipe")
        return False
    
    # 간단한 테스트 비디오 생성 (손 검출)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    width, height = 640, 480
    fps = 30
    duration = 3
    total_frames = fps * duration
    
    output_path = "codec_test_output/mediapipe_test.mp4"
    
    try:
        # 가장 호환성 좋은 코덱 사용
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"❌ VideoWriter 초기화 실패")
            return False
        
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            for frame_num in range(total_frames):
                # 간단한 배경 생성
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # 가짜 손 랜드마크 그리기 (실제 손 없이)
                center_x = int((frame_num / total_frames) * (width - 200)) + 100
                center_y = height // 2
                
                # 손 모양 시뮬레이션
                for i in range(5):  # 5개 손가락
                    finger_x = center_x + (i - 2) * 30
                    finger_y = center_y - 50
                    cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 0), -1)
                
                # 손바닥
                cv2.circle(frame, (center_x, center_y), 15, (0, 255, 0), -1)
                
                # 정보 텍스트
                cv2.putText(frame, f'MediaPipe Test - Frame: {frame_num}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Hand Tracking Simulation', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                out.write(frame)
        
        out.release()
        
        # 결과 확인
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ MediaPipe 통합 테스트 성공!")
            logger.info(f"   📁 파일: {output_path}")
            logger.info(f"   📏 크기: {file_size:,} bytes")
            return True
        else:
            logger.error(f"❌ MediaPipe 테스트 파일이 생성되지 않음")
            return False
            
    except Exception as e:
        logger.error(f"❌ MediaPipe 통합 테스트 오류: {e}")
        return False

def cleanup_test_files():
    """테스트 파일 정리"""
    test_folder = "codec_test_output"
    if os.path.exists(test_folder):
        import shutil
        try:
            shutil.rmtree(test_folder)
            logger.info(f"🧹 테스트 파일 정리 완료")
        except Exception as e:
            logger.warning(f"⚠️ 테스트 파일 정리 실패: {e}")

if __name__ == "__main__":
    logger.info("🚀 비디오 저장 문제 진단 시작")
    logger.info("=" * 60)
    
    # 1. 코덱 테스트
    successful_codecs = test_video_codecs()
    
    # 2. MediaPipe 통합 테스트
    if successful_codecs:
        test_mediapipe_integration()
    
    # 3. 결과 및 권장사항
    logger.info("\n" + "=" * 60)
    logger.info("🎯 최종 결과 및 권장사항")
    
    if successful_codecs:
        best_codec = successful_codecs[0]
        logger.info(f"✅ 권장 설정:")
        logger.info(f"   코덱: {best_codec['codec']}")
        logger.info(f"   확장자: {best_codec['extension']}")
        logger.info(f"   설명: {best_codec['description']}")
        
        logger.info(f"\n📝 app.py에서 사용할 코드:")
        logger.info(f"   fourcc = cv2.VideoWriter_fourcc(*'{best_codec['codec']}')")
        logger.info(f"   out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))")
        
    else:
        logger.error(f"❌ 사용 가능한 코덱이 없습니다.")
        logger.error(f"시스템 설정이나 OpenCV 설치를 확인해주세요.")
    
    # 정리할지 묻기
    choice = input(f"\n🧹 테스트 파일을 삭제하시겠습니까? (y/N): ").lower()
    if choice == 'y':
        cleanup_test_files()
    else:
        logger.info(f"📁 테스트 파일은 'codec_test_output' 폴더에 보관됩니다.")
    
    logger.info(f"✅ 진단 완료!")