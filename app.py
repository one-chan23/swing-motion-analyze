#!/usr/bin/env python3
"""
야구 타격자세 분석기 Flask 서버 - 비디오 저장 문제 해결 버전
전자통신공학과 지능시스템 및 실험 최종 프로젝트
"""

# 표준 라이브러리
import os
import sys
import json
import uuid
import math
import time
import threading
import traceback
import logging
import re
import shutil
from datetime import datetime
from flask import Response, Flask, render_template, request, jsonify, send_file, session, send_from_directory
from werkzeug.utils import secure_filename
import mimetypes

# 컴퓨터 비전 라이브러리
import cv2
import mediapipe as mp
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask 앱 초기화
app = Flask(__name__)
app.secret_key = 'baseball_analyzer_secret_key_2024'

# 프로젝트 설정
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
PROCESSED_VIDEOS_FOLDER = 'static/processed_videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# 폴더 생성
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, PROCESSED_VIDEOS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Flask 설정
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 제한


class BaseballAnalyzer:
    """야구 타격자세 분석 클래스"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.analysis_results = {}
        
        # 사용 가능한 비디오 코덱 확인
        self.available_codecs = self._check_available_codecs()
        
        # AI 모델 참조 데이터 (프로 선수 기준)
        self.reference_data = {
            'professional': {
                'shoulder_balance': 0.03,
                'knee_angle_left': 162,
                'knee_angle_right': 160,
                'foot_distance': 0.38,
                'shoulder_rotation': 28,
                'hip_rotation': 22,
                'elbow_angle_left': 118,
                'elbow_angle_right': 115,
                'wrist_speed': 0.18,
                'overall_score': 92
            }
        }

    def _check_available_codecs(self):
        """사용 가능한 비디오 코덱 확인"""
        logger.info("비디오 코덱 확인 중...")
        
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        codecs_to_test = [
            ('X264', '.mp4'),  # MP4V - 가장 호환성 좋음
            ('XVID', '.avi'),  # XVID 
            ('MJPG', '.avi'),  # MJPEG
            ('mp4v', '.mp4'),  # X264
            ('H264', '.mp4'),  # H264
        ]
        
        available = []
        
        for codec_name, extension in codecs_to_test:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                test_path = f'test_codec_{codec_name}.mp4'
                test_writer = cv2.VideoWriter(test_path, fourcc, 30, (100, 100))
                
                if test_writer.isOpened():
                    test_writer.write(test_frame)
                    test_writer.release()
                    
                    if os.path.exists(test_path):
                        file_size = os.path.getsize(test_path)
                        if file_size > 0:
                            available.append((codec_name, extension))
                            logger.info(f"✅ 코덱 {codec_name} 사용 가능")
                        os.remove(test_path)
                    else:
                        logger.warning(f"❌ 코덱 {codec_name}: 파일 생성 실패")
                else:
                    logger.warning(f"❌ 코덱 {codec_name}: VideoWriter 초기화 실패")
                    
            except Exception as e:
                logger.warning(f"❌ 코덱 {codec_name} 오류: {e}")
        
        if not available:
            logger.error("❌ 사용 가능한 비디오 코덱이 없습니다!")
            available = [('x264', '.mp4')]  # 기본값
        
        logger.info(f"사용 가능한 코덱: {[codec[0] for codec in available]}")
        return available

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def calculate_angle(self, a, b, c):
        """세 점으로 각도 계산"""
        try:
            a, b, c = np.array(a), np.array(b), np.array(c)
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            return 360 - angle if angle > 180.0 else angle
        except:
            return 0

    def calculate_distance(self, point1, point2):
        """두 점 사이의 거리 계산"""
        try:
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except:
            return 0

    def get_body_landmarks(self, landmarks):
        """신체 랜드마크 추출"""
        try:
            return {
                'left_shoulder': [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y],
                'right_shoulder': [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                'left_elbow': [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y],
                'right_elbow': [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y],
                'left_wrist': [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y],
                'right_wrist': [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y],
                'left_hip': [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y],
                'right_hip': [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y],
                'left_knee': [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y],
                'right_knee': [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y],
                'left_ankle': [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y],
                'right_ankle': [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y]
            }
        except:
            return None

    def analyze_stance(self, body_points):
        """타격 자세 분석"""
        try:
            shoulder_level = abs(body_points['left_shoulder'][1] - body_points['right_shoulder'][1])
            left_knee_angle = self.calculate_angle(body_points['left_hip'], body_points['left_knee'], body_points['left_ankle'])
            right_knee_angle = self.calculate_angle(body_points['right_hip'], body_points['right_knee'], body_points['right_ankle'])
            foot_distance = self.calculate_distance(body_points['left_ankle'], body_points['right_ankle'])
            
            return {
                'shoulder_level': shoulder_level,
                'left_knee_angle': left_knee_angle,
                'right_knee_angle': right_knee_angle,
                'foot_distance': foot_distance
            }
        except:
            return {'shoulder_level': 0, 'left_knee_angle': 0, 'right_knee_angle': 0, 'foot_distance': 0}

    def analyze_swing(self, body_points, prev_body_points=None):
        """스윙 분석"""
        try:
            shoulder_vector = np.array(body_points['right_shoulder']) - np.array(body_points['left_shoulder'])
            shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0]) * 180 / np.pi
            
            left_elbow_angle = self.calculate_angle(body_points['left_shoulder'], body_points['left_elbow'], body_points['left_wrist'])
            right_elbow_angle = self.calculate_angle(body_points['right_shoulder'], body_points['right_elbow'], body_points['right_wrist'])
            
            hip_vector = np.array(body_points['right_hip']) - np.array(body_points['left_hip'])
            hip_angle = np.arctan2(hip_vector[1], hip_vector[0]) * 180 / np.pi
            
            wrist_speed = 0
            if prev_body_points:
                wrist_speed = self.calculate_distance(body_points['right_wrist'], prev_body_points['right_wrist'])
            
            return {
                'shoulder_angle': shoulder_angle,
                'left_elbow_angle': left_elbow_angle,
                'right_elbow_angle': right_elbow_angle,
                'hip_angle': hip_angle,
                'wrist_speed': wrist_speed
            }
        except:
            return {'shoulder_angle': 0, 'left_elbow_angle': 0, 'right_elbow_angle': 0, 'hip_angle': 0, 'wrist_speed': 0}

    def calculate_ai_score(self, stance_analysis, swing_analysis):
        """AI 기반 점수 계산"""
        try:
            pro_data = self.reference_data['professional']
            current_data = {
                'shoulder_balance': stance_analysis['shoulder_level'],
                'knee_angle_left': stance_analysis['left_knee_angle'],
                'knee_angle_right': stance_analysis['right_knee_angle'],
                'foot_distance': stance_analysis['foot_distance'],
                'shoulder_rotation': abs(swing_analysis['shoulder_angle']),
                'hip_rotation': abs(swing_analysis['hip_angle']),
                'elbow_angle_left': swing_analysis['left_elbow_angle'],
                'elbow_angle_right': swing_analysis['right_elbow_angle'],
                'wrist_speed': swing_analysis['wrist_speed']
            }
            
            weights = {
                'shoulder_balance': 0.15, 'knee_angle_left': 0.1, 'knee_angle_right': 0.1,
                'foot_distance': 0.1, 'shoulder_rotation': 0.2, 'hip_rotation': 0.15,
                'elbow_angle_left': 0.1, 'elbow_angle_right': 0.1, 'wrist_speed': 0.1
            }
            
            total_score = 0
            for key, current_value in current_data.items():
                pro_value = pro_data[key]
                if pro_value != 0:
                    diff_ratio = abs(current_value - pro_value) / max(abs(pro_value), 1)
                    similarity = max(0, 1 - diff_ratio)
                    total_score += similarity * weights[key] * 100
            
            return min(100, max(0, total_score))
        except:
            return 50

    def get_grade(self, score):
        """점수에 따른 등급 반환"""
        if score >= 90: return 'Excellent'
        elif score >= 80: return 'Good'
        elif score >= 70: return 'Average'
        elif score >= 60: return 'Below Average'
        else: return 'Poor'

    def get_recommendations(self, stance_analysis, swing_analysis, score):
        """개선 제안 생성"""
        recommendations = []
        try:
            if stance_analysis['shoulder_level'] > 0.05:
                recommendations.append("어깨의 균형을 맞춰주세요")
            if stance_analysis['left_knee_angle'] < 150:
                recommendations.append("왼쪽 무릎을 조금 더 펴주세요")
            if stance_analysis['right_knee_angle'] < 150:
                recommendations.append("오른쪽 무릎을 조금 더 펴주세요")
            if stance_analysis['foot_distance'] < 0.3:
                recommendations.append("발 간격을 좀 더 넓혀주세요")
            elif stance_analysis['foot_distance'] > 0.5:
                recommendations.append("발 간격을 좀 더 좁혀주세요")
            if swing_analysis['left_elbow_angle'] < 100:
                recommendations.append("왼팔 팔꿈치를 좀 더 펴주세요")
            if swing_analysis['right_elbow_angle'] < 100:
                recommendations.append("오른팔 팔꿈치를 좀 더 펴주세요")
            if score < 70:
                recommendations.append("기본 자세 연습을 더 해보세요")
                recommendations.append("프로 선수의 영상을 참고하여 폼을 개선하세요")
            return recommendations if recommendations else ["훌륭한 자세입니다"]
        except:
            return ["분석을 완료했습니다"]

    def draw_pose_landmarks(self, image, landmarks):
        """포즈 랜드마크 그리기 (관절 점 + 연결선) - 안정화된 버전"""
        try:
            if not landmarks:
                return
            
            h, w, _ = image.shape
            important_landmarks = [
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                mp.solutions.pose.PoseLandmark.LEFT_WRIST, mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
                mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
                mp.solutions.pose.PoseLandmark.LEFT_ANKLE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
            ]
            
            connections = [
                (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
                (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
                (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
                (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
                (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
                (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
                (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
                (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
                (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
                (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
                (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
                (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE)
            ]
            
            # 연결선 그리기 (흰색)
            for start_landmark, end_landmark in connections:
                try:
                    start_point = landmarks[start_landmark.value]
                    end_point = landmarks[end_landmark.value]
                    if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                        start_x, start_y = int(start_point.x * w), int(start_point.y * h)
                        end_x, end_y = int(end_point.x * w), int(end_point.y * h)
                        
                        # 화면 경계 확인
                        if (0 <= start_x < w and 0 <= start_y < h and 
                            0 <= end_x < w and 0 <= end_y < h):
                            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
                except:
                    continue
            
            # 관절점 그리기 (흰색 점)
            for landmark_type in important_landmarks:
                try:
                    landmark = landmarks[landmark_type.value]
                    if landmark.visibility > 0.5:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        
                        # 화면 경계 확인
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(image, (x, y), 4, (255, 255, 255), -1)
                            cv2.circle(image, (x, y), 4, (0, 0, 0), 1)
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"포즈 그리기 오류: {e}")

    def draw_info_overlay(self, image, score, grade, swing_count, swing_intensity=0):
        """정보 오버레이"""
        try:
            cv2.putText(image, f'Score: {score:.1f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f'Grade: {grade}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f'Swings: {swing_count}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if swing_intensity > 0:
                intensity_color = (0, 255, 0) if swing_intensity > 0.35 else (0, 255, 255)
                cv2.putText(image, f'Intensity: {swing_intensity:.2f}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, intensity_color, 2)
        except Exception as e:
            logger.warning(f"오버레이 오류: {e}")

    def detect_swing_sequence(self, body_points, prev_body_points, frame_count, fps):
        """스윙 감지"""
        if not prev_body_points:
            return False, 0
        
        try:
            # 손목과 팔꿈치 움직임
            right_wrist_movement = self.calculate_distance(body_points['right_wrist'], prev_body_points['right_wrist'])
            left_wrist_movement = self.calculate_distance(body_points['left_wrist'], prev_body_points['left_wrist'])
            right_elbow_movement = self.calculate_distance(body_points['right_elbow'], prev_body_points['right_elbow'])
            
            # 어깨 회전
            current_shoulder_angle = np.arctan2(
                body_points['right_shoulder'][1] - body_points['left_shoulder'][1],
                body_points['right_shoulder'][0] - body_points['left_shoulder'][0]
            ) * 180 / np.pi
            
            prev_shoulder_angle = np.arctan2(
                prev_body_points['right_shoulder'][1] - prev_body_points['left_shoulder'][1],
                prev_body_points['right_shoulder'][0] - prev_body_points['left_shoulder'][0]
            ) * 180 / np.pi
            
            shoulder_rotation = abs(current_shoulder_angle - prev_shoulder_angle)
            
            # 스윙 강도 계산
            swing_intensity = (
                min(right_wrist_movement * 3, 1.0) * 0.3 +
                min(left_wrist_movement * 3, 1.0) * 0.2 +
                min(right_elbow_movement * 4, 1.0) * 0.2 +
                min(shoulder_rotation / 30, 1.0) * 0.3
            )
            
            return swing_intensity > 0.35, swing_intensity
        except:
            return False, 0

    def process_video(self, video_path, session_id):
        """개선된 비디오 처리 및 분석"""
        try:
            logger.info(f"🎬 비디오 분석 시작: {session_id}")
            
            # 입력 비디오 열기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
                return None
            
            # 비디오 정보
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"📹 비디오 정보 - FPS: {fps}, 총 프레임: {total_frames}, 해상도: {width}x{height}")
            
            # 출력 비디오 설정 - 여러 코덱으로 시도
            output_path = None
            out = None
            processed_video_url = None
            
            for codec_name, extension in self.available_codecs:
                try:
                    output_filename = f'{session_id}_processed{extension}'
                    temp_output_path = os.path.join(PROCESSED_VIDEOS_FOLDER, output_filename)
                    
                    logger.info(f"🔧 코덱 {codec_name} 시도 중...")
                    
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    temp_out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                    
                    if temp_out.isOpened():
                        logger.info(f"✅ 코덱 {codec_name} 성공!")
                        output_path = temp_output_path
                        out = temp_out
                        processed_video_url = f'/static/processed_videos/{output_filename}'
                        break
                    else:
                        logger.warning(f"❌ 코덱 {codec_name} 실패")
                        
                except Exception as e:
                    logger.warning(f"코덱 {codec_name} 오류: {e}")
                    continue
            
            if out is None or not out.isOpened():
                logger.error("❌ 사용 가능한 비디오 코덱을 찾을 수 없습니다")
                cap.release()
                return None
            
            logger.info(f"📁 출력 파일: {output_path}")
            logger.info(f"🌐 웹 URL: {processed_video_url}")
            
            # 분석 변수 초기화
            frame_count = 0
            processed_frames = 0
            swing_count = 0
            swing_analyses = []
            prev_body_points = None
            swing_in_progress = False
            swing_start_frame = 0
            swing_peak_intensity = 0
            frames_since_last_swing = 0
            min_swing_gap = fps * 1.0
            
            # 초기 상태 설정
            self.analysis_results[session_id] = {
                'progress': 0,
                'current_score': 0,
                'current_grade': 'Poor',
                'swing_count': 0,
                'status': 'processing'
            }
            
            with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    try:
                        # 이미지 처리
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(image)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        current_score = 0
                        current_grade = 'Poor'
                        swing_intensity = 0
                        
                        if results.pose_landmarks:
                            # 포즈 랜드마크 그리기
                            self.draw_pose_landmarks(image, results.pose_landmarks.landmark)
                            
                            body_points = self.get_body_landmarks(results.pose_landmarks.landmark)
                            if body_points:
                                stance_analysis = self.analyze_stance(body_points)
                                swing_analysis = self.analyze_swing(body_points, prev_body_points)
                                
                                current_score = self.calculate_ai_score(stance_analysis, swing_analysis)
                                current_grade = self.get_grade(current_score)
                                
                                # 스윙 감지
                                is_swinging, swing_intensity = self.detect_swing_sequence(
                                    body_points, prev_body_points, frame_count, fps
                                )
                                
                                frames_since_last_swing += 1
                                
                                # 스윙 시작 감지
                                if (is_swinging and not swing_in_progress and frames_since_last_swing > min_swing_gap):
                                    swing_in_progress = True
                                    swing_start_frame = frame_count
                                    swing_peak_intensity = swing_intensity
                                    frames_since_last_swing = 0
                                    logger.info(f"🏏 스윙 시작 감지: 프레임 {frame_count}")
                                
                                # 스윙 진행 중
                                elif swing_in_progress and is_swinging:
                                    swing_peak_intensity = max(swing_peak_intensity, swing_intensity)
                                
                                # 스윙 종료 감지
                                elif (swing_in_progress and not is_swinging and 
                                      (frame_count - swing_start_frame) > fps * 0.3 and 
                                      swing_peak_intensity > 0.4):
                                    
                                    swing_in_progress = False
                                    swing_count += 1
                                    swing_duration = (frame_count - swing_start_frame) / fps
                                    
                                    recommendations = self.get_recommendations(stance_analysis, swing_analysis, current_score)
                                    
                                    swing_analyses.append({
                                        'swing_number': swing_count,
                                        'frame': frame_count,
                                        'duration': swing_duration,
                                        'peak_intensity': swing_peak_intensity,
                                        'timestamp': frame_count / fps,
                                        'ai_score': current_score,
                                        'grade': current_grade,
                                        'recommendations': recommendations
                                    })
                                    
                                    logger.info(f"✅ 스윙 완료: #{swing_count}, 점수: {current_score:.1f}")
                                    swing_peak_intensity = 0
                                
                                prev_body_points = body_points
                        
                        # 정보 오버레이 (점수, 등급, 스윙 횟수 표시)
                        self.draw_info_overlay(image, current_score, current_grade, swing_count, swing_intensity)
                        
                        # 처리된 프레임을 출력 비디오에 저장
                        out.write(image)
                        processed_frames += 1
                        
                        # 진행률 업데이트 (매 10프레임마다)
                        if frame_count % 10 == 0:
                            progress = (frame_count / total_frames) * 100
                            self.analysis_results[session_id].update({
                                'progress': progress,
                                'current_score': current_score,
                                'current_grade': current_grade,
                                'swing_count': swing_count,
                                'status': 'processing',
                                'frames_processed': frame_count,
                                'total_frames': total_frames
                            })
                            
                            # 진행률 로그 (매 100프레임마다)
                            if frame_count % 100 == 0:
                                logger.info(f"📊 진행률: {progress:.1f}% ({frame_count}/{total_frames})")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ 프레임 {frame_count} 처리 오류: {e}")
                        # 오류가 발생해도 원본 프레임은 저장
                        try:
                            out.write(frame)
                            processed_frames += 1
                        except:
                            pass
            
            # 비디오 스트림 정리
            cap.release()
            out.release()
            
            # 파일 생성 및 무결성 확인
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ 처리된 비디오 파일 생성 완료!")
                logger.info(f"   📁 파일 경로: {output_path}")
                logger.info(f"   📏 파일 크기: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                logger.info(f"   🎬 처리된 프레임: {processed_frames}/{total_frames}")
                
                # 생성된 비디오 무결성 확인
                test_cap = cv2.VideoCapture(output_path)
                if test_cap.isOpened():
                    test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    test_duration = test_cap.get(cv2.CAP_PROP_FRAME_COUNT) / test_cap.get(cv2.CAP_PROP_FPS)
                    test_cap.release()
                    logger.info(f"   ✅ 저장된 프레임 수: {test_frame_count}")
                    logger.info(f"   ⏱️ 비디오 길이: {test_duration:.1f}초")
                    
                    if test_frame_count > 0 and file_size > 1024:  # 최소 1KB 이상
                        # 최종 결과 계산
                        if swing_analyses:
                            avg_score = sum(s['ai_score'] for s in swing_analyses) / len(swing_analyses)
                            max_score = max(s['ai_score'] for s in swing_analyses)
                            min_score = min(s['ai_score'] for s in swing_analyses)
                        else:
                            avg_score = max_score = min_score = 50
                        
                        # 최종 결과 저장
                        final_results = {
                            'status': 'completed',
                            'session_id': session_id,
                            'video_info': {
                                'fps': fps,
                                'total_frames': total_frames,
                                'processed_frames': processed_frames,
                                'duration': total_frames / fps,
                                'processed_video': processed_video_url,
                                'original_resolution': f'{width}x{height}',
                                'file_size': file_size
                            },
                            'summary': {
                                'total_swings': swing_count,
                                'avg_score': round(avg_score, 1),
                                'max_score': round(max_score, 1),
                                'min_score': round(min_score, 1),
                                'final_grade': self.get_grade(avg_score)
                            },
                            'swing_analyses': swing_analyses,
                            'analysis_date': datetime.now().isoformat(),
                            'general_recommendations': self.get_recommendations(
                                stance_analysis if 'stance_analysis' in locals() else {'shoulder_level': 0, 'left_knee_angle': 0, 'right_knee_angle': 0, 'foot_distance': 0},
                                swing_analysis if 'swing_analysis' in locals() else {'shoulder_angle': 0, 'left_elbow_angle': 0, 'right_elbow_angle': 0, 'hip_angle': 0, 'wrist_speed': 0},
                                avg_score
                            ) if swing_analyses else ["분석할 스윙이 감지되지 않았습니다."],
                            'redirect_to_results': True,
                            'progress': 100
                        }
                        
                        # 결과 파일 저장
                        results_file = os.path.join(RESULTS_FOLDER, f'{session_id}_results.json')
                        with open(results_file, 'w', encoding='utf-8') as f:
                            json.dump(final_results, f, indent=2, ensure_ascii=False)
                        
                        self.analysis_results[session_id] = final_results
                        logger.info(f"🎉 비디오 분석 완료: {session_id}")
                        logger.info(f"📊 총 스윙: {swing_count}회, 평균 점수: {avg_score:.1f}점")
                        return final_results
                        
                    else:
                        logger.error("❌ 생성된 비디오가 손상되었거나 비어있습니다")
                else:
                    logger.error("❌ 생성된 비디오를 읽을 수 없습니다")
            else:
                logger.error(f"❌ 처리된 비디오 파일이 생성되지 않았습니다: {output_path}")
            
            # 실패 시 기본 결과 반환
            error_result = {
                'status': 'error',
                'error': '비디오 파일 생성 실패',
                'message': '처리된 비디오 파일을 생성할 수 없습니다.'
            }
            self.analysis_results[session_id] = error_result
            return error_result
            
        except Exception as e:
            logger.error(f"❌ 비디오 처리 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            
            error_result = {
                'status': 'error',
                'error': str(e),
                'message': '비디오 분석 중 오류가 발생했습니다.'
            }
            self.analysis_results[session_id] = error_result
            return error_result


# 전역 분석기 인스턴스
analyzer = BaseballAnalyzer()


# Flask 라우트들 (기존과 동일)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if file and analyzer.allowed_file(file.filename):
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_{filename}')
            file.save(file_path)
            
            logger.info(f"📁 파일 업로드 성공: {session_id}_{filename}")
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Video uploaded successfully'
            })
        
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"❌ 업로드 오류: {e}")
        return jsonify({'success': False, 'error': f'Upload error: {str(e)}'}), 500


@app.route('/analyze/<session_id>')
def analyze_page(session_id):
    return render_template('analyze.html', session_id=session_id)


@app.route('/start_analysis/<session_id>')
def start_analysis(session_id):
    try:
        video_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(session_id)]
        
        if not video_files:
            return jsonify({'success': False, 'error': 'Video file not found'}), 404
        
        video_path = os.path.join(UPLOAD_FOLDER, video_files[0])
        
        def run_analysis():
            analyzer.process_video(video_path, session_id)
        
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        logger.info(f"🚀 분석 시작: {session_id}")
        return jsonify({'success': True, 'message': 'Analysis started'})
        
    except Exception as e:
        logger.error(f"❌ 분석 시작 오류: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analysis_status/<session_id>')
def analysis_status(session_id):
    if session_id in analyzer.analysis_results:
        return jsonify(analyzer.analysis_results[session_id])
    else:
        return jsonify({'status': 'not_started', 'progress': 0})


@app.route('/results/<session_id>')
def results_page(session_id):
    try:
        results_file = os.path.join(RESULTS_FOLDER, f'{session_id}_results.json')
        
        if not os.path.exists(results_file):
            logger.warning(f"❌ 결과 파일이 없습니다: {results_file}")
            return "Results not found", 404
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 비디오 파일 존재 확인 및 로깅
        if results.get('video_info', {}).get('processed_video'):
            video_url = results['video_info']['processed_video']
            video_filename = video_url.split('/')[-1]
            video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, video_filename)
            
            logger.info(f"🎬 비디오 URL: {video_url}")
            logger.info(f"📁 비디오 파일 경로: {video_path}")
            logger.info(f"✅ 비디오 파일 존재: {os.path.exists(video_path)}")
            
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                logger.info(f"📏 비디오 파일 크기: {file_size:,} bytes")
        
        return render_template('results.html', results=results, session_id=session_id)
        
    except Exception as e:
        logger.error(f"❌ 결과 페이지 오류: {e}")
        return f"Error loading results: {str(e)}", 500


@app.route('/download_results/<session_id>')
def download_results(session_id):
    try:
        results_file = os.path.join(RESULTS_FOLDER, f'{session_id}_results.json')
        if os.path.exists(results_file):
            return send_file(results_file, as_attachment=True, 
                            download_name=f'baseball_analysis_{session_id}.json')
        else:
            return "Results not found", 404
    except Exception as e:
        logger.error(f"❌ 결과 다운로드 오류: {e}")
        return f"Error downloading results: {str(e)}", 500


@app.route('/static/processed_videos/<filename>')
def serve_processed_video(filename):
    try:
        video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, filename)
        
        logger.info(f"🎬 비디오 서빙 요청: {filename}")
        logger.info(f"📁 비디오 파일 경로: {video_path}")
        logger.info(f"✅ 파일 존재: {os.path.exists(video_path)}")
        
        if not os.path.exists(video_path):
            logger.error(f"❌ 비디오 파일이 존재하지 않음: {video_path}")
            return "Video file not found", 404
        
        file_size = os.path.getsize(video_path)
        logger.info(f"📏 비디오 파일 크기: {file_size:,} bytes")
        
        # Range 요청 처리
        range_header = request.headers.get('Range', None)
        if range_header:
            return serve_video_with_range(video_path, range_header)
        
        # MIME 타입 명시적 설정
        if filename.endswith('.mp4'):
            mimetype = 'video/mp4'
        elif filename.endswith('.avi'):
            mimetype = 'video/x-msvideo'
        else:
            mimetype = 'video/mp4'
        
        return send_file(video_path, mimetype=mimetype, conditional=True)
        
    except Exception as e:
        logger.error(f"❌ 비디오 파일 서빙 오류: {e}")
        return f"Error serving video: {str(e)}", 500


def serve_video_with_range(video_path, range_header):
    """Range 요청 처리"""
    try:
        file_size = os.path.getsize(video_path)
        byte_start = 0
        byte_end = file_size - 1
        
        if range_header:
            match = re.search(r'bytes=(\d+)-(\d*)', range_header)
            if match:
                byte_start = int(match.group(1))
                if match.group(2):
                    byte_end = int(match.group(2))
        
        content_length = byte_end - byte_start + 1
        
        def generate():
            with open(video_path, 'rb') as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        return Response(
            generate(),
            206,
            headers={
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(content_length),
                'Content-Type': 'video/mp4'
            }
        )
    except Exception as e:
        logger.error(f"❌ Range 처리 오류: {e}")
        return "Error processing range request", 500


@app.route('/static/results/<filename>')
def serve_results_file(filename):
    try:
        return send_from_directory(RESULTS_FOLDER, filename)
    except Exception as e:
        logger.error(f"❌ 결과 파일 서빙 오류: {e}")
        return "Results file not found", 404


# 디버깅 및 테스트 라우트 추가
@app.route('/debug/video_codecs')
def debug_video_codecs():
    """비디오 코덱 디버깅 페이지"""
    try:
        logger.info("🔧 비디오 코덱 디버깅 시작")
        
        # OpenCV 버전 정보
        opencv_version = cv2.__version__
        
        # 사용 가능한 코덱 확인
        available_codecs = analyzer.available_codecs
        
        # 테스트 비디오 생성 시도
        test_results = []
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        for codec_name, extension in available_codecs:
            try:
                test_filename = f'test_{codec_name}{extension}'
                test_path = os.path.join(PROCESSED_VIDEOS_FOLDER, test_filename)
                
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                test_writer = cv2.VideoWriter(test_path, fourcc, 30, (100, 100))
                
                if test_writer.isOpened():
                    for i in range(30):  # 1초 분량
                        test_writer.write(test_frame)
                    test_writer.release()
                    
                    if os.path.exists(test_path):
                        file_size = os.path.getsize(test_path)
                        test_results.append({
                            'codec': codec_name,
                            'extension': extension,
                            'status': 'Success',
                            'file_size': file_size,
                            'test_file': test_filename
                        })
                        logger.info(f"✅ 테스트 성공: {codec_name} - {file_size} bytes")
                    else:
                        test_results.append({
                            'codec': codec_name,
                            'extension': extension,
                            'status': 'Failed - No file created',
                            'file_size': 0
                        })
                else:
                    test_results.append({
                        'codec': codec_name,
                        'extension': extension,
                        'status': 'Failed - VideoWriter not opened',
                        'file_size': 0
                    })
                    
            except Exception as e:
                test_results.append({
                    'codec': codec_name,
                    'extension': extension,
                    'status': f'Error: {str(e)}',
                    'file_size': 0
                })
        
        debug_info = {
            'opencv_version': opencv_version,
            'available_codecs': available_codecs,
            'test_results': test_results,
            'processed_videos_folder': PROCESSED_VIDEOS_FOLDER,
            'folder_exists': os.path.exists(PROCESSED_VIDEOS_FOLDER),
            'folder_writable': os.access(PROCESSED_VIDEOS_FOLDER, os.W_OK)
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"❌ 디버깅 오류: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/debug/test_video/<codec_name>')
def test_specific_codec(codec_name):
    """특정 코덱으로 테스트 비디오 생성"""
    try:
        logger.info(f"🧪 코덱 {codec_name} 테스트 시작")
        
        # 테스트용 비디오 생성
        width, height = 320, 240
        fps = 30
        duration = 3
        total_frames = fps * duration
        
        # 확장자 결정
        extension = '.mp4' if codec_name in ['x264', 'H264', 'avc1'] else '.avi'
        output_filename = f'test_{codec_name}_{int(time.time())}{extension}'
        output_path = os.path.join(PROCESSED_VIDEOS_FOLDER, output_filename)
        
        # 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            return jsonify({'error': f'VideoWriter failed to open with codec {codec_name}'}), 400
        
        # 테스트 프레임 생성
        for frame_num in range(total_frames):
            # 색깔이 변하는 프레임 생성
            color = int((frame_num / total_frames) * 255)
            frame = np.full((height, width, 3), [color, 255 - color, 128], dtype=np.uint8)
            
            # 프레임 번호 텍스트
            cv2.putText(frame, f'{frame_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        # 결과 확인
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            
            # 생성된 비디오 검증
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_fps = test_cap.get(cv2.CAP_PROP_FPS)
                test_cap.release()
                
                result = {
                    'success': True,
                    'codec': codec_name,
                    'file_path': output_path,
                    'file_size': file_size,
                    'frame_count': test_frame_count,
                    'fps': test_fps,
                    'url': f'/static/processed_videos/{output_filename}',
                    'message': f'테스트 비디오 생성 성공'
                }
                
                logger.info(f"✅ 테스트 성공: {codec_name} - {file_size} bytes, {test_frame_count} frames")
                return jsonify(result)
            else:
                return jsonify({'error': 'Created video file cannot be read'}), 500
        else:
            return jsonify({'error': 'Video file was not created'}), 500
            
    except Exception as e:
        logger.error(f"❌ 코덱 테스트 오류: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 500MB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"❌ Internal server error: {e}")
    return jsonify({'success': False, 'error': 'Internal server error occurred.'}), 500


if __name__ == '__main__':
    logger.info("🚀 Flask 서버 시작 중...")
    logger.info(f"📁 업로드 폴더: {UPLOAD_FOLDER}")
    logger.info(f"📁 결과 폴더: {RESULTS_FOLDER}")
    logger.info(f"📁 처리된 비디오 폴더: {PROCESSED_VIDEOS_FOLDER}")
    
    # 필요한 라이브러리 확인
    try:
        import cv2
        import mediapipe as mp
        logger.info("✅ 필수 라이브러리 확인 완료")
        logger.info(f"   - OpenCV: {cv2.__version__}")
        logger.info(f"   - MediaPipe: {mp.__version__}")
    except ImportError as e:
        logger.error(f"❌ 필수 라이브러리 누락: {e}")
        sys.exit(1)
    
    # 포트 시도
    ports_to_try = [8080, 8000, 3000, 5000]
    
    for port in ports_to_try:
        try:
            logger.info(f"🌐 포트 {port}에서 서버 시작...")
            logger.info(f"   접속 주소: http://localhost:{port}")
            logger.info(f"   디버깅 URL: http://localhost:{port}/debug/video_codecs")
            app.run(debug=True, host='127.0.0.1', port=port, threaded=True, use_reloader=False)
            break
        except OSError as e:
            logger.warning(f"⚠️  포트 {port} 사용 불가, 다른 포트 시도...")
            continue
    else:
        logger.error("❌ 사용 가능한 포트를 찾을 수 없습니다.")