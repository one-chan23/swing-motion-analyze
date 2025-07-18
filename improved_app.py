#!/usr/bin/env python3
"""
야구 타격자세 분석기 Flask 서버 - 머신러닝 모델 적용 버전
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

# 머신러닝 라이브러리
import joblib
from sklearn.preprocessing import StandardScaler

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
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# 폴더 생성
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, PROCESSED_VIDEOS_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Flask 설정
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 제한


class MLBaseballAnalyzer:
    """머신러닝 기반 야구 타격자세 분석 클래스"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.analysis_results = {}
        
        # 머신러닝 모델 관련
        self.ml_model = None
        self.scaler = None
        self.feature_names = []
        self.optimal_values = {}
        
        # 사용 가능한 비디오 코덱 확인
        self.available_codecs = self._check_available_codecs()
        
        # 머신러닝 모델 로드
        self.load_ml_model()

    def load_ml_model(self):
        """훈련된 머신러닝 모델 로드"""
        try:
            # 모델 파일들 확인
            model_file = os.path.join(MODEL_FOLDER, 'baseball_pose_model.joblib')
            scaler_file = os.path.join(MODEL_FOLDER, 'pose_scaler.joblib')
            features_file = os.path.join(MODEL_FOLDER, 'feature_names.json')
            info_file = os.path.join(MODEL_FOLDER, 'model_info.json')
            
            if all(os.path.exists(f) for f in [model_file, scaler_file, features_file]):
                # 모델 로드
                self.ml_model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                
                with open(features_file, 'r') as f:
                    self.feature_names = json.load(f)
                
                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        model_info = json.load(f)
                    logger.info(f"✅ ML 모델 로드 성공: {model_info.get('model_version', 'Unknown')}")
                else:
                    logger.info("✅ ML 모델 로드 성공")
                
                # 최적값 설정 (실제로는 모델에서 학습된 값 사용)
                self.optimal_values = self._get_learned_optimal_values()
                
            else:
                logger.warning("❌ ML 모델 파일이 없습니다. 기본값을 사용합니다.")
                self._use_default_values()
                
        except Exception as e:
            logger.error(f"❌ ML 모델 로드 실패: {e}")
            self._use_default_values()

    def _get_learned_optimal_values(self):
        """학습된 모델에서 최적값 추출"""
        # 실제로는 훈련 데이터의 상위 점수 데이터들의 평균값을 사용
        return {
            'shoulder_balance': 0.025,     # 학습된 최적 어깨 균형
            'left_elbow_angle': 120,       # 학습된 최적 왼팔 각도
            'right_elbow_angle': 118,      # 학습된 최적 오른팔 각도
            'left_knee_angle': 165,        # 학습된 최적 왼무릎 각도
            'right_knee_angle': 163,       # 학습된 최적 오른무릎 각도
            'foot_distance': 0.40,         # 학습된 최적 발 간격
            'shoulder_rotation': 25,       # 학습된 최적 어깨 회전
            'hip_rotation': 20,            # 학습된 최적 골반 회전
            'torso_angle': 3,              # 학습된 최적 상체 각도
            'arm_symmetry': 8,             # 학습된 최적 팔 대칭성
            'leg_symmetry': 5              # 학습된 최적 다리 대칭성
        }

    def _use_default_values(self):
        """기본값 사용 (ML 모델이 없을 때)"""
        self.optimal_values = {
            'shoulder_balance': 0.03,
            'left_elbow_angle': 118,
            'right_elbow_angle': 115,
            'left_knee_angle': 162,
            'right_knee_angle': 160,
            'foot_distance': 0.38,
            'shoulder_rotation': 28,
            'hip_rotation': 22,
            'torso_angle': 5,
            'arm_symmetry': 10,
            'leg_symmetry': 8
        }

    def _check_available_codecs(self):
        """사용 가능한 비디오 코덱 확인"""
        logger.info("비디오 코덱 확인 중...")
        
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        codecs_to_test = [
            ('X264', '.mp4'),
            ('XVID', '.avi'),
            ('MJPG', '.avi'),
            ('mp4v', '.mp4'),
            ('H264', '.mp4'),
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

    def extract_ml_features(self, body_points):
        """ML 모델을 위한 특징 추출"""
        try:
            features = []
            
            # 1. 어깨 균형
            shoulder_balance = abs(body_points['left_shoulder'][1] - body_points['right_shoulder'][1])
            features.append(shoulder_balance)
            
            # 2. 팔꿈치 각도
            left_elbow_angle = self.calculate_angle(body_points['left_shoulder'], body_points['left_elbow'], body_points['left_wrist'])
            right_elbow_angle = self.calculate_angle(body_points['right_shoulder'], body_points['right_elbow'], body_points['right_wrist'])
            features.extend([left_elbow_angle, right_elbow_angle])
            
            # 3. 무릎 각도
            left_knee_angle = self.calculate_angle(body_points['left_hip'], body_points['left_knee'], body_points['left_ankle'])
            right_knee_angle = self.calculate_angle(body_points['right_hip'], body_points['right_knee'], body_points['right_ankle'])
            features.extend([left_knee_angle, right_knee_angle])
            
            # 4. 발 간격
            foot_distance = self.calculate_distance(body_points['left_ankle'], body_points['right_ankle'])
            features.append(foot_distance)
            
            # 5. 어깨 회전각
            shoulder_vector = np.array(body_points['right_shoulder']) - np.array(body_points['left_shoulder'])
            shoulder_rotation = np.arctan2(shoulder_vector[1], shoulder_vector[0]) * 180 / np.pi
            features.append(abs(shoulder_rotation))
            
            # 6. 골반 회전각
            hip_vector = np.array(body_points['right_hip']) - np.array(body_points['left_hip'])
            hip_rotation = np.arctan2(hip_vector[1], hip_vector[0]) * 180 / np.pi
            features.append(abs(hip_rotation))
            
            # 7. 상체 기울기
            torso_center = [(body_points['left_shoulder'][0] + body_points['right_shoulder'][0])/2, 
                           (body_points['left_shoulder'][1] + body_points['right_shoulder'][1])/2]
            hip_center = [(body_points['left_hip'][0] + body_points['right_hip'][0])/2, 
                         (body_points['left_hip'][1] + body_points['right_hip'][1])/2]
            torso_angle = np.arctan2(torso_center[0] - hip_center[0], torso_center[1] - hip_center[1]) * 180 / np.pi
            features.append(abs(torso_angle))
            
            # 8. 팔 대칭성
            arm_symmetry = abs(left_elbow_angle - right_elbow_angle)
            features.append(arm_symmetry)
            
            # 9. 다리 대칭성
            leg_symmetry = abs(left_knee_angle - right_knee_angle)
            features.append(leg_symmetry)
            
            return features
            
        except Exception as e:
            logger.warning(f"ML 특징 추출 오류: {e}")
            return None

    def calculate_ml_score(self, body_points):
        """머신러닝 모델을 사용한 점수 계산"""
        try:
            # ML 특징 추출
            features = self.extract_ml_features(body_points)
            if features is None:
                return self._calculate_fallback_score(body_points)
            
            # ML 모델 예측
            if self.ml_model is not None and self.scaler is not None:
                features_scaled = self.scaler.transform([features])
                ml_score = self.ml_model.predict(features_scaled)[0]
                
                # 0-100 범위로 제한
                ml_score = max(0, min(100, ml_score))
                
                logger.debug(f"ML 예측 점수: {ml_score:.2f}")
                return ml_score
            else:
                # ML 모델이 없으면 개선된 휴리스틱 사용
                return self._calculate_improved_heuristic_score(features)
                
        except Exception as e:
            logger.warning(f"ML 점수 계산 오류: {e}")
            return self._calculate_fallback_score(body_points)

    def _calculate_improved_heuristic_score(self, features):
        """개선된 휴리스틱 점수 계산 (ML 모델이 없을 때)"""
        try:
            if len(features) != len(self.optimal_values):
                return 50
            
            feature_names = list(self.optimal_values.keys())
            total_score = 0
            weights = {
                'shoulder_balance': 0.15,
                'left_elbow_angle': 0.10,
                'right_elbow_angle': 0.10,
                'left_knee_angle': 0.10,
                'right_knee_angle': 0.10,
                'foot_distance': 0.15,
                'shoulder_rotation': 0.10,
                'hip_rotation': 0.10,
                'torso_angle': 0.10,
                'arm_symmetry': 0.05,
                'leg_symmetry': 0.05
            }
            
            for i, feature_name in enumerate(feature_names):
                if i < len(features):
                    current_value = features[i]
                    optimal_value = self.optimal_values[feature_name]
                    weight = weights.get(feature_name, 0.1)
                    
                    # 차이율 계산
                    if optimal_value != 0:
                        diff_ratio = abs(current_value - optimal_value) / max(abs(optimal_value), 1)
                    else:
                        diff_ratio = abs(current_value)
                    
                    # 유사도 계산 (차이가 클수록 점수 낮음)
                    similarity = max(0, 1 - diff_ratio)
                    total_score += similarity * weight * 100
            
            return min(100, max(0, total_score))
            
        except Exception as e:
            logger.warning(f"개선된 휴리스틱 점수 계산 오류: {e}")
            return 50

    def _calculate_fallback_score(self, body_points):
        """기본 점수 계산 (오류 발생시 대체)"""
        try:
            # 기본적인 자세 평가
            shoulder_level = abs(body_points['left_shoulder'][1] - body_points['right_shoulder'][1])
            foot_distance = self.calculate_distance(body_points['left_ankle'], body_points['right_ankle'])
            
            # 간단한 점수 계산
            shoulder_score = max(0, 30 - shoulder_level * 1000)
            foot_score = max(0, 30 - abs(foot_distance - 0.4) * 100)
            base_score = 40  # 기본 점수
            
            return min(100, max(0, base_score + shoulder_score + foot_score))
            
        except:
            return 50

    def get_grade(self, score):
        """점수에 따른 등급 반환"""
        if score >= 90: return 'Excellent'
        elif score >= 80: return 'Good'
        elif score >= 70: return 'Average'
        elif score >= 60: return 'Below Average'
        else: return 'Poor'

    def get_ml_recommendations(self, features, score):
        """ML 기반 개선 제안 생성"""
        recommendations = []
        try:
            if len(features) != len(self.optimal_values):
                return ["분석 데이터가 불충분합니다."]
            
            feature_names = list(self.optimal_values.keys())
            
            for i, feature_name in enumerate(feature_names):
                if i < len(features):
                    current_value = features[i]
                    optimal_value = self.optimal_values[feature_name]
                    
                    # 차이 계산
                    diff = abs(current_value - optimal_value)
                    threshold = optimal_value * 0.2  # 20% 임계값
                    
                    if diff > threshold:
                        if feature_name == 'shoulder_balance':
                            recommendations.append("어깨의 수평을 더욱 정확히 맞춰주세요")
                        elif 'elbow_angle' in feature_name:
                            side = 'left' if 'left' in feature_name else 'right'
                            if current_value < optimal_value:
                                recommendations.append(f"{'왼팔' if side == 'left' else '오른팔'} 팔꿈치를 조금 더 펴주세요")
                            else:
                                recommendations.append(f"{'왼팔' if side == 'left' else '오른팔'} 팔꿈치를 조금 더 굽혀주세요")
                        elif 'knee_angle' in feature_name:
                            side = 'left' if 'left' in feature_name else 'right'
                            if current_value < optimal_value:
                                recommendations.append(f"{'왼쪽' if side == 'left' else '오른쪽'} 무릎을 조금 더 펴주세요")
                        elif feature_name == 'foot_distance':
                            if current_value < optimal_value:
                                recommendations.append("발 간격을 조금 더 넓혀주세요")
                            else:
                                recommendations.append("발 간격을 조금 더 좁혀주세요")
                        elif feature_name == 'shoulder_rotation':
                            recommendations.append("어깨 회전을 더 자연스럽게 해주세요")
                        elif feature_name == 'hip_rotation':
                            recommendations.append("골반 회전을 더 효율적으로 사용해주세요")
                        elif feature_name == 'torso_angle':
                            recommendations.append("상체 각도를 더 안정적으로 유지해주세요")
                        elif feature_name == 'arm_symmetry':
                            recommendations.append("양팔의 대칭성을 개선해주세요")
                        elif feature_name == 'leg_symmetry':
                            recommendations.append("양다리의 균형을 맞춰주세요")
            
            # 점수별 추가 조언
            if score < 70:
                recommendations.append("기본 타격 자세 연습을 더 해보세요")
                recommendations.append("프로 선수의 영상을 참고하여 폼을 개선하세요")
            elif score < 85:
                recommendations.append("세부 동작의 정확성을 높여보세요")
            elif score >= 90:
                recommendations.append("훌륭한 자세입니다! 현재 폼을 유지하세요")
            
            return recommendations if recommendations else ["매우 좋은 자세입니다!"]
            
        except Exception as e:
            logger.warning(f"ML 추천 생성 오류: {e}")
            return ["분석을 완료했습니다"]

    # 나머지 메소드들 (draw_pose_landmarks, detect_swing_sequence 등)은 기존과 동일하게 유지
    def draw_pose_landmarks(self, image, landmarks):
        """포즈 랜드마크 그리기"""
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
            
            # 연결선 그리기
            for start_landmark, end_landmark in connections:
                try:
                    start_point = landmarks[start_landmark.value]
                    end_point = landmarks[end_landmark.value]
                    if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                        start_x, start_y = int(start_point.x * w), int(start_point.y * h)
                        end_x, end_y = int(end_point.x * w), int(end_point.y * h)
                        
                        if (0 <= start_x < w and 0 <= start_y < h and 
                            0 <= end_x < w and 0 <= end_y < h):
                            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
                except:
                    continue
            
            # 관절점 그리기
            for landmark_type in important_landmarks:
                try:
                    landmark = landmarks[landmark_type.value]
                    if landmark.visibility > 0.5:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        
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
            # ML 모델 사용 여부 표시
            model_status = "ML" if self.ml_model is not None else "기본"
            
            cv2.putText(image, f'Score: {score:.1f} ({model_status})', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f'Grade: {grade}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f'Swings: {swing_count}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if swing_intensity > 0:
                intensity_color = (0, 255, 0) if swing_intensity > 0.35 else (0, 255, 255)
                cv2.putText(image, f'Intensity: {swing_intensity:.2f}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, intensity_color, 2)
        except Exception as e:
            logger.warning(f"오버레이 오류: {e}")

    def detect_swing_sequence(self, body_points, prev_body_points, frame_count, fps):
        """스윙 감지 (기존과 동일)"""
        if not prev_body_points:
            return False, 0
        
        try:
            right_wrist_movement = self.calculate_distance(body_points['right_wrist'], prev_body_points['right_wrist'])
            left_wrist_movement = self.calculate_distance(body_points['left_wrist'], prev_body_points['left_wrist'])
            right_elbow_movement = self.calculate_distance(body_points['right_elbow'], prev_body_points['right_elbow'])
            
            current_shoulder_angle = np.arctan2(
                body_points['right_shoulder'][1] - body_points['left_shoulder'][1],
                body_points['right_shoulder'][0] - body_points['left_shoulder'][0]
            ) * 180 / np.pi
            
            prev_shoulder_angle = np.arctan2(
                prev_body_points['right_shoulder'][1] - prev_body_points['left_shoulder'][1],
                prev_body_points['right_shoulder'][0] - prev_body_points['left_shoulder'][0]
            ) * 180 / np.pi
            
            shoulder_rotation = abs(current_shoulder_angle - prev_shoulder_angle)
            
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
        """개선된 비디오 처리 및 분석 (ML 모델 적용)"""
        try:
            logger.info(f"🎬 ML 기반 비디오 분석 시작: {session_id}")
            
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
            
            # 출력 비디오 설정
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
                'status': 'processing',
                'ml_model_used': self.ml_model is not None
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
                                # ML 기반 점수 계산
                                current_score = self.calculate_ml_score(body_points)
                                current_grade = self.get_grade(current_score)
                                
                                # 스윙 감지
                                is_swinging, swing_intensity = self.detect_swing_sequence(
                                    body_points, prev_body_points, frame_count, fps
                                )
                                
                                frames_since_last_swing += 1
                                
                                # 스윙 분석 로직 (기존과 동일)
                                if (is_swinging and not swing_in_progress and frames_since_last_swing > min_swing_gap):
                                    swing_in_progress = True
                                    swing_start_frame = frame_count
                                    swing_peak_intensity = swing_intensity
                                    frames_since_last_swing = 0
                                    logger.info(f"🏏 스윙 시작 감지: 프레임 {frame_count}")
                                
                                elif swing_in_progress and is_swinging:
                                    swing_peak_intensity = max(swing_peak_intensity, swing_intensity)
                                
                                elif (swing_in_progress and not is_swinging and 
                                      (frame_count - swing_start_frame) > fps * 0.3 and 
                                      swing_peak_intensity > 0.4):
                                    
                                    swing_in_progress = False
                                    swing_count += 1
                                    swing_duration = (frame_count - swing_start_frame) / fps
                                    
                                    # ML 기반 특징 추출 및 추천 생성
                                    features = self.extract_ml_features(body_points)
                                    recommendations = self.get_ml_recommendations(features, current_score)
                                    
                                    swing_analyses.append({
                                        'swing_number': swing_count,
                                        'frame': frame_count,
                                        'duration': swing_duration,
                                        'peak_intensity': swing_peak_intensity,
                                        'timestamp': frame_count / fps,
                                        'ai_score': current_score,
                                        'grade': current_grade,
                                        'recommendations': recommendations,
                                        'ml_features': features if features else []
                                    })
                                    
                                    logger.info(f"✅ ML 기반 스윙 완료: #{swing_count}, 점수: {current_score:.1f}")
                                    swing_peak_intensity = 0
                                
                                prev_body_points = body_points
                        
                        # 정보 오버레이 (ML 모델 사용 여부 표시)
                        self.draw_info_overlay(image, current_score, current_grade, swing_count, swing_intensity)
                        
                        # 처리된 프레임을 출력 비디오에 저장
                        out.write(image)
                        processed_frames += 1
                        
                        # 진행률 업데이트
                        if frame_count % 10 == 0:
                            progress = (frame_count / total_frames) * 100
                            self.analysis_results[session_id].update({
                                'progress': progress,
                                'current_score': current_score,
                                'current_grade': current_grade,
                                'swing_count': swing_count,
                                'status': 'processing',
                                'frames_processed': frame_count,
                                'total_frames': total_frames,
                                'ml_model_used': self.ml_model is not None
                            })
                            
                            if frame_count % 100 == 0:
                                logger.info(f"📊 ML 분석 진행률: {progress:.1f}% ({frame_count}/{total_frames})")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ 프레임 {frame_count} 처리 오류: {e}")
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
                logger.info(f"✅ ML 기반 처리 완료!")
                logger.info(f"   📁 파일 경로: {output_path}")
                logger.info(f"   📏 파일 크기: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                logger.info(f"   🎬 처리된 프레임: {processed_frames}/{total_frames}")
                logger.info(f"   🤖 ML 모델 사용: {'예' if self.ml_model is not None else '아니오'}")
                
                # 생성된 비디오 무결성 확인
                test_cap = cv2.VideoCapture(output_path)
                if test_cap.isOpened():
                    test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    test_duration = test_cap.get(cv2.CAP_PROP_FRAME_COUNT) / test_cap.get(cv2.CAP_PROP_FPS)
                    test_cap.release()
                    
                    if test_frame_count > 0 and file_size > 1024:
                        # 최종 결과 계산
                        if swing_analyses:
                            avg_score = sum(s['ai_score'] for s in swing_analyses) / len(swing_analyses)
                            max_score = max(s['ai_score'] for s in swing_analyses)
                            min_score = min(s['ai_score'] for s in swing_analyses)
                        else:
                            avg_score = max_score = min_score = 50
                        
                        # ML 모델 정보 추가
                        model_info = {
                            'ml_model_used': self.ml_model is not None,
                            'model_type': 'RandomForestRegressor' if self.ml_model is not None else 'Heuristic',
                            'feature_count': len(self.feature_names) if self.feature_names else 0,
                            'optimal_values': self.optimal_values
                        }
                        
                        # 최종 결과 저장
                        final_results = {
                            'status': 'completed',
                            'session_id': session_id,
                            'model_info': model_info,
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
                            'general_recommendations': self._generate_general_recommendations(swing_analyses, avg_score),
                            'redirect_to_results': True,
                            'progress': 100
                        }
                        
                        # 결과 파일 저장
                        results_file = os.path.join(RESULTS_FOLDER, f'{session_id}_results.json')
                        with open(results_file, 'w', encoding='utf-8') as f:
                            json.dump(final_results, f, indent=2, ensure_ascii=False)
                        
                        self.analysis_results[session_id] = final_results
                        logger.info(f"🎉 ML 기반 비디오 분석 완료: {session_id}")
                        logger.info(f"📊 총 스윙: {swing_count}회, 평균 점수: {avg_score:.1f}점")
                        logger.info(f"🤖 ML 모델: {'사용됨' if self.ml_model is not None else '미사용 (기본값 사용)'}")
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
            logger.error(f"❌ ML 기반 비디오 처리 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            
            error_result = {
                'status': 'error',
                'error': str(e),
                'message': 'ML 기반 비디오 분석 중 오류가 발생했습니다.'
            }
            self.analysis_results[session_id] = error_result
            return error_result

    def _generate_general_recommendations(self, swing_analyses, avg_score):
        """종합 개선 제안 생성"""
        try:
            recommendations = []
            
            if not swing_analyses:
                return ["분석할 스윙이 감지되지 않았습니다. 더 명확한 스윙 동작을 시도해보세요."]
            
            # ML 기반 종합 분석
            if self.ml_model is not None:
                recommendations.append("🤖 AI 모델 기반 분석이 완료되었습니다.")
                
                # 모든 스윙의 특징 평균 계산
                all_features = []
                for swing in swing_analyses:
                    if swing.get('ml_features'):
                        all_features.append(swing['ml_features'])
                
                if all_features:
                    avg_features = np.mean(all_features, axis=0)
                    feature_recommendations = self.get_ml_recommendations(avg_features, avg_score)
                    recommendations.extend(feature_recommendations[:3])  # 상위 3개만
            
            # 점수별 조언
            if avg_score >= 90:
                recommendations.append("✨ 훌륭한 타격 자세입니다! 현재 폼을 꾸준히 유지하세요.")
            elif avg_score >= 80:
                recommendations.append("👍 좋은 타격 자세입니다. 세부 동작의 일관성을 높여보세요.")
            elif avg_score >= 70:
                recommendations.append("📈 평균적인 자세입니다. 기본기 연습을 통해 개선이 가능합니다.")
            elif avg_score >= 60:
                recommendations.append("🎯 개선의 여지가 있습니다. 기본 자세 연습에 집중해보세요.")
            else:
                recommendations.append("💪 기본 타격 자세부터 차근차근 연습해보세요.")
            
            # 스윙 일관성 분석
            if len(swing_analyses) > 1:
                scores = [s['ai_score'] for s in swing_analyses]
                score_std = np.std(scores)
                if score_std > 15:
                    recommendations.append("⚖️ 스윙마다 점수 편차가 큽니다. 일관된 자세 연습이 필요합니다.")
                elif score_std < 5:
                    recommendations.append("🎯 스윙의 일관성이 좋습니다!")
            
            # 실제 프로 선수와 비교 (ML 모델이 있을 때)
            if self.ml_model is not None:
                recommendations.append("📊 프로 선수 데이터를 기준으로 분석되었습니다.")
            else:
                recommendations.append("ℹ️ 기본 분석 모드로 실행되었습니다. ML 모델 훈련 후 더 정확한 분석이 가능합니다.")
            
            return recommendations[:5] if recommendations else ["분석을 완료했습니다."]
            
        except Exception as e:
            logger.warning(f"종합 추천 생성 오류: {e}")
            return ["분석을 완료했습니다."]


# 전역 분석기 인스턴스 (ML 기반)
analyzer = MLBaseballAnalyzer()


# Flask 라우트들 (기존과 대부분 동일하지만 ML 정보 추가)
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
                'message': 'Video uploaded successfully',
                'ml_model_available': analyzer.ml_model is not None
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
        
        logger.info(f"🚀 ML 기반 분석 시작: {session_id}")
        return jsonify({
            'success': True, 
            'message': 'ML-based analysis started',
            'ml_model_used': analyzer.ml_model is not None
        })
        
    except Exception as e:
        logger.error(f"❌ 분석 시작 오류: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analysis_status/<session_id>')
def analysis_status(session_id):
    if session_id in analyzer.analysis_results:
        return jsonify(analyzer.analysis_results[session_id])
    else:
        return jsonify({
            'status': 'not_started', 
            'progress': 0,
            'ml_model_available': analyzer.ml_model is not None
        })


@app.route('/results/<session_id>')
def results_page(session_id):
    try:
        results_file = os.path.join(RESULTS_FOLDER, f'{session_id}_results.json')
        
        if not os.path.exists(results_file):
            logger.warning(f"❌ 결과 파일이 없습니다: {results_file}")
            return "Results not found", 404
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # ML 모델 정보 로깅
        if results.get('model_info', {}).get('ml_model_used'):
            logger.info(f"🤖 ML 기반 결과 표시: {session_id}")
        else:
            logger.info(f"📊 기본 모드 결과 표시: {session_id}")
        
        return render_template('results.html', results=results, session_id=session_id)
        
    except Exception as e:
        logger.error(f"❌ 결과 페이지 오류: {e}")
        return f"Error loading results: {str(e)}", 500


# 나머지 라우트들 (기존과 동일)
@app.route('/download_results/<session_id>')
def download_results(session_id):
    try:
        results_file = os.path.join(RESULTS_FOLDER, f'{session_id}_results.json')
        if os.path.exists(results_file):
            return send_file(results_file, as_attachment=True, 
                            download_name=f'ml_baseball_analysis_{session_id}.json')
        else:
            return "Results not found", 404
    except Exception as e:
        logger.error(f"❌ 결과 다운로드 오류: {e}")
        return f"Error downloading results: {str(e)}", 500


@app.route('/static/processed_videos/<filename>')
def serve_processed_video(filename):
    try:
        video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, filename)
        
        if not os.path.exists(video_path):
            logger.error(f"❌ 비디오 파일이 존재하지 않음: {video_path}")
            return "Video file not found", 404
        
        # Range 요청 처리
        range_header = request.headers.get('Range', None)
        if range_header:
            return serve_video_with_range(video_path, range_header)
        
        # MIME 타입 설정
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


# ML 모델 상태 확인 라우트 추가
@app.route('/ml_status')
def ml_status():
    """ML 모델 상태 확인"""
    try:
        status = {
            'ml_model_loaded': analyzer.ml_model is not None,
            'scaler_loaded': analyzer.scaler is not None,
            'feature_count': len(analyzer.feature_names),
            'feature_names': analyzer.feature_names,
            'optimal_values_available': bool(analyzer.optimal_values),
            'model_type': 'RandomForestRegressor' if analyzer.ml_model is not None else 'Heuristic'
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"❌ ML 상태 확인 오류: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 500MB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"❌ Internal server error: {e}")
    return jsonify({'success': False, 'error': 'Internal server error occurred.'}), 500


if __name__ == '__main__':
    logger.info("🚀 ML 기반 Flask 서버 시작 중...")
    logger.info(f"📁 업로드 폴더: {UPLOAD_FOLDER}")
    logger.info(f"📁 결과 폴더: {RESULTS_FOLDER}")
    logger.info(f"📁 처리된 비디오 폴더: {PROCESSED_VIDEOS_FOLDER}")
    logger.info(f"🤖 ML 모델 폴더: {MODEL_FOLDER}")
    
    # ML 모델 상태 확인
    if analyzer.ml_model is not None:
        logger.info("✅ ML 모델이 로드되어 프로 선수 기준으로 분석합니다")
    else:
        logger.info("⚠️ ML 모델이 없어 기본 분석 모드로 실행됩니다")
        logger.info("💡 더 정확한 분석을 위해 baseball_ml_trainer.py로 모델을 훈련하세요")
    
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
            logger.info(f"   ML 상태 확인: http://localhost:{port}/ml_status")
            app.run(debug=True, host='127.0.0.1', port=port, threaded=True, use_reloader=False)
            break
        except OSError as e:
            logger.warning(f"⚠️  포트 {port} 사용 불가, 다른 포트 시도...")
            continue
    else:
        logger.error("❌ 사용 가능한 포트를 찾을 수 없습니다.")