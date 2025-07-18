#!/usr/bin/env python3
"""
야구 타격자세 분석 - 머신러닝 모델 훈련
프로 선수 영상을 통한 최적 타격 자세 학습
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseballPoseAnalyzer:
    """야구 타격자세 분석을 위한 머신러닝 클래스"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        
    def extract_pose_features(self, video_path):
        """비디오에서 포즈 특징 추출"""
        cap = cv2.VideoCapture(video_path)
        features_list = []
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB 변환
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                
                if results.pose_landmarks:
                    features = self._calculate_pose_features(results.pose_landmarks.landmark)
                    if features:
                        features_list.append(features)
        
        cap.release()
        
        if features_list:
            # 평균값 반환 (안정적인 자세 특징)
            return np.mean(features_list, axis=0)
        return None
    
    def _calculate_pose_features(self, landmarks):
        """포즈 랜드마크에서 특징 계산"""
        try:
            # 주요 관절 포인트 추출
            left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # 특징 계산
            features = []
            
            # 1. 어깨 균형 (수평도)
            shoulder_balance = abs(left_shoulder[1] - right_shoulder[1])
            features.append(shoulder_balance)
            
            # 2. 팔꿈치 각도
            left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            features.extend([left_elbow_angle, right_elbow_angle])
            
            # 3. 무릎 각도
            left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            features.extend([left_knee_angle, right_knee_angle])
            
            # 4. 발 간격
            foot_distance = self._calculate_distance(left_ankle, right_ankle)
            features.append(foot_distance)
            
            # 5. 어깨 회전각
            shoulder_vector = np.array(right_shoulder) - np.array(left_shoulder)
            shoulder_rotation = np.arctan2(shoulder_vector[1], shoulder_vector[0]) * 180 / np.pi
            features.append(abs(shoulder_rotation))
            
            # 6. 골반 회전각
            hip_vector = np.array(right_hip) - np.array(left_hip)
            hip_rotation = np.arctan2(hip_vector[1], hip_vector[0]) * 180 / np.pi
            features.append(abs(hip_rotation))
            
            # 7. 상체 기울기
            torso_center = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
            hip_center = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
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
            logger.warning(f"특징 계산 오류: {e}")
            return None
    
    def _calculate_angle(self, a, b, c):
        """세 점으로 각도 계산"""
        try:
            a, b, c = np.array(a), np.array(b), np.array(c)
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            return 360 - angle if angle > 180.0 else angle
        except:
            return 0
    
    def _calculate_distance(self, point1, point2):
        """두 점 사이의 거리 계산"""
        try:
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except:
            return 0
    
    def prepare_training_data(self, data_folder):
        """훈련 데이터 준비"""
        logger.info("훈련 데이터 준비 중...")
        
        features_list = []
        scores_list = []
        
        # 프로 선수 폴더 구조: data_folder/pro_players/score_XX/video.mp4
        pro_folder = os.path.join(data_folder, 'pro_players')
        
        if not os.path.exists(pro_folder):
            logger.error(f"프로 선수 데이터 폴더가 없습니다: {pro_folder}")
            return None, None
        
        for score_folder in os.listdir(pro_folder):
            if score_folder.startswith('score_'):
                try:
                    score = float(score_folder.split('_')[1])
                    score_path = os.path.join(pro_folder, score_folder)
                    
                    for video_file in os.listdir(score_path):
                        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                            video_path = os.path.join(score_path, video_file)
                            features = self.extract_pose_features(video_path)
                            
                            if features is not None:
                                features_list.append(features)
                                scores_list.append(score)
                                logger.info(f"추출 완료: {video_file} (점수: {score})")
                
                except Exception as e:
                    logger.warning(f"폴더 처리 오류 {score_folder}: {e}")
        
        if len(features_list) == 0:
            logger.error("훈련 데이터가 없습니다!")
            return None, None
        
        # 특징 이름 설정
        self.feature_names = [
            'shoulder_balance', 'left_elbow_angle', 'right_elbow_angle',
            'left_knee_angle', 'right_knee_angle', 'foot_distance',
            'shoulder_rotation', 'hip_rotation', 'torso_angle',
            'arm_symmetry', 'leg_symmetry'
        ]
        
        X = np.array(features_list)
        y = np.array(scores_list)
        
        logger.info(f"훈련 데이터 준비 완료: {len(X)}개 샘플, {len(self.feature_names)}개 특징")
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """모델 훈련"""
        logger.info("모델 훈련 시작...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 데이터 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 랜덤 포레스트 모델 훈련 (하이퍼파라미터 조정)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # 예측 및 평가
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        logger.info(f"훈련 MSE: {train_mse:.4f}, 테스트 MSE: {test_mse:.4f}")
        logger.info(f"훈련 R²: {train_r2:.4f}, 테스트 R²: {test_r2:.4f}")
        
        # 특징 중요도 출력
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("특징 중요도:")
        for _, row in feature_importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance
        }
    
    def save_model(self, model_path='models'):
        """모델 저장"""
        os.makedirs(model_path, exist_ok=True)
        
        # 모델 저장
        model_file = os.path.join(model_path, 'baseball_pose_model.joblib')
        joblib.dump(self.model, model_file)
        
        # 스케일러 저장
        scaler_file = os.path.join(model_path, 'pose_scaler.joblib')
        joblib.dump(self.scaler, scaler_file)
        
        # 특징 이름 저장
        features_file = os.path.join(model_path, 'feature_names.json')
        with open(features_file, 'w') as f:
            json.dump(self.feature_names, f)
        
        # 모델 정보 저장
        model_info = {
            'model_type': 'RandomForestRegressor',
            'features_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'created_date': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        
        info_file = os.path.join(model_path, 'model_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"모델 저장 완료: {model_path}")
    
    def load_model(self, model_path='models'):
        """모델 불러오기"""
        try:
            # 모델 불러오기
            model_file = os.path.join(model_path, 'baseball_pose_model.joblib')
            self.model = joblib.load(model_file)
            
            # 스케일러 불러오기
            scaler_file = os.path.join(model_path, 'pose_scaler.joblib')
            self.scaler = joblib.load(scaler_file)
            
            # 특징 이름 불러오기
            features_file = os.path.join(model_path, 'feature_names.json')
            with open(features_file, 'r') as f:
                self.feature_names = json.load(f)
            
            logger.info(f"모델 불러오기 완료: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"모델 불러오기 실패: {e}")
            return False
    
    def predict_score(self, features):
        """점수 예측"""
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다!")
            return None
        
        try:
            features_scaled = self.scaler.transform([features])
            score = self.model.predict(features_scaled)[0]
            return max(0, min(100, score))  # 0-100 범위로 제한
        except Exception as e:
            logger.error(f"점수 예측 오류: {e}")
            return None
    
    def get_optimal_values(self):
        """학습된 데이터의 최적값 반환"""
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다!")
            return None
        
        # 높은 점수를 받은 데이터들의 평균값을 최적값으로 사용
        # 실제 구현시에는 훈련 데이터에서 상위 10% 점수의 평균을 계산
        optimal_values = {
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
        
        return optimal_values


def create_sample_data():
    """샘플 데이터 생성 (실제 프로젝트에서는 실제 영상 사용)"""
    logger.info("샘플 데이터 생성 중...")
    
    # 샘플 데이터 폴더 생성
    os.makedirs('data/pro_players', exist_ok=True)
    
    # 다양한 점수 폴더 생성
    scores = [85, 88, 90, 92, 95, 98]
    
    for score in scores:
        folder_path = f'data/pro_players/score_{score}'
        os.makedirs(folder_path, exist_ok=True)
        
        # 실제로는 여기에 해당 점수에 맞는 프로 선수 영상들을 넣어야 함
        logger.info(f"폴더 생성: {folder_path}")
        logger.info(f"  → 이 폴더에 점수 {score}점에 해당하는 프로 선수 타격 영상들을 넣어주세요")


def main():
    """메인 실행 함수"""
    logger.info("야구 타격자세 분석 모델 훈련 시작")
    
    # 분석기 초기화
    analyzer = BaseballPoseAnalyzer()
    
    # 샘플 데이터 구조 생성
    create_sample_data()
    
    print("\n" + "="*60)
    print("📊 야구 타격자세 분석 - 머신러닝 모델 훈련")
    print("="*60)
    print("1. 먼저 'data/pro_players/score_XX' 폴더에 프로 선수 영상을 넣어주세요")
    print("2. 점수별로 폴더를 나누어 영상을 배치해주세요")
    print("   예: score_95/player1.mp4, score_88/player2.mp4")
    print("3. 영상 준비가 완료되면 다음 코드를 실행하세요:")
    print("="*60)
    
    # 실제 훈련 코드 (영상이 있을 때 실행)
    training_code = """
# 훈련 데이터 준비
X, y = analyzer.prepare_training_data('data')

if X is not None:
    # 모델 훈련
    results = analyzer.train_model(X, y)
    
    # 모델 저장
    analyzer.save_model('models')
    
    print("🎉 모델 훈련 및 저장 완료!")
    print(f"📈 테스트 R² 점수: {results['test_r2']:.4f}")
else:
    print("❌ 훈련 데이터가 없습니다. 프로 선수 영상을 먼저 준비해주세요.")
"""
    
    print("\n💻 훈련 실행 코드:")
    print(training_code)


if __name__ == "__main__":
    main()