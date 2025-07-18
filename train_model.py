#!/usr/bin/env python3
"""
ML 모델 훈련 실행 스크립트
"""

from baseball_ml_trainer import BaseballPoseAnalyzer
import os

def main():
    print("=" * 50)
    print("야구 타격자세 ML 모델 훈련")
    print("=" * 50)
    
    # 1. 데이터 확인
    print("\n1. 데이터 확인 중...")
    data_folder = 'data/pro_players'
    total_videos = 0
    
    if not os.path.exists(data_folder):
        print(f"ERROR: {data_folder} 폴더가 없습니다!")
        return
    
    for score_folder in os.listdir(data_folder):
        if score_folder.startswith('score_'):
            folder_path = os.path.join(data_folder, score_folder)
            video_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            print(f"  {score_folder}: {len(video_files)}개 영상")
            total_videos += len(video_files)
            
            # 처음 3개 파일명 표시
            for video in video_files[:3]:
                print(f"    - {video}")
    
    print(f"\n총 {total_videos}개 영상 발견")
    
    if total_videos == 0:
        print("ERROR: 훈련할 영상이 없습니다!")
        print("data/pro_players/score_XX/ 폴더에 영상 파일을 넣어주세요.")
        return
    
    # 2. 모델 훈련
    print("\n2. ML 모델 훈련 시작...")
    analyzer = BaseballPoseAnalyzer()
    
    # 훈련 데이터 준비
    print("  - 훈련 데이터 준비 중...")
    X, y = analyzer.prepare_training_data('data')
    
    if X is not None:
        print(f"  - 데이터 준비 완료: {len(X)}개 샘플")
        
        # 모델 훈련
        print("  - 모델 훈련 시작...")
        results = analyzer.train_model(X, y)
        
        # 모델 저장
        print("  - 모델 저장 중...")
        analyzer.save_model('models')
        
        print("\n3. 훈련 완료!")
        print(f"  - 테스트 R2 점수: {results['test_r2']:.4f}")
        print(f"  - 테스트 MSE: {results['test_mse']:.4f}")
        print(f"  - 훈련 MSE: {results['train_mse']:.4f}")
        
        # 특징 중요도 출력
        print("\n4. 특징 중요도:")
        feature_importance = results['feature_importance']
        for _, row in feature_importance.head(5).iterrows():
            print(f"  - {row['feature']}: {row['importance']:.4f}")
        
        print("\n5. 저장된 파일:")
        print("  - models/baseball_pose_model.joblib")
        print("  - models/pose_scaler.joblib") 
        print("  - models/feature_names.json")
        print("  - models/model_info.json")
        
        print("\n성공! 이제 improved_app.py를 실행하세요:")
        print("python improved_app.py")
        
    else:
        print("ERROR: 훈련 데이터를 준비할 수 없습니다.")
        print("영상 파일들을 확인해주세요.")

if __name__ == "__main__":
    main()