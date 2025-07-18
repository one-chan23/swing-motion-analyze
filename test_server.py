#!/usr/bin/env python3
"""
간단한 테스트 서버 - 포트 권한 문제 해결용
"""

from flask import Flask, render_template
import socket

app = Flask(__name__)

@app.route('/')
def test_index():
    return '''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>서버 테스트</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 50px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                margin: 0;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
                max-width: 500px;
                margin: 0 auto;
            }
            h1 { font-size: 2.5em; margin-bottom: 20px; }
            .status { font-size: 1.2em; margin: 20px 0; }
            .next-steps {
                background: rgba(255, 255, 255, 0.2);
                padding: 20px;
                border-radius: 10px;
                margin-top: 30px;
                text-align: left;
            }
            .success { color: #4CAF50; font-weight: bold; }
            .port-info { 
                background: rgba(255, 255, 255, 0.3);
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-family: monospace;
                font-size: 1.1em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎉 서버 연결 성공!</h1>
            <div class="status success">Flask 서버가 정상적으로 실행되고 있습니다.</div>
            
            <div class="port-info">
                현재 포트: <span id="current-port"></span>
            </div>
            
            <div class="next-steps">
                <h3>📋 다음 단계:</h3>
                <ol>
                    <li>이 창을 닫고 터미널로 돌아가세요</li>
                    <li><strong>Ctrl+C</strong>로 서버를 중지하세요</li>
                    <li><strong>python app.py</strong>를 다시 실행하세요</li>
                    <li>표시되는 포트 번호로 접속하세요</li>
                </ol>
                
                <p><strong>💡 팁:</strong><br>
                만약 app.py에서도 같은 오류가 발생하면,<br>
                <strong>관리자 권한</strong>으로 명령 프롬프트를 실행하세요.</p>
            </div>
        </div>
        
        <script>
            // 현재 포트 번호 표시
            document.getElementById('current-port').textContent = window.location.port || '80';
        </script>
    </body>
    </html>
    '''

def find_free_port():
    """사용 가능한 포트 찾기"""
    ports_to_try = [8080, 8000, 3000, 5000, 7000, 9000, 4000, 6000]
    
    for port in ports_to_try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    
    # 동적으로 포트 찾기
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

if __name__ == '__main__':
    port = find_free_port()
    print(f"🚀 테스트 서버 시작 중...")
    print(f"📡 사용 가능한 포트: {port}")
    print(f"🌐 브라우저에서 접속: http://localhost:{port}")
    print(f"🔄 서버 중지: Ctrl+C")
    print("=" * 50)
    
    try:
        app.run(debug=False, host='127.0.0.1', port=port, use_reloader=False)
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        print("\n🔧 해결 방법:")
        print("1. 관리자 권한으로 명령 프롬프트 실행")
        print("2. Windows 방화벽 설정 확인")
        print("3. 바이러스 백신 프로그램 확인")