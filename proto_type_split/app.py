import os
import time
import grpc
import threading
import json
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np
from datetime import datetime
import wave
import io
import requests
from queue import Queue

# CLOVA Speech gRPC 모듈
import nest_pb2
import nest_pb2_grpc

# 설정 파일 로드
def load_config():
    with open('server_conf.json', 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()

app = Flask(__name__)
app.config['SECRET_KEY'] = CONFIG['apps']['app']['secret_key']
socketio = SocketIO(app, 
                   cors_allowed_origins=CONFIG['cors']['allowed_origins'], 
                   allow_upgrades=True)

# 브라우저별 세션 관리
active_sessions = {}  # { session_id: STTSession }

# CLOVA Speech 설정
CLOVA_HOST = CONFIG['clova']['host']
CLOVA_PORT = CONFIG['clova']['port']
CLOVA_INVOKE_URL = CONFIG['clova']['invoke_url']
CLIENT_SECRET = CONFIG['clova']['client_secret']

# 출력 디렉토리 설정
OUTPUT_DIR = CONFIG['directories']['recordings']
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AudioAccumulator:
    def __init__(self):
        self.accumulated_audio = bytearray()
        self.lock = threading.Lock()

    def add_audio(self, audio_data):
        with self.lock:
            self.accumulated_audio.extend(audio_data)

    def get_all_audio(self):
        with self.lock:
            return bytes(self.accumulated_audio)

class STTSession:
    def __init__(self, sid):
        self.sid = sid
        self.is_active = True
        self.lock = threading.Lock()
        
        # 세션 시작 시간
        self.start_time = datetime.now()
        self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # 파일 경로 설정
        self.orig_file_path = os.path.join(OUTPUT_DIR, f"{self.session_id}_orig.txt")
        self.orig2_file_path = os.path.join(OUTPUT_DIR, f"{self.session_id}_orig2.txt")
        self.long_file_path = os.path.join(OUTPUT_DIR, f"{self.session_id}_long.txt")
        
        print(f"[DEBUG] Initialized file paths:")
        print(f"[DEBUG] orig_file_path: {self.orig_file_path}")
        print(f"[DEBUG] orig2_file_path: {self.orig2_file_path}")
        print(f"[DEBUG] long_file_path: {self.long_file_path}")
        
        # 오디오 누적기 초기화
        self.audio_accumulator = AudioAccumulator()
        
        # gRPC 채널 생성
        self.channel = None
        self.stub = None
        self.request_generator = None
        self.response_future = None
        
        # 스트림 처리를 위한 큐
        self.audio_queue = []
        
        # 장문 인식 간격 설정 (초)
        self.long_recognition_interval = 10
        
        # 파일 기록을 위한 플래그 및 현재 라인
        self.need_timestamp = True
        self.current_line = ""
        
        # gRPC 스트림 시작
        self.initialize_grpc()
        
        # 장문 인식 스레드 시작
        self.start_long_recognition_thread()

    def initialize_grpc(self):
        """gRPC 연결 초기화"""
        try:
            creds = grpc.ssl_channel_credentials()
            self.channel = grpc.secure_channel(f"{CLOVA_HOST}:{CLOVA_PORT}", creds)
            self.stub = nest_pb2_grpc.NestServiceStub(self.channel)
            
            # 응답 처리 스레드 시작
            self.start_response_thread()
            print(f"[{self.sid}] gRPC 초기화 완료")
        except Exception as e:
            print(f"[{self.sid}] gRPC 초기화 실패: {str(e)}")

    def start_response_thread(self):
        """응답 처리 스레드 시작"""
        def process_responses():
            try:
                metadata = [("authorization", f"Bearer {CLIENT_SECRET}")]
                responses = self.stub.recognize(self.create_request_generator(), metadata=metadata)
                
                for response in responses:
                    if not self.is_active:
                        break
                    
                    try:
                        result = json.loads(response.contents)
                        if 'transcription' in result:
                            text = result['transcription'].get('text', '').strip()
                            if text:
                                is_partial = result.get('isPartial', True)
                                has_final_mark = text.endswith(('.', '?', '!', '。'))
                                
                                # orig 파일 기록 (기존대로 유지)
                                with open(self.orig_file_path, 'a', encoding='utf-8') as f_orig:
                                    if is_partial:
                                        f_orig.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Partial] {text}\n")
                                    else:
                                        f_orig.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {text}\n")
                                
                                # orig2 파일 기록 로직 수정
                                with self.lock:
                                    with open(self.orig2_file_path, 'a', encoding='utf-8') as f_orig2:
                                        if self.need_timestamp:
                                            # 새로운 줄을 시작하며 타임스탬프 기록
                                            f_orig2.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {text}")
                                            self.need_timestamp = False
                                        else:
                                            # 현재 줄에 단어 추가
                                            f_orig2.write(f" {text}")
                                        
                                        if not is_partial or has_final_mark:
                                            # partial이거나 문장 종료 시 줄바꿈
                                            f_orig2.write("\n")
                                            self.need_timestamp = True
                                
                                # 소켓으로 결과 전송 (기존대로 유지)
                                if is_partial:
                                    socketio.emit('stt_partial', {'text': text}, room=self.sid)
                                else:
                                    socketio.emit('stt_final', {'text': text}, room=self.sid)
                                    
                    except Exception as e:
                        print(f"[{self.sid}] 응답 처리 중 오류: {str(e)}")
                        
            except Exception as e:
                print(f"[{self.sid}] gRPC 스트림 처리 중 오류: {str(e)}")
                
        self.response_thread = threading.Thread(target=process_responses)
        self.response_thread.daemon = True
        self.response_thread.start()

    def process_long_recognition(self):
        """장문 음성 인식 처리"""
        try:
            # 누적된 오디오 데이터 가져오기
            audio_data = self.audio_accumulator.get_all_audio()
            if not audio_data:
                return

            # WAV 파일로 변환
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)

            # 임시 WAV 파일 저장
            temp_wav_path = os.path.join(OUTPUT_DIR, f"{self.session_id}_temp.wav")
            with open(temp_wav_path, 'wb') as f:
                f.write(wav_buffer.getvalue())

            # CLOVA Speech API 호출
            headers = {
                'Accept': 'application/json;UTF-8',
                'X-CLOVASPEECH-API-KEY': CLIENT_SECRET
            }
            
            files = {
                'media': open(temp_wav_path, 'rb'),
                'params': (None, json.dumps({
                    'language': 'ko-KR',
                    'completion': 'sync',
                    'diarization': {
                        'enable': True,
                        'speakerCountMin': 1,
                        'speakerCountMax': 10
                    }
                }), 'application/json')
            }

            response = requests.post(
                CLOVA_INVOKE_URL + '/recognizer/upload',
                headers=headers,
                files=files
            )

            # 임시 파일 삭제
            os.remove(temp_wav_path)

            if response.status_code == 200:
                result = response.json()
                with open(self.long_file_path, 'w', encoding='utf-8') as f:
                    if 'segments' in result:
                        for segment in result['segments']:
                            speaker = segment.get('speaker', 'Unknown')
                            text = segment.get('text', '').strip()
                            if text:
                                f.write(f"화자_{speaker}: {text}\n")
                print(f"장문 인식 결과 저장 완료: {self.long_file_path}")
            else:
                print(f"장문 인식 API 오류: {response.status_code}")

        except Exception as e:
            print(f"장문 인식 처리 중 오류: {str(e)}")

    def start_long_recognition_thread(self):
        """장문 인식 처리 스레드 시작"""
        def run_long_recognition():
            while self.is_active:
                self.process_long_recognition()
                time.sleep(self.long_recognition_interval)
        
        self.long_recognition_thread = threading.Thread(target=run_long_recognition)
        self.long_recognition_thread.daemon = True
        self.long_recognition_thread.start()

    def create_request_generator(self):
        """gRPC 요청 생성기"""
        yield nest_pb2.NestRequest(
            type=nest_pb2.RequestType.CONFIG,
            config=nest_pb2.NestConfig(
                config=json.dumps({
                    "transcription": {"language": "ko"},
                    "semanticEpd": {
                        "useWordEpd": "true",
                        "syllableThreshold": "2",
                        "durationThreshold": "2000",
                        "gapThreshold": "500"
                    }
                })
            )
        )
        
        while self.is_active:
            with self.lock:
                if self.audio_queue:
                    audio_data = self.audio_queue.pop(0)
                    # Float32Array를 Int16Array로 변환
                    audio_data = (np.array(audio_data) * 32767).astype(np.int16).tobytes()
                    # 오디오 데이터 누적
                    self.audio_accumulator.add_audio(audio_data)
                    yield nest_pb2.NestRequest(
                        type=nest_pb2.RequestType.DATA,
                        data=nest_pb2.NestData(
                            chunk=audio_data,
                            extra_contents=json.dumps({"seqId": 0, "epFlag": False})
                        )
                    )
            time.sleep(0.01)

    def process_audio(self, audio_data):
        """오디오 데이터를 큐에 추가"""
        if self.is_active:
            try:
                with self.lock:
                    self.audio_queue.append(audio_data)
            except Exception as e:
                print(f"[{self.sid}] 오디오 처리 중 오류: {str(e)}")

    def finish(self):
        """세션 종료"""
        self.is_active = False
        if self.channel:
            self.channel.close()
        # 마지막으로 장문 인식 실행
        self.process_long_recognition()
        print(f"[{self.sid}] 세션 종료됨")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/main")
def main():
    base_url = CONFIG['server']['base_url']
    urls = {
        'url_app2': f"{base_url}:{CONFIG['apps']['app2_result']['port']}",
        'url_app3': f"{base_url}:{CONFIG['apps']['app3_result']['port']}",
        'url_app4': f"{base_url}:{CONFIG['apps']['app4_result']['port']}",
        'url_final_test': f"{base_url}:{CONFIG['apps']['final_test_4']['port']}"
    }
    return render_template("main.html", **urls)

@socketio.on('connect')
def on_connect(auth):
    sid = request.sid
    print(f"[{sid}] 브라우저 소켓 연결됨")
    active_sessions[sid] = STTSession(sid)

@socketio.on('audio_data')
def handle_audio_data(data):
    try:
        sid = request.sid
        if sid in active_sessions:
            session = active_sessions[sid]
            # ArrayBuffer를 NumPy 배열로 변환
            audio_array = np.frombuffer(data, dtype=np.float32)
            session.process_audio(audio_array)
    except Exception as e:
        print(f"오디오 데이터 처리 중 오류: {str(e)}")

@socketio.on('disconnect')
def on_disconnect(auth):
    sid = request.sid
    if sid in active_sessions:
        session = active_sessions[sid]
        session.finish()
        del active_sessions[sid]
        print(f"[{sid}] 브라우저 소켓 해제 및 세션 종료")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=33178, debug=True)
