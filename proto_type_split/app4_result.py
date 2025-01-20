import os
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import json

def load_config():
    with open('server_conf.json', 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()

app = Flask(__name__)
app.config['SECRET_KEY'] = CONFIG['apps']['app4_result']['secret_key']
socketio = SocketIO(app, cors_allowed_origins=CONFIG['cors']['allowed_origins'])

# 모니터링할 디렉토리 설정
MONITOR_DIR = CONFIG['directories']['recordings']
os.makedirs(MONITOR_DIR, exist_ok=True)

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.file_positions = {}  # {파일경로: 마지막 읽은 위치}
        self.current_files = {}   # {파일경로: 마지막 수정 시간}
        self.check_existing_files()
        print("파일 모니터링 핸들러 초기화 완료")

    def check_existing_files(self):
        """기존 파일들의 현재 상태 저장"""
        for filename in os.listdir(MONITOR_DIR):
            if any(filename.endswith(ext) for ext in ['_syc_qa.txt', '_syc_search.txt', '_syc_summary.txt']):
                filepath = os.path.join(MONITOR_DIR, filename)
                self.current_files[filepath] = os.path.getmtime(filepath)
                self.file_positions[filepath] = 0
                print(f"기존 파일 발견: {filepath}")

    def on_modified(self, event):
        if event.is_directory:
            return

        try:
            filepath = event.src_path
            if not any(filepath.endswith(ext) for ext in ['_syc_qa.txt', '_syc_search.txt', '_syc_summary.txt']):
                return

            # 파일 수정 시간 체크
            current_mtime = os.path.getmtime(filepath)
            last_mtime = self.current_files.get(filepath, 0)
            
            if current_mtime == last_mtime:
                return

            self.current_files[filepath] = current_mtime
            print(f"\n파일 변경 감지: {filepath}")
            
            current_pos = self.file_positions.get(filepath, 0)
            content = self.read_file_safely(filepath, current_pos)
            
            if content:
                file_type = ''
                if filepath.endswith('_syc_qa.txt'):
                    file_type = 'qa'
                elif filepath.endswith('_syc_search.txt'):
                    file_type = 'search'
                elif filepath.endswith('_syc_summary.txt'):
                    file_type = 'summary'

                # 새로운 내용이 있을 때만 전송
                if content.strip():
                    socketio.emit('file_update', {
                        'type': file_type,
                        'content': content,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # 파일 위치 업데이트
                    self.file_positions[filepath] = self.get_file_size(filepath)
                
        except Exception as e:
            print(f"파일 처리 중 오류: {str(e)}")

    def get_file_size(self, filepath):
        """파일 크기 반환"""
        try:
            return os.path.getsize(filepath)
        except OSError:
            return 0

    def read_file_safely(self, filepath, start_pos=0):
        """안전하게 파일 읽기"""
        if not os.path.exists(filepath):
            return None

        encodings = ['utf-8', 'cp949', 'euc-kr']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    f.seek(start_pos)
                    content = f.read()
                    return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"파일 읽기 오류 ({encoding}): {str(e)}")

        return None

@app.route('/')
def index():
    return render_template('yc_result.html')

def start_file_monitoring():
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITOR_DIR, recursive=False)
    observer.start()
    print(f"파일 모니터링 시작됨 (디렉토리: {MONITOR_DIR})")
    return observer

if __name__ == '__main__':
    observer = start_file_monitoring()
    try:
        socketio.run(
            app, 
            host=CONFIG['common']['host'],
            port=CONFIG['apps']['app4_result']['port'],
            debug=CONFIG['common']['debug']
        )
    except KeyboardInterrupt:
        observer.stop()
    observer.join()