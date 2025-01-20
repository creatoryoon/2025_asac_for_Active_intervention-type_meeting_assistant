import os
import time
import json
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'monitor_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 모니터링할 디렉토리 설정
MONITOR_DIR = 'recordings'
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
            if filename.endswith('_orig2.txt') or filename.endswith('_long.txt'):
                filepath = os.path.join(MONITOR_DIR, filename)
                self.current_files[filepath] = os.path.getmtime(filepath)
                self.file_positions[filepath] = 0
                print(f"기존 파일 발견: {filepath}")

    def on_modified(self, event):
        if event.is_directory:
            return

        try:
            filepath = event.src_path
            if not (filepath.endswith('_orig2.txt') or filepath.endswith('_long.txt')):
                return

            # 파일 수정 시간 체크
            current_mtime = os.path.getmtime(filepath)
            last_mtime = self.current_files.get(filepath, 0)
            
            if current_mtime == last_mtime:
                return  # 실제 변경이 없으면 무시

            self.current_files[filepath] = current_mtime
            print(f"\n파일 변경 감지: {filepath}")
            
            # 파일 타입에 따라 처리
            if filepath.endswith('_orig2.txt'):
                self.handle_orig2_file(filepath)
            elif filepath.endswith('_long.txt'):
                self.handle_long_file(filepath)
                
        except Exception as e:
            print(f"파일 처리 중 오류: {str(e)}")
            traceback.print_exc()
    
    def handle_orig2_file(self, filepath):
        """orig2 파일 처리"""
        try:
            current_pos = self.file_positions.get(filepath, 0)
            content, new_pos = self.read_file_safely(filepath, current_pos)
            
            if content and new_pos > current_pos:
                self.file_positions[filepath] = new_pos
                lines = content.strip().split('\n')
                
                for line in lines:
                    if not line.strip():
                        continue
    
                    # 줄바꿈 문자가 실제로 있는지 확인
                    original_line = line
                    has_newline = content.find(line + '\n') != -1
    
                    socketio.emit('text_update', {
                        'type': 'accumulated',
                        'text': original_line,
                        'hasNewline': has_newline
                    })
                    print(f"orig2 텍스트 전송: {original_line}, 줄바꿈: {has_newline}")
                    
        except Exception as e:
            print(f"orig2 파일 처리 오류: {str(e)}")
            traceback.print_exc()
    def handle_long_file(self, filepath):
        """화자 분리 파일 처리"""
        try:
            # 파일 전체 내용 읽기
            content = self.read_file_safely(filepath)[0]
            if content is not None:
                # 전체 내용을 그대로 전송
                socketio.emit('text_update', {
                    'type': 'speaker',
                    'text': content,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                print(f"화자 분리 텍스트 전송 - 전체 내용 업데이트")
                        
        except Exception as e:
            print(f"화자 분리 파일 처리 오류: {str(e)}")
            traceback.print_exc()

    def read_file_safely(self, filepath, start_pos=None):
        """안전하게 파일 읽기"""
        encodings = ['utf-8', 'cp949', 'euc-kr']
        
        if not os.path.exists(filepath):
            return None, 0

        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    if start_pos is not None:
                        f.seek(start_pos)
                    content = f.read()
                    new_pos = f.tell()
                    print(f"파일 성공적으로 읽음 (인코딩: {encoding})")
                    return content, new_pos
            except UnicodeDecodeError:
                print(f"{encoding} 인코딩으로 읽기 실패, 다음 인코딩 시도...")
                continue
            except Exception as e:
                print(f"파일 읽기 중 오류 발생 ({encoding}): {str(e)}")

        print(f"모든 인코딩 시도 실패: {filepath}")
        return None, 0



@app.route('/')
def index():
    return render_template('monitor.html')

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
        socketio.run(app, host="0.0.0.0", port=33179, debug=True)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()