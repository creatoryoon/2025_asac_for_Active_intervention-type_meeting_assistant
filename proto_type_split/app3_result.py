import os
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import traceback
import json
import re

def load_config():
    with open('server_conf.json', 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()

app = Flask(__name__)
app.config['SECRET_KEY'] = CONFIG['apps']['app3_result']['secret_key']
socketio = SocketIO(app, cors_allowed_origins=CONFIG['cors']['allowed_origins'])

# 모니터링할 디렉토리 설정
MONITOR_DIR = CONFIG['directories']['recordings']
os.makedirs(MONITOR_DIR, exist_ok=True)

def start_file_monitoring():
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITOR_DIR, recursive=False)
    observer.start()
    print(f"파일 모니터링 시작됨 (디렉토리: {MONITOR_DIR})")
    return observer

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.file_positions = {}  # {파일경로: 마지막 읽은 위치}
        self.current_files = {}   # {파일경로: 마지막 수정 시간}
        self.check_existing_files()
        print("파일 모니터링 핸들러 초기화 완료")

    # def check_existing_files(self):
    #     """기존 파일들의 현재 상태 저장"""
    #     for filename in os.listdir(MONITOR_DIR):
    #         if filename.endswith('_summary.txt') or filename.endswith('_gpt.txt'):
    #             filepath = os.path.join(MONITOR_DIR, filename)
    #             self.current_files[filepath] = os.path.getmtime(filepath)
    #             self.file_positions[filepath] = 0
    #             print(f"기존 파일 발견: {filepath}")
    def check_existing_files(self):
        """기존 파일들의 현재 상태 저장"""
        for filename in os.listdir(MONITOR_DIR):
            # datetime_summary.txt 형식 또는 _gpt.txt 파일 처리
            if (re.match(r'\d{8}_\d{6}_summary\.txt$', filename) or 
                filename.endswith('_gpt.txt')):
                filepath = os.path.join(MONITOR_DIR, filename)
                self.current_files[filepath] = os.path.getmtime(filepath)
                self.file_positions[filepath] = 0
                print(f"기존 파일 발견: {filepath}")

    # def on_modified(self, event):
    #     if event.is_directory:
    #         return

    #     try:
    #         filepath = event.src_path
    #         if not (filepath.endswith('_summary.txt') or filepath.endswith('_gpt.txt')):
    #             return

    #         current_mtime = os.path.getmtime(filepath)
    #         last_mtime = self.current_files.get(filepath, 0)
            
    #         if current_mtime == last_mtime:
    #             return

    #         self.current_files[filepath] = current_mtime
    #         print(f"\n파일 변경 감지: {filepath}")
            
    #         if filepath.endswith('_summary.txt'):
    #             self.handle_summary_file(filepath)
    #         elif filepath.endswith('_gpt.txt'):
    #             self.handle_gpt_file(filepath)
                
    #     except Exception as e:
    #         print(f"파일 처리 중 오류: {str(e)}")
    #         traceback.print_exc()
    def on_modified(self, event):
        if event.is_directory:
            return
    
        try:
            filepath = event.src_path
            filename = os.path.basename(filepath)
    
            # gpt 파일은 기존대로 처리
            if filename.endswith('_gpt.txt'):
                self.process_file(filepath, 'gpt')
            # summary 파일은 datetime 패턴 확인 후 처리
            elif re.match(r'\d{8}_\d{6}_summary\.txt$', filename):
                self.process_file(filepath, 'summary')
            # 다른 파일은 무시
            else:
                return
                    
        except Exception as e:
            print(f"파일 처리 중 오류: {str(e)}")
            traceback.print_exc()

    def process_file(self, filepath, file_type):
        """공통 파일 처리 로직"""
        current_mtime = os.path.getmtime(filepath)
        last_mtime = self.current_files.get(filepath, 0)
        
        if current_mtime == last_mtime:
            return  # 실제 변경이 없으면 무시
    
        self.current_files[filepath] = current_mtime
        print(f"\n파일 변경 감지: {filepath}")
        
        if file_type == 'summary':
            self.handle_summary_file(filepath)
        elif file_type == 'gpt':
            self.handle_gpt_file(filepath)

    def read_file_safely(self, filepath):
        """안전하게 파일 읽기"""
        encodings = ['utf-8', 'cp949', 'euc-kr']
        
        if not os.path.exists(filepath):
            return None, 0

        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                    return content, f.tell()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"파일 읽기 중 오류 발생 ({encoding}): {str(e)}")

        return None, 0

    def handle_summary_file(self, filepath):
        try:
            content = self.read_file_safely(filepath)[0]
            if content is not None:
                socketio.emit('text_update', {
                    'type': 'summary',
                    'text': content,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                print(f"요약 텍스트 전송 - 전체 내용 업데이트")
                    
        except Exception as e:
            print(f"요약 파일 처리 오류: {str(e)}")
            traceback.print_exc()

    def handle_gpt_file(self, filepath):
        try:
            content = self.read_file_safely(filepath)[0]
            if content is not None:
                socketio.emit('text_update', {
                    'type': 'gpt',
                    'text': content,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                print(f"GPT 텍스트 전송 - 전체 내용 업데이트")
                    
        except Exception as e:
            print(f"GPT 파일 처리 오류: {str(e)}")
            traceback.print_exc()

@app.route('/')
def index():
    return render_template('summary.html')

if __name__ == '__main__':
    observer = start_file_monitoring()
    try:
        socketio.run(
            app, 
            host=CONFIG['common']['host'],
            port=CONFIG['apps']['app3_result']['port'],
            debug=CONFIG['common']['debug']
        )
    except KeyboardInterrupt:
        observer.stop()
    observer.join()