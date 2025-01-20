import subprocess
import time
import signal
import os
import psutil
import json
from datetime import datetime

def load_config():
    with open('server_conf.json', 'r', encoding='utf-8') as f:
        return json.load(f)

class ServerManager:
    def __init__(self):
        self.config = load_config()
        self.processes = {}
        self.server_files = [
            'app.py',
            'app2_result.py',
            'app3_result.py',
            'app4_result.py',
            'gpt_sum.py',
            'final_test_4.py',
            'agent_ssh_final2.py'
        ]
        self.log_dir = "server_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def start_servers(self):
        """모든 서버 프로세스 시작"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for server_file in self.server_files:
            try:
                # 로그 파일 설정
                log_file = os.path.join(self.log_dir, f"{server_file}_{timestamp}.log")
                with open(log_file, 'w') as f:
                    # 각 프로세스를 새로운 프로세스 그룹에서 실행
                    process = subprocess.Popen(
                        ['python', server_file],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid
                    )
                self.processes[server_file] = process
                print(f"Started {server_file} (PID: {process.pid})")
                time.sleep(2)  # 각 서버 시작 사이에 약간의 딜레이
            except Exception as e:
                print(f"Error starting {server_file}: {e}")

    def stop_servers(self):
        """모든 서버 프로세스 중지 시도"""
        for server_file, process in self.processes.items():
            try:
                # 프로세스 그룹 전체에 SIGTERM 시그널 전송
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                print(f"Sent SIGTERM to {server_file} (PID: {process.pid})")
            except Exception as e:
                print(f"Error stopping {server_file}: {e}")

    def force_kill_servers(self):
        """남아있는 프로세스 강제 종료"""
        for server_file, process in self.processes.items():
            try:
                # 프로세스와 그 자식 프로세스들 찾기
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                
                # 자식 프로세스들 먼저 종료
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass

                # 부모 프로세스 종료
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                
                print(f"Force killed {server_file} (PID: {process.pid})")
            except Exception as e:
                print(f"Error force killing {server_file}: {e}")

    def check_ports(self):
        """설정된 포트들이 사용 가능한지 확인"""
        import socket
        
        for app_name, app_config in self.config['apps'].items():
            port = app_config['port']
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('0.0.0.0', port))
                sock.close()
            except socket.error:
                print(f"Warning: Port {port} for {app_name} is already in use")
                return False
        return True

    def cleanup_zombie_processes(self):
        """좀비 프로세스 정리"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.cmdline()
                    if len(cmdline) >= 2 and 'python' in cmdline[0] and any(server in cmdline[1] for server in self.server_files):
                        proc.kill()
                        print(f"Cleaned up zombie process: {proc.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            print(f"Error cleaning up zombie processes: {e}")

def main():
    manager = ServerManager()
    
    # 시작 전 포트 체크
    if not manager.check_ports():
        print("Port conflict detected. Cleaning up existing processes...")
        manager.cleanup_zombie_processes()
        time.sleep(2)

    try:
        print("Starting all servers...")
        manager.start_servers()
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down servers gracefully...")
        manager.stop_servers()
        time.sleep(3)  # 정상 종료 대기
        
        print("Force killing any remaining processes...")
        manager.force_kill_servers()
        manager.cleanup_zombie_processes()
        
        print("All servers have been shut down.")

if __name__ == "__main__":
    main()