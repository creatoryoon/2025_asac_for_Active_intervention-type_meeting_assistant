해당 버전은 server에서 돌리기 위한 버전입니다.
우분투 기준이며 
conda로 가상환경이 python 3.10.13으로 생성,
git clone
req.txt로 의존성 설치.
python3 -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. nest.proto 로 컴파일.
server_conf.json에 각 api와 포트 입력
manage_servers.py로 실행.
