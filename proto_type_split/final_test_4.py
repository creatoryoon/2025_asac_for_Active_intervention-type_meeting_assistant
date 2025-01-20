from openai import OpenAI
import html
import glob
import time
import json
import threading
from threading import Lock
import requests
import re
import os 
from typing import Dict, List
from operator import itemgetter
from queue import Queue
from datetime import datetime
import traceback
from watchdog.observers import Observer
import atexit
from watchdog.events import FileSystemEventHandler

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

from flask import Flask, render_template
from flask_socketio import SocketIO

# 설정 파일 로드
def load_config():
    with open('server_conf.json', 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()

# OpenAI API Key
os.environ['OPENAI_API_KEY'] = CONFIG['apis']['openai']['api_key']

# Naver Search API
naver_search_id = CONFIG['apis']['naver_search']['client_id']
naver_search_pw = CONFIG['apis']['naver_search']['client_secret']

app = Flask(__name__)
app.config['SECRET_KEY'] = CONFIG['apps']['final_test_4']['secret_key']
socketio = SocketIO(app, cors_allowed_origins=CONFIG['cors']['allowed_origins'])


def create_memory():
    try:
        return ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    except Exception as e:
        print(f"Memory initialization error: {e}")
        raise

summary_memory = create_memory()

# 프롬프트 템플릿
summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        당신은 회의나 발표의 주요 내용을 실시간으로 요약하는 AI 어시스턴트입니다. 
        목표는 중요한 세부 사항을 놓치지 않으면서 간결하고 유익한 요약을 제공하는 것입니다. 아래는 요약을 작성할 때 따를 수 있는 템플릿입니다. 모든 내용을 처리할 때 이 구조를 따르세요.
        
        실시간 문장에 다음과 같은 양식으로 사용자에게 정보를 제공합니다.
        *********
        회의/발표 기본 템플릿 (적용 가능한 경우)

        회의 안건/프레젠테이션 주제: # 문장에서 주제가 나오면 이 양식으로 출력합니다.
        날짜 및 시간: # 날짜를 알 수 있으면 그 다음으로 출력합니다.
        발표자/연설자 이름: # 알 수 있으면 실시합니다.
        
        각 섹션 요약 
        각각의 회의나 발표 섹션에 대해 실시간으로 내용을 기록하며, 관련된 주요 내용만을 요약하십시오.
        핵심적인 논의 내용이나 발표의 주요 포인트를 간결하게 요약합니다.
        각 발언자의 핵심 아이디어 및 주장
        각 구체적인 내용을 섹션별로
        참석자들의 질문 및 발표자/연설자의 답변 요약(해당시)
        하나의 안건, 주장에 대한 요약 
        새로운 아이디어, 주제시 위의 섹션요약으로 다시 시작
        논의된 주제의 결론 또는 후속 조치
        향후 계획 또는 진행 사항
        *********
        실시간 음성을 입력하고 있기 때문에, 프로젝트,회의와 관련된 문장에 대해서만 형식에 맞게 제공하면 됩니다.
        회의와 관련없는 **무조건 무응답해야합니다**.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        당신은 사용자의 요청에 따라 gpt 학습된 데이터를 통해 질의응답을 하는 에이전트입니다.
        없는 정보를 만들어내서는 안됩니다.
        사용자의 요청에 따라 성심성의껏 중요한 내용만 요약해야합니다.
        """),
        ("human", "{input}"),
    ]
)

confernce_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        당신은 사용자의 요청에 따라 gpt 학습된 데이터를 통해 질의응답을 하는 에이전트입니다.
        없는 정보를 만들어내서는 안됩니다.
        사용자의 요청에 따라 성심성의껏 작성해야합니다
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

controll_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 다른 에이전트들을 관리하는 Controll Agent입니다.
    실시간으로 문장들이 완성돼서 입력됩니다. 그렇기에 문장을 파악해서 다른 에이전트를 작동시킬지 정확히 파악해야합니다.

    당신이 관리해야할 에이전트는 총 2가지 에이전트입니다.

    **summary 에이전트**: 회의나 발표의 실시간 내용을 요약하는 에이전트입니다.
    아래와 같은 경우에 summary 에이전트를 활성화해야 합니다:
    - 회의나 발표의 주요 내용이 포함된 경우
    - 중요한 정보나 의사결정이 포함된 경우
    - 참석자들의 의견이나 토론 내용이 포함된 경우
    - 업무 관련 대화나 프로젝트 논의가 포함된 경우
    - 미래 계획이나 전략에 대한 논의가 포함된 경우
    - 회의 목표나 안건에 관한 내용이 포함된 경우
    
     **qa 에이전트**: GPT 내부에 있는 데이터를 가지고 사용자의 질문에 응답할때 사용하는 에이전트 입니다.
     무조건 검색이라고 seach 에이전트를 호출하는 것이 아닌, gpt가 학습한 데이터로 충분히 답변이 가능하거나 gpt와의 질의응답이 필요하거나, 회의에 대한 요약, 의견 조율이 필요할때 이 에이전트를 활성화 시킵니다.
     수신 받은 문장이 진행중인 회의 내용일 경우 호출시키면 안됩니다. 'gpt야 물어볼게'와 같이 정보를 요구하면서 외부검색이 아닌경우만 실행시킵니다.

    **search 에이전트**: 외부 검색이 반드시 필요한 경우에만 사용하는 에이전트입니다.
    다음과 같은 경우에만 search 에이전트를 활성화합니다:
    - 실시간 뉴스나 최신 정보가 필요한 경우
    - 제품 가격이나 구매 정보가 필요한 경우
    - 특정 장소나 위치 정보가 필요한 경우

    기본적으로 모든 대화와 회의 내용은 summary 에이전트로 처리하는 것을 우선으로 합니다.
    qa나 search는 매우 명확한 필요성이 있을 때만 사용해야 합니다.
    
    에이전트 선택시:
    search 에이전트를 활성화 시킬땐 "search"
    qa 에이전트를 활성화 시킬땐 "qa"
    summary 에이전트를 활성화 시킬땐 "summary"
    아무런 호출이 필요없을땐 "N"으로 응답합니다.
    """),
    ("human", "{user_input}")
])

search_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        당신은 사용자의 자연어 질문을 최적화된 검색 쿼리로 변환하고, 네이버 검색 API를 통해 정보를 제공하는 전문가입니다.
        def naver_Search(search: str)를 사용하기 위해선  단하나의 'str'로  "query,o  bj"로 입력됩니다
        입력변수 search='query'+','+'obj'입니다. 
        - query: 요청에서 검색에 최적화된 키워드로 변환합니다.  
        - obj: 검색 유형을 지정합니다
          - 'news': 뉴스 검색 (특정 주제의 최신 뉴스나 기사를 찾을 때 사용)
          - 'local': 지역 정보 검색 (맛집, 가게, 장소 등을 찾을 때 사용)
          - 'shop': 상품 검색 (제품 정보나 가격을 찾을 때 사용)
	    도구를 사용시 해당 양식은 반드시 지켜져야 합니다.
        검색어에 해당하는 query는 최대한 간단하고 명확하게 구성되어야 합니다.
     
     
        # 주요 역할과 기능
        1. 문맥 파악:
            - 사용자의 질문에서 실제 의도와 필요한 정보를 정확히 파악
            - 불필요한 문구, 조사, 부가 설명 등을 제거하고 핵심 키워드 추출
            - "지금 삼성전자 주가가 어떻게 되나요?" → "삼성전자 주가"처럼 최적화
        
        2. 검색 유형 결정:
            - news: 뉴스, 속보, 동향, 현황 정보가 필요한 경우
            - shop: 제품 정보, 구매, 추천이 필요한 경우
            - local: 장소, 위치 정보가 필요한 경우
        
        3. 쿼리 최적화:
            A. 제품 검색 (shop) 예시:
                - "게이밍용 마우스 추천해줘" → "게이밍 마우스,shop"
                - "로지텍 무선 마우스 가격 알려줘" → "로지텍 무선마우스,shop"
            
            B. 뉴스 검색 (news) 예시:
                - "요즘 삼성전자 주가가 어떻게 되나요?" → "삼성전자 주가,news"
            
            C. 지역 검색 (local) 예시:
                - "근처 맛집 추천해줘" → "맛집,local"
        
        4. 중요 규칙:
            - 검색어는 최대한 간단하고 명확하게 구성
            - 불필요한 조사, 어미 제거
            - 문맥을 고려하여 필요한 키워드 추가
            - 검색 목적에 맞는 적절한 타입 선택
            - 문맥상 동의어나 유사어 고려
        
        5. 예시
            - '요즘 회의 할 때 마이크 필요한거 같은데, 마이크 추천해줘' 라고 입력이 되면, (마이크, 추천)이라는 키워드를 인식해서 '마이크 추천'으로 검색
            - '너무 덥다. 내일 서울 마포구 날씨 검색해봐' 라고 이야기 하면 (내일,서울,마포구,날씨)에 대해서 검색을 진행
            - '삼성전자 주가 너무 많이 떨어지는거 같은데, 지금 삼성전자 주가 얼마인거야' 라고 물어보면 '삼성전자 주가' 이렇게 인식을 하고 검색을 진행
        
        # 검색 후 결과 표시 형식
        
        1. 뉴스 검색 결과:
        ```
        📰 [주제] 관련 뉴스 요약
        ============================

        분석한 기사 수: [X]개

        [기사 1]
        제목: [기사 제목]
        요약: [주요 내용 요약]
        원문: [링크]

        [기사 2]
        ...

        종합 분석:
        [전체 기사들의 핵심 내용과 트렌드 분석]
        ============================
        ```

        2. 상품 검색 결과:
        ```
        🛍️ [검색어] 상품 검색 결과
        ============================

        추천 상품 목록:

        [상품 1]
        제품명: [상품명]
        가격: [가격]원
        판매처: [판매처명]
        특징: [주요 특징]
        구매링크: [링크]

        [상품 2]
        ...

        구매 시 참고사항:
        [가격대 분석, 인기 모델, 주의사항 등]
        ============================
        ```

        3. 장소 검색 결과:
        ```
        📍 [검색어] 검색 결과
        ============================

        검색된 장소 목록:

        [장소 1]
        이름: [장소명]
        분류: [업종/카테고리]
        주소: [상세주소]
        특징: [주요 특징]

        [장소 2]
        ...

        참고사항:
        [영업시간, 주차정보, 기타 유용한 정보]
        ============================
        ```
        """,
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])



class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor
        self.file_positions = {}
        self._initialize_file_positions()
        print("파일 모니터링 핸들러 초기화됨")

    def _initialize_file_positions(self):
        try:
            for filepath in glob.glob(os.path.join(Config.BASE_PATH, '*_orig2.txt')):
                self.file_positions[filepath] = os.path.getsize(filepath)
        except Exception as e:
            print(f"파일 위치 초기화 오류: {e}")

    def on_modified(self, event):
        if event.is_directory or not (event.src_path.endswith('_orig2.txt') or event.src_path.endswith('_smg_orig2.txt')):
            return

        try:
            content = self._read_new_content(event.src_path)
            if content:
                print(f"새로운 내용 감지됨: {event.src_path}")
                self.processor.process_text(content)
        except Exception as e:
            print(f"파일 변경 처리 오류: {e}")
            traceback.print_exc()

    def _read_new_content(self, filepath):
        try:
            current_position = self.file_positions.get(filepath, 0)
            with open(filepath, 'r', encoding='utf-8') as f:
                f.seek(current_position)
                content = f.read()
                self.file_positions[filepath] = f.tell()
                return content if content else None
        except Exception as e:
            print(f"파일 읽기 오류: {e}")
            return None
    # def _read_new_content(self, filepath):
    #     try:
    #         with open(filepath, 'r', encoding='utf-8') as f:
    #             # 파일 위치를 추적하지 않고 매번 새로 읽기
    #             lines = f.readlines()
    #             if not lines:
    #                 return None
                    
    #             # 마지막 줄 가져오기
    #             last_line = lines[-1].strip()
                
    #             # 이전에 처리한 내용과 다른 경우에만 반환
    #             if last_line != self.file_positions.get(filepath):
    #                 self.file_positions[filepath] = last_line  # 현재 내용을 저장
    #                 return last_line
                
    #             return None
                
    #     except Exception as e:
    #         print(f"파일 읽기 오류: {e}")
    #     return None

class SummaryAgent:
    def __init__(self):
        self.model_structure = ChatOpenAI(
            model="gpt-4",
            temperature=0,
        )
        self.memory = summary_memory
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                |itemgetter(self.memory.memory_key)
            )
            | summary_prompt
            | self.model_structure
            | StrOutputParser()
        )
        self.task_queue = Queue()
        self.result_queue = Queue()  # 결과 큐 추가
        self.sentence = ""
        self.count = 0
        self.last_summary = None
        self.is_running = True
        self.processing_lock = threading.Lock()  # 처리 동기화를 위한 락 추가
        self.thread = threading.Thread(target=self.process_tasks)
        self.thread.daemon = True
        self.thread.start()

    def process_tasks(self):
        while self.is_running:
            try:
                if not self.task_queue.empty():
                    with self.processing_lock:
                        self.sentence += self.task_queue.get()
                        self.count += 1
                        if self.count >= 1:  # 문장이 완성되면 처리
                            self.summary_process_tasks()
                time.sleep(0.1)  # CPU 사용률 감소
            except Exception as e:
                print(f"요약 처리 중 오류 발생: {str(e)}")
                traceback.print_exc()

    def summary_process_tasks(self):
        try:
            answer = self.chain.invoke({"input": self.sentence})
            print('요약 에이전트 응답:', answer)
            
            # 컨텍스트 저장
            self.memory.save_context(
                inputs={"human": self.sentence},
                outputs={"ai": answer}
            )
            
            # 결과 저장 및 큐에 추가
            self.last_summary = answer
            self.result_queue.put({
                'type': 'summary',
                'content': answer,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # 상태 초기화
            self.sentence = ""
            self.count = 0
            
        except Exception as e:
            print(f"요약 생성 중 오류 발생: {str(e)}")
            traceback.print_exc()
            
class QAAgent:
    def __init__(self):
        self.model_structure = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
        )
        self.memory = summary_memory
        
        # 디버깅을 위한 로거 설정
        self.debug_mode = True  # 디버깅 모드 활성화
        
        # QA 체인 초기화
        try:
            self.chain = (
                RunnablePassthrough.assign(
                    chat_history=RunnableLambda(self.memory.load_memory_variables)
                    |itemgetter("chat_history")
                )
                | qa_prompt
                | self.model_structure
            )
            print("QA 체인 초기화 성공")
        except Exception as e:
            print(f"QA 체인 초기화 실패: {str(e)}")
            traceback.print_exc()

        self.task_queue = Queue()
        self.result_queue = Queue()
        self.sentence = ""
        self.count = 0
        self.last_response = None
        self.is_running = True
        self.processing_lock = threading.Lock()
        
        # 디버깅용 상태 추적
        self.processing_status = {
            'last_input': None,
            'last_error': None,
            'total_processed': 0,
            'successful_responses': 0,
            'failed_responses': 0
        }
        
        self.thread = threading.Thread(target=self.process_tasks)
        self.thread.daemon = True
        self.thread.start()
        
        print("QAAgent 초기화 완료")

    def _debug_log(self, message: str, level: str = 'info'):
        """디버깅 로그 출력"""
        if self.debug_mode:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[QA-{level.upper()}] {timestamp} - {message}")

    def process_tasks(self):
        """태스크 처리 메인 루프"""
        while self.is_running:
            try:
                if not self.task_queue.empty():
                    with self.processing_lock:
                        new_text = self.task_queue.get()
                        self._debug_log(f"새 태스크 수신: {new_text}")
                        
                        if not new_text.strip():
                            self._debug_log("빈 텍스트 무시", 'warn')
                            continue
                            
                        self.sentence = new_text
                        self.count += 1
                        self.processing_status['total_processed'] += 1
                        self.processing_status['last_input'] = new_text
                        
                        if self.count >= 1:
                            self.qa_process_tasks()
                            
                time.sleep(0.1)
            except Exception as e:
                self._debug_log(f"태스크 처리 중 오류: {str(e)}", 'error')
                self.processing_status['last_error'] = str(e)
                traceback.print_exc()

    def qa_process_tasks(self):
        try:
            self._debug_log(f"QA 처리 시작 - 입력: {self.sentence}")

            # 입력 검증
            if not isinstance(self.sentence, str):
                raise ValueError(f"Invalid input type: {type(self.sentence)}")

            # 메모리에서 대화 히스토리를 가져올 때, 빈 dict를 인자로 전달
            memory_data = self.memory.load_memory_variables({})  
            chat_history = memory_data.get("chat_history", [])

            # QA 처리
            response = self.chain.invoke({
                "input": self.sentence,
                "chat_history": chat_history
            })

            self._debug_log("QA 응답 생성 완료")

            if not hasattr(response, 'content'):
                self._debug_log("응답에 content 속성이 없음", 'warn')
                content = str(response)
            else:
                content = response.content

            # 메모리에 컨텍스트 저장
            self.memory.save_context(
                {"input": self.sentence},
                {"output": content}
            )

            # 결과 저장 및 전송
            self.last_response = content
            self.result_queue.put({
                'type': 'qa',
                'content': content,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            self._debug_log(f"처리 완료 - 응답: {content[:100]}...")
            self.processing_status['successful_responses'] += 1

            # 상태 초기화
            self.sentence = ""
            self.count = 0

        except Exception as e:
            self._debug_log(f"QA 처리 중 오류 발생: {str(e)}", 'error')
            self.processing_status['failed_responses'] += 1
            self.processing_status['last_error'] = str(e)
            traceback.print_exc()

            # 에러 발생시 사용자에게 알림
            error_message = "죄송합니다. 답변 생성 중 문제가 발생했습니다. 다시 시도해주세요."
            self.result_queue.put({
                'type': 'qa',
                'content': error_message,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            })

    def get_debug_status(self) -> dict:
        """디버깅 상태 정보 반환"""
        return {
            'status': self.processing_status,
            'queue_size': self.task_queue.qsize(),
            'is_running': self.is_running,
            'memory_size': len(self.memory.load_memory_variables().get("chat_history", [])),
            'last_response': self.last_response[:100] if self.last_response else None
        }

    def force_process(self, text: str) -> str:
        """디버깅용 강제 처리 메서드"""
        try:
            self._debug_log(f"강제 처리 시작 - 입력: {text}")
            response = self.chain.invoke({"input": text})
            self._debug_log("강제 처리 완료")
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            self._debug_log(f"강제 처리 중 오류: {str(e)}", 'error')
            return f"Error: {str(e)}"


class SearchAgent:
    def __init__(self):
        self.model_structure = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.last_result = None
        self.processing_lock = threading.Lock()
        self.is_running = True
        
        # 검색 키워드 사전 정의
        self.search_keywords = {
            'shop': [
                "가격", "제품", "물건", "구매", "사다", "구입", "살", "노트북", "컴퓨터",
                "사줘", "추천", "사세요", "구매", "구입", "장바구니", "쇼핑", "아이템",
                "제품", "상품", "물건", "브랜드", "모델", "스펙", "사양", "최저가",
                "견적", "세트", "패키지", "부품", "액세서리", "주문"
            ],
            'local': [
                "맛집", "장소", "위치", "가게", "식당", "어디", "주변", "근처",
                "찾아가", "방문", "주소", "오픈", "영업", "매장", "지점", "지하철",
                "버스", "약국", "병원", "카페", "편의점", "마트", "은행", "학교",
                "도서관", "공원", "주차장", "호텔", "숙소"
            ],
            'news': [
                "뉴스", "소식", "기사", "속보", "발표", "보도", "최신", "새로운",
                "정보", "동향", "현황", "상황", "업데이트", "이슈", "문제", "사건",
                "사고", "정책", "시장", "산업", "기업", "주가", "증시", "날씨",
                "예보", "전망", "분석", "평가", "리뷰"
            ]
        }
        
        # 제외할 스톱워드 정의
        self.stop_words = [
            "해줘", "알려줘", "추천해줘", "보여줘", "좀", 
            "주세요", "할까요", "하나요", "있나요", "일까요", "될까요",
            "그래서", "그러면", "그리고", "하지만", "그런데"
        ]

        self.thread = threading.Thread(target=self.process_tasks)
        self.thread.daemon = True
        self.thread.start()

    def naver_search(self, search_query: str) -> str:
        try:
            parts = search_query.split(",")
            if len(parts) == 2:
                query, obj = [p.strip() for p in parts]  # 공백 제거 추가
            else:
                query = search_query
                obj = 'news'  # 기본값

            if not query:  # 빈 쿼리 체크
                return "검색어가 없습니다"

            sort = 'sim'
            encText = requests.utils.quote(query)
            display = 3 if obj == 'news' else 5

            print(f"검색 요청: query={query}, type={obj}")  # 디버깅용

            url = f"https://openapi.naver.com/v1/search/{obj}?query={encText}&display={display}&sort={sort}"
            if obj == 'shop':
                url += '&exclude=used:cbshop'

            headers = {
                "X-Naver-Client-Id": naver_search_id,
                "X-Naver-Client-Secret": naver_search_pw
            }

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return f"Error Code: {response.status_code}"

            info = response.json()
            if not info.get('items'):
                return f"검색 결과가 없습니다. (검색어: {query})"

            if obj == 'news':
                return self._process_news_results(info)
            elif obj == 'shop':
                return self._process_shop_results(info)
            elif obj == 'local':
                return self._process_local_results(info)
            return "지원되지 않는 검색 유형입니다."

        except Exception as e:
            print(f"Naver Search error: {str(e)}")
            traceback.print_exc()
            return f"검색 중 오류가 발생했습니다: {str(e)}"

    def process_tasks(self):
        while self.is_running:
            try:
                if not self.task_queue.empty():
                    with self.processing_lock:
                        query = self.task_queue.get()
                        search_type = self._determine_search_type(query)
                        cleaned_query = self._clean_query(query, search_type)
                        search_param = f"{cleaned_query},{search_type}"
                        print(f"검색 파라미터: {search_param}")  # 디버깅용
                        result = self.naver_search(search_param)
                        
                        if result:
                            self.last_result = result
                            self.result_queue.put({
                                'type': 'search',
                                'content': result,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                time.sleep(0.1)
            except Exception as e:
                print(f"검색 처리 중 오류 발생: {str(e)}")
                traceback.print_exc()

    def _determine_search_type(self, query: str) -> str:
        """개선된 검색 유형 결정"""
        query_lower = query.lower()
        
        # 각 검색 유형별 키워드 매칭 점수 계산
        scores = {
            'shop': 0,
            'local': 0,
            'news': 0
        }
        
        for search_type, keywords in self.search_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[search_type] += 1
        
        # 가장 높은 점수를 가진 검색 유형 반환
        max_score_type = max(scores.items(), key=lambda x: x[1])
        
        # 아무 키워드도 매칭되지 않은 경우 컨텍스트 기반 판단
        if max_score_type[1] == 0:
            # 가격, 구매 관련 숫자가 포함된 경우 shop으로 판단
            if re.search(r'\d+원|\d+만원|\d+천원|\d+억원', query_lower):
                return 'shop'
            # 위치 관련 조사가 포함된 경우 local로 판단
            elif re.search(r'에서|으로|로|에', query_lower):
                return 'local'
            # 기본값은 news
            return 'news'
        
        return max_score_type[0]

    def _clean_query(self, query: str, search_type: str) -> str:
        """검색 타입별 최적화된 쿼리 정제"""
        # 기본 전처리
        query = query.strip()
        
        # 스톱워드 처리 개선
        stop_words_pattern = r'\b(' + '|'.join(self.stop_words) + r')\b'
        query = re.sub(stop_words_pattern, '', query)
        
        # 검색 타입별 특수 처리
        if search_type == 'shop':
            # 가격 관련 부분 추출
            price_matches = re.findall(r'\d+만원|\d+천원|\d+원|\d+억원', query)
            if price_matches:
                price_part = ' '.join(price_matches)
                query = f"{query} {price_part}"
            
            # 제품 사양이나 모델명 보존
            model_matches = re.findall(r'[A-Za-z0-9-]+', query)
            if model_matches:
                model_part = ' '.join(model_matches)
                if model_part not in query:
                    query = f"{query} {model_part}"
                    
        elif search_type == 'local':
            # 지역명 보존
            location_pattern = r'(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)'
            location_matches = re.findall(location_pattern, query)
            if location_matches:
                location = location_matches[0]
                query = f"{location} {query}"
                
        elif search_type == 'news':
            # 날짜 관련 정보 보존
            date_matches = re.findall(r'\d+년|\d+월|\d+일|최근|오늘|어제|이번주|이번달', query)
            if date_matches:
                date_part = ' '.join(date_matches)
                if date_part not in query:
                    query = f"{query} {date_part}"
        
        # 불필요한 공백 제거 및 정리
        query = re.sub(r'\s+', ' ', query).strip()
        
        # 빈 검색어 방지
        if not query.strip():
            return "검색어가 없습니다"
        
        return query

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _process_news_results(self, info: dict) -> str:
        """뉴스 검색 결과 처리"""
        news_text = "📰 뉴스 검색 결과\n"
        news_text += "=" * 50 + "\n\n"
        
        for idx, item in enumerate(info['items'], 1):
            title = self._clean_text(item.get('title', ''))
            description = self._clean_text(item.get('description', ''))
            link = item.get('originallink', '')
            
            news_text += f"[뉴스 {idx}]\n"
            news_text += f"제목: {title}\n"
            news_text += f"내용: {description}\n"
            news_text += f"원문링크: {link}\n"
            news_text += "-" * 40 + "\n\n"
        
        news_text += "=" * 50
        return news_text

    def _process_shop_results(self, info: dict) -> str:
        """상품 검색 결과 처리"""
        shop_text = "🛍️ 상품 검색 결과\n"
        shop_text += "=" * 50 + "\n\n"
        
        for idx, item in enumerate(info['items'], 1):
            title = self._clean_text(item.get('title', ''))
            price = item.get('lprice', '가격 정보 없음')
            mall = item.get('mallName', '')
            link = item.get('link', '')
            
            shop_text += f"[상품 {idx}]\n"
            shop_text += f"상품명: {title}\n"
            shop_text += f"가격: {price}원\n"
            shop_text += f"판매처: {mall}\n"
            shop_text += f"구매링크: {link}\n"
            shop_text += "-" * 40 + "\n\n"
        
        shop_text += "=" * 50
        return shop_text

    def _process_local_results(self, info: dict) -> str:
        """지역 검색 결과 처리"""
        local_text = "📍 장소 검색 결과\n"
        local_text += "=" * 50 + "\n\n"
        
        for idx, item in enumerate(info['items'], 1):
            title = self._clean_text(item.get('title', ''))
            address = item.get('address', '주소 정보 없음')
            category = item.get('category', '')
            
            local_text += f"[장소 {idx}]\n"
            local_text += f"이름: {title}\n"
            local_text += f"분류: {category}\n"
            local_text += f"주소: {address}\n"
            local_text += "-" * 40 + "\n\n"
        
        local_text += "=" * 50
        return local_text

class MultiAgentProcessor:
    def __init__(self, socketio):
        self.socketio = socketio
        self.model_structure = ChatOpenAI(
            model='gpt-4',
            temperature=0,
        )
        self.summary_agent = SummaryAgent()
        self.qa_agent = QAAgent()
        self.search_agent = SearchAgent()
        self.chain = RunnablePassthrough()|controll_prompt|self.model_structure
        
        # 텍스트 처리를 위한 큐와 변수들
        self.text_queue = Queue()
        self.last_texts = ""
        self.sentence_queue = Queue()
        self.is_running = True
        self.sentence_lock = threading.Lock()
        self.file_lock = threading.Lock()
        
        # 파일 시스템 설정
        self.base_path = os.path.join('.', 'recordings')
        self.paths = {
            'realtime': os.path.join(self.base_path, 'realtime'),
            'summary': os.path.join(self.base_path, 'summary'),
            'qa': os.path.join(self.base_path, 'qa'),
            'search': os.path.join(self.base_path, 'search')
        }
        
        # 디렉토리 생성
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
            
        # 파일 모니터링 설정
        self.event_handler = FileChangeHandler(self)
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler, 
            self.base_path,
            recursive=False
        )
        self.observer.start()

        # 처리 스레드 시작
        self.processing_thread = threading.Thread(target=self._process_text_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()

    def process_text(self, text):
        """외부에서 접근 가능한 텍스트 처리 메서드"""
        if not text:
            return
        
        self.text_queue.put(text)

    def _process_text_queue(self):
        """내부 텍스트 큐 처리"""
        while self.is_running:
            try:
                if not self.text_queue.empty():
                    with self.sentence_lock:
                        new_text = self.text_queue.get()
                        if not new_text.strip():
                            continue
                            
                        self.last_texts += new_text
                        processed_sentences = self._process_sentences(self.last_texts)
                        
                        if processed_sentences:
                            latest_sentence = processed_sentences[-1]
                            self._handle_sentence(latest_sentence)
                            
                time.sleep(0.1)
            except Exception as e:
                print(f"텍스트 큐 처리 중 오류: {str(e)}")
                traceback.print_exc()

    def _process_sentences(self, text):
        """문장 처리 및 중복 제거"""
        sentences = re.split(r"(?<=[.!?])\s*", text.strip())
        processed_sentences = []
        
        for sentence in sentences[:-1]:  # 마지막 미완성 문장 제외
            cleaned_sentence = self._clean_sentence(sentence)
            if cleaned_sentence:
                processed_sentences.append(cleaned_sentence)
                self.sentence_queue.put(cleaned_sentence)
        
        self.last_texts = sentences[-1] if sentences else ""
        return processed_sentences

    def _clean_sentence(self, sentence):
        """문장 정제 및 중복 단어 제거"""
        words = sentence.split()
        cleaned_words = []
        prev_word = None
        repeating = False
        
        for word in words:
            if word != prev_word:
                cleaned_words.append(word)
                repeating = False
            elif not repeating:
                cleaned_words.append(word)
                repeating = True
            prev_word = word
            
        return ' '.join(cleaned_words).strip()

    def _handle_sentence(self, sentence):
        """개별 문장 처리"""
        try:
            # 실시간 텍스트 저장
            self._save_realtime_text(sentence)
            
            # 컨트롤러 응답 처리
            response = self.chain.invoke(sentence)
            agent_type = response.content.lower().strip()
            print(f"\n입력 텍스트: [{sentence}]")
            print(f"컨트롤러 응답: [{agent_type}]")

            # 에이전트 작업 처리
            self._dispatch_to_agents(sentence, agent_type)
            
            # 소켓 이벤트 발송
            self.socketio.emit('realtime_text', {'text': sentence})
            
        except Exception as e:
            print(f"문장 처리 중 오류: {str(e)}")
            traceback.print_exc()

    def _dispatch_to_agents(self, text, agent_type):
        """
        컨트롤러(prompt)에서 나온 agent_type에 따라 
        해당 에이전트를 호출하는 로직.
        """
        # 1) summary 에이전트는 항상 호출 (회의내용 요약)
        self.summary_agent.task_queue.put(text)

        # 2) '질문 키워드'가 있는지 체크해서, 있으면 qa로 강제 override
        question_keywords = ["?", "알려줘", "무엇", "어떻게", "왜", "질문", "궁금", "설명", "가르쳐"]
        # 간단한 소문자 변환
        lower_text = text.lower()

        if any(keyword in lower_text for keyword in question_keywords):
            agent_type = "qa"

        # 3) 최종 agent_type 분기
        if agent_type == "qa":
            print("QA 작업 시작...")
            self.qa_agent.task_queue.put(text)
        elif agent_type == "search":
            print("검색 작업 시작...")
            text = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}: ', '', text)
            self.search_agent.task_queue.put(text)
        else:
            # "summary"나 "N"이면 별도 동작 없이 넘어감
            # 이미 summary_agent는 위에서 호출했으므로 패스
            pass


    def _process_results(self):
        """결과 처리"""
        while self.is_running:
            try:
                for agent, queue in [
                    ('summary', self.summary_agent.result_queue),
                    ('qa', self.qa_agent.result_queue),
                    ('search', self.search_agent.result_queue)
                ]:
                    if not queue.empty():
                        result = queue.get()
                        self.save_agent_output(
                            result['type'],
                            result['content'],
                            result['timestamp']
                        )
                time.sleep(0.1)
            except Exception as e:
                print(f"결과 처리 중 오류: {str(e)}")
                traceback.print_exc()

    def _save_realtime_text(self, text):
        """실시간 텍스트 저장"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            filepath = os.path.join(
                self.paths['realtime'],
                f"realtime_{datetime.now().strftime('%Y%m%d')}.txt"
            )
            
            with self.file_lock:
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {text}\n")
        except Exception as e:
            print(f"실시간 텍스트 저장 오류: {str(e)}")

    def save_agent_output(self, agent_type: str, content: str, timestamp: str = None):
        """에이전트 출력 저장"""
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            current_date = datetime.now().strftime("%Y%m%d")
            filename = f"{current_date}_smg_{agent_type}.txt"
            filepath = os.path.join(self.paths[agent_type], filename)
            
            with self.file_lock:
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(f"{timestamp}: {content}\n")
            
            # 소켓 이벤트 발송
            self.socketio.emit('llm_update', {
                'type': agent_type,
                'content': content,
                'timestamp': timestamp
            })
            
        except Exception as e:
            print(f"{agent_type} 결과 저장 중 오류: {str(e)}")
            traceback.print_exc()

    def cleanup(self):
        """리소스 정리"""
        try:
            self.is_running = False
            
            # Observer 정리
            if hasattr(self, 'observer'):
                self.observer.stop()
                self.observer.join(timeout=5)

            # 각 에이전트 정리
            for agent in [self.summary_agent, self.qa_agent, self.search_agent]:
                if agent:
                    agent.is_running = False
                    if hasattr(agent, 'thread'):
                        agent.thread.join(timeout=5)

            # 스레드 정리
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=5)
            if hasattr(self, 'result_thread'):
                self.result_thread.join(timeout=5)
                
        except Exception as e:
            print(f"Cleanup 중 오류 발생: {str(e)}")
            traceback.print_exc()

processor = None

# Flask routes
@app.route('/')
def index():
    return render_template('ai.html')

@app.route('/summary')
def view_summary():
    summary_path = os.path.join('.', 'recordings', 'summary')
    try:
        # 오늘 날짜의 파일만 읽기
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"{current_date}_smg_summary.txt"
        filepath = os.path.join(summary_path, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content if content else "오늘의 요약 내용이 없습니다."
        else:
            return "오늘의 요약 파일이 아직 생성되지 않았습니다."
        
    except Exception as e:
        print(f"요약 읽기 오류: {str(e)}")
        return f"요약 파일 읽기 오류: {str(e)}"

@socketio.on('connect')
def handle_connect():
    global processor
    if processor is None:
        processor = MultiAgentProcessor(socketio)
    print("클라이언트 연결됨")

@socketio.on('disconnect')
def handle_disconnect():
    print("클라이언트 연결 해제됨")

if __name__ == "__main__":
    socketio.run(
        app,
        host=CONFIG['common']['host'],
        port=CONFIG['apps']['final_test_4']['port'],
        debug=CONFIG['common']['debug']
    )