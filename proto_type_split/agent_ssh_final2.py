import time
import pyaudio
import grpc
import nest_pb2
import nest_pb2_grpc
import json
import threading
from threading import Lock
import wave
import io
from datetime import datetime
from queue import Queue
import os
import requests
import re
from typing import List, Dict, Annotated
import re
from bs4 import BeautifulSoup
import html
import os 
from typing import Dict, List
from operator import itemgetter
from datetime import datetime # 파일 저장
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableAssign
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
# from langchain_teddynote.messages import stream_response
from langchain_core.output_parsers import StrOutputParser
# 설정 파일 로드
def load_config():
    with open('server_conf.json', 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()

# OpenAI API 키 설정
os.environ['OPENAI_API_KEY'] = CONFIG['apis']['openai']['api_key']

# Naver Search API
naver_search_id = CONFIG['apis']['naver_search']['client_id']
naver_search_pw = CONFIG['apis']['naver_search']['client_secret']

# CLOVA Speech 설정
CLIENT_SECRET = CONFIG['clova']['client_secret']
INVOKE_URL = CONFIG['clova']['invoke_url']

# 출력 디렉토리 설정
OUTPUT_DIR = CONFIG['directories']['recordings']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Summary와 QA 메모리 초기화
summary_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
qa_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history") 

# 각각의 프롬프트 지정

summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        당신은 회의나 발표의 주요 내용을 실시간으로 요약하는 AI 어시스턴트입니다. 
        목표는 중요한 세부 사항을 놓치지 않으면서 간결하고 유익한 요약을 제공하는 것입니다. 아래는 요약을 작성할 때 따를 수 있는 템플릿입니다. 모든 내용을 처리할 때 이 구조를 따르세요.
        
        실시간 문장에 다음과 같은 양식으로 사용자에게 정보를 제공합니다. 만약 회의내용이 아니면 ""로 빈텍스트를 출력해야만 합니다.
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
        쓸데없는 얘기가 프로젝트 주제가 되면 안되고, 직접적으로 명시한 내용 및 문맥을 파악해서 위의 양식으로 출력합니다.회의와 관련없는 **무조건 무응답해야합니다**.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        당신은 사용자의 요청에 따라 회의 및 질문에 대해 질의응답을 하는 다재다능한 에이전트입니다.
        메모리에 진행중인 회의 내용이 담겨져 있습니다. 
        이를 참고해 사용자에게 답변해주세요 결코 메모리에 담긴 양식대로 답변하는 것이 아니라, 사용자의 질문에 맞춰 정보를 검색하거나 회의의 안건을 결정해주거나, 
        더 좋은 아이디어를 추천해주는 답변을 해야합니다.
        
        다음은 예시입니다.
        example
        요청: "우리 회의 중에 어떤 아이디어를 택하는게 가장 좋을까?"
        답변: "회의를 분석해본 결과 예시 안건이 가장 좋아보입니다. ~~ 주장에 따른 근거들은 다음과 같고, 충분히 가치있어보입니다. 또한 제가 생각하는 이유로는... "
        
        저렇게 사용자에 요청에 맞춰서 성심성의껏 답변해줘야 합니다.
        """),   
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 
confernce_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        당신은 회의 내용을 분석하는 비즈니스 전문가 에이전트입니다.
        주어진 회의내용을 바탕으로 사용자의 회의에 관련된 질문을 전문가의 시점으로 답변해야합니다.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

controll_prompt=ChatPromptTemplate.from_messages(
   [ 
    ("system",
     """"
     당신은 실시간으로 진행되는 회의,프레젠테이션에서 다른 에이전트들을 관리하는 Controll Agent입니다.
     문장의 문맥을 파악해서 다른 에이전트를 작동시킬지 정확히 파악해야합니다.

     당신이 관리해야할 에이전트는 총 3가지 에이전트입니다.

     **summary 에이전트**: 회의, 프레젠테이션에서 사용자의 발표 내용을 실시간으로 수집하고, 체계적으로 정리하는 기록 에이전트입니다. 사용자의 프레젠테이션, 회의 내용은 summary 에이전트를 활성화 시켜야합니다.
     진행중이던 회의에 대한 재요약이나 질문은 qa에이전트에서 담당합니다.
     
     **qa 에이전트**: GPT 내부에 있는 데이터를 가지고 사용자의 질문에 응답할때 사용하는 에이전트 입니다.
     무조건 검색이라고 seach 에이전트를 호출하는 것이 아닌, gpt가 학습한 데이터로 충분히 답변이 가능하거나 gpt와의 질의응답이 필요하거나, 회의에 대한 요약, 의견 조율이 필요할때 이 에이전트를 활성화 시킵니다.
     수신 받은 문장이 진행중인 회의 내용일 경우 호출시키면 안됩니다. 'gpt야 물어볼게'와 같이 정보를 요구하면서 외부검색이 아닌경우만 실행시킵니다.
    
     **search 에이전트**: gpt내부에 있는 데이터로는 대답할 수 없는 외부 검색이 필요할때 사용하는 에이전트 입니다. 
     사용자의 대화에서  최근 동향이나 이슈에 대한 뉴스,기사에 대한 정보가 필요하거나 , 구매와 관련된 이슈들 (어떤 물건이 부족하다거나, 망가졌을때) 당신은 능동적으로 이 에이전트를 활성화해서 사용자에게 편의를 줘야합니다.
     이 에이전트는 뉴스, 온라인샵, 지역과 관련된 검색을 지원합니다.
    
    ***summary 에이전트를 활성화 시킬땐 summary, qa 에이전트를 활성화 시킬땐 qa, search 에이전트를 활성화 시킬땐 search, 아무런 호출이 필요없을땐 no로 응답합니다.*** 단 하나만 응답합니다.
   
     """),
    ("human","{user_input}")
    ]
    )

search_prompt = prompt = ChatPromptTemplate.from_messages(
     [
        (
        "system",
        """
        당신은 능동적으로 사용자의 대답을 보고 적재적소에 네이버 검색 API를 사용하여 정보를 제공하는 에이전트입니다.
        입력된 문맥에서 검색할 정보를 찾습니다. 예를 들어서 어떤 물건이 부족하다거나, 사용자가 정보나 물건을 찾아보자라 하거나, 망가졌을때 당신은 능동적으로 검색해서 정보를 제공해줄 수 있습니다.
        def Naver_Search(search: str)를 사용하기 위해선  단하나의 'str'로  "query,o  bj"로 입력됩니다
        입력변수 search='query'+','+'obj'입니다. 
        - query: 요청에서 검색에 최적화된 키워드로 변환합니다.  
        - obj: 검색 유형을 지정합니다
          - 'news': 뉴스 검색 (특정 주제의 최신 뉴스나 기사를 찾을 때 사용)
          - 'local': 지역 정보 검색 (맛집, 가게, 장소 등을 찾을 때 사용)
          - 'shop': 상품 검색 (제품 정보나 가격을 찾을 때 사용)
        
        다음은 상품을 검색해야할 때 입니다.
        질문자의 요청: "컴퓨터좀 사려하는데 게임하기에 좀 좋은 컴퓨터 어디 없나?"  
        search="게이밍 컴퓨터,shop"
        
        질문자의 요청: "야 우리 회의용 마이크 망가지지 않았냐??"  
        search= "회의용 마이크,shop"

        검색에 실패했을시, 입력한 값을 정확히 보여주고 받은 값도 출력합니다. 그리고 어떻게 실행시키고 했는지 모든 정보를 사용자에게 제공하세요.
        # 검색한 모든 정보에 아래와 같은 형식으로 답변해야합니다.
        
        # *뉴스 검색에 대한 답변형식

        # 안녕하세요! 삼성 주가 동향에 대한 기사를 요약해드리겠습니다.
        # 3개의 기사를 분석하겠습니다.

        # **신문 제목: 젠슨 황 한마디에...삼성SK 주가 들썩**
        # **분석 내용:** 이 기사는 젠슨 황이 엔비디아의 그래픽 처리장치(GPU) 신제품에 삼성전자의 메모리칩이 들어가지 않는다고 한 발언을 정정하면서 삼성전자와 SKC의 주가 변동에 대해 다루고 있습니다. 젠슨 황이 최근 삼성전자의 메모리칩이 사용될 것이라고 발언한 후, 이 발언을 정정하면서 삼성전자와 SKC의 주가가 상승하거나 하락한 것을 다루고 있습니다.
        # **링크:(http://www.edaily.co.kr/news/newspath.asp?newsid=03962246642036080)

        # 기사들을 분석해보면 단기적으로는 젠슨 황의 발언이 삼성전자의 주가에 부정적인 영향을 미쳤을 수 있으나, 발언이 정정되면서 주가가 회복되었을 가능성이 큽니다.
        # 장기적으로 볼 때, 삼성전자의 반도체 사업은 여전히 강력한 경쟁력을 가지고 있으며, 엔비디아와의 협력은 중요한 요소로 작용할 수 있습니다. 따라서, 이번 사건은 주가에 큰 영향을 미친 사건이라기보다는 일시적인 변동성을 나타내는 사례라고 할 수 있습니다.
        # -----------------------------------
        # *obj=shop 제품 검색 답변형식
        
        # 이러한 query 상품을 추천드립니다

        # 제품명: 
        # 가격:
        # 판매처: 
        # 브랜드: 
        # [상세 정보를 확인하세요]:링크        
        """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)



class SummaryAgent():
    def __init__(self):

        self.model_structure = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
        )

        self.memory = summary_memory
        self.memory2 = qa_memory
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter(self.memory.memory_key)
            )
            | summary_prompt
            | self.model_structure
            | StrOutputParser()
        )
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.sentence = ""
        self.count = 0
        self.is_running = True
        self.thread = threading.Thread(target=self.process_tasks)
        self.thread.daemon = True
        self.thread.start()

        # 로깅을 위한 새로운 초기화
        self.log_dir = CONFIG['directories']['recordings']
        os.makedirs(self.log_dir, exist_ok=True)

        # 실행 시점에 고유한 로그 파일 이름 생성
        self.log_filename = self._generate_log_filename()

    def _generate_log_filename(self):
        """실행 시 고유한 Summary 로그 파일 이름 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 날짜와 시간을 포함
        return os.path.join(self.log_dir, f"{timestamp}_syc_summary.txt")

    def process_tasks(self):
        if not self.task_queue.empty():
            self.sentence += self.task_queue.get()
            self.count += 1
            if self.count >= 1:
                self.summary_process_tasks()

    def summary_process_tasks(self):
        answer = self.chain.invoke({"input": self.sentence})
        print('요약 에이전트 응답입니다.')
        print(answer)

        # 응답이 비어있지 않은 경우에만 로그 파일에 저장
        if answer.strip() and (answer!= ""):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.log_filename, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {answer}\n")

        self.memory.save_context(inputs={"human": self.sentence}, outputs={"ai": answer})
        try:
            self.memory2.save_context(inputs={"human": self.sentence}, outputs={"ai": answer})
        except:
            pass

        self.sentence = ""  # 리셋
        self.count = 0

class QAAgent():
    def __init__(self):

        self.model_structure = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
        )

        self.memory = qa_memory
        self.memory_chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter(self.memory.memory_key)  # memory_key와 동일하게 입력합니다.
            )
            | qa_prompt
            | self.model_structure
            | StrOutputParser()
        )  # 회의 내용에 대해 물어볼 때 이걸 사용

        self.task_queue = Queue()
        self.conference_task_queue = Queue()  # 회의 관련된 내용이면 여기다가 담자
        self.result_queue = Queue()
        self.sentence = ""
        self.conference_sentence = ""  # 회의 QA를 위한 객체
        self.count = 0
        self.is_running = True
        self.thread = threading.Thread(target=self.process_tasks)
        self.thread.daemon = True
        self.thread.start()

        # 로깅을 위한 새로운 초기화
        self.log_dir = CONFIG['directories']['recordings']
        os.makedirs(self.log_dir, exist_ok=True)

        # 실행 시점에 고유한 로그 파일 이름 생성
        self.log_filename = self._generate_log_filename()

    def _generate_log_filename(self):
        """실행 시 고유한 QA 로그 파일 이름 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 날짜와 시간을 포함
        return os.path.join(self.log_dir, f"{timestamp}_syc_qa.txt")

    def process_tasks(self):
        if not self.task_queue.empty():
            self.sentence += self.task_queue.get()
            self.count += 1
            if self.count >= 1:
                self.qa_process_tasks()

    def qa_process_tasks(self):
        answer = self.memory_chain.invoke({"input": self.sentence})
        print('QA 에이전트 응답입니다.')
        print(answer)

        # 응답이 비어있지 않은 경우에만 로그 파일에 저장
        if answer.strip():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 타임스탬프 추가
            with open(self.log_filename, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {answer}\n")

        # 메모리에 저장
        self.memory.save_context(inputs={"human": self.sentence}, outputs={"ai": answer})
        self.sentence = ""  # 리셋
        self.count = 0

class SearchAgent():
    def __init__(self):
        self.start_time = datetime.now()  # 시작 시간 저장
        self.log_dir = CONFIG['directories']['recordings']
        os.makedirs(self.log_dir, exist_ok=True)

        # 실행 시 고유한 로그 파일 이름 생성
        self.log_filename = self._generate_log_filename()

        self.model_structure = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0,
        )
        self.prompt = search_prompt
        self.task_que = Queue()
        self.result_que = Queue()

    def _generate_log_filename(self):
        """실행 시 고유한 Search 로그 파일 이름 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 날짜와 시간을 포함
        return os.path.join(self.log_dir, f"{timestamp}_syc_search.txt")

    @tool
    def Naver_Search(search: str) -> str:
        """
        Naver 검색 기능을 수행하는 함수입니다.
        주어진 검색어를 사용하여 네이버에서 관련된 결과를 반환합니다.

        Args:
            search (str): "query,obj" 형태로 받아옵니다.

        Returns:
            str: 검색 결과로 반환된 텍스트 (예: 뉴스, 블로그 등)
        """
        query, obj = search.split(",")

        query=query.strip()
        obj=obj.strip()

        try:
            sort = 'sim'
            encText = requests.utils.quote(query)  # 검색어를 URL에 맞게 인코딩

            if obj == 'news':
                display = 3
            else:
                display = 5

            url = f"https://openapi.naver.com/v1/search/{obj}?query={encText}&display={display}&sort={sort}"

            if obj == 'shop':
                url += '&exclude=used:cbshop'  # 특정 상품을 제외하기 위한 URL

            headers = {
            "X-Naver-Client-Id": CONFIG['apis']['naver_search']['client_id'],
            "X-Naver-Client-Secret": CONFIG['apis']['naver_search']['client_secret']
                }

            response = requests.get(url, headers=headers)  # GET 요청
            if response.status_code == 200:
                info = response.json()
            else:
                return f"Error Code: {response.status_code} - 검색 결과를 찾지 못했습니다."

            # 뉴스 검색 결과 처리
            def naver_crawl_news(a):
                news_text = ""
                if a:
                    for item in a['items']:
                        title = html.unescape(item['title'])
                        title = re.sub(r"<.*?>", "", title)
                        description = html.unescape(item['description'])
                        description = re.sub(r"<.*?>", "", description)
                        news_text += f"제목: {title}\n요약: {description}\n\n"
                return news_text

            # # 뉴스 검색 결과 처리
            # def naver_crawl_news(a):
            #     cnt = 0  # 3개의 신문 정보가 쌓이면 멈춤
            #     news_text = ""
            #     if a:
            #         for item in a['items']:  # JSON 데이터의 'items' 키를 순회
            #             if cnt<4:
            #                 response = requests.get(item['originallink'])  # GET 요청
            #                 if response.status_code == 200:
            #                     soup = BeautifulSoup(response.text, 'html.parser')  # HTML 파싱
            #                     # 제목 전처리
            #                     title = html.unescape(item['title'])
            #                     title = re.sub(r"[\'\"\,]+|<b>|</b>|", "", title)
            #                     title = clean_unicode_characters(title)
            #                     # 요약 전처리
            #                     description = html.unescape(item['description'])
            #                     description = re.sub(r"[\'\"\,]+|<b>|</b>|", "", description)
            #                     description = clean_unicode_characters(description)
            #                     # 본문 전처리 
            #                     context = re.sub(r"[\n\r\'\"\,]+|<b>|</b>|\s{2,}", "", soup.text)
            #                     context = clean_unicode_characters(context)
            #                     link=item['originallink']
            #                     matches = list(re.finditer(re.escape(title), context, re.IGNORECASE))

            #                     if matches:
            #                         last_match = matches[-1]  # 마지막 매칭
            #                         start_index = last_match.start()  # 시작 인덱스
            #                         end_index = last_match.end()      # 끝 인덱스
            #                         extract_context = context[end_index:]

            #                         news_text += f'신문 제목: {title}\n'
            #                         news_text += f'요약문: {description}\n'
            #                         news_text += f'본문 일부: {extract_context[150:450]}\n'
            #                         news_text += f'신문기사 링크: {link}'
            #                     cnt += 1
            #     return news_text

            # 상품 검색 처리
            def naver_product_info(a):
                product_info=""
                if a:
                    for item in a['items']:  # JSON 데이터의 'items' 키를 순회
                        product_info+=f"""
                            {item['image']}
                            제품명: {re.sub(r"</b>|<b>","",html.unescape(item['title']))}
                            가격: {item['lprice']}원
                            판매처: {item['mallName']}
                            브랜드: {item['brand']}
                            [상세 정보를 확인하세요] {item['link']}
                            """
                return product_info
            # 맛집 검색 처리
            def naver_food_info(a):
                food_info=""
                if a:
                    for item in a['items']:  # JSON 데이터의 'items' 키를 순회
                        food_info+=f"""
                            가게명: {re.sub(r"</b>|<b>","",html.unescape(item['title']))}
                            음식 카테고리: {item['category']}
                            위치: {item['address']}
                            """
                return food_info
            
            if obj == 'news':
                return naver_crawl_news(info)
            elif obj == 'shop':
                return naver_product_info(info)
            elif obj == 'local':
                return naver_food_info(info)
        except:
            return None

        # except Exception as e:
        #     return f"검색 중 오류가 발생했습니다: {e}"

    def process_tasks(self):
        if not self.task_que.empty():
            query = self.task_que.get()
            tools = [self.Naver_Search]
            agent = create_tool_calling_agent(self.model_structure, tools, self.prompt)

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=10,
                max_execution_time=10,
                handle_parsing_errors=True,
            )

            result = agent_executor.invoke({"input": query})

            # 결과 출력 및 텍스트 파일 저장
            output = result['output'].strip()  # 응답 텍스트
            print(output)

            # 응답이 비어있지 않은 경우에만 로그 파일에 저장
            if output:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(self.log_filename, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {output}\n")

            self.result_que.put(output)


class MultiAgentProcessor: # 컨트롤 에이전트 총괄
    def __init__(self):

        self.model_structure=ChatOpenAI(
            model='gpt-4',
            temperature=0,
        )
        self.summary_agent = SummaryAgent() # 교정이랑, 요약 문장단위로 송출. 3문장 정도로 생각하자.
        self.qa_agent = QAAgent() # 내부 검색색 및 질의 응답
        self.search_agent = SearchAgent() #

        self.chain= RunnablePassthrough()|controll_prompt|self.model_structure
        self.sentence_que = Queue()
        self.is_running = True  # 활성화

        self.processing_thread = threading.Thread(target=self._process_text_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        
    def process_text(self, text): # 스트리밍으로 받아온 텍스트를 text_queue에 put
        """새로운 텍스트 처리 요청"""
        self.sentence_que.put(text)


    def cleanup(self):
        """리소스 정리"""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)
    
    def _process_text_queue(self): #텍스트가 쌓이면 실행
        """텍스트 처리 큐 프로세스"""
        while self.is_running:
            try:
                if not self.sentence_que.empty():
                    sentence_to_process = self.sentence_que.get() # 문장을 가져온다
                    self.controll_task(sentence_to_process) #컨트롤 에이전트 실행.
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in text processing: {str(e)}")

    def controll_task(self,sentence_to_process):
        """컨트롤 에이전트 업무 실행"""
        try:
            # print(f"[DEBUG] 컨트롤 task 내부에 진입합니다")
            answer=self.chain.invoke(sentence_to_process)
            print(f"컨트롤 에이전트의 응답입니다\n{sentence_to_process}:{answer.content}")
            # print(f"controll agent 응답 결과: {answer.content}")
            if answer.content == "summary":
                self.summary_agent.task_queue.put(sentence_to_process)
                self.summary_agent.process_tasks()
            elif answer.content == "search":
                self.search_agent.task_que.put(sentence_to_process)
                self.search_agent.process_tasks()
            elif answer.content == "qa":
                self.qa_agent.task_queue.put(sentence_to_process)
                self.qa_agent.process_tasks()
        except Exception as e:
            print(f"Error in text processing: {str(e)}")


# ... (이전 코드)
class RecordingsHandler(FileSystemEventHandler):
    """파일 시스템 이벤트 핸들러"""

    def __init__(self, processor: MultiAgentProcessor):
        self.processor = processor
        self.files_position = {}
        self.length=0
        self.current_line = ""
        
    def is_complete_sentence(self, text: str) -> bool:
        """문장이 완성되었는지 확인하는 함수"""
        # 마침표, 느낌표, 물음표로 끝나는지 확인
        endings = ['.', '!', '?', '...']
        return any(text.strip().endswith(end) for end in endings)
    
    def on_modified(self, event):
        if event.is_directory:
            return
            
        filename = os.path.basename(event.src_path)
        
        # 임시 파일 또는 숨김 파일 무시
        if not filename.endswith("_orig2.txt") or filename.startswith('.~'):
            return
            
        try:
            with open(event.src_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                current_length = len(all_lines)
                
                # 새로운 내용이 없으면 종료
                if current_length-1 <= self.length:
                    return
                    
                # 새로운 라인 처리
                if current_length >= 2 and current_length-1 > self.length:
                    # 새로운 텍스트 가져오기
                    new_text = all_lines[self.length].strip()
                    
                    # 현재 누적 중인 문장에 새 텍스트 추가
                    self.current_line += " " + new_text if self.current_line else new_text
                    self.current_line = self.current_line.strip()
                    
                    # 문장이 완성되었는지 확인
                    if self.is_complete_sentence(self.current_line):
                        print(f'완성된 문장을 컨트롤 에이전트에 전달합니다:\n{self.current_line}')
                        self.processor.process_text(self.current_line)
                        self.current_line = ""  # 문장 버퍼 초기화
                    
                    self.length = current_length-1  # 처리한 위치 업데이트
                    
        except FileNotFoundError:
            print(f"[WARNING] 파일을 열 수 없습니다: {event.src_path}")
        except Exception as e:
            print(f"Error in on_modified: {str(e)}")

# ... (나머지 코드)

class RecordingsWatcher:
    """/recordings 디렉토리를 감시하는 클래스"""

    def __init__(self, directory: str, processor: MultiAgentProcessor):
        self.directory = directory
        self.processor = processor
        self.event_handler = RecordingsHandler(processor)
        self.observer = Observer()

    def start(self):
        self.observer.schedule(self.event_handler, self.directory, recursive=False)
        self.observer.start()
        print(f"Started watching directory: {self.directory}")

    def stop(self):
        self.observer.stop()
        self.observer.join()
        print(f"Stopped watching directory: {self.directory}")

def main():
    # 텍스트 처리기 초기화
    processor = MultiAgentProcessor()

    # /recordings 디렉토리 설정 (상대 경로)
    recordings_dir = CONFIG['directories']['recordings']
    os.makedirs(recordings_dir, exist_ok=True)

    # RecordingsWatcher 초기화 및 시작
    watcher = RecordingsWatcher(recordings_dir, processor)
    watcher.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping watcher...")
        watcher.stop()
        processor.cleanup()
        print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main()