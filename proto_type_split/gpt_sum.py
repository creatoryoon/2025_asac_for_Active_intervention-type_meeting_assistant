import time
import os
import json
from openai import OpenAI
from pathlib import Path
from typing import Dict, List

# 설정 파일 로드
def load_config():
    with open('server_conf.json', 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()

###############################################################################
# 0. OpenAI API 설정
###############################################################################
client = OpenAI(api_key=CONFIG['apis']['openai']['api_key'])


###############################################################################
# 1. GPT 교정 & 요약 함수
###############################################################################
def gpt_correct_text(text: str, partial_context: str = "") -> str:
    """
    (필요하다면) partial_context로 최근 부분 인식 맥락도 전달할 수 있음.
    """
    system_prompt = (
        "당신은 스크립트(회의록) 교정 전문가입니다.\n"
        "주어진 텍스트를 맞춤법과 표현이 자연스럽도록 교정해 주세요.\n\n"
        "아래는 부분 인식 정보(오타가 많을 수 있으니 맥락 참고용)입니다:\n"
        f"{partial_context}\n"
        "---------------------------------\n"
    )
    user_prompt = text

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=2048
    )
    corrected_text = response.choices[0].message.content.strip()
    return corrected_text

def gpt_summarize_text(text: str) -> str:
    """
    교정된 텍스트를 요약해준다.
    """
    system_prompt = (
        "당신은 전문 요약가입니다.\n"
        "주어진 텍스트의 핵심을 간단하고 명확하게 요약해 주세요. "
        "중요한 인물, 이슈, 결론이 빠지지 않도록 해주세요.\n"
    )
    user_prompt = text

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    summary_text = response.choices[0].message.content.strip()
    return summary_text

###############################################################################
# 2. 파일 모니터링 + 데이터 관리
###############################################################################
class ASRDataStore:
    """
    - partial_lines: orig.txt에서 계속 누적되는 부분 인식 결과
    - long_lines: long.txt에서 누적되는 장문 인식 결과(확정)
    """
    def __init__(self):
        self.partial_lines: List[str] = []
        self.long_lines: List[str] = []

asr_data_map: Dict[str, ASRDataStore] = {}
file_offsets: Dict[str, int] = {}

def get_file_key(fname: str) -> str:
    """
    예: '20250114_111418_orig.txt' -> '20250114_111418'
        '20250114_111418_long.txt' -> '20250114_111418'
    """
    stem = Path(fname).stem  # 예: '20250114_111418_orig'
    if "_orig" in stem:
        return stem.replace("_orig", "")
    elif "_long" in stem:
        return stem.replace("_long", "")
    return stem  # fallback

import time
from pathlib import Path
from typing import List, Tuple

def read_new_lines(
    file_path: Path,
    offset: int,
    max_retry: int = 5,
    retry_interval: float = 0.05
) -> Tuple[List[str], int]:
    """
    파일을 offset부터 이진 모드로 읽은 뒤,
    1) UTF-8로 디코딩(errors='replace')
       - 실제로 깨진 부분은 �로 대체
    2) (옵션) 그래도 문제가 있으면 CP949(errors='replace')도 시도
    - 둘 다 실패하면 재시도 (일시적 파일 깨짐 대비)
    - max_retry: 최대 재시도 횟수
    - retry_interval: 재시도 사이 대기 시간(초)
    """
    if not file_path.exists():
        return [], offset

    for attempt in range(1, max_retry + 1):
        # offset부터 끝까지의 바이트 읽기
        data = file_path.read_bytes()[offset:]

        # 먼저 UTF-8(errors='replace') 시도
        try:
            text = data.decode('utf-8', errors='replace')
            # UTF-8로 성공(설령 일부 문자가 �로 대체됐어도),
            # 여기서는 굳이 UnicodeDecodeError가 나지 않으므로 바로 return
            lines = text.splitlines(True)
            new_offset = offset + len(data)
            return lines, new_offset

        except UnicodeDecodeError:
            # UTF-8(errors='replace')마저 실패할 일이 많진 않지만,
            # 혹시 대비해 CP949(errors='replace')도 시도
            try:
                text = data.decode('cp949', errors='replace')
                lines = text.splitlines(True)
                new_offset = offset + len(data)
                return lines, new_offset
            except UnicodeDecodeError as e:
                print(f"[WARN] 인코딩 에러(UTF-8/CP949) 발생: {e} | "
                      f"{attempt}/{max_retry} 재시도 중...")
                if attempt < max_retry:
                    time.sleep(retry_interval)
                else:
                    print(f"[ERROR] 파일 '{file_path.name}' 디코딩 실패: "
                          f"최대 재시도({max_retry}) 초과.")
                    return [], offset

    # 이론상 여기에 도달하지 않지만, 안전상
    return [], offset


    
###############################################################################
# 3. 폴더를 돌며 orig/long 파일 갱신 감지 → ASRDataStore 업데이트
###############################################################################
def poll_asr_folder(folder_path: str, poll_interval: float = 0.05):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"폴더가 아님: {folder_path}")

    while True:
        # 1) 폴더 내 *_orig.txt, *_long.txt 파일 목록 가져오기
        orig_files = sorted(folder.glob("*_orig.txt"))
        long_files = sorted(folder.glob("*_long.txt"))

        # 2) 각 파일에 대해 새로운 라인 읽기
        for fpath in (orig_files + long_files):
            fname = fpath.name
            file_key = get_file_key(fname)

            if fname not in file_offsets:
                file_offsets[fname] = 0

            new_lines, new_offset = read_new_lines(fpath, file_offsets[fname])
            if new_lines:
                # 업데이트
                file_offsets[fname] = new_offset
                if file_key not in asr_data_map:
                    asr_data_map[file_key] = ASRDataStore()

                # 부분 인식 / 장문 인식 구분해서 저장
                if fname.endswith("_orig.txt"):
                    asr_data_map[file_key].partial_lines.extend(
                        [l.strip() for l in new_lines if l.strip()]
                    )
                else:  # _long.txt
                    asr_data_map[file_key].long_lines.extend(
                        [l.strip() for l in new_lines if l.strip()]
                    )

                # 장문 인식이 새로 들어오면 → GPT 교정 & 결과 파일 생성
                if fname.endswith("_long.txt"):
                    do_correction_and_save(file_key, folder)

        time.sleep(poll_interval)

###############################################################################
# 4. GPT 교정 & 결과 저장
###############################################################################
def do_correction_and_save(file_key: str, folder: Path):
    """
    file_key에 해당하는 partial_lines와 long_lines를 참고하여
    최종 교정본과 요약본을 생성하고 저장한다.
    """
    data_store = asr_data_map[file_key]

    # (1) 부분 인식 결과 모두 합친 것 (참고용)
    partial_text = "\n".join(data_store.partial_lines)

    # (2) 장문 인식(확정) 결과가 현재까지 누적된 것
    long_text = "\n".join(data_store.long_lines)

    # GPT 교정 호출
    corrected_text = gpt_correct_text(long_text, partial_context=partial_text)
    summary_text = gpt_summarize_text(corrected_text)

    # 저장 경로: (file_key)_gpt.txt, (file_key)_summary.txt
    corrected_file_path = folder / f"{file_key}_gpt.txt"
    summary_file_path = folder / f"{file_key}_summary.txt"

    with corrected_file_path.open("w", encoding="utf-8") as f_c:
        f_c.write(corrected_text)

    with summary_file_path.open("w", encoding="utf-8") as f_s:
        f_s.write(summary_text)

    print(f"[INFO] 교정본, 요약본 생성 완료: {corrected_file_path.name}, {summary_file_path.name}")

###############################################################################
# 5. 메인 실행
###############################################################################
if __name__ == "__main__":
    target_folder = CONFIG['directories']['recordings']  # 설정 파일에서 가져온 recordings 경로
    print(f"[INFO] 폴더 모니터링 시작: {target_folder}")
    
    try:
        poll_asr_folder(target_folder, poll_interval=0.5)
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {str(e)}")
