# build_knowledge_base.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import json

# ── 1. 기상 특보 기준 (기상청 공식 기준) ──────────────
weather_docs = [
    Document(
        page_content="""해상 기상 특보 발표 기준 (기상청 공식)
풍랑주의보: 해상 풍속 14m/s 이상이 3시간 이상 지속되거나 유의파고 3m 이상 예상 시
풍랑경보: 해상 풍속 21m/s 이상이 3시간 이상 지속되거나 유의파고 5m 이상 예상 시
태풍주의보: 태풍으로 강풍/풍랑/호우 현상이 주의보 기준 도달 예상 시
태풍경보: 태풍으로 강풍/풍랑 경보 기준 도달 또는 총강우량 200mm 이상 예상 시""",
        metadata={"source": "기상청", "category": "기상특보기준", "alert_level": "YELLOW~RED"}
    ),
    Document(
        page_content="""해상 경보 레벨 판단 기준
GREEN (정상): 풍속 14m/s 미만, 파고 3m 미만, 주간, 일반 선박 활동
YELLOW (경계): 풍속 14~21m/s 또는 파고 3~5m 또는 야간 미확인 선박 또는 군함 탐지
RED (위험): 풍속 21m/s 초과 또는 파고 5m 초과 또는 오물폭탄/드론 탐지 또는 야간 군함""",
        metadata={"source": "해안경계기준", "category": "경보레벨기준", "alert_level": "ALL"}
    ),
    Document(
        page_content="""풍속별 해상 위험도 분류
0~7m/s: 위험도 낮음 - 정상 해상 활동 가능
7~14m/s: 위험도 보통 - 소형 선박 주의 필요
14~21m/s: 위험도 높음 - 풍랑주의보 수준, 조업 자제 권고
21m/s 이상: 위험도 매우 높음 - 풍랑경보 수준, 즉각 대피 필요""",
        metadata={"source": "기상청", "category": "풍속위험도", "alert_level": "GREEN~RED"}
    ),
]

# ── 2. 선박 유형별 대응 절차 ──────────────────────────
vessel_docs = [
    Document(
        page_content="""군함 탐지 시 대응 절차
1. 즉각 상황 보고 (지휘관 보고)
2. 군함 식별 정보 수집 (크기, 이동방향, 속도)
3. 경보 레벨 YELLOW 이상 발령
4. 해양경찰청 통보
5. 지속 감시 및 이동 경로 추적
야간 군함 탐지 시 경보 레벨 RED 즉각 발령""",
        metadata={"source": "해안경계절차", "category": "선박대응", "vessel_type": "군함"}
    ),
    Document(
        page_content="""무인항공기(드론) 탐지 시 대응 절차
1. 즉각 경보 레벨 YELLOW 발령
2. 드론 비행 방향 및 고도 추적
3. 북한 발 드론 여부 식별
4. 관계 부처 즉각 통보
5. 안티드론 장비 가동 검토
드론이 군사 시설 방향 접근 시 RED 발령""",
        metadata={"source": "해안경계절차", "category": "선박대응", "vessel_type": "드론"}
    ),
    Document(
        page_content="""오물폭탄 탐지 시 대응 절차
1. 즉각 경보 레벨 RED 발령
2. 생화학 오염 대응 절차 가동
3. 접근 금지 구역 설정
4. 관계 부처 긴급 통보 (국방부, 환경부)
5. 전문 처리팀 출동 요청
오물폭탄은 탐지 즉시 RED 발령 필수""",
        metadata={"source": "해안경계절차", "category": "위협대응", "vessel_type": "오물폭탄"}
    ),
    Document(
        page_content="""어선 탐지 시 대응 절차
일반 어선: 경보 레벨 GREEN 유지, 정상 모니터링
야간 어선: 경보 레벨 YELLOW 검토, 식별 요청
강풍 시 어선: 어선 안전 확인, 필요 시 대피 안내
미확인 어선: 선박 AIS 확인, 미등록 시 해경 통보""",
        metadata={"source": "해안경계절차", "category": "선박대응", "vessel_type": "어선"}
    ),
    Document(
        page_content="""상선 탐지 시 대응 절차
일반 상선: 경보 레벨 GREEN 유지
항로 이탈 상선: 해양교통관제센터(VTS) 통보
야간 미확인 상선: 경보 레벨 YELLOW, 식별 요청
위험 물질 운반 의심: 해경 즉각 통보""",
        metadata={"source": "해안경계절차", "category": "선박대응", "vessel_type": "상선"}
    ),
]

# ── 3. 야간/기상 조건별 대응 ──────────────────────────
condition_docs = [
    Document(
        page_content="""야간 해안 경계 강화 기준
야간(일몰 후~일출 전) 시 기본 경계 수준 상향
야간 미확인 선박 탐지: 즉각 식별 요청 및 경보 YELLOW
야간 군함 탐지: 즉각 경보 RED 발령
야간 강풍(14m/s 이상): 경보 YELLOW, 조업 선박 안전 확인
야간 시정 불량: 레이더 감시 강화""",
        metadata={"source": "해안경계절차", "category": "야간대응", "alert_level": "YELLOW~RED"}
    ),
    Document(
        page_content="""해무 발생 시 대응 절차
해무 1~2단계: 경계 강화, 레이더 감시 병행
해무 3단계(짙은 해무): 육안 감시 불가, 레이더 전환
해무 시 선박 충돌 위험 증가
VTS(해양교통관제센터) 통보 및 협조 요청
해무 + 야간 조건: 경보 레벨 YELLOW 자동 상향""",
        metadata={"source": "해안경계절차", "category": "기상대응", "weather": "해무"}
    ),
    Document(
        page_content="""폭풍/악천후 시 해안 경계 대응
풍속 14~21m/s (풍랑주의보): 경보 YELLOW, 소형 선박 대피 권고
풍속 21m/s 이상 (풍랑경보): 경보 RED, 전 선박 대피 명령
강수량 30mm/h 이상: 감시 카메라 시정 확인, 보조 장비 가동
태풍 접근 시: 비상 대응 체계 가동, 관계 부처 상황 공유""",
        metadata={"source": "해안경계절차", "category": "기상대응", "weather": "폭풍"}
    ),
]

# ── 4. 유사 상황 사례 ─────────────────────────────────
case_docs = [
    Document(
        page_content="""사례 001: 야간 군함 탐지 → RED 발령
상황: 2024년 8월, 연평도 인근 야간 CCTV에서 군함 1척 탐지
기상: 풍속 8m/s, 맑음, 야간
조치: 즉각 RED 발령 → 지휘관 보고 → 해경 통보 → 추적 감시
결과: 북한 경비정으로 식별, 영해 침범 전 자진 이탈
교훈: 야간 군함 탐지 시 기상 조건 무관 RED 발령 필수""",
        metadata={"source": "사례집", "category": "유사사례", "alert_level": "RED"}
    ),
    Document(
        page_content="""사례 002: 강풍 + 어선 조난 → YELLOW 발령
상황: 2024년 10월, 백령도 인근 풍속 16m/s 강풍 중 어선 1척 표류
기상: 풍속 16m/s, 파고 3.5m, 주간
조치: YELLOW 발령 → 해경 구조 요청 → 어선 안전 유도
결과: 어선 안전 대피 완료
교훈: 풍랑주의보 수준 기상에서 표류 선박 발견 시 즉각 해경 통보""",
        metadata={"source": "사례집", "category": "유사사례", "alert_level": "YELLOW"}
    ),
    Document(
        page_content="""사례 003: 오물폭탄 탐지 → RED 발령
상황: 2024년 6월, 연평도 해안에서 오물폭탄 낙하 탐지
기상: 풍속 5m/s, 맑음, 주간
조치: 즉각 RED 발령 → 접근 금지 → 생화학 대응팀 출동
결과: 비위험 물질 확인 후 수거
교훈: 오물폭탄은 기상 조건 무관 RED 즉각 발령""",
        metadata={"source": "사례집", "category": "유사사례", "alert_level": "RED"}
    ),
    Document(
        page_content="""사례 004: 해무 + 미확인 선박 → YELLOW 발령
상황: 2024년 5월, 백령도 해무 3단계 상황에서 레이더 미확인 선박 탐지
기상: 해무 3단계, 시정 500m 이하, 야간
조치: YELLOW 발령 → 레이더 집중 감시 → 해경 통보
결과: 중국 어선으로 식별, 영해 이탈 유도
교훈: 해무 + 야간 조건에서 미확인 선박은 YELLOW 즉각 발령""",
        metadata={"source": "사례집", "category": "유사사례", "alert_level": "YELLOW"}
    ),
]

# ── ChromaDB 구축 ─────────────────────────────────────
all_docs = weather_docs + vessel_docs + condition_docs + case_docs

# 텍스트 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs = splitter.split_documents(all_docs)
print(f"총 문서: {len(all_docs)}개 → 청크: {len(split_docs)}개")

# 임베딩 모델 (한국어 지원)
print("임베딩 모델 로딩 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    model_kwargs={"device": "cuda"}
)

# ChromaDB 저장
print("ChromaDB 구축 중...")
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="/home/hail/pan/VLM-project/chroma_db"
)
vectordb.persist()
print(f"ChromaDB 저장 완료! 총 {vectordb._collection.count()}개 벡터")

# ── 검색 테스트 ───────────────────────────────────────
print("\n=== 검색 테스트 ===")
test_queries = [
    "야간에 군함 탐지됐을 때 어떻게 해야 하나요?",
    "풍속 20m/s일 때 경보 레벨은?",
    "오물폭탄 탐지 시 대응 절차",
    "해무 상황에서 미확인 선박 탐지",
]

for query in test_queries:
    print(f"\n질문: {query}")
    results = vectordb.similarity_search(query, k=2)
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content[:100]}...")