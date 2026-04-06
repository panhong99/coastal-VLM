# verify_vectordb.py
import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from collections import Counter

CHROMA_PATH = "/home/hail/pan/VLM-project/chroma_db"

# 임베딩 모델 로드
print("임베딩 모델 로딩 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    model_kwargs={"device": "cuda"}
)

# ChromaDB 로드
print("ChromaDB 로딩 중...")
vectordb = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

# ── 1. 기본 통계 ──────────────────────────────────────
total = vectordb._collection.count()
print(f"\n=== 기본 통계 ===")
print(f"총 벡터 수: {total}개")

# 메타데이터 분포 확인
print("메타데이터 수집 중...")
BATCH_SIZE = 1000
alert_levels, objects, times, sources = [], [], [], []

for i in range(0, total, BATCH_SIZE):
    batch = vectordb._collection.get(
        limit=BATCH_SIZE,
        offset=i
    )
    for m in batch['metadatas']:
        alert_levels.append(m.get('alert_level', '미상'))
        objects.append(m.get('object', '미상'))
        times.append(m.get('time', '미상'))
        sources.append(m.get('source', '미상'))

    if i % 10000 == 0:
        print(f"  {i}/{total} 처리 중...")

print(f"\n경보 레벨 분포:")
for k, v in Counter(alert_levels).items():
    print(f"  {k}: {v}개 ({v/total*100:.1f}%)")

print(f"\n탐지 객체 분포:")
for k, v in Counter(objects).most_common():
    print(f"  {k}: {v}개 ({v/total*100:.1f}%)")

print(f"\n주야간 분포:")
for k, v in Counter(times).items():
    print(f"  {k}: {v}개 ({v/total*100:.1f}%)")

print(f"\n데이터 소스 분포:")
for k, v in Counter(sources).items():
    print(f"  {k}: {v}개 ({v/total*100:.1f}%)")

# ── 2. 유사도 검색 테스트 ─────────────────────────────
print(f"\n=== 유사도 검색 테스트 ===")

test_queries = [
    {
        "query":    "야간에 군함 탐지됐을 때 대응 방법",
        "expected": "YELLOW or RED"
    },
    {
        "query":    "풍속 강하고 어선 발견 주간",
        "expected": "YELLOW"
    },
    {
        "query":    "오물폭탄 탐지 긴급 상황",
        "expected": "RED"
    },
    {
        "query":    "맑은 날 주간 어선 정상 운항",
        "expected": "GREEN"
    },
]

for test in test_queries:
    print(f"\n질문: {test['query']}")
    print(f"예상 레벨: {test['expected']}")

    results = vectordb.similarity_search_with_relevance_scores(
        test['query'], k=3
    )
    for i, (doc, score) in enumerate(results):
        print(f"  [{i+1}] 유사도: {1-score:.3f} | "
              f"경보: {doc.metadata['alert_level']} | "
              f"객체: {doc.metadata['object']} | "
              f"시간: {doc.metadata['time']}")

# ── 3. 메타데이터 필터링 테스트 ───────────────────────
print(f"\n=== 메타데이터 필터링 테스트 ===")

# 야간 + 군함만 검색
filtered = vectordb.similarity_search(
    "군함 탐지",
    k=3,
    filter={"$and": [
        {"time":        {"$eq": "야간"}},
        {"object":      {"$eq": "군함"}},
        {"alert_level": {"$eq": "YELLOW"}}
    ]}
)
print(f"\n[야간 + 군함 + YELLOW 필터] 검색 결과: {len(filtered)}개")
for doc in filtered:
    print(f"  {doc.metadata}")

# RED 케이스만 검색
red_filtered = vectordb.similarity_search(
    "위험 상황 긴급 대응",
    k=5,
    filter={"alert_level": {"$eq": "RED"}}
)
print(f"\n[RED 케이스 필터] 검색 결과: {len(red_filtered)}개")
for doc in red_filtered:
    print(f"  객체: {doc.metadata['object']} | 시간: {doc.metadata['time']}")

print(f"\n✅ 벡터DB 검증 완료!")