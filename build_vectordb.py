# build_vectordb.py
import json
import torch
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 파인튜닝 데이터 → 벡터DB용 텍스트 변환
with open('/home/hail/pan/VLM-project/finetune_data_merged.json') as f:
    data = json.load(f)

print(f"총 {len(data)}개 데이터 변환 중...")

docs = []
for d in data:
    # output에서 핵심 정보 추출
    output   = d['output']
    input_   = d['input']

    # 텍스트로 변환
    text = f"{input_}\n{output}"

    # 메타데이터 추출
    alert_level = "GREEN"
    for level in ["RED", "YELLOW", "GREEN"]:
        if level in output:
            alert_level = level
            break

    object_type = "미상"
    for obj in ["군함", "어선", "상선", "드론", "오물폭탄", "삐라", "항공기"]:
        if obj in output:
            object_type = obj
            break

    time_str = "야간" if "야간" in input_ else "주간"
    location = "연평도" if "연평도" in input_ else "백령도" if "백령도" in input_ else "미상"

    docs.append(Document(
        page_content=text,
        metadata={
            "image":       d['image'],
            "alert_level": alert_level,
            "object":      object_type,
            "time":        time_str,
            "location":    location,
            "source":      d.get('source', 'real')
        }
    ))

print(f"총 {len(docs)}개 문서 생성 완료")

# 임베딩 모델
print("임베딩 모델 로딩 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    model_kwargs={"device": "cuda"}
)

# ChromaDB 저장 (배치로 나눠서)
print("ChromaDB 구축 중...")
BATCH_SIZE  = 5000
chroma_path = "/home/hail/pan/VLM-project/chroma_db"

vectordb = None
for i in range(0, len(docs), BATCH_SIZE):
    batch = docs[i:i+BATCH_SIZE]
    print(f"배치 {i//BATCH_SIZE + 1}/{len(docs)//BATCH_SIZE + 1} 처리 중... ({len(batch)}개)")

    if vectordb is None:
        vectordb = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory=chroma_path
        )
    else:
        vectordb.add_documents(batch)

vectordb.persist()
print(f"\nChromaDB 저장 완료! 총 {vectordb._collection.count()}개 벡터")

# 검색 테스트
print("\n=== 검색 테스트 ===")
test_queries = [
    "야간에 군함 탐지됐을 때",
    "풍속 강하고 어선 발견",
    "오물폭탄 탐지 상황",
    "해무 상황에서 미확인 선박",
]

for query in test_queries:
    print(f"\n질문: {query}")
    results = vectordb.similarity_search(
        query, k=3,
    )
    for i, doc in enumerate(results):
        alert = doc.metadata['alert_level']
        obj   = doc.metadata['object']
        time  = doc.metadata['time']
        print(f"  [{i+1}] {alert} / {obj} / {time}")
        print(f"       {doc.page_content[100:200].strip()}...")