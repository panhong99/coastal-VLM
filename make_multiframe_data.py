"""
make_multiframe_data.py

finetune_data_merged.json의 단일 이미지 샘플을
8프레임 멀티프레임 시퀀스로 변환합니다.

실제 데이터(I1, I2):
  - 시퀀스당 평균 49프레임이 이미지 폴더에 존재
  - 어노테이션된 프레임을 기준으로 앞 7장 + 현재 = 8장

합성 데이터(EO_SU_DT 등):
  - 시퀀스당 1~4프레임만 존재
  - 반복(cyclic)으로 8장 채움
"""

import json
from pathlib import Path
from collections import defaultdict

# ── 경로 설정 ──────────────────────────────────────────────
DATA_PATH   = "/home/hail/pan/VLM-project/finetune_data_merged.json"
OUTPUT_PATH = "/home/hail/pan/VLM-project/finetune_data_multiframe.json"
NUM_FRAMES  = 8

REAL_IMG_DIRS = [
    "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I1_images",
    "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I2_images",
]
SYN_IMG_DIRS = [
    "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_SU_DT",
    "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_SU_NT",
    "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_WI_DT",
    "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/IR_SU_NT",
]


# ── 파일명 파싱 ─────────────────────────────────────────────
def parse_filename(filename: str):
    """
    Returns (seq_key, frame_no, source)
    - 실제: I1_S0_C5_0183009 -> ('I1_S0_C5_0183', 9, 'real')
    - 합성: EO_WI_DT_W4_H2_B2D2_0007 -> ('EO_WI_DT_W4_H2_B2D2', 7, 'synthetic')
    """
    name = filename.replace(".jpg", "")
    if name.startswith("I1") or name.startswith("I2"):
        parts = name.rsplit("_", 1)
        seq_id = parts[0] + "_" + parts[1][:4]   # 시퀀스 번호(앞4자리)
        frame  = int(parts[1][4:])                 # 프레임 번호(뒤3자리)
        return seq_id, frame, "real"
    else:
        parts = name.rsplit("_", 1)
        return parts[0], int(parts[1]), "synthetic"


# ── 이미지 디렉토리 → 시퀀스 맵 ────────────────────────────
def build_seq_map(dirs):
    """
    Returns:
        seq_map: dict  seq_key -> sorted list of (frame_no, abs_path)
        img_map: dict  filename -> abs_path
    """
    seq_map = defaultdict(list)
    img_map = {}

    for d in dirs:
        p = Path(d)
        if not p.exists():
            print(f"  [경고] 경로 없음: {d}")
            continue
        for f in p.glob("*.jpg"):
            try:
                seq_key, frame_no, _ = parse_filename(f.name)
                seq_map[seq_key].append((frame_no, str(f)))
                img_map[f.name] = str(f)
            except Exception:
                pass

    for k in seq_map:
        seq_map[k].sort(key=lambda x: x[0])

    return seq_map, img_map


# ── 8프레임 선택 ────────────────────────────────────────────
def select_frames(frames, anchor_frame, source):
    """
    frames: sorted list of (frame_no, path)
    anchor_frame: frame number of annotated image
    source: 'real' or 'synthetic'
    """
    if source == "real":
        frame_nos = [f[0] for f in frames]
        try:
            anchor_idx = frame_nos.index(anchor_frame)
        except ValueError:
            # anchor가 없으면 마지막 프레임을 앵커로
            anchor_idx = len(frames) - 1

        start    = max(0, anchor_idx - NUM_FRAMES + 1)
        selected = frames[start : anchor_idx + 1]

        # 앞이 부족하면 첫 프레임으로 패딩
        while len(selected) < NUM_FRAMES:
            selected = [selected[0]] + selected

        return [p for _, p in selected[-NUM_FRAMES:]]

    else:  # synthetic: cyclic repeat
        cycle_paths = []
        while len(cycle_paths) < NUM_FRAMES:
            cycle_paths.extend(p for _, p in frames)
        return [p for p in cycle_paths[:NUM_FRAMES]]


# ── 메인 ────────────────────────────────────────────────────
def main():
    print("=== 멀티프레임 데이터 생성 ===")

    print("\n[1/3] 이미지 시퀀스 맵 구성 중...")
    seq_map, img_map = build_seq_map(REAL_IMG_DIRS + SYN_IMG_DIRS)
    print(f"  시퀀스 수: {len(seq_map)}  |  이미지 수: {len(img_map)}")

    print("\n[2/3] 단일프레임 데이터 로드 중...")
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  원본 샘플 수: {len(data)}")

    print("\n[3/3] 멀티프레임 시퀀스 생성 중...")
    results  = []
    skipped  = 0
    padded   = 0

    for item in data:
        try:
            seq_key, anchor_frame, source = parse_filename(item["image"])
        except Exception:
            skipped += 1
            continue

        if seq_key not in seq_map:
            skipped += 1
            continue

        frames = seq_map[seq_key]

        # 프레임이 1장뿐인 합성 데이터는 반복으로 채움
        if len(frames) < NUM_FRAMES:
            padded += 1

        selected_paths = select_frames(frames, anchor_frame, source)
        image_names    = [Path(p).name for p in selected_paths]

        results.append({
            "images": image_names,       # 8장 파일명 리스트
            "input":  item["input"],
            "output": item["output"],
            "source": item.get("source", source),
        })

    print(f"\n결과:")
    print(f"  생성된 멀티프레임 샘플: {len(results):,}개")
    print(f"  스킵(매핑 실패):        {skipped:,}개")
    print(f"  패딩(프레임 부족):      {padded:,}개")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUTPUT_PATH}")

    # 샘플 확인
    print("\n=== 샘플 확인 ===")
    s = results[0]
    print(f"  images ({len(s['images'])}장): {s['images']}")
    print(f"  input:  {s['input'][:80]}...")
    print(f"  output: {s['output'][:80]}...")
    print(f"  source: {s['source']}")


if __name__ == "__main__":
    main()
