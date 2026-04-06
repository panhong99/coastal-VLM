"""
make_demo_video.py
------------------
두 가지 모드로 데모용 짧은 영상을 만드는 스크립트

Mode 1 (sequential) : 동일 씬의 연속 프레임 이미지들 → mp4
Mode 2 (ken_burns)  : 이미지 1장 → 팬/줌 효과로 CCTV 영상처럼 보이는 mp4

Usage:
    # 씬 0001 (주간) 영상 생성
    python make_demo_video.py --mode sequential --scene 0001 --type I1

    # 씬 0002 (야간) 영상 생성
    python make_demo_video.py --mode sequential --scene 0002 --type I1

    # 이미지 1장으로 영상 생성
    python make_demo_video.py --mode ken_burns --image path/to/image.jpg

    # 여러 씬을 한번에 (demo 폴더에 모두 저장)
    python make_demo_video.py --mode all
"""

import cv2
import numpy as np
import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
DATASET_DIR = Path("/home/hail/pan/VLM-project/dataset/open_data/data/Training/images")
OUTPUT_DIR  = Path("/home/hail/pan/VLM-project/demo_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

FPS        = 5    # CCTV 느낌: 5fps
RESOLUTION = (640, 480)


# ── CCTV 오버레이 (타임스탬프 + 위치) ──────────────────────────────────────────
def add_cctv_overlay(frame, location="연평도", cam_id="CAM-05", frame_idx=0):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 반투명 상단 바
    cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # 타임스탬프 (프레임마다 1초씩 증가하는 척)
    base_time = datetime(2025, 6, 15, 10, 30, 0)
    from datetime import timedelta
    t = base_time + timedelta(seconds=frame_idx)
    ts = t.strftime("%Y-%m-%d %H:%M:%S")

    cv2.putText(frame, f"{location} | {cam_id} | {ts}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 녹화 표시 (빨간 점)
    cv2.circle(frame, (w - 20, 15), 6, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (w - 50, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return frame


# ── Re-encode to H.264 for browser compatibility ───────────────────────────────
def reencode_h264(src_path: str) -> str:
    """
    Re-encode a cv2-written mp4v file to H.264 using ffmpeg.
    Overwrites the original file with the browser-compatible version.
    """
    tmp_path = src_path.replace(".mp4", "_tmp.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", src_path,
            "-vcodec", "libx264",
            "-crf", "23",          # quality (lower = better, 18-28 is typical)
            "-pix_fmt", "yuv420p", # required for browser/QuickTime compatibility
            tmp_path
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.replace(tmp_path, src_path)
    return src_path


# ── Mode 1: 연속 프레임 → 영상 ─────────────────────────────────────────────────
def make_sequential_video(scene_id="0001", img_type="I1",
                          location="연평도", max_frames=60):
    img_dir = DATASET_DIR / f"{img_type}_images"
    frames  = sorted(img_dir.glob(f"{img_type}_S0_C5_{scene_id}*.jpg"))

    if not frames:
        print(f"[ERROR] 씬 {scene_id} 이미지 없음: {img_dir}")
        return None

    frames = frames[:max_frames]
    print(f"[sequential] 씬 {scene_id} ({img_type}) : {len(frames)}장 → 영상 생성")

    out_path = OUTPUT_DIR / f"demo_{img_type}_scene{scene_id}.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        RESOLUTION
    )

    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img = cv2.resize(img, RESOLUTION)
        img = add_cctv_overlay(img, location=location, frame_idx=i)
        writer.write(img)

    writer.release()
    reencode_h264(str(out_path))
    duration = len(frames) / FPS
    print(f"  → saved: {out_path}  ({duration:.1f}s)")
    return str(out_path)


# ── Mode 2: 이미지 1장 → Ken Burns 효과 영상 ──────────────────────────────────
def make_ken_burns_video(image_path, location="연평도",
                         duration_sec=10, zoom_in=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 이미지 읽기 실패: {image_path}")
        return None

    ih, iw = img.shape[:2]
    tw, th  = RESOLUTION
    total_frames = duration_sec * FPS

    stem     = Path(image_path).stem
    out_path = OUTPUT_DIR / f"demo_kenburns_{stem}.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        RESOLUTION
    )

    # 줌 범위: 100% ~ 120% (또는 반대)
    zoom_start = 1.0 if zoom_in else 1.2
    zoom_end   = 1.2 if zoom_in else 1.0

    # 팬 방향: 좌→우 (또는 우→좌)
    pan_x_start = 0.0
    pan_x_end   = 0.1  # 이미지 너비의 10% 이동

    for i in range(total_frames):
        t = i / max(total_frames - 1, 1)  # 0.0 ~ 1.0

        zoom  = zoom_start + t * (zoom_end - zoom_start)
        pan_x = pan_x_start + t * (pan_x_end - pan_x_start)

        # 크롭 영역 계산
        crop_w = int(iw / zoom)
        crop_h = int(ih / zoom)
        x_off  = int(pan_x * (iw - crop_w))
        y_off  = (ih - crop_h) // 2

        x_off = max(0, min(x_off, iw - crop_w))
        y_off = max(0, min(y_off, ih - crop_h))

        cropped = img[y_off:y_off + crop_h, x_off:x_off + crop_w]
        resized = cv2.resize(cropped, RESOLUTION)

        # 미세한 노이즈 (CCTV 질감)
        noise  = np.random.normal(0, 3, resized.shape).astype(np.int16)
        resized = np.clip(resized.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        resized = add_cctv_overlay(resized, location=location, frame_idx=i)
        writer.write(resized)

    writer.release()
    reencode_h264(str(out_path))
    print(f"[ken_burns] {Path(image_path).name} → {out_path}  ({duration_sec}s)")
    return str(out_path)


# ── Mode 3: 여러 이미지 슬라이드쇼 (씬 전환 효과) ─────────────────────────────
def make_slideshow_video(image_paths, location="연평도",
                         sec_per_image=3, transition_frames=10):
    """
    여러 이미지를 페이드 전환으로 이어붙인 영상
    - 이미지가 몇 장 없어도 자연스러운 영상 생성 가능
    """
    if not image_paths:
        print("[ERROR] 이미지 없음")
        return None

    out_path = OUTPUT_DIR / "demo_slideshow.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        RESOLUTION
    )

    frames_per_image = sec_per_image * FPS
    global_idx = 0

    for idx, ip in enumerate(image_paths):
        img = cv2.imread(ip)
        if img is None:
            continue
        img = cv2.resize(img, RESOLUTION)

        next_img = None
        if idx + 1 < len(image_paths):
            ni = cv2.imread(image_paths[idx + 1])
            if ni is not None:
                next_img = cv2.resize(ni, RESOLUTION)

        for f in range(frames_per_image):
            frame = img.copy()

            # 마지막 transition_frames 프레임에서 페이드 아웃/인
            if next_img is not None and f >= frames_per_image - transition_frames:
                alpha = (f - (frames_per_image - transition_frames)) / transition_frames
                frame = cv2.addWeighted(img, 1 - alpha, next_img, alpha, 0)

            frame = add_cctv_overlay(frame.astype(np.uint8),
                                     location=location, frame_idx=global_idx)
            writer.write(frame)
            global_idx += 1

    writer.release()
    reencode_h264(str(out_path))
    total_sec = global_idx / FPS
    print(f"[slideshow] {len(image_paths)} images → {out_path}  ({total_sec:.1f}s)")
    return str(out_path)


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="데모 영상 생성기")
    parser.add_argument("--mode", choices=["sequential", "ken_burns", "slideshow", "all"],
                        default="all")
    parser.add_argument("--scene",    default="0001",  help="씬 ID (sequential 모드)")
    parser.add_argument("--type",     default="I1",    help="I1(주간) / I2(야간)")
    parser.add_argument("--image",    default=None,    help="이미지 경로 (ken_burns 모드)")
    parser.add_argument("--location", default="연평도", help="위치 표시 텍스트")
    args = parser.parse_args()

    if args.mode == "sequential":
        make_sequential_video(args.scene, args.type, args.location)

    elif args.mode == "ken_burns":
        if args.image is None:
            # 기본 이미지 사용
            default_img = str(DATASET_DIR / "I1_images" / "I1_S0_C5_0001001.jpg")
            print(f"--image 미지정, 기본값 사용: {default_img}")
            args.image = default_img
        make_ken_burns_video(args.image, args.location)

    elif args.mode == "slideshow":
        # I1 씬0001에서 5장 샘플
        imgs = sorted((DATASET_DIR / "I1_images").glob("I1_S0_C5_0001*.jpg"))[:5]
        make_slideshow_video([str(p) for p in imgs], args.location)

    elif args.mode == "all":
        print("=== 데모 영상 일괄 생성 ===\n")
        # 1) 주간 씬 (I1 scene 0001)
        make_sequential_video("0001", "I1", "연평도 (주간)")
        # 2) 야간 씬 (I1 scene 0002)
        make_sequential_video("0002", "I1", "연평도 (야간)")
        # 3) 단일 이미지 Ken Burns
        sample_img = str(DATASET_DIR / "I1_images" / "I1_S0_C5_0001001.jpg")
        make_ken_burns_video(sample_img, "연평도")
        # 4) 슬라이드쇼 (씬0001 처음 5장)
        imgs = sorted((DATASET_DIR / "I1_images").glob("I1_S0_C5_0001*.jpg"))[:5]
        make_slideshow_video([str(p) for p in imgs], "연평도")

        print(f"\n=== 완료! 저장 위치: {OUTPUT_DIR} ===")
        for f in sorted(OUTPUT_DIR.glob("*.mp4")):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
