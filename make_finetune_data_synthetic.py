# make_finetune_data_synthetic.py
import json
import os
from pathlib import Path

# 클래스 매핑
SUB_CLASS_MAP = {
    11: "어선", 12: "군함", 13: "상선",
    21: "고정익 항공기", 22: "회전익 항공기", 23: "무인항공기(드론)",
    31: "조류",
    41: "삐라", 42: "오물폭탄"
}

WEATHER_MAP = {
    1: "맑음", 2: "흐림", 3: "비", 4: "눈",
    5: "해무 1단계", 6: "해무 2단계", 7: "해무 3단계"
}

WAVE_MAP = {
    1: "1단계(잔잔)", 2: "2단계(약간)", 3: "3단계(보통)",
    4: "4단계(높음)", 5: "5단계(매우높음)",
    6: "6단계(위험)", 7: "7단계(매우위험)"
}

def get_weather_risk(weather, wave, night):
    """기상/파도/야간 조건으로 위험도 판단"""
    if wave >= 5 or weather in [3, 4] or (night == 2 and wave >= 3):
        return "높음"
    elif wave >= 3 or weather in [2, 5, 6, 7]:
        return "보통"
    return "낮음"

def get_alert_level(sub_class, weather_risk, night):
    """경보 레벨 판단"""
    # 위협 객체
    if sub_class in [42]:  # 오물폭탄
        return "RED"
    if sub_class in [41]:  # 삐라
        return "YELLOW"
    if sub_class == 12:    # 군함
        if weather_risk == "높음":
            return "RED"
        return "YELLOW"
    if sub_class == 23:    # 드론
        return "YELLOW"
    # 기상 위험도 기반
    if weather_risk == "높음":
        return "YELLOW"
    return "GREEN"

def get_action(alert_level, class_name):
    actions = {
        "GREEN": f"정상 모니터링 유지. {class_name} 동향 지속 관찰.",
        "YELLOW": f"경계 강화. {class_name} 식별 및 이동 경로 추적 요망.",
        "RED": f"즉각 상황 보고. {class_name} 긴급 대응 절차 가동."
    }
    return actions[alert_level]

def convert_synthetic(json_data):
    img  = json_data["image"]
    env  = json_data["env"]
    ann  = json_data["annotations"][0]

    sub_class    = ann["sub_class"]
    class_name   = SUB_CLASS_MAP.get(sub_class, "미상")
    weather_str  = WEATHER_MAP.get(env["weather"], "미상")
    wave_str     = WAVE_MAP.get(env["wave"], "미상")
    night        = env["night"]
    time_str     = "야간" if night == 2 else "주간"
    season_str   = "하절기" if env["season"] == 1 else "동절기"
    weather_risk = get_weather_risk(env["weather"], env["wave"], night)
    alert_level  = get_alert_level(sub_class, weather_risk, night)

    input_prompt = f"""해안 감시 카메라 이미지입니다. [{season_str} / {time_str}]
기상정보: 날씨 {weather_str}, 파도 {wave_str}

다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요."""

    output_report = f"""## 해안 경계 상황 리포트

**탐지 객체:** {class_name} {json_data['object_num']}척
**촬영 환경:** {season_str} / {time_str}
**기상 상태:** {weather_str}, 파도 {wave_str} → 위험도 {weather_risk}
**경보 레벨:** {alert_level}
**상황 요약:** {json_data['caption']}
**권고 조치:** {get_action(alert_level, class_name)}"""

    return {
        "image": img["filename"],
        "input": input_prompt,
        "output": output_report,
        "source": "synthetic"
    }

def main():
    base = "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/jsons"
    folders = ["EO_SU_DT", "EO_SU_NT", "EO_WI_DT", "IR_SU_NT"]

    results = []
    errors  = 0

    for folder in folders:
        files = list(Path(f"{base}/{folder}").glob("*.json"))
        print(f"{folder}: {len(files)}개 처리 중...")

        for fpath in files:
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                results.append(convert_synthetic(data))
            except Exception as e:
                errors += 1
                continue

    output_path = "/home/hail/pan/VLM-project/finetune_data_synthetic.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n완료! 총 {len(results)}개 생성 (에러: {errors}개)")

    # 경보 레벨 분포 확인
    from collections import Counter
    alerts  = [d['output'].split('**경보 레벨:** ')[1].split('\n')[0] for d in results]
    objects = [d['output'].split('**탐지 객체:** ')[1].split(' ')[0] for d in results]

    print("\n경보 레벨 분포:")
    for k, v in Counter(alerts).items():
        print(f"  {k}: {v}개 ({v/len(results)*100:.1f}%)")

    print("\n탐지 객체 분포:")
    for k, v in Counter(objects).items():
        print(f"  {k}: {v}개 ({v/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()