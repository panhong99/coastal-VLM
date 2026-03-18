# make_finetune_data.py
import json
import os
from pathlib import Path

# 클래스 매핑
CLASS_MAP = {
    "0": "어선",
    "1": "상선", 
    "2": "군함",
    "3": "사람",
    "4": "유조류",
    "5": "선박"
}

def get_weather_risk(weather):
    """기상 데이터로 위험도 판단"""
    wind_speed = weather[2]  # 풍속
    rain = weather[3]         # 강수량
    
    if wind_speed >= 14 or rain > 10:
        return "높음"
    elif wind_speed >= 7 or rain > 0:
        return "보통"
    else:
        return "낮음"

def get_alert_level(class_id, weather_risk, scenario):
    """경보 레벨 판단"""
    # 군함 탐지 시 상향
    if class_id == "2":
        if weather_risk == "높음":
            return "RED"
        return "YELLOW"
    # 기상 위험도 기반
    if weather_risk == "높음":
        return "YELLOW"
    return "GREEN"

def get_action(alert_level, class_name):
    """경보 레벨별 권고 조치"""
    actions = {
        "GREEN": f"정상 모니터링 유지. {class_name} 동향 지속 관찰.",
        "YELLOW": f"경계 강화. {class_name} 식별 및 이동 경로 추적 요망.",
        "RED": f"즉각 상황 보고. {class_name} 에 대한 긴급 대응 절차 가동."
    }
    return actions[alert_level]

def get_scenario_text(scenario):
    if scenario == 1:
        return "이동 중"
    elif scenario == 2:
        return "정지 중"
    return "미상"

def convert_to_report(json_data):
    """JSON 데이터를 파인튜닝용 리포트로 변환"""
    meta = json_data["meta"]
    ann  = json_data["annotations"][0]
    
    weather      = meta["weather"]
    class_id     = ann["class"]
    class_name   = CLASS_MAP.get(class_id, "미상")
    weather_risk = get_weather_risk(weather)
    alert_level  = get_alert_level(class_id, weather_risk, meta["scenario"])
    scenario_txt = get_scenario_text(meta["scenario"])
    
    # 위치 매핑
    location_map = {0: "백령도", 1: "연평도"}
    location = location_map.get(meta["location"], "미상")
    time_map = {0: "주간", 1: "야간"}
    time_str = time_map.get(meta["time"], "미상")

    # input 프롬프트
    input_prompt = f"""해안 감시 카메라 이미지입니다. [{location} / {time_str}]
기상정보: 기온 {weather[0]}°C, 풍향 {weather[1]}°, 풍속 {weather[2]}m/s, 강수량 {weather[3]}mm, 습도 {weather[6]}%

다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요."""

    # output 리포트
    output_report = f"""## 해안 경계 상황 리포트

**탐지 객체:** {class_name} 1척 ({scenario_txt})
**촬영 환경:** {location} / {time_str}
**기상 상태:** 기온 {weather[0]}°C, 풍속 {weather[2]}m/s, 습도 {weather[6]}% → 위험도 {weather_risk}
**경보 레벨:** {alert_level}
**상황 요약:** {ann['caption']}
**권고 조치:** {get_action(alert_level, class_name)}"""

    return {
        "image": ann["filename"],
        "input": input_prompt,
        "output": output_report
    }

# 실행
def main():
    json_dirs = [
        "/home/hail/pan/VLM-project/dataset/open_data/data/Training/jsons/I1_jsons",
        "/home/hail/pan/VLM-project/dataset/open_data/data/Training/jsons/I2_jsons"
    ]
    
    results = []
    errors  = 0
    
    for json_dir in json_dirs:
        files = list(Path(json_dir).glob("*.json"))
        print(f"{json_dir}: {len(files)}개 처리 중...")
        
        for fpath in files:
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                results.append(convert_to_report(data))
            except Exception as e:
                errors += 1
                continue
    
    # 저장
    output_path = "/home/hail/pan/VLM-project/finetune_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n완료! 총 {len(results)}개 생성 (에러: {errors}개)")
    print(f"저장 위치: {output_path}")
    
    # 샘플 출력
    print("\n=== 샘플 확인 ===")
    print("INPUT:\n",  results[0]["input"])
    print("\nOUTPUT:\n", results[0]["output"])

if __name__ == "__main__":
    main()