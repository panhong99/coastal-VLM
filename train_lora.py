# train_lora_deepspeed.py
import json
import torch
from pathlib import Path
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import TrainerCallback
from tqdm import tqdm
import time
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

MODEL_PATH  = "/home/hail/pan/VLM-project/models/Qwen2.5-VL-7B-Instruct"
DATA_PATH   = "/home/hail/pan/VLM-project/finetune_data_multiframe.json"   # 멀티프레임 데이터
OUTPUT_PATH = "/home/hail/pan/VLM-project/checkpoints/lora_multiframe"

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

class TimeTrackingCallback(TrainerCallback):
    def __init__(self):
        self.train_start  = None
        self.epoch_start  = None
        self.step_times   = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        print(f"\n{'='*50}")
        print(f"학습 시작!")
        print(f"총 스텝: {state.max_steps}")
        print(f"{'='*50}\n")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        epoch = int(state.epoch) + 1
        print(f"\n[Epoch {epoch}/{int(args.num_train_epochs)}] 시작")

    def on_step_end(self, args, state, control, **kwargs):
        self.step_times.append(time.time())

        if state.global_step % args.logging_steps == 0:
            # 평균 스텝 소요시간
            if len(self.step_times) >= 2:
                avg_step_time = (
                    self.step_times[-1] - self.step_times[-args.logging_steps]
                ) / args.logging_steps
            else:
                avg_step_time = 0

            # 남은 시간 계산
            remaining_steps = state.max_steps - state.global_step
            eta_seconds     = remaining_steps * avg_step_time
            eta_str         = time.strftime("%H시간 %M분 %S초", time.gmtime(eta_seconds))

            # 경과 시간
            elapsed         = time.time() - self.train_start
            elapsed_str     = time.strftime("%H시간 %M분 %S초", time.gmtime(elapsed))

            print(f"  Step {state.global_step}/{state.max_steps} | "
                  f"경과: {elapsed_str} | "
                  f"남은시간(ETA): {eta_str} | "
                  f"스텝당: {avg_step_time:.1f}초")

            wandb.log({
                "elapsed_seconds": elapsed,
                "eta_seconds":     eta_seconds,
                "step_time":       avg_step_time,
            }, step=state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time  = time.time() - self.epoch_start
        epoch_str   = time.strftime("%H시간 %M분 %S초", time.gmtime(epoch_time))
        total_str   = time.strftime("%H시간 %M분 %S초", time.gmtime(time.time() - self.train_start))

        print(f"\n[Epoch {int(state.epoch)}] 완료 | "
              f"소요: {epoch_str} | "
              f"총 경과: {total_str}")
        
        wandb.log({
            "epoch_time_seconds": epoch_time,
            "epoch":              int(state.epoch),
        }, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start
        total_str  = time.strftime("%H시간 %M분 %S초", time.gmtime(total_time))
        print(f"\n{'='*50}")
        print(f"학습 완료! 총 소요시간: {total_str}")
        print(f"{'='*50}\n")
        
        wandb.log({"total_time_seconds": total_time})
        wandb.finish()

def build_image_map(dirs):
    img_map = {}
    for d in dirs:
        for f in Path(d).glob("*.jpg"):
            img_map[f.name] = str(f)
    return img_map

print("이미지 경로 매핑 중...")
IMAGE_MAP = build_image_map(REAL_IMG_DIRS + SYN_IMG_DIRS)
print(f"총 {len(IMAGE_MAP)}개 이미지 매핑 완료")

# 프로세서
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28
)

class CoastDataset(Dataset):
    def __init__(self, data_path, image_map, max_samples=None):
        with open(data_path) as f:
            data = json.load(f)
        if max_samples:
            data = data[:max_samples]
        # 멀티프레임: 'images' 키(리스트)로 모든 프레임이 image_map에 있어야 함
        self.data = [
            d for d in data
            if all(img in image_map for img in d['images'])
        ]
        self.image_map = image_map
        print(f"유효 데이터: {len(self.data)}개")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 8프레임 이미지 content 구성 (추론의 vlm_analyze와 동일한 포맷)
        image_content = [
            {
                "type":           "image",
                "image":          self.image_map[img_name],
                "resized_height": 224,
                "resized_width":  224,
            }
            for img_name in item['images']
        ]

        messages = [
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": item['input']}]
            },
            {
                "role": "assistant",
                "content": item['output']
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=False,
        )

        return {
            "input_ids":      inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values":   inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "labels":         inputs["input_ids"].squeeze(0).clone(),
        }

def data_collator(features):
    from torch.nn.utils.rnn import pad_sequence

    input_ids      = pad_sequence([f["input_ids"] for f in features],
                                   batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = pad_sequence([f["attention_mask"] for f in features],
                                   batch_first=True, padding_value=0)
    labels         = pad_sequence([f["labels"] for f in features],
                                   batch_first=True, padding_value=-100)
    pixel_values   = torch.cat([f["pixel_values"] for f in features], dim=0)
    image_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "pixel_values":   pixel_values,
        "image_grid_thw": image_grid_thw,
        "labels":         labels,
    }

# 모델 로드
print("모델 로딩 중...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map=None  # DeepSpeed가 관리하므로 None
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 데이터셋 (6000개로 제한 → 약 6시간)
dataset    = CoastDataset(DATA_PATH, IMAGE_MAP, max_samples=6000)
val_size   = int(len(dataset) * 0.1)
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

wandb.init(
    project="coastal-surveillance-vlm",
    name="qwen2.5-vl-7b-lora-multiframe-v2",   # 멀티프레임 실험
    config={
        "model":         "Qwen2.5-VL-7B-Instruct",
        "lora_r":        8,
        "lora_alpha":    16,
        "batch_size":    1,
        "grad_accum":    8,
        "lr":            2e-4,
        "epochs":        3,
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
        "image_size":    "224x224",
        "num_frames":    8,              # 멀티프레임
        "zero_stage":    2,
        "dataset":       "real+synthetic 멀티프레임(8장)",
    }
)

# TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,   # 실질적 배치 = 8
    learning_rate=2e-4,
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    deepspeed="/home/hail/pan/VLM-project/ds_config.json",  # DeepSpeed 연결
    report_to="wandb",
    run_name="qwen2.5-vl-7b-lora-v1",
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[TimeTrackingCallback()]
)

print("\n학습 시작!")
trainer.train()

# 저장
model.save_pretrained(f"{OUTPUT_PATH}/final")
processor.save_pretrained(f"{OUTPUT_PATH}/final")
print("완료!")