# -*- coding: utf-8 -*-
"""
QLoRA 微调：DeepSeek-R1-Distill-Qwen-1.5B + 二次元角色扮演混合数据集
仿照原项目 Unsloth LoRA 流程，Windows 原生环境改用 PEFT + TRL + bitsandbytes 实现：
  - 4bit NF4 动态量化加载基座（若 bitsandbytes 不可用则自动回退 bf16 + LoRA）
  - LoRA r=32，注入 q/k/v/o + gate/up/down 全部线性层
产物：weights/lora_adapter（LoRA 权重）、train/training_log.json（loss 与显存记录）
"""
import json
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "DeepSeek-R1-Distill-Qwen-1.5B")
DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned_roleplay_dataset")
ADAPTER_DIR = os.path.join(BASE_DIR, "weights", "lora_adapter")
LOG_PATH = os.path.join(BASE_DIR, "train", "training_log.json")

MAX_SEQ_LENGTH = 2048


def load_base_model():
    """优先 4bit NF4 量化加载；bitsandbytes 不可用时回退 bf16。"""
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, quantization_config=bnb_config, device_map="auto",
        )
        print(">> 4bit NF4 量化加载成功（bitsandbytes）")
        return model, "4bit-nf4"
    except Exception as e:  # noqa: BLE001
        print(f">> bitsandbytes 4bit 不可用（{e}），回退 bf16 加载")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto",
        )
        return model, "bf16"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model, quant_mode = load_base_model()

    # LoRA 参数注入（与原项目一致：r=32，全部注意力 + MLP 线性层）
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = load_from_disk(DATA_DIR)
    print(f">> 训练集: {len(dataset)} 条")

    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    total_mem = round(gpu_stats.total_memory / 1024**3, 3)
    print(f">> GPU = {gpu_stats.name}, 总显存 {total_mem} GB, 已占用 {start_mem} GB")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=5,
            output_dir=os.path.join(BASE_DIR, "train", "checkpoints"),
            optim="paged_adamw_8bit" if quant_mode == "4bit-nf4" else "adamw_torch",
            report_to=[],
            bf16=True,
            save_strategy="no",
            seed=24,
        ),
    )

    trainer_stats = trainer.train()

    peak_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    log = {
        "quant_mode": quant_mode,
        "gpu": gpu_stats.name,
        "gpu_total_gb": total_mem,
        "peak_reserved_gb": peak_mem,
        "train_samples": len(dataset),
        "train_runtime_s": round(trainer_stats.metrics.get("train_runtime", 0), 2),
        "train_loss": trainer_stats.metrics.get("train_loss"),
        "loss_history": [
            {"step": h.get("step"), "loss": h.get("loss")}
            for h in trainer.state.log_history if "loss" in h
        ],
    }
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f">> 训练日志已保存: {LOG_PATH}")
    print(f">> 训练峰值显存: {peak_mem} GB / {total_mem} GB")

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f">> LoRA adapter 已保存: {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
