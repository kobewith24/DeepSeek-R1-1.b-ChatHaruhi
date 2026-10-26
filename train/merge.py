# -*- coding: utf-8 -*-
"""
合并 LoRA adapter 到基座模型，导出 fp16 完整权重。
产物：weights/DeepSeek-R1-1.5B-roleplay-fp16（safetensors，可直接被 Ollama 导入）
"""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "DeepSeek-R1-Distill-Qwen-1.5B")
ADAPTER_DIR = os.path.join(BASE_DIR, "weights", "lora_adapter")
OUT_DIR = os.path.join(BASE_DIR, "weights", "DeepSeek-R1-1.5B-roleplay-fp16")


def main():
    base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    merged = model.merge_and_unload()
    merged.save_pretrained(OUT_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f">> 合并完成，fp16 模型已保存: {OUT_DIR}")


if __name__ == "__main__":
    main()
