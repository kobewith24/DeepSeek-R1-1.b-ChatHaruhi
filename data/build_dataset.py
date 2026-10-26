# -*- coding: utf-8 -*-
"""
数据集构建：ChatHaruhi-54K（二次元角色扮演） + FineTome-100k（通用对话）
仿照原项目 OpenMathReasoning + FineTome-100k 的 25% / 75% 混合配比思路，
本项目中：25% 角色扮演数据 + 75% 通用对话数据，微调角色扮演能力的同时避免灾难性遗忘。
产物：data/cleaned_roleplay_dataset（datasets.save_to_disk 格式，含 "text" 字段）
"""
import os
import random

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像，免翻墙

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "DeepSeek-R1-Distill-Qwen-1.5B")
OUT_DIR = os.path.join(BASE_DIR, "data", "cleaned_roleplay_dataset")

CHAT_PERCENTAGE = 0.75          # 通用对话占比 75%，角色扮演占比 25%
ROLEPLAY_N = 2500               # 角色扮演样本数
SEED = 24


def load_roleplay():
    """ChatHaruhi-54K 角色扮演对话 -> messages 列表"""
    ds = load_dataset("silk-road/ChatHaruhi-54K-Role-Playing-Dialogue", split="train")
    print(f"ChatHaruhi-54K 总量: {len(ds)}")
    idx = random.Random(SEED).sample(range(len(ds)), ROLEPLAY_N)
    convs = []
    for i in idx:
        r = ds[i]
        system = (
            f"你现在扮演{ r['agent_role'] }。请完全以{ r['agent_role'] }的身份、语气、性格和口癖"
            "进行回复，沉浸于角色之中，不要跳出角色，不要提及你是 AI 助手。"
        )
        user = (f"{r['user_role']}：" if r["user_role"] else "") + r["user_question"]
        convs.append([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["agent_response"]},
        ])
    return convs


def load_general():
    """FineTome-100k 通用对话（sharegpt 格式） -> messages 列表"""
    ds = load_dataset("mlabonne/FineTome-100k", split="train")
    print(f"FineTome-100k 总量: {len(ds)}")
    n_general = int(ROLEPLAY_N * (CHAT_PERCENTAGE / (1 - CHAT_PERCENTAGE)))  # 75/25
    idx = random.Random(SEED).sample(range(len(ds)), n_general)
    convs = []
    for i in idx:
        msgs = [{"role": m["from"].replace("human", "user").replace("gpt", "assistant"),
                 "content": m["value"]} for m in ds[i]["conversations"]]
        convs.append(msgs)
    return convs


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    roleplay_convs = load_roleplay()
    general_convs = load_general()
    print(f"角色扮演样本: {len(roleplay_convs)}, 通用对话样本: {len(general_convs)}")

    texts = [
        tokenizer.apply_chat_template(c, tokenize=False)
        for c in roleplay_convs + general_convs
    ]
    random.Random(SEED).shuffle(texts)

    combined = Dataset.from_pandas(pd.DataFrame(pd.Series(texts, name="text")))
    combined.save_to_disk(OUT_DIR)
    print(f"已保存混合数据集到 {OUT_DIR}，共 {len(combined)} 条")
    print("---- 样例 ----")
    print(combined[0]["text"][:500])


if __name__ == "__main__":
    main()
