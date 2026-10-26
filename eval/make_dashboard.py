# -*- coding: utf-8 -*-
"""
生成 EvalScope Dashboard 风格的成绩环状图（仿原项目 Dashboard 饼图）。
数据来源：eval/evalscope/**/reports/<model>/<dataset>.json（EvalScope 评测报告）
输出：eval/reports/dashboard.png + eval/reports/dashboard_scores.json
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Wedge

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ES_DIR = os.path.join(BASE_DIR, "eval", "evalscope")
OUT_PNG = os.path.join(BASE_DIR, "eval", "reports", "dashboard.png")
OUT_JSON = os.path.join(BASE_DIR, "eval", "reports", "dashboard_scores.json")

DATASETS = ["gsm8k", "arc"]
DATASET_LABEL = {"gsm8k": "GSM8K（数学推理）", "arc": "ARC（常识推理）"}


def load_score(run_dir, dataset):
    """从 EvalScope 报告里取主指标分数（0~1）"""
    pattern = os.path.join(run_dir, "reports", "*", f"{dataset}.json")
    files = glob.glob(pattern)
    if not files:
        return None
    data = json.load(open(files[0], encoding="utf-8"))
    for m in data.get("metrics", []):
        name = (m.get("identity") or {}).get("name", "")
        if name.lower() in ("accuracy", "acc", "avg_accuracy", "score"):
            return float(m["score"])
    return None


def score_color(frac):
    """红(0) -> 黄(0.5) -> 绿(1) 色阶，与 EvalScope Dashboard 一致"""
    return plt.get_cmap("RdYlGn")(float(np.clip(frac, 0, 1)))


def draw_ring(ax, values, radius, width, text_r_offset=0.07):
    n = len(values)
    step = 360.0 / n
    pad = 1.2  # 段间留白（度）
    for i, v in enumerate(values):
        frac = (v / 100.0) if v is not None else 0.0
        start = 90.0 - (i + 1) * step + pad
        end = 90.0 - i * step - pad
        ax.add_patch(Wedge((0, 0), radius + width, start, end, width=width,
                           facecolor=score_color(frac), edgecolor="#0d1117",
                           linewidth=1.5, alpha=0.96))
        mid = np.deg2rad((start + end) / 2)
        label = f"{v:.0f}%" if v is not None else "N/A"
        # 分数写在环带中间
        ax.text(np.cos(mid) * (radius + width / 2),
                np.sin(mid) * (radius + width / 2),
                label, ha="center", va="center", fontsize=13,
                color="#0d1117", fontweight="bold")


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    runs = {
        "base": {"gsm8k": os.path.join(ES_DIR, "base_gsm8k"),
                 "arc": os.path.join(ES_DIR, "base_arc")},
        "ft": {"gsm8k": os.path.join(ES_DIR, "ft_gsm8k"),
               "arc": os.path.join(ES_DIR, "ft_arc")},
    }
    scores = {"base": {}, "ft": {}}
    for model, ds in runs.items():
        for d in DATASETS:
            s = load_score(ds[d], d)
            scores[model][d] = round(s * 100, 1) if s is not None else None

    json.dump(scores, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    fig = plt.figure(figsize=(12, 9), facecolor="#0d1117")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0d1117")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.4)

    base_vals = [scores["base"][d] for d in DATASETS]
    ft_vals = [scores["ft"][d] for d in DATASETS]

    # 内环：基座；外环：微调后
    draw_ring(ax, base_vals, radius=0.75, width=0.55)
    draw_ring(ax, ft_vals, radius=1.40, width=0.55)

    ax.text(0, 0.16, "EvalScope 评测成绩", ha="center", va="center",
            fontsize=14, color="#c9d1d9")
    ax.text(0, -0.05, "DeepSeek-R1-1.5B\n基座 vs 微调后", ha="center", va="center",
            fontsize=9.5, color="#8b949e")

    # 数据集名称标注（环外，左右对齐避免裁切）
    n = len(DATASETS)
    for i in range(n):
        mid = np.deg2rad(90 - (i + 0.5) * (360.0 / n))
        x = np.cos(mid) * 2.12
        y = np.sin(mid) * 2.12
        ax.plot([np.cos(mid) * 1.98, x - 0.06 * np.sign(x)], [np.sin(mid) * 1.98, y],
                color="#484f58", lw=1)
        ax.text(x, y, DATASET_LABEL[DATASETS[i]], ha="left" if x > 0 else "right",
                va="center", fontsize=11, color="#c9d1d9")

    legend = [
        Patch(facecolor="#6b7280", label="内环：基座 deepseek-r1:1.5b"),
        Patch(facecolor="#6b7280", label="外环：微调 deepseek-r1-roleplay"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False, fontsize=10, labelcolor="#c9d1d9")
    ax.text(0, -2.18, "样本量各 10 条（EvalScope，limit=10，max_tokens=1024）；分数 = 主指标 Accuracy",
            ha="center", va="center", fontsize=9, color="#8b949e")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    print(f">> 仪表盘已保存: {OUT_PNG}")
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
