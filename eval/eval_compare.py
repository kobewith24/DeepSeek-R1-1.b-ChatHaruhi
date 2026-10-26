# -*- coding: utf-8 -*-
"""
微调前后对比评测：通过 Ollama OpenAI 兼容 API 分别询问基座模型与微调模型。
- 角色扮演用例：检验垂域能力提升（凉宫春日 / 御坂美琴 / 韦小宝）
- 通用用例：检验是否发生灾难性遗忘（数学推理 + 日常对话）
产物：eval/reports/compare_report.md
"""
import datetime
import json
import os

from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "eval", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

BASE_MODEL = "deepseek-r1:1.5b"
FT_MODEL = "deepseek-r1-roleplay"

CASES = [
    {
        "type": "roleplay",
        "name": "凉宫春日",
        "messages": [
            {"role": "system", "content": "你现在扮演凉宫春日。请完全以凉宫春日的身份、语气、性格和口癖进行回复，沉浸于角色之中，不要跳出角色，不要提及你是 AI 助手。"},
            {"role": "user", "content": "你好！今天社团活动做什么呀？"},
        ],
    },
    {
        "type": "roleplay",
        "name": "御坂美琴",
        "messages": [
            {"role": "system", "content": "你现在扮演御坂美琴。请完全以御坂美琴的身份、语气、性格和口癖进行回复，沉浸于角色之中，不要跳出角色，不要提及你是 AI 助手。"},
            {"role": "user", "content": "你在学园都市最喜欢的地方是哪里？"},
        ],
    },
    {
        "type": "roleplay",
        "name": "韦小宝",
        "messages": [
            {"role": "system", "content": "你现在扮演韦小宝。请完全以韦小宝的身份、语气、性格和口癖进行回复，沉浸于角色之中，不要跳出角色，不要提及你是 AI 助手。"},
            {"role": "user", "content": "康熙：小桂子，朕听说你最近又发了笔横财？"},
        ],
    },
    {
        "type": "general",
        "name": "数学推理（防遗忘检查）",
        "messages": [{"role": "user", "content": "Solve (x + 2)^2 = 0. Show your reasoning."}],
    },
    {
        "type": "general",
        "name": "日常对话（防遗忘检查）",
        "messages": [{"role": "user", "content": "你好，给我推荐三部适合周末看的电影吧。"}],
    },
]


def chat(model, messages):
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.6, top_p=0.95, max_tokens=1024,
    )
    return resp.choices[0].message.content


def main():
    report = [f"# 微调前后对比评测报告\n",
              f"生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n",
              f"基座模型：`{BASE_MODEL}`　微调模型：`{FT_MODEL}`\n"]
    results = []
    for case in CASES:
        print(f">> 评测用例: {case['name']}")
        base_out = chat(BASE_MODEL, case["messages"])
        ft_out = chat(FT_MODEL, case["messages"])
        results.append({"case": case["name"], "type": case["type"],
                        "base": base_out, "finetuned": ft_out})
        report += [
            f"\n## [{case['type']}] {case['name']}\n",
            f"**输入**：{case['messages'][-1]['content']}\n",
            f"**基座模型回复**：\n\n{base_out}\n",
            f"**微调模型回复**：\n\n{ft_out}\n",
        ]

    path = os.path.join(REPORT_DIR, "compare_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    with open(os.path.join(REPORT_DIR, "compare_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f">> 报告已保存: {path}")


if __name__ == "__main__":
    main()
