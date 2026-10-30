# -*- coding: utf-8 -*-
r"""
角色扮演 LLM-as-a-Judge 评测：
  1) 用 Ollama 分别让基座模型与微调模型回答 30 条角色扮演测试集
  2) 用裁判模型（默认 dsr-v4.1-f）做 A/B 盲评，交换顺序各评一次，消除位置偏差
  3) 汇总：各维度均分、胜/平/负、按问题类型分组

用法：
  set JUDGE_BASE_URL=https://xxx/v1
  set JUDGE_API_KEY=sk-xxx
  set JUDGE_MODEL=dsr-v4.1-f
  python eval\judge_roleplay.py            # 完整流程
  python eval\judge_roleplay.py --dry-run  # 只跑生成 + 假裁判，验证流程
"""
import argparse
import json
import os
import random
import sys
import time

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET = os.path.join(BASE_DIR, "eval", "roleplay_testset.json")
OUT_DIR = os.path.join(BASE_DIR, "eval", "roleplay")
REPORT_DIR = os.path.join(BASE_DIR, "eval", "reports")

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
BASE_MODEL = os.environ.get("BASE_MODEL", "deepseek-r1:1.5b")
FT_MODEL = os.environ.get("FT_MODEL", "deepseek-r1-roleplay")

JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-flash")

DIMENSIONS = ["consistency", "immersion", "quality"]

JUDGE_SYSTEM = (
    "你是一名严格、公正的角色扮演质量评测裁判。你要比较两个模型对同一个角色扮演问题的回答，"
    "判断哪个更像角色本人。评分只看文本本身，不因为回答更长就给更高分。"
)

JUDGE_TEMPLATE = """【角色设定】
{system}

【用户问题】
{question}

【回答 A】
{a}

【回答 B】
{b}

请从以下维度给两个回答各打 1~5 分（整数，5 分最好）：
1. consistency（角色一致性）：语气、口癖、性格、身份设定是否符合该角色
2. immersion（沉浸度）：是否出戏、是否自称 AI / 语言模型、是否跳出角色
3. quality（回答质量）：是否切题、自然、有内容，不空泛

然后判断哪个回答更好（winner 取 A、B 或 tie）。

只输出 JSON，不要输出其它内容，格式：
{{"winner": "A", "a": {{"consistency": 4, "immersion": 5, "quality": 4}}, "b": {{"consistency": 2, "immersion": 1, "quality": 3}}, "reason": "一句话说明"}}"""


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate(model, system, question, temperature=0.6):
    resp = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "options": {"temperature": temperature, "num_predict": 512},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def judge_once(system, question, a, b, dry_run=False):
    """调用裁判模型一次，返回解析后的 dict"""
    if dry_run:
        return {
            "winner": random.choice(["A", "B", "tie"]),
            "a": {d: random.randint(2, 5) for d in DIMENSIONS},
            "b": {d: random.randint(2, 5) for d in DIMENSIONS},
            "reason": "[dry-run] mock judge",
        }

    url = JUDGE_BASE_URL.rstrip("/") + "/chat/completions"
    prompt = JUDGE_TEMPLATE.format(system=system, question=question, a=a, b=b)
    body = {
        "model": JUDGE_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    last_err = None
    for _ in range(3):
        try:
            r = requests.post(
                url, json=body, timeout=300,
                headers={"Authorization": f"Bearer {JUDGE_API_KEY}"},
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            data = parse_judge_json(text)
            if data.get("winner") not in ("A", "B", "tie"):
                raise ValueError("bad winner")
            return data
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"judge failed: {last_err}")


def parse_judge_json(text):
    """解析裁判输出：优先严格 JSON，失败则用正则兜底（兼容未加引号的键）"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    raw = text[start:end + 1] if start != -1 and end != -1 else text
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        pass

    # 兜底：逐字段正则提取
    import re
    out = {}
    m = re.search(r'winner\W*:?\s*"?(A|B|tie)"?', raw, re.I)
    if m:
        out["winner"] = m.group(1) if m.group(1).lower() != "tie" else "tie"
    for side in ("a", "b"):
        block = re.search(rf'{side}\W*:\s*(\{{.*?\}})', raw, re.S)
        scores = {}
        if block:
            for d in DIMENSIONS:
                mm = re.search(rf'{d}\W*:?\s*(\d)', block.group(1))
                if mm:
                    scores[d] = int(mm.group(1))
        out[side] = scores
    return out

DIM_LABEL = {"consistency": "角色一致性", "immersion": "沉浸度（不出戏）", "quality": "回答质量"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="用假裁判验证流程")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    ap.add_argument("--gen-only", action="store_true", help="只生成回答，不评分")
    ap.add_argument("--aggregate-only", action="store_true", help="跳过生成与评分，直接用已有数据重算平均分")
    args = ap.parse_args()

    testset = read_json(TESTSET)
    cases = testset["cases"]
    if args.limit:
        cases = cases[: args.limit]

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    answers_path = os.path.join(OUT_DIR, "answers.json")
    answers = read_json(answers_path, {"meta": {}, "items": {}})
    items = answers.setdefault("items", {})

    judge_path = os.path.join(OUT_DIR, "judge_raw.json")
    judged = read_json(judge_path, {"meta": {}, "items": {}})
    jitems = judged.setdefault("items", {})

    if not args.aggregate_only:
        # ---------- 1. 生成回答 ----------
        for i, c in enumerate(cases, 1):
            cid = c["id"]
            if cid in items and items[cid].get("base") and items[cid].get("ft"):
                continue
            print(f"[{i}/{len(cases)}] 生成 {cid} ({c['character']}) ...")
            items[cid] = {
                "case": c,
                "base": generate(BASE_MODEL, c["system"], c["question"]),
                "ft": generate(FT_MODEL, c["system"], c["question"]),
            }
            write_json(answers_path, answers)
        print(f">> 回答已保存: {answers_path}")

        if args.gen_only:
            return

        # ---------- 2. 盲评（两种顺序） ----------
        for i, c in enumerate(cases, 1):
            cid = c["id"]
            if cid in jitems and jitems[cid].get("order_ab") and jitems[cid].get("order_ba"):
                continue
            print(f"[{i}/{len(cases)}] 裁判评分 {cid} ...")
            jitems[cid] = {
                "order_ab": judge_once(c["system"], c["question"], items[cid]["base"], items[cid]["ft"], args.dry_run),
                "order_ba": judge_once(c["system"], c["question"], items[cid]["ft"], items[cid]["base"], args.dry_run),
            }
            write_json(judge_path, judged)

        judged["meta"] = {
            "judge_model": JUDGE_MODEL,
            "base_model": BASE_MODEL,
            "ft_model": FT_MODEL,
            "n": len(cases),
            "dry_run": args.dry_run,
        }
        write_json(judge_path, judged)
        print(f">> 原始评分已保存: {judge_path}")

    # ---------- 3. 汇总：只比较平均得分 ----------
    def scores_of(side, rec):
        """某模型在两条顺序里的各维度得分（取平均）"""
        out = {d: [] for d in DIMENSIONS}
        for key in ("order_ab", "order_ba"):
            r = rec[key]
            s = r.get(side) if key == "order_ab" else r.get("b" if side == "a" else "a")
            if s:
                for d in DIMENSIONS:
                    if isinstance(s.get(d), (int, float)):
                        out[d].append(s[d])
        return {d: (sum(v) / len(v) if v else None) for d, v in out.items()}

    def overall(s):
        vals = [s[d] for d in DIMENSIONS if s[d] is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def mean(v):
        vals = [x for x in v if x is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    per_case = []
    by_type = {}
    for cid, rec in jitems.items():
        if cid not in items:
            continue
        c = items[cid]["case"]
        sb, sf = scores_of("a", rec), scores_of("b", rec)
        ob, of = overall(sb), overall(sf)
        per_case.append({
            "id": cid, "character": c["character"], "type": c["type"],
            "base_scores": sb, "ft_scores": sf,
            "base_overall": ob, "ft_overall": of,
        })
        by_type.setdefault(c["type"], {"base": [], "ft": []})
        by_type[c["type"]]["base"].append(ob)
        by_type[c["type"]]["ft"].append(of)

    dim_avg = {side: {d: mean([p[f"{side}_scores"][d] for p in per_case]) for d in DIMENSIONS}
               for side in ("base", "ft")}
    base_all = mean([p["base_overall"] for p in per_case])
    ft_all = mean([p["ft_overall"] for p in per_case])

    summary = {
        "n": len(per_case),
        "judge_model": JUDGE_MODEL,
        "base_model": BASE_MODEL,
        "ft_model": FT_MODEL,
        "dim_avg": dim_avg,
        "overall": {"base": base_all, "ft": ft_all,
                    "delta": round(ft_all - base_all, 2) if (base_all is not None and ft_all is not None) else None},
        "by_type": {t: {"base": mean(v["base"]), "ft": mean(v["ft"])} for t, v in by_type.items()},
        "per_case": per_case,
    }
    write_json(os.path.join(REPORT_DIR, "roleplay_judge.json"), summary)

    def fmt(x):
        return "—" if x is None else f"{x:.2f}"

    lines = [
        "# 角色扮演 LLM-as-a-Judge 评测",
        "",
        f"- 裁判模型：`{summary['judge_model']}`（temperature=0，A/B 盲评 + 交换顺序后取平均）",
        f"- 被测：`{BASE_MODEL}`（微调前） vs `{FT_MODEL}`（微调后）",
        f"- 样本：{summary['n']} 条（6 角色 × 5 类问题）",
        "",
        "## 平均得分（1~5 分）",
        "",
        "| 维度 | 微调前 | 微调后 | 提升 |",
        "|---|---|---|---|",
    ]
    for d in DIMENSIONS:
        b, f = dim_avg["base"][d], dim_avg["ft"][d]
        delta = None if (b is None or f is None) else round(f - b, 2)
        lines.append(f"| {DIM_LABEL[d]} | {fmt(b)} | {fmt(f)} | {('+' if delta and delta > 0 else '') + fmt(delta) if delta is not None else '—'} |")
    o = summary["overall"]
    lines.append(f"| **综合** | **{fmt(o['base'])}** | **{fmt(o['ft'])}** | **{('+' if o['delta'] and o['delta'] > 0 else '') + fmt(o['delta'])}** |")

    lines += ["", "## 按问题类型（三项均分）", "", "| 类型 | 微调前 | 微调后 | 提升 |", "|---|---|---|---|"]
    for t, v in summary["by_type"].items():
        delta = None if (v["base"] is None or v["ft"] is None) else round(v["ft"] - v["base"], 2)
        lines.append(f"| {t} | {fmt(v['base'])} | {fmt(v['ft'])} | {('+' if delta and delta > 0 else '') + fmt(delta) if delta is not None else '—'} |")

    lines += ["", "## 逐条得分", "", "| id | 角色 | 类型 | 微调前 | 微调后 |", "|---|---|---|---|---|"]
    for p in sorted(per_case, key=lambda x: x["id"]):
        lines.append(f"| {p['id']} | {p['character']} | {p['type']} | {fmt(p['base_overall'])} | {fmt(p['ft_overall'])} |")

    md_path = os.path.join(REPORT_DIR, "roleplay_judge.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f">> 汇总报告: {md_path}")
    print(f">> 综合均分: 微调前 {fmt(o['base'])} → 微调后 {fmt(o['ft'])}（{o['delta']:+}）")


if __name__ == "__main__":
    sys.exit(main())
