# -*- coding: utf-8 -*-
"""
压测：对 Ollama 服务连续发送 20 次请求，统计成功率与吞吐（tok/s）。
Ollama 原生 API 返回 eval_count（生成 token 数）与 eval_duration（纳秒），可精确计算吞吐。
产物：eval/reports/benchmark.json
"""
import json
import os
import time

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "eval", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL = "deepseek-r1-roleplay"
N_REQUESTS = 20
PROMPT = "给我讲一个关于机器人的短故事。"


def main():
    ok = 0
    total_tokens = 0
    total_eval_ns = 0
    details = []
    for i in range(N_REQUESTS):
        try:
            t0 = time.time()
            r = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": MODEL, "stream": False,
                      "messages": [{"role": "user", "content": PROMPT}],
                      "options": {"num_predict": 256}},
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
            ok += 1
            total_tokens += data.get("eval_count", 0)
            total_eval_ns += data.get("eval_duration", 0)
            details.append({"i": i, "eval_count": data.get("eval_count"),
                            "wall_s": round(time.time() - t0, 2)})
            print(f"[{i+1}/{N_REQUESTS}] ok, {data.get('eval_count')} tokens")
        except Exception as e:  # noqa: BLE001
            details.append({"i": i, "error": str(e)})
            print(f"[{i+1}/{N_REQUESTS}] FAILED: {e}")

    result = {
        "model": MODEL,
        "requests": N_REQUESTS,
        "success": ok,
        "success_rate": ok / N_REQUESTS,
        "total_tokens": total_tokens,
        "throughput_tok_per_s": round(total_tokens / (total_eval_ns / 1e9), 1) if total_eval_ns else 0,
        "details": details,
    }
    path = os.path.join(REPORT_DIR, "benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n成功 {ok}/{N_REQUESTS}，吞吐 {result['throughput_tok_per_s']} tok/s")
    print(f">> 结果已保存: {path}")


if __name__ == "__main__":
    main()
