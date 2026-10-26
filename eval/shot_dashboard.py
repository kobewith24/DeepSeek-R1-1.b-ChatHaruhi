# -*- coding: utf-8 -*-
"""截图 EvalScope 生成的报告页中的环图（sunburst），拼成微调前后对比仪表盘"""
import io
import os
import sys

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"D:\DeepSeek-R1-FT"
OUT_DIR = os.path.join(BASE, "eval", "reports")
FINAL = os.path.join(OUT_DIR, "dashboard.png")

PAGES = {
    "base": {
        "html": os.path.join(BASE, "eval", "dashboard_outputs", "combined_base", "reports", "report.html"),
        "title": "微调前：deepseek-r1:1.5b（GSM8K 20% / ARC 30%）",
    },
    "ft": {
        "html": os.path.join(BASE, "eval", "dashboard_outputs", "combined_ft", "reports", "report.html"),
        "title": "微调后：deepseek-r1-roleplay（GSM8K 60% / ARC 35%）",
    },
}


def font(size):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        p = os.path.join(r"C:\Windows\Fonts", name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def shoot(key, cfg):
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1400, "height": 1100}, device_scale_factor=2)
        pg.goto("file:///" + cfg["html"], wait_until="networkidle", timeout=60000)
        pg.wait_for_timeout(3500)
        # 精确定位环图：plotly 每个图都有空的 sunburstlayer 占位，取有实际内容的那个
        handle = pg.evaluate_handle(
            "() => { const g = [...document.querySelectorAll('g.sunburstlayer')]"
            ".find(el => el.childElementCount > 0);"
            "return g ? g.closest('.js-plotly-plot') : null; }")
        el = handle.as_element()
        el.scroll_into_view_if_needed()
        pg.wait_for_timeout(600)
        buf = el.screenshot()
        b.close()
    return Image.open(io.BytesIO(buf)).convert("RGB")


def main():
    imgs = {k: shoot(k, v) for k, v in PAGES.items()}

    # 统一宽度
    w = max(im.width for im in imgs.values())
    imgs = {k: im.resize((w, int(im.height * w / im.width))) for k, im in imgs.items()}

    pad, header_h, title_h = 30, 90, 40
    cols = len(imgs)
    cell_w = w
    body_h = max(im.height for im in imgs.values())
    total_w = pad * (cols + 1) + cell_w * cols
    total_h = header_h + title_h + body_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), "#0d1117")
    d = ImageDraw.Draw(canvas)
    d.text((pad, 26), "EvalScope 评测报告 · 微调前后对比（图表由 EvalScope 自动生成）",
           font=font(30), fill="#c9d1d9")

    x = pad
    for k in ("base", "ft"):
        im = imgs[k]
        d.text((x, header_h + 6), PAGES[k]["title"], font=font(22), fill="#8b949e")
        canvas.paste(im, (x, header_h + title_h))
        x += cell_w + pad

    canvas.save(FINAL)
    print("saved", FINAL, canvas.size)


if __name__ == "__main__":
    main()
