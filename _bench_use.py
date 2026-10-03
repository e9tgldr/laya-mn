"""Inference speed + VRAM measurement for the exported laya-mn model.

Measures ONE classification = one text + N options scored in a single forward pass.
This is what a product actually calls, so it is the number that decides use cases.

Usage: venv\Scripts\python.exe _bench_use.py
"""
import json
import os
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(ROOT, "out_mn")

CATS = ["invoice_payment", "school", "government", "vendor_spam", "personal"]

CASES = {
    "SHORT (4 words)": "Re: tөслийн явц",
    "MEDIUM (~20 words)": ("Сайн байна уу, 10-р сарын сургуулийн төлбөрийн нэхэмжлэхийг "
                           "хавсаргав. Төлбөрийг сарын 25-ны дотор төлнө үү."),
    "LONG (~60 words)": ("Сайн байна уу. Манай сургуулийн 2026-2027 оны хичээлийн жилийн "
                         "эхний улирлын төлбөрийн нэхэмжлэхийг хавсаргаж байна. Нэхэмжлэхэд "
                         "дундаж сургалтын төлбөр, хоолны зардал, номын сангийн хураамж "
                         "тусгагдсан болно. Төлбөрийг сарын 25-ны дотор багтааж шилжүүлнэ үү. "
                         "Хэрэв асуулт байвал сургалтын албатай холбогдоно уу. Баярлалаа."),
}


def score(model, tok, cfg, build_sequence, QTYPES, text, keys, ins, device):
    q = {"t": "choice", "ins": ins, "crit": {k: None for k in keys}}
    seq, markers = build_sequence(tok, text, q, cfg["max_len"], cfg["head_max_len"])
    n = len(keys)
    if len(markers) != n:
        return None, "truncation ate an option marker (%d vs %d)" % (len(markers), n)
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    am = torch.ones_like(ids)
    mp = torch.tensor([markers], dtype=torch.long, device=device)
    mm = torch.ones_like(mp, dtype=torch.bool)
    qt = torch.tensor([QTYPES["choice"]], dtype=torch.long, device=device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
    with torch.no_grad(), ctx:
        lg, _ = model(ids, am, mp, mm, qt)
    lg = lg.float().masked_fill(~mm, -1e4)[0, :n]
    p = torch.softmax(lg, -1).cpu().numpy()
    t = float(cfg.get("temperature", [1.0, 1.0, 1.0])[QTYPES["choice"]])
    if t != 1.0:
        z = np.log(np.clip(p, 1e-12, None)) / t
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
    return p, None


def main():
    from laya.common import build_model, build_sequence, QTYPES
    from safetensors.torch import load_file

    with open(os.path.join(ML, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["max_len"], cfg["head_max_len"] = 192, 128

    import mn_data
    tok = mn_data.load_tokenizer(ML)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, encoder_dir=os.path.join(ML, "encoder"))
    model.load_state_dict(load_file(os.path.join(ML, "model.safetensors")), strict=False)
    model.to(device).eval()

    params = sum(p.numel() for p in model.parameters())
    print("device: %s | params: %.1fM | max_len: %s | options in test: %d"
          % (device, params / 1e6, cfg["max_len"], len(CATS)))
    print("temperature (choice): %s" % cfg.get("temperature", [None])[0])
    print("-" * 78)

    ins = "Which mailbox folder does this message belong to?"
    for label, text in CASES.items():
        p, err = score(model, tok, cfg, build_sequence, QTYPES, text, CATS, ins, device)
        if p is None:
            print("%-18s ERROR %s" % (label, err))
            continue
        # warmup, then timed run
        for _ in range(3):
            score(model, tok, cfg, build_sequence, QTYPES, text, CATS, ins, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        N = 20
        for _ in range(N):
            score(model, tok, cfg, build_sequence, QTYPES, text, CATS, ins, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) / N * 1000
        top = CATS[int(np.argmax(p))]
        print("%-18s %7.1f ms  | %6.0f /sec | top=%s p=%.3f"
              % (label, ms, 1000.0 / ms, top, float(p.max())))

    if device.type == "cuda":
        print("-" * 78)
        print("peak VRAM this run: %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
        print("model weights fp32 would be ~%.2f GB" % (params * 4 / 1024**3))


if __name__ == "__main__":
    main()
