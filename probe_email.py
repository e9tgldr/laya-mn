"""OFF-DISTRIBUTION PROBE: can the exported Laya-MN model route REAL Mongolian email prose
into ARBITRARY user-supplied categories (not MASSIVE scenarios)?

MASSIVE mn-MN is short voice-style utterances professionally translated from English.
This tests the actual transfer question for an email-sorting product.

Usage: venv\\Scripts\\python.exe probe_email.py
"""
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- the categories a real inbox would actually need (NOT MASSIVE scenarios) -------------
CATS = [
    "school_parent",     # parent/guardian writing about a child
    "school_staff",      # teacher/colleague internal message
    "government",        # ministry / khoroo / licence / official document
    "invoice_payment",   # money owed, e-barimt, bank transfer
    "vendor_spam",       # marketing, ads, cold outreach
    "personal",          # friends, family, social
]
CAT_CRIT = {c: None for c in CATS}

# ---- realistic Mongolian inbox lines (deliberately NOT MASSIVE voice-command style) -------
CASES = [
    ("school_parent",
     "Сайн байна уу. Манай хүү Дорж 3-р ангид сурдаг. Маргааш эцэг эхийн хурал болох уу? "
     "Хэрэв болбол хэдэн цагт эхлэхийг хэлж өгөөч."),
    ("school_staff",
     "Багш аа, 5-р ангийн математикийн шалгалтын дүнг Excel файлаар хавсаргалаа. "
     "Дүнгүүдийг системд оруулж өгнө үү."),
    ("government",
     "Таны ирүүлсэн хүсэлтийг хянан үзэж, 2026 оны 10 дугаар сарын 5-ны дотор "
     "хариу өгөхийг мэдэгдэж байна. Дахин ирүүлэх шаардлагагүй."),
    ("invoice_payment",
     "Эрхэм хамтран ажиллагч, 9 дүгээр сарын нэхэмжлэх 1,250,000 төгрөг. "
     "Төлбөрийг 10 дугаар сарын 1-ний дотор шилжүүлнэ үү. Дансны дугаар хавсралтад."),
    ("vendor_spam",
     "ЯАРАЛТАЙ! Зөвхөн өнөөдөр 70% хямдрал! Манай шинэ ухаалаг утсыг худалдан авбал "
     "бэлэг авах боломжтой. Дарахад л хангалттай!"),
    ("personal",
     "Сайн уу? Өчигдөр оройн хоолонд ирсэнд баярлалаа. Дараа долоо хоногт дахин уулзах уу? "
     "Гэртээ аюулгүй хүрээрэй."),
]

# control: a genuine MASSIVE-style utterance the model WAS trained on
CONTROL = ("энэ долоо хоног намайг таван цагт сэрээгээрэй", "alarm")


def score(model, tok, cfg, build_sequence, QTYPES, text, keys, ins, qtype, device):
    q = {"t": "choice", "ins": ins, "crit": {k: None for k in keys}}
    seq, markers = build_sequence(tok, text, q, cfg["max_len"], cfg["head_max_len"])
    n = len(keys)
    if len(markers) != n:
        return None, None, "truncation ate an option marker (%d markers vs %d options)" % (len(markers), n)
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
    return p, keys, None


def main():
    import argparse
    from laya.common import build_model, build_sequence, QTYPES

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out_mn_cal_email", help="model dir (absolute or under ROOT)")
    args = ap.parse_args()
    ml = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    with open(os.path.join(ml, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["max_len"], cfg["head_max_len"] = 192, 128

    import mn_data
    tok = mn_data.load_tokenizer(ml)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, encoder_dir=os.path.join(ml, "encoder"))
    from safetensors.torch import load_file
    missing = model.load_state_dict(load_file(os.path.join(ml, "model.safetensors")), strict=False)
    model.to(device).eval()
    print("device:", device, "| temperature:", cfg.get("temperature"), flush=True)
    print("missing keys:", len(missing.missing_keys), "| unexpected:", len(missing.unexpected_keys), flush=True)
    print("=" * 78, flush=True)

    ins = "Which mailbox folder does this message belong to?"

    # --- control: trained-domain sanity check
    p, keys, err = score(model, tok, cfg, build_sequence, QTYPES, CONTROL[0],
                         ["alarm", "weather", "news", "calendar"], ins, "scenario", device)
    print("[CONTROL trained-domain] gold=%s -> %s p=%.4f %s"
          % (CONTROL[1], keys[int(np.argmax(p))] if p is not None else "ERR",
             float(p.max()) if p is not None else 0.0, err or ""), flush=True)

    # --- the real question: arbitrary categories on email prose
    correct = 0
    confs = []
    print("\n[OFF-DISTRIBUTION] 6-way arbitrary inbox categories on email prose:", flush=True)
    for gold, text in CASES:
        p, keys, err = score(model, tok, cfg, build_sequence, QTYPES, text, CATS, ins, "scenario", device)
        if p is None:
            print("  %-16s ERROR %s" % (gold, err), flush=True)
            continue
        order = np.argsort(-p)[:3]
        hit = int(keys[int(np.argmax(p))] == gold)
        correct += hit
        confs.append(float(p.max()))
        print("  %-16s -> %-16s p=%.3f  %s   | top3: %s"
              % (gold, keys[int(np.argmax(p))], float(p.max()),
                 "OK " if hit else "MISS",
                 ", ".join("%s %.2f" % (keys[i], p[i]) for i in order)), flush=True)

    print("\noff-distribution top-1 accuracy: %d/%d = %.3f  (random = %.3f)"
          % (correct, len(CASES), correct / len(CASES), 1.0 / len(CATS)), flush=True)
    if confs:
        print("mean max-confidence: %.3f" % float(np.mean(confs)), flush=True)
        print("NOTE: confidence on untrained categories is not calibration - treat as ranking only.",
              flush=True)


if __name__ == "__main__":
    main()
