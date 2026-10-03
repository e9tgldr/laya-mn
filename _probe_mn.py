"""Does the BASE model already understand Mongolian, or does fine-tuning create it?

  A. Tokenization: Cyrillic Mongolian -> real subwords, or junk byte-pieces?
  B. Cross-lingual geometry: same meaning MN/EN/RU - is MN a neighbor of EN?
  C. Base vs tuned on the real MASSIVE mn-MN test set: top1/top5/top10 + gold rank.

Base = jhu-clsp/mmBERT-base (1833 languages), weights in model/multilingual/model.safetensors.
"""
import collections
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(ROOT, "out_mn")
BASE = os.path.join(ROOT, "model", "multilingual")


def main():
    from laya.common import build_model, QTYPES
    from safetensors.torch import load_file
    import mn_data

    cfg = json.load(open(os.path.join(ML, "rl_agent_config.json"), encoding="utf-8"))
    cfg["max_len"], cfg["head_max_len"] = 192, 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = mn_data.load_tokenizer(BASE)

    print("=" * 78)
    print("A. TOKENIZATION of Cyrillic Mongolian")
    print("=" * 78)
    for lang, text in [
        ("MN", "Энэ долоо хоног намайг таван цагт сэрээгээрэй"),
        ("MN", "Сургуулийн төлбөрийн нэхэмжлэх хавсаргав"),
        ("EN", "Remind me to wake up at five o'clock this week"),
        ("EN", "The school tuition invoice is attached"),
    ]:
        ids = tok(text)["input_ids"]
        pieces = tok.convert_ids_to_tokens(ids)
        print("\n%s (%d chars -> %d tokens, %.2f chars/token)"
              % (lang, len(text), len(ids), len(text) / max(1, len(ids))))
        print("   %s" % " | ".join(pieces[:26]))

    print()
    print("=" * 78)
    print("B. CROSS-LINGUAL GEOMETRY of the base encoder")
    print("=" * 78)
    base = build_model(cfg, encoder_dir=os.path.join(BASE, "encoder"))
    sd = load_file(os.path.join(BASE, "model.safetensors"))
    miss, unexp = base.load_state_dict(sd, strict=False)
    print("base loaded: missing=%d unexpected=%d" % (len(miss), len(unexp)))
    base.to(device).eval()

    pairs = [
        ("MN", "энэ долоо хоног намайг таван цагт сэрээгээрэй"),
        ("EN", "remind me to wake up at five o'clock this week"),
        ("RU", "напомни мне проснуться в пять часов на этой неделе"),
        ("MN2", "маргааш өглөө намайг зургаан цагт сэрээгээрэй"),
        ("EN2", "wake me up tomorrow morning at six o'clock"),
    ]
    vecs = {}
    with torch.no_grad():
        for tag, t in pairs:
            b = tok(t, return_tensors="pt")
            ids = b["input_ids"].to(device)
            am = b["attention_mask"].to(device)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
            with ctx:
                out = base.encoder(ids, attention_mask=am)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else (
                out[0] if isinstance(out, tuple) else out)
            h = h.float()
            m = am.unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            v = torch.nn.functional.normalize(v, dim=-1)
            vecs[tag] = v[0].cpu().numpy()

    tags = list(vecs)
    print("\ncosine similarity between sentence embeddings:")
    print("        " + "".join("%9s" % t for t in tags))
    for a in tags:
        print("%-8s%s" % (a, "".join("%9.3f" % float(vecs[a] @ vecs[b]) for b in tags)))
    print("\nMN<->EN vs MN<->RU tells whether Mongolian lands in the same")
    print("meaning space as English, or is just Cyrillic noise to the encoder.")

    print()
    print("=" * 78)
    print("C. BASE vs TUNED on the real MASSIVE mn-MN test set")
    print("=" * 78)
    items_path = os.path.join(ROOT, "items_192_128.pt")
    if not os.path.exists(items_path):
        print("items_192_128.pt missing - skipping")
        return
    test_items = torch.load(items_path, weights_only=False)["items"]["test"]
    print("test items available:", len(test_items))

    def evaluate(model, tag, limit=400):
        model.to(device).eval()
        subset = test_items[:limit]
        hits = collections.Counter()
        ranks, conf, nopt = [], [], []
        with torch.no_grad():
            for it in subset:
                ids = torch.tensor([it["ids"]], dtype=torch.long, device=device)
                am = torch.ones_like(ids)
                mk = torch.tensor([it["markers"]], dtype=torch.long, device=device)
                mm = torch.ones_like(mk, dtype=torch.bool)
                qt = torch.tensor([QTYPES["choice"]], dtype=torch.long, device=device)
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
                with ctx:
                    lg, _ = model(ids, am, mk, mm, qt)
                lg = lg.float().masked_fill(~mm, -1e4)[0, : len(it["keys"])]
                p = torch.softmax(lg, -1).cpu().numpy()
                gold = int(np.argmax(it["target"]))
                r = int(np.where(np.argsort(-p) == gold)[0][0])
                ranks.append(r)
                nopt.append(len(it["keys"]))
                if r == 0:
                    hits["top1"] += 1
                if r < 5:
                    hits["top5"] += 1
                if r >= 0 and r < min(10, len(it["keys"])):
                    hits["top10"] += 1
                if len(it["keys"]) >= 10:
                    hits["n10"] += 1
                    if r < 10:
                        hits["top10_of_ge10"] += 1
                conf.append(float(p.max()))
        n = len(subset)
        print("\n[%s]  %d decisions, mean options %.1f" % (tag, n, np.mean(nopt)))
        print("   top-1  %.4f" % (hits["top1"] / n))
        print("   top-5  %.4f" % (hits["top5"] / n))
        if hits["n10"]:
            print("   top-10 %.4f   (on the %d items that HAVE >=10 options)"
                  % (hits["top10_of_ge10"] / hits["n10"], hits["n10"]))
        print("   mean rank of gold: %.2f" % float(np.mean(ranks)))
        print("   mean max-confidence: %.4f" % float(np.mean(conf)))

    evaluate(base, "BASE  jhu-clsp/mmBERT-base (NOT trained on Mongolian)")
    tuned = build_model(cfg, encoder_dir=os.path.join(ML, "encoder"))
    tuned.load_state_dict(load_file(os.path.join(ML, "model.safetensors")), strict=False)
    evaluate(tuned, "TUNED 8-epoch Mongolian")


if __name__ == "__main__":
    main()
