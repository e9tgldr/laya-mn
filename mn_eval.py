"""Evaluate a Laya checkpoint on the Mongolian MASSIVE test split.

Reports scenario + intent accuracy against their random baselines, plus ECE.
Usage:  python mn_eval.py --model <dir> --tag baseline
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(model_dir, items_path, tag, out_dir):
    from laya.common import build_model, ece_score, proper_reward
    from safetensors.torch import load_file
    import mn_data
    from mn_train import collate_train_batch

    cfg = mn_data.load_cfg(model_dir)
    tok = mn_data.load_tokenizer(model_dir)
    blob = torch.load(items_path, weights_only=False)
    test_items = blob["items"]["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=False)
    model.to(device).eval()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    stats = {}
    confs, corrects, losses = [], [], []
    import time
    t0 = time.time()
    n_cases = 0

    with torch.no_grad():
        for i in range(0, len(test_items), 16):
            chunk = test_items[i:i + 16]
            b = collate_train_batch(chunk, pad_id)
            dev = {k: v.to(device) for k, v in b.items()}
            ctx = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else torch.no_grad()
            with ctx:
                lg, _ = model(dev["input_ids"], dev["attention_mask"], dev["marker_pos"],
                              dev["marker_mask"], dev["qtype"])
            lg = lg.float().masked_fill(~dev["marker_mask"], -1e4)
            p = torch.softmax(lg, -1).cpu().numpy()
            lb = b["label"].numpy()
            tg = b["target"].numpy()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                pr = p[r, :k]
                # apply fitted temperature if present
                ti = it["qtype"]
                t = float(cfg.get("temperature", [1.0, 1.0, 1.0])[ti])
                if t != 1.0:
                    z = np.log(np.clip(pr, 1e-12, None)) / t
                    z -= z.max()
                    pr = np.exp(z)
                    pr /= pr.sum()
                pred = int(np.argmax(pr))
                ok = int(pred == lb[r])
                s = stats.setdefault(it["qid"], {"n": 0, "correct": 0})
                s["n"] += 1
                s["correct"] += ok
                confs.append(float(pr.max()))
                corrects.append(ok)
                losses.append(float(-np.log(np.clip(pr[lb[r]], 1e-12, None))))
                n_cases += 1

    res = {
        "tag": tag,
        "model_dir": model_dir,
        "n_decisions": n_cases,
        "per_question": {},
        "mean_nll": float(np.mean(losses)) if losses else None,
    }
    for qid, s in sorted(stats.items()):
        res["per_question"][qid] = {
            "accuracy": s["correct"] / s["n"],
            "n": s["n"],
        }
    res["accuracy_overall"] = float(np.mean(corrects)) if corrects else 0.0
    res["ece"] = float(ece_score(np.array(confs), np.array(corrects))) if confs else None
    res["random_baseline"] = {"scenario": 1.0 / 18, "intent": 0.25}
    res["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, "eval_%s.json" % tag)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print("saved ->", fp, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default=os.path.join(ROOT, "items_192_128.pt"))
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports"))
    a = ap.parse_args()
    run(a.model, a.items, a.tag, a.out)
