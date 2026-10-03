"""Post-hoc DOMAIN calibration for a trained Laya-MN checkpoint.

The training script fits its temperature on `train_items[::25][:400]` -- for the small
email set that is only ~115 items drawn from the TRAINING pool, which is both thin and
not held out. This script refits the temperature on a clean held-out split (default:
the validation split of whatever item file you pass) and writes a NEW model directory,
leaving the source checkpoint untouched.

Temperature scaling cannot change argmax, so accuracy is identical before and after;
the only thing it can move is ECE and NLL. That is exactly what this reports.

Usage:
  python mn_calib.py --model out_mn_email --out out_mn_email_cal \
      --items items_email.pt --fit-split validation --tag email
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))


def collect_logits(model_dir, items, device, dtype, batch=16):
    """Raw per-question logits + targets, exactly as the trainer's calibration block does."""
    from laya.common import build_model
    from safetensors.torch import load_file
    import mn_data
    from mn_train import collate_train_batch

    cfg = mn_data.load_cfg(model_dir)
    tok = mn_data.load_tokenizer(model_dir)
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=False)
    model.to(device).eval()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    preds = []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            b = collate_train_batch(chunk, pad_id)
            with torch.autocast("cuda", dtype=dtype):
                lg, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["marker_pos"].to(device), b["marker_mask"].to(device),
                              b["qtype"].to(device))
            lg = lg.float().cpu().numpy()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                preds.append((it["qtype"], lg[r, :k], it["target"]))
    return preds, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="source model dir (not modified)")
    ap.add_argument("--out", required=True, help="destination model dir")
    ap.add_argument("--items", default=os.path.join(ROOT, "items_email.pt"))
    ap.add_argument("--fit-split", default="validation", choices=["train", "validation"])
    ap.add_argument("--qid", default=None, help="fit on this question id only (e.g. intent)")
    ap.add_argument("--tag", default="calib")
    ap.add_argument("--out-reports", default=os.path.join(ROOT, "reports"))
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    from mn_train import fit_one_temp
    import mn_eval

    blob = torch.load(args.items, weights_only=False)
    fit_items = blob["items"][args.fit_split]
    if args.qid:
        fit_items = [i for i in fit_items if i["qid"] == args.qid]
        print("fit filter qid=%s -> %d items" % (args.qid, len(fit_items)), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # ---- 1. ECE BEFORE (source checkpoint, its own temperature) -----------------------
    before = mn_eval.run(args.model, args.items, "%s_before" % args.tag, args.out_reports)

    # ---- 2. refit temperature on the held-out split ----------------------------------
    preds, n_fit = collect_logits(args.model, fit_items, device, dtype)
    temps = []
    for qt in range(3):
        sel = [(z, t) for q, z, t in preds if q == qt]
        temps.append(fit_one_temp(sel) if sel else 1.2)
    old_temps = json.load(open(os.path.join(args.model, "rl_agent_config.json"),
                               encoding="utf-8")).get("temperature", [1.0, 1.0, 1.0])
    print("fit items used: %d" % n_fit, flush=True)
    print("temperature before (choice, score, noul): [%s]"
          % ", ".join("%.3f" % t for t in old_temps), flush=True)
    print("temperature after  (choice, score, noul): [%s]"
          % ", ".join("%.3f" % t for t in temps), flush=True)

    # ---- 3. write the new model dir (weights copied, checkpoints skipped) -------------
    if os.path.abspath(args.model) == os.path.abspath(args.out):
        raise SystemExit("refusing to overwrite the source checkpoint in place")
    shutil.copytree(args.model, args.out, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.pt"))
    cp = os.path.join(args.out, "rl_agent_config.json")
    cfg = json.load(open(cp, encoding="utf-8"))
    cfg["temperature"] = temps
    cfg["calibration"] = {
        "fit_split": args.fit_split,
        "fit_qid": args.qid or "all",
        "fit_items": n_fit,
        "source_model": os.path.abspath(args.model),
        "temperature_before": old_temps,
    }
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("wrote ->", args.out, flush=True)

    # ---- 4. ECE AFTER -----------------------------------------------------------------
    after = mn_eval.run(args.out, args.items, "%s_after" % args.tag, args.out_reports)

    rec = {
        "tag": args.tag,
        "items": args.items,
        "source_model": os.path.abspath(args.model),
        "calibrated_model": os.path.abspath(args.out),
        "fit_split": args.fit_split,
        "fit_qid": args.qid or "all",
        "fit_items": n_fit,
        "temperature_before": old_temps,
        "temperature_after": temps,
        "before": {"accuracy": before["accuracy_overall"], "ece": before["ece"],
                   "mean_nll": before["mean_nll"], "per_question": before["per_question"]},
        "after": {"accuracy": after["accuracy_overall"], "ece": after["ece"],
                  "mean_nll": after["mean_nll"], "per_question": after["per_question"]},
    }
    os.makedirs(args.out_reports, exist_ok=True)
    fp = os.path.join(args.out_reports, "calibration_%s.json" % args.tag)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    print("\n=== CALIBRATION RESULT (%s) ===" % args.tag, flush=True)
    print("accuracy : %.4f -> %.4f   (unchanged, temperature cannot move argmax)"
          % (before["accuracy_overall"], after["accuracy_overall"]), flush=True)
    print("ECE      : %.4f -> %.4f" % (before["ece"], after["ece"]), flush=True)
    print("mean NLL : %.4f -> %.4f" % (before["mean_nll"], after["mean_nll"]), flush=True)
    print("saved ->", fp, flush=True)


if __name__ == "__main__":
    main()
