"""Single-GPU RLCD fine-tune of Laya-multilingual on Mongolian (MASSIVE mn-MN).

Faithful port of the official 2xT4 notebook recipe (notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
to one 6 GB GPU: GRPO-style REINFORCE on a strictly proper scoring reward + soft cross-entropy
guidance, then post-hoc temperature calibration.

Verified recipe constants kept from upstream:
  EPOCHS 4, GROUP_SIZE 4, LR_ENCODER 2.5e-5, LR_HEAD 1e-4, sigma 0.4 -> 0.1, w_sph 0.75, CE weight 1.0
"""
import argparse
import gc
import json
import math
import os
import random
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- batching
def collate_train_batch(items, pad_id):
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    """Temperature that makes reported probabilities honest (upstream LBFGS fitter)."""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    try:
        opt.step(closure)
        return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())
    except Exception:  # noqa: BLE001
        return 1.0


@torch.no_grad()
def evaluate(model, items, device, dtype, pad_id, batch=16):
    model.eval()
    correct = n = 0
    by_qid = {}
    probs_all, corr_all = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        b = collate_train_batch(chunk, pad_id)
        with torch.autocast("cuda", dtype=dtype):
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["marker_pos"].to(device), b["marker_mask"].to(device),
                              b["qtype"].to(device))
        logits = logits.float().masked_fill(~b["marker_mask"].to(device), -1e4)
        p = torch.softmax(logits, -1).cpu().numpy()
        lb = b["label"].numpy()
        for r, it in enumerate(chunk):
            k = len(it["markers"])
            pred = int(np.argmax(p[r, :k]))
            conf = float(p[r, :k].max())
            ok = int(pred == lb[r])
            s = by_qid.setdefault(it["qid"], [0, 0])
            s[0] += ok
            s[1] += 1
            correct += ok
            n += 1
            probs_all.append(conf)
            corr_all.append(ok)
    model.train()
    acc = correct / max(1, n)
    per = {k: v[0] / max(1, v[1]) for k, v in sorted(by_qid.items())}
    return {"accuracy": acc, "n": n, "per_qid": per,
            "mean_conf": float(np.mean(probs_all)) if probs_all else 0.0}


def start_keep_awake():
    """Hold the system awake so a long unattended run is not frozen by sleep.

    A CUDA context does not survive a Windows sleep/resume cycle: training stalls
    with the GPU pinned at 100% and no further progress. Display-off is still
    allowed - we request only ES_SYSTEM_REQUIRED, not ES_DISPLAY_REQUIRED.
    """
    import threading

    def _run():
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            k = ctypes.windll.kernel32
            while True:
                k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
                time.sleep(30)
        except Exception as e:  # noqa: BLE001
            print("keep-awake unavailable:", e, flush=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print("keep-awake thread started (system sleep blocked; display may still blank)", flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--out", default=os.path.join(ROOT, "out_mn"))
    ap.add_argument("--max-hours", type=float, default=9.0, help="hard wall-clock budget")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--stop-file", default=None, help="stop gracefully when this path exists")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    start_keep_awake()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    from safetensors.torch import load_file, save_file
    from laya.common import build_model, proper_reward
    import mn_data

    model_dir = mn_data.ensure_model()
    ml_dir = os.path.join(model_dir, "multilingual")
    cfg = mn_data.load_cfg(ml_dir)
    tok = mn_data.load_tokenizer(ml_dir)

    ip = args.items or os.path.join(ROOT, "items_192_128.pt")
    blob = torch.load(ip, weights_only=False)
    cfg["max_len"] = blob["max_len"]
    cfg["head_max_len"] = blob["head_max_len"]
    train_items = blob["items"]["train"]
    val_items = blob["items"]["validation"]
    random.shuffle(train_items)
    print("train items %d | val items %d | max_len %d head_max_len %d"
          % (len(train_items), len(val_items), cfg["max_len"], cfg["head_max_len"]), flush=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True

    # architecture from saved encoder config; ALL weights from the shipped state dict
    model = build_model(cfg, encoder_dir=os.path.join(ml_dir, "encoder"))
    sd = load_file(os.path.join(ml_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("state_dict loaded | missing=%d unexpected=%d" % (len(missing), len(unexpected)), flush=True)
    if missing:
        print("  MISSING:", missing[:8], flush=True)

    try:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as e:  # noqa: BLE001
        print("encoder checkpointing unavailable:", e, flush=True)
    model.head_checkpointing = True
    model.to(device)

    start_epoch = 0
    best_acc = -1.0
    if args.resume:
        ck_path = os.path.join(args.out, "last.pt")
        if os.path.exists(ck_path):
            ck = torch.load(ck_path, weights_only=False, map_location=device)
            model.load_state_dict(ck["model"])
            start_epoch = int(ck.get("epoch", 0))
            print("RESUMED from last.pt at epoch %d (val acc %.4f)"
                  % (start_epoch, ck.get("val", {}).get("accuracy", -1.0)), flush=True)
            bp = os.path.join(args.out, "best.pt")
            if os.path.exists(bp):
                bc = torch.load(bp, weights_only=False, map_location="cpu")
                best_acc = float(bc.get("val", {}).get("accuracy", -1.0))
                print("  existing best.pt kept as baseline: %.4f (epoch %d)"
                      % (best_acc, bc.get("epoch", -1)), flush=True)
        else:
            print("RESUME requested but last.pt missing -> starting fresh", flush=True)

    enc_p = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_p = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    print("encoder params %.1fM | head params %.1fM"
          % (sum(p.numel() for p in enc_p) / 1e6, sum(p.numel() for p in head_p) / 1e6), flush=True)

    opt = torch.optim.AdamW([{"params": enc_p, "lr": args.lr_encoder},
                             {"params": head_p, "lr": args.lr_head}], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "fp16"))

    steps_per_epoch = max(1, len(train_items) // (args.micro_batch * args.grad_accum))
    total_updates = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_updates), eta_min=1e-6)

    os.makedirs(args.out, exist_ok=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    history = []
    if args.resume:
        hp = os.path.join(args.out, "train_history.json")
        if os.path.exists(hp):
            try:
                with open(hp, encoding="utf-8") as f:
                    history = json.load(f)
                print("loaded %d prior epoch records" % len(history), flush=True)
            except Exception as e:  # noqa: BLE001
                print("could not read prior history:", e, flush=True)
    mb = args.micro_batch

    def stop_requested():
        return bool(args.stop_file) and os.path.exists(args.stop_file)

    stopped = False
    for epoch in range(start_epoch, start_epoch + args.epochs):
        random.shuffle(train_items)
        ep_loss = 0.0
        nb = 0
        accum = 0
        opt.zero_grad(set_to_none=True)
        progress = (epoch - start_epoch) / max(1, args.epochs - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress

        bi = 0
        while bi < len(train_items):
            chunk = train_items[bi:bi + mb]
            if not chunk:
                break
            b = collate_train_batch(chunk, pad_id)
            dev = {k: v.to(device) for k, v in b.items()}
            r = None
            try:
                with torch.autocast("cuda", dtype=dtype):
                    logits, act = model(dev["input_ids"], dev["attention_mask"],
                                        dev["marker_pos"], dev["marker_mask"], dev["qtype"])
                logits = logits.float()
                mask = dev["marker_mask"]
                kk = mask.sum(-1, keepdim=True).float()
                target = dev["target"]

                # 1. G noisy logit distributions, zero-mean projected over the question's options
                eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
                eps = (eps - eps.sum(-1, keepdim=True) / kk) * mask
                z = logits.detach().unsqueeze(0) + eps
                q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

                # 2. strictly proper scoring reward -> GRPO-style advantage
                with torch.no_grad():
                    r = proper_reward(q, target.unsqueeze(0), dev["qtype"], mask,
                                      w_sph=0.75, w_rps=1.0)
                    adv = r - r.mean(0, keepdim=True)
                    adv = adv / (adv.std() + 1e-6)

                # 3. REINFORCE on the noise + full soft cross-entropy guidance
                logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
                loss_rl = -(adv * logp).mean()
                loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
                loss = (loss_rl + 1.0 * loss_ce) / args.grad_accum

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except torch.cuda.OutOfMemoryError:
                if mb <= 1:
                    raise
                print("OOM at micro_batch=%d -> halving, retrying same chunk" % mb, flush=True)
                opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                mb = max(1, mb // 2)
                continue  # retry this chunk with the smaller batch

            bi += mb
            accum += 1
            if accum % args.grad_accum == 0 or bi >= len(train_items):
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            ep_loss += loss.item() * args.grad_accum
            nb += 1
            if nb % 50 == 0:
                el = time.time() - t0
                rw = r.mean().item() if r is not None else float("nan")
                print("  ep%d step %d loss %.4f reward %.3f lr %.2e mb %d elapsed %.1f min"
                      % (epoch + 1, nb, loss.item() * args.grad_accum,
                         rw, sched.get_last_lr()[0], mb, el / 60), flush=True)
                if el > args.max_hours * 3600:
                    print("WALL CLOCK BUDGET HIT -> stopping early", flush=True)
                    stopped = True
                    break
                if stop_requested():
                    print("STOP FILE SEEN -> wrapping up this epoch and stopping", flush=True)
                    stopped = True
                    break

        vm = evaluate(model, val_items, device, dtype, pad_id)
        print("=== epoch %d done | avg loss %.4f | val acc %.4f (scenario %.3f intent %.3f) | %.1f min"
              % (epoch + 1, ep_loss / max(1, nb), vm["accuracy"],
                 vm["per_qid"].get("scenario", 0.0), vm["per_qid"].get("intent", 0.0),
                 (time.time() - t0) / 60), flush=True)
        history.append({"epoch": epoch + 1, "loss": ep_loss / max(1, nb), "val": vm})

        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val": vm, "cfg": cfg},
                   os.path.join(args.out, "last.pt"))
        if vm["accuracy"] > best_acc:
            best_acc = vm["accuracy"]
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val": vm, "cfg": cfg},
                       os.path.join(args.out, "best.pt"))
            print("  new best val acc %.4f -> best.pt" % best_acc, flush=True)
        if time.time() - t0 > args.max_hours * 3600:
            print("WALL CLOCK BUDGET HIT (epoch boundary)", flush=True)
            stopped = True
        if stop_requested():
            print("STOP FILE SEEN (epoch boundary)", flush=True)
            stopped = True
        if stopped:
            break

    # ---------------------------------------------------------------- calibration + export
    ck = torch.load(os.path.join(args.out, "best.pt"), weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print("fitting calibration temperatures on the EXPORTED weights (epoch %d)..."
          % ck["epoch"], flush=True)
    calib = train_items[::25][:400]
    preds = []
    with torch.no_grad():
        for i in range(0, len(calib), 16):
            chunk = calib[i:i + 16]
            b = collate_train_batch(chunk, pad_id)
            with torch.autocast("cuda", dtype=dtype):
                lg, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["marker_pos"].to(device), b["marker_mask"].to(device),
                              b["qtype"].to(device))
            lg = lg.float().cpu().numpy()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                preds.append((it["qtype"], lg[r, :k], it["target"]))
    temps = []
    for qt in range(3):
        sel = [(z, t) for q, z, t in preds if q == qt]
        temps.append(fit_one_temp(sel) if sel else 1.2)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps], flush=True)

    model.to("cpu")
    sd_out = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd_out, os.path.join(args.out, "model.safetensors"))
    os.makedirs(os.path.join(args.out, "encoder"), exist_ok=True)
    model.encoder.config.save_pretrained(os.path.join(args.out, "encoder"))
    tok.save_pretrained(os.path.join(args.out, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-multilingual-mn"
    cfg["temperature"] = temps
    with open(os.path.join(args.out, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.out, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("DONE -> %s (best val acc %.4f, epoch %d)" % (args.out, best_acc, ck["epoch"]), flush=True)


if __name__ == "__main__":
    main()
