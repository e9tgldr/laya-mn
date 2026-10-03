"""Pipeline smoke test: validates sequence building, collate, forward/backward, VRAM and speed
WITHOUT the 647 MB weights (architecture-only random init). Run before the overnight job.
"""
import json
import os
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    from datasets import load_dataset
    from laya.common import build_model, proper_reward
    import mn_data
    from mn_train import collate_train_batch, fit_one_temp

    ml = os.path.join(ROOT, "model", "multilingual")
    cfg = mn_data.load_cfg(ml)
    tok = mn_data.load_tokenizer(ml)
    print("tokenizer vocab:", len(tok), "| encoder:", cfg["encoder"], flush=True)
    print("pad id:", tok.pad_token_id, "| mask id:", tok.mask_token_id, flush=True)

    simap = mn_data.scenario_intent_map("train")
    print("scenarios:", len(simap), "| intents:", sum(len(v) for v in simap.values()), flush=True)

    ds_s = load_dataset(mn_data.SCEN_DS, split="train").select(range(128))
    ds_i = load_dataset(mn_data.INT_DS, split="train").select(range(128))
    items, dropped = mn_data.build_items(ds_s, ds_i, simap, tok, cfg, 192, 128)
    print("built items:", len(items), "| dropped:", dropped, flush=True)
    assert items, "no items built"
    lens = [len(it["ids"]) for it in items]
    ks = [len(it["markers"]) for it in items]
    print("seq len min/mean/max: %d / %.0f / %d" % (min(lens), sum(lens) / len(lens), max(lens)), flush=True)
    print("options per item min/max:", min(ks), max(ks), flush=True)
    print("sample qid:", items[0]["qid"], "| options:", len(items[0]["keys"]),
          "| gold:", items[0]["gold"], flush=True)
    print("sample ids[:12]:", items[0]["ids"][:12], flush=True)

    # ---- memory / speed on architecture-only model
    cfg["max_len"], cfg["head_max_len"] = 192, 128
    model = build_model(cfg, encoder_dir=os.path.join(ml, "encoder"))
    n = sum(p.numel() for p in model.parameters())
    print("params: %.1fM" % (n / 1e6), flush=True)
    try:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as e:  # noqa: BLE001
        print("no encoder checkpointing:", e, flush=True)
    model.head_checkpointing = True
    dev = torch.device("cuda")
    model.to(dev).train()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for mb in (8, 4, 2):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            chunk = items[:mb]
            t0 = time.time()
            for it in range(3):
                b = collate_train_batch(chunk, pad_id)
                d = {k: v.to(dev) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, act = model(d["input_ids"], d["attention_mask"], d["marker_pos"],
                                        d["marker_mask"], d["qtype"])
                logits = logits.float()
                mask = d["marker_mask"]
                kk = mask.sum(-1, keepdim=True).float()
                eps = torch.randn((4,) + logits.shape, device=dev) * 0.3 * mask
                eps = (eps - eps.sum(-1, keepdim=True) / kk) * mask
                z = logits.detach().unsqueeze(0) + eps
                q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
                r = proper_reward(q, d["target"].unsqueeze(0), d["qtype"], mask, w_sph=0.75)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
                logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * 0.3 ** 2)
                loss = -(adv * logp).mean() - (d["target"] * torch.log_softmax(
                    logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            dt = (time.time() - t0) / 3
            peak = torch.cuda.max_memory_allocated() / 1e9
            print("mb=%d OK | %.3f s/step | peak VRAM %.2f GB | loss %.3f" % (mb, dt, peak, loss.item()),
                  flush=True)
            est = len(items)  # perf only; real run uses full set
            print("   -> epoch of 23028 items would be ~%.1f min at mb=%d"
                  % (23028 / mb * dt / 60, mb), flush=True)
            break
        except torch.cuda.OutOfMemoryError:
            print("mb=%d OOM (expected on 6GB, auto-halving works)" % mb, flush=True)
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

    print("temperature fitter:", round(fit_one_temp([([1.0, 0.0], [1.0, 0.0])] * 20), 3), flush=True)
    print("SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
