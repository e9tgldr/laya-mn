"""Build Laya-format RLCD training items from MASSIVE Mongolian (mn-MN).

State = one Mongolian user utterance.
Q1 "scenario" = choice over all 18 MASSIVE scenarios      (coarse routing)
Q2 "intent"   = choice over the intents of the GOLD scenario (fine routing, 3-4 options)

Gold answers are one-hot distributions; Laya's proper-scoring reward consumes them directly.
MASSIVE mn-MN is the professionally-translated Mongolian half of MASSIVE (CC-BY-4.0).
"""
import json
import os
import random
import time
import collections

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "convaiinnovations/laya"
SCEN_DS = "SetFit/amazon_massive_scenario_mn-MN"
INT_DS = "SetFit/amazon_massive_intent_mn-MN"


def ensure_model(root=ROOT, allow=None):
    """Download only the multilingual checkpoint (retry loop: HF CDN stalls from Mongolia)."""
    from huggingface_hub import snapshot_download
    d = os.path.join(root, "model")
    marker = os.path.join(d, "multilingual", "model.safetensors")
    if os.path.exists(marker):
        return d
    pats = allow or ["multilingual/*", "rl_agent_config.json", "MODEL_LICENSE"]
    last = None
    for attempt in range(8):
        try:
            snapshot_download(MODEL_ID, local_dir=d, allow_patterns=pats, max_workers=4)
            if os.path.exists(marker):
                return d
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError("model download failed: %r" % (last,))


def load_cfg(model_dir):
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    try:
        from laya.agent import _fix_tokenizer_config
        _fix_tokenizer_config(model_dir)
    except Exception:  # noqa: BLE001
        pass
    return AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))


def scenario_intent_map(split="train"):
    """Derive scenario -> [intents] from the id-aligned MASSIVE subsets."""
    from datasets import load_dataset
    ds_s = load_dataset(SCEN_DS, split=split)
    ds_i = load_dataset(INT_DS, split=split)
    m = collections.defaultdict(set)
    for a, b in zip(ds_s, ds_i):
        m[a["label_text"]].add(b["label_text"])
    return {k: sorted(v) for k, v in m.items()}


def build_items(ds_s, ds_i, simap, tok, cfg, max_len, head_max_len):
    from laya.common import build_sequence, render_options, QTYPES

    scenarios = sorted(simap.keys())
    scen_crit = {s: None for s in scenarios}
    items, dropped = [], 0

    for a, b in zip(ds_s, ds_i):
        state = a["text"]
        gold_scen = a["label_text"]
        gold_int = b["label_text"]
        ints = simap.get(gold_scen)
        if not ints or gold_int not in ints:
            dropped += 1
            continue

        qs = [
            ("scenario", "Which application scenario does this user request belong to?",
             scen_crit, gold_scen),
            ("intent", "The request is about %s. Which specific action does the user want?" % gold_scen,
             {k: None for k in ints}, gold_int),
        ]
        for qid, ins, crit, gold in qs:
            q = {"t": "choice", "ins": ins, "crit": crit}
            keys = list(crit.keys())
            if gold not in keys:
                dropped += 1
                continue
            k = len(render_options(q))
            if k < 2:
                # degenerate question (a scenario with a single intent): no learning signal,
                # its reward is maximised by always answering the only option
                dropped += 1
                continue
            seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
            if len(markers) != k:  # truncation ate an option marker
                dropped += 1
                continue
            target = [1.0 if i == keys.index(gold) else 0.0 for i in range(k)]
            items.append({
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES["choice"],
                "target": target,
                "label": target.index(1.0),
                "qid": qid,
                "keys": keys,
                "gold": gold,
                "text": state,
            })
    return items, dropped


def main():
    import argparse
    import torch
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--head-max-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows per split")
    args = ap.parse_args()

    model_dir = ensure_model()
    ml_dir = os.path.join(model_dir, "multilingual")
    cfg = load_cfg(ml_dir)
    tok = load_tokenizer(ml_dir)
    print("encoder        :", cfg["encoder"], flush=True)
    print("tokenizer len  :", len(tok), flush=True)

    simap = scenario_intent_map("train")
    print("scenarios      :", len(simap), flush=True)
    print("intents        :", sum(len(v) for v in simap.values()), flush=True)
    dist = collections.Counter(len(v) for v in simap.values())
    print("intents-per-scenario distribution:", dict(sorted(dist.items())), flush=True)
    degenerate = [k for k, v in simap.items() if len(v) < 2]
    if degenerate:
        print("scenarios with a single intent (their intent question is dropped):", degenerate,
              flush=True)

    out = {}
    for split in ("train", "validation", "test"):
        ds_s = load_dataset(SCEN_DS, split=split)
        ds_i = load_dataset(INT_DS, split=split)
        if split != "train" and args.limit:
            ds_s = ds_s.select(range(min(args.limit, len(ds_s))))
            ds_i = ds_i.select(range(min(args.limit, len(ds_i))))
        items, dropped = build_items(ds_s, ds_i, simap, tok, cfg, args.max_len, args.head_max_len)
        out[split] = items
        print("%-10s rows=%-6d items=%-6d dropped=%d" % (split, len(ds_s), len(items), dropped),
              flush=True)

    tp = os.path.join(ROOT, "items_%d_%d.pt" % (args.max_len, args.head_max_len))
    torch.save({"items": out, "max_len": args.max_len, "head_max_len": args.head_max_len,
                "simap": simap, "cfg": cfg}, tp)
    print("saved ->", tp, flush=True)


if __name__ == "__main__":
    main()
