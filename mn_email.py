"""Build an EMAIL-FOCUSED item set from the cached MASSIVE mn-MN items.

Why not email-only: the scenario question has 18 options. If every training utterance is
an email, its gold is always `email`, so the scenario head simply learns to answer "email"
for everything and stops discriminating. That head is then worthless as an email/not-email
filter. So we keep the email scenario items (all of them) and add an equal number of
NON-email scenario items as negatives -- the scenario head stays live while every intent
question in the set is an email intent.

Set composition:
  * scenario question, gold == 'email'          -> all
  * intent   question, gold startswith email_    -> all   (4 intents)
  * scenario question, gold != 'email'          -> sampled 1:1 as negatives
"""
import argparse
import collections
import os
import random

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))


def build(src_blob, negatives=True, ratio=1.0, seed=42):
    rng = random.Random(seed)
    items = src_blob["items"]
    out = {}
    for split in ("train", "validation", "test"):
        it = items[split]
        em_scen = [i for i in it if i["qid"] == "scenario" and i["gold"] == "email"]
        em_int = [i for i in it if i["qid"] == "intent" and i["gold"].startswith("email_")]
        neg = [i for i in it if i["qid"] == "scenario" and i["gold"] != "email"]
        if negatives:
            n = min(len(neg), int(len(em_scen) * ratio))
            neg = rng.sample(neg, n)
        else:
            neg = []
        # shuffle so the two question types are interleaved through training
        sel = em_scen + em_int + neg
        rng.shuffle(sel)
        out[split] = sel
        print("%-11s email_scenario=%-5d email_intent=%-5d non_email_neg=%-5d TOTAL=%d"
              % (split, len(em_scen), len(em_int), len(neg), len(sel)), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "items_192_128.pt"))
    ap.add_argument("--dst", default=os.path.join(ROOT, "items_email.pt"))
    ap.add_argument("--no-negatives", action="store_true",
                    help="email-only (scenario head will collapse to always-email)")
    ap.add_argument("--ratio", type=float, default=1.0, help="negatives per email scenario item")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    blob = torch.load(args.src, weights_only=False)
    print("source:", args.src, "| splits:", {k: len(v) for k, v in blob["items"].items()}, flush=True)

    items = build(blob, negatives=not args.no_negatives, ratio=args.ratio, seed=args.seed)

    tr = items["train"]
    byq = collections.Counter(i["qid"] for i in tr)
    print("train by qid:", dict(byq), flush=True)
    intents = [i for i in tr if i["qid"] == "intent"]
    print("train email intents:", dict(collections.Counter(i["gold"] for i in intents)), flush=True)
    print("train intent option-counts:",
          dict(collections.Counter(len(i["keys"]) for i in intents)), flush=True)

    torch.save({"items": items, "max_len": blob["max_len"], "head_max_len": blob["head_max_len"],
                "simap": blob["simap"], "cfg": blob["cfg"]}, args.dst)
    print("saved ->", args.dst, flush=True)


if __name__ == "__main__":
    main()
