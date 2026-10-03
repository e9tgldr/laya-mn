"""Count MASSIVE mn-MN email-subset items in the cached item file (CPU only, no Laya)."""
import collections
import os

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
d = torch.load(os.path.join(ROOT, "items_192_128.pt"), weights_only=False)
items = d["items"]
simap = d["simap"]

print("max_len / head_max_len:", d["max_len"], d["head_max_len"])
print("18 scenarios:", sorted(simap.keys()))
print("email intents:", simap.get("email"))

for split in ("train", "validation", "test"):
    it = items[split]
    by_q = collections.Counter(i["qid"] for i in it)
    em = [i for i in it if i["gold"] == "email" and i["qid"] == "intent"]
    em_scen = [i for i in it if i["qid"] == "scenario"]
    em_scen_email = [i for i in em_scen if i["gold"] == "email"]
    print("\n[%s] total items=%d  by_qid=%s" % (split, len(it), dict(by_q)))
    print("   scenario items=%d of which gold=email: %d" % (len(em_scen), len(em_scen_email)))
    print("   email-intent items=%d" % len(em))
    print("   email intent dist:", dict(collections.Counter(i["gold"] for i in em)))
    print("   option-count dist on email-intent items:",
          dict(collections.Counter(len(i["keys"]) for i in em)))
    if em:
        t = em[0]
        print("   sample keys:", t["keys"])
        print("   sample text:", t["text"][:120])
