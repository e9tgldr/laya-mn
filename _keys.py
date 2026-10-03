import json, os
from safetensors import safe_open

BASE = r'C:\Users\hi\Documents\laya-mn\model\multilingual'
p = os.path.join(BASE, 'model.safetensors')
with safe_open(p, framework='pt') as f:
    keys = list(f.keys())
print('total tensors:', len(keys))
pref = {}
for k in keys:
    root = k.split('.')[0]
    pref.setdefault(root, []).append(k)
for root, ks in sorted(pref.items(), key=lambda kv: -len(kv[1])):
    print('%-22s %5d  e.g. %s' % (root, len(ks), ks[0]))
print()
print('cfg:', json.dumps(json.load(open(os.path.join(BASE, 'rl_agent_config.json'))), indent=None)[:400])
