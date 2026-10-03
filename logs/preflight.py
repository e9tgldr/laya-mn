import os, torch, mn_data
from laya.common import build_model
from safetensors.torch import load_file
ml = os.path.join(r'C:\Users\hi\Documents\laya-mn\model', 'multilingual')
cfg = mn_data.load_cfg(ml)
m = build_model(cfg, encoder_dir=os.path.join(ml, 'encoder'))
sd = load_file(os.path.join(ml, 'model.safetensors'))
missing, unexpected = m.load_state_dict(sd, strict=False)
print('PREFLIGHT keys=%d missing=%d unexpected=%d' % (len(sd), len(missing), len(unexpected)))
print('PREFLIGHT missing_sample=%s' % (missing[:5],))
print('PREFLIGHT OK' if len(missing) == 0 else 'PREFLIGHT WARN: untrained params present')
