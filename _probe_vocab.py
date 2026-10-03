"""Is Mongolian actually in the tokenizer's vocabulary, or only its Cyrillic characters?

If Mongolian subwords exist -> it was in the training mixture.
If only single Cyrillic chars/bytes tokenize cleanly -> it was never trained, and the
earlier MN<->EN embedding similarity comes from Cyrillic proximity + transfer.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "model", "multilingual")
sys.path.insert(0, ROOT)

import mn_data

tok = mn_data.load_tokenizer(BASE)
vocab = tok.get_vocab()
vocab_lower = {k.lower().replace("\u2581", ""): k for k in vocab}

print("vocab size:", len(vocab))

# whole Mongolian words we care about - do they exist as single tokens?
words = [
    "баярлалаа", "сургууль", "сургуулийн", "төлбөр", "нэхэмжлэх", "багш",
    "сурагч", "хүүхэд", "энэ", "долоо", "хоног", "маргааш", "өглөө",
    "монгол", "хэл", "цаг", "өдөр", "хурлын", "мэдэгдэл", "хянах",
]
print("\n== whole-word token test ==")
hits = 0
for w in words:
    found = vocab_lower.get(w.lower())
    if found:
        hits += 1
    print("  %-14s %s" % (w, "IN VOCAB as '%s'" % found if found else "--- not a token"))

print("\n%d/%d whole Mongolian words present as single tokens" % (hits, len(words)))

# does every Cyrillic char at least tokenize without <unk>?
print("\n== character coverage (should be no <unk>) ==")
sample = "Баярлалаа. Сургуулийн төлбөрийн нэхэмжлэх хавсаргав. Үүнийг хянана уу?"
ids = tok(sample)["input_ids"]
pieces = tok.convert_ids_to_tokens(ids)
unks = [p for p in pieces if p == "<unk>"]
print("  %d chars -> %d tokens, chars/token = %.2f" % (len(sample), len(ids), len(sample) / len(ids)))
print("  <unk> tokens: %d" % len(unks))
single = sum(1 for p in pieces if len(p.replace("\u2581", "")) == 1)
print("  single-character pieces: %d / %d  (%.0f%%)" % (single, len(pieces), 100.0 * single / len(pieces)))
print("  pieces: %s" % " ".join(pieces[:30]))

# compare with another Cyrillic language the model definitely knows
print("\n== control: Russian + Ukrainian (known-covered Cyrillic) ==")
for lang, text in [
    ("RU", "Здравствуйте, счёт за обучение прилагается. Проверьте пожалуйста."),
    ("UK", "Вітаю, рахунок за навчання додається. Перевірте будь ласка."),
]:
    i = tok(text)["input_ids"]
    p = tok.convert_ids_to_tokens(i)
    s = sum(1 for x in p if len(x.replace("\u2581", "")) == 1)
    print("  %s: %d chars -> %d tokens (%.2f chars/token), single-char pieces %d/%d"
          % (lang, len(text), len(i), len(text) / len(i), s, len(p)))

# English control
print("\n== control: English ==")
for text in ["The school tuition invoice is attached. Please review it.",
             "Thank you very much, see you tomorrow morning."]:
    i = tok(text)["input_ids"]
    p = tok.convert_ids_to_tokens(i)
    s = sum(1 for x in p if len(x.replace("\u2581", "")) == 1)
    print("  %d chars -> %d tokens (%.2f chars/token), single-char pieces %d/%d"
          % (len(text), len(i), len(text) / len(i), s, len(p)))
