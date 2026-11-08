"""Write outputs/synthetic/long_docs.json: 2,048 synthetic ~50K-token documents
(seed 0), so every 8K/32K packed window holds one document. Used by the
_long_documents() fake-PG configs in kimi_k3_fake_configs.py."""
import json
import os
import random

random.seed(0)
words = "the of and to in a is that for it as was with be by on not he this are or his from at which but have an they you were her she there been one all we their has would when if so no".split()
os.makedirs("outputs/synthetic", exist_ok=True)
with open("outputs/synthetic/long_docs.json", "w") as f:
    for i in range(2048):
        n = 0
        parts = []
        while n < 60000:
            w = random.choice(words) if random.random() < 0.7 else "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(random.randint(3, 9)))
            parts.append(w)
            n += len(w) + 1
        f.write(json.dumps({"text": " ".join(parts)}) + "\n")
