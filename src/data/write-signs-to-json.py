import json

import pandas as pd

data_dir = "../../data"
labels_filename = "KARSL-502_Labels"
signs = pd.read_excel(
    f"{data_dir}/{labels_filename}.xlsx", usecols=["Sign-Arabic", "Sign-English"]
)
AR_WORDS, EN_WORDS = signs.to_dict(orient="list").items()
with open(f"{data_dir}/{labels_filename}.json", "w+", encoding="utf-8") as f:
    json.dump({"AR_WORDS": AR_WORDS[1], "EN_WORDS": EN_WORDS[1]}, f, ensure_ascii=False)

with open(f"{data_dir}/{labels_filename}.json", "r", encoding="utf-8") as f:
    signs = json.load(f)
AR_WORDS, EN_WORDS = signs["AR_WORDS"], signs["EN_WORDS"]
print(AR_WORDS)
