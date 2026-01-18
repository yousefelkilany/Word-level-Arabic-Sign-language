import pandas as pd
import json

data_dir = "../../data"
labels_filename = "KARSL-502_Labels"
words = pd.read_excel(
    f"{data_dir}/{labels_filename}.xlsx", usecols=["Sign-Arabic", "Sign-English"]
)
AR_WORDS, EN_WORDS = words.to_dict(orient="list").items()
with open(f"{data_dir}/{labels_filename}.json", "w+", encoding="utf-8") as f:
    json.dump({"AR_WORDS": AR_WORDS[1], "EN_WORDS": EN_WORDS[1]}, f, ensure_ascii=False)

with open(f"{data_dir}/{labels_filename}.json", "r", encoding="utf-8") as f:
    words = json.load(f)
AR_WORDS, EN_WORDS = words["AR_WORDS"], words["EN_WORDS"]
print(AR_WORDS)
