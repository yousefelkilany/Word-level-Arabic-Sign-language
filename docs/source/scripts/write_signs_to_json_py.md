---
title: write_signs_to_json.py
date: 2026-02-05
lastmod: 2026-02-01
src_hash: d9b757b27159b09bebb3db3e0942f0eb50b2e380dd318baea3bd073cc4854893
aliases: ["Label Converter", "Excel to JSON"]
---

# write_signs_to_json.py

#source #scripts #data #preprocessing

**File Path**: `scripts/write_signs_to_json.py`

**Purpose**: Converts the Master Excel label file into a JSON dictionary for use in the application.

## Overview

The script reads the sign language labels (Arabic and English) from an Excel sheet and creates a lightweight JSON file (`labels.json`). This ensures the live application has fast, indexed access to sign names without requiring an Excel parser at runtime.

## Logic

1. **Read Excel**: Uses `pandas` to load `KARSL-502_Labels.xlsx`.
2. **Extract columns**: Grabs "Sign-Arabic" and "Sign-English".
3. **Save JSON**: Writes to `LABELS_JSON_PATH` with `ensure_ascii=False` to preserve Arabic characters.

## Related Documentation

**Depends On**:
- `KARSL-502_Labels.xlsx` (stored in `LOCAL_INPUT_DATA_DIR`).

**Used By**:
- [[../source/api/main_py|main.py]] - To load labels for recognition output.
