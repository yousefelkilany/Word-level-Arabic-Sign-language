---
title: write_signs_to_json.py
date: 2026-01-28
lastmod: 2026-02-01
src_hash: d9b757b27159b09bebb3db3e0942f0eb50b2e380dd318baea3bd073cc4854893
aliases: ["Labels Format Converter", "Excel to JSON Script"]
---

# source/data/write_signs_to_json.py

#source-code #data #processing #json

**File Path**: `src/data/write-signs-to-json.py`

**Purpose**: Utility script to convert Excel labels to JSON format.

## Overview

Reads the raw Excel file containing Arabic and English sign labels and exports them to a structured JSON file. This JSON file is then consumed by the application to load class labels.

## Process
1. **Load Excel**: Reads `KARSL-502_Labels.xlsx` from the input directory.
2. **Extract Columns**: Selects `Sign-Arabic` and `Sign-English`.
3. **Format**: Converts to dictionary format.
4. **Save**: Writes to `labels.json`.
5. **Verify**: Reloads the JSON to verify integrity.

## Variables
- `LABELS_JSON_PATH`: Output destination.
- `LOCAL_INPUT_DATA_DIR`: Input directory containing the Excel file.

## Related Documentation
- [[../core/constants_py|constants.py]] - Defines paths.
- [[dataloader_py|dataloader.py]] - Uses the generated JSON.

---

**File Location**: `src/data/write-signs-to-json.py`
