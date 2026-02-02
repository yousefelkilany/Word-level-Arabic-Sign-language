---
description: Automatically sync documentation content with source code changes using the metadata script and SOURCE_DOCS_GUIDE.
---

1. Run the metadata synchronization script to detect out-of-sync files.
// turbo
uv run scripts/sync_docs_metadata.py

2. Analyze the terminal output from the previous step.
   - If "ACTION REQUIRED" is not found, the workflow is complete.
   - If "ACTION REQUIRED" is found, extract the list of flagged `.md` files and their corresponding source files.

3. For each flagged file in the list:
   - Read the source file (e.g., `src/core/constants.py`).
   - Read the current documentation file (e.g., `docs/source/core/constants_py.md`).
   - Read `docs/SOURCE_DOCS_GUIDE.md` to ensure compliance with standards.
   - Update the documentation content to reflect the source changes while preserving metadata.

4. After processing all files, re-run the sync script to verify that all hashes are now matching and "ACTION REQUIRED" is no longer present.
// turbo
uv run scripts/sync_docs_metadata.py
