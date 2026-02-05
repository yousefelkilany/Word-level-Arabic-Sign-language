---
description: Automatically sync documentation content with source code changes using the metadata script and SOURCE_DOCS_GUIDE.
---

1. Run the metadata synchronization script to detect out-of-sync files.
// turbo
uv run scripts/sync_docs_metadata.py

2. Analyze the terminal output from the previous step.
   - If "ACTION REQUIRED" and "ORPHANED DOCUMENTATION" are not found, the workflow is complete.
   - If "ACTION REQUIRED" is found, extract the list of flagged `.md` files and their source.
   - If "ORPHANED DOCUMENTATION" is found, identify the stale `.md` files.

3. Handle Flagged and Orphaned Files:
   - **For actioned files**: Update content to reflect source changes while preserving metadata.
   - **For orphaned files**: Verify if source was deleted or moved. If deleted, remove the `.md`. If moved, relocate it.
   - **Propagate Changes**: Identify and update related high-level conceptual documents in `docs/` that are peers to `docs/source/` (e.g., `docs/frontend/`, `docs/api/`, `docs/models/`).
     - *Frontend Note*: If source files related to the frontend (`static/js/`, etc.) are updated, ensure high-level frontend docs in `docs/frontend/` are synchronized.

4. After processing all files, re-run the sync script to verify that all hashes are now matching and "ACTION REQUIRED" is no longer present.
// turbo
uv run scripts/sync_docs_metadata.py

5. **Final Review**: Ensure that the `architecture_overview.md` and `getting_started.md` are still accurate if major changes occurred.

6. **Update Main README**: Run the `update-readme` workflow to ensure the project's root `README.md` reflects any major documentation or source changes.
   - // turbo
   - /update-readme
