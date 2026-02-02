---
title: SOURCE_DOCS_GUIDE
date: 2026-02-01
lastmod: 2026-02-01
---

# SOURCE_DOCS_GUIDE: AI-Driven Documentation Maintenance

This guide provides instructions for AI agents (LLMs) to create and update source-linked documentation in `docs/source/`. 

## 🤖 Rule for AI Agents
When tasked with updating documentation based on source code changes:
1.  **Analyze the Source**: Read the target file in `src/`.
2.  **Compare with MD**: Read the corresponding `.md` in `docs/source/`.
3.  **Identify Changes**: Look for new/modified classes, functions, constants, or logic.
4.  **Update Content**: Apply changes to the `.md` file while preserving the structure and frontmatter.
5.  **Sync Metadata**: Update `lastmod` and `src_hash` (if the automation script is being run).

## 📂 File Mappings & Naming
- **Source**: `src/path/file.py` -> **Doc**: `docs/source/path/file_py.md`
- **Source**: `static/js/file.js` -> **Doc**: `docs/source/frontend/file_js.md`
- **Rule**: Replace the dot in extension with an underscore (e.g., `.py` becomes `_py`).

## 📝 Frontmatter Template
Every source doc MUST have:
```yaml
---
title: <filename>.<ext>
date: <creation_date>
lastmod: <today_date>
src_hash: <sha256_of_source_content>
aliases: ["Optional Display Name", "Alternative Title"]
---
```

## 🏗️ Content Structure
Follow this standard sections hierarchy:

### 1. Header
- `# <filename>.<ext>`
- `#source #<module_tags>`
- **File Path**: `src/path/file.py`
- **Purpose**: One-sentence summary.

### 2. Overview
High-level description of what the file does and its place in the system.

### 3. Detailed Documentation
Document all major entities:
- **Classes**: Purpose, attributes, and methods.
- **Functions**: Parameters, return types, and core logic.
- **Constants**: Significance and values (if not sensitive).

**Linking Rule**: Use **Relative Wiki Links**. Use only relative links.

- Same directory: `[[file_py#entity|Entity]]`
- Parent directory: `[[../parent_file_py#entity|Entity]]`
- Neighbor directory: `[[../neighbor_dir/file_py#entity|Entity]]`

### 4. Relationships (Bidirectional)
- **Used By**: List of files/functions that import or call this file.
- **Calls**: List of files/functions this file calls.

### 5. Related Documentation
Links to conceptual guides using relative paths (e.g., `[[../../core/mediapipe_integration|MediaPipe Integration]]`).

## 🔄 Updating Logic
- **New Feature**: Add a new section or entry in the existing section.
- **Modified Logic**: Update the description of the function/class.
- **Deleted Code**: Remove the corresponding documentation.
- **Redesign**: If the file is heavily refactored, rewrite the Overview and Detailed sections.

## ⚡ Automation Signal
If `scripts/sync_docs_metadata.py` outputs a hash mismatch, use it as a trigger to perform a full diff-based update of the `.md` content.
