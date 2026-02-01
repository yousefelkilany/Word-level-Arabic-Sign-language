---
title: Documentation Status & Generation Guide
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Documentation Maintenance", "Writer's Guide"]
---

# Documentation Status & Generation Guide

This document tracks the documentation generation progress.

## 📊 Progress Summary

- **Completed**: 57 files
- **Remaining**: 0 files
- **Total**: 57 files
- **Progress**: 100%

## ✅ Completed Documentation

### Main Documentation
- ✅ `index.md`
- ✅ `getting_started.md`
- ✅ `architecture_overview.md`
- ✅ `README.md`

### Indexes
- ✅ `indexes/function_index.md`
- ✅ `indexes/class_index.md`

### API Concepts
- ✅ `api/fastapi_application.md`
- ✅ `api/live_processing_pipeline.md`
- ✅ `api/websocket_communication.md`

### Core Concepts
- ✅ `core/keypoint_visualization.md`
- ✅ `core/mediapipe_integration.md`

### Data Concepts
- ✅ `data/data_preparation_pipeline.md`
- ✅ `data/dataset_overview.md`
- ✅ `data/memory_mapped_datasets.md`

### Model Concepts
- ✅ `models/architecture_design.md`
- ✅ `models/onnx_export_process.md`
- ✅ `models/training_process.md`

### Frontend Concepts
- ✅ `frontend/web_interface_design.md`
- ✅ `frontend/websocket_client_implementation.md`

### Deployment & Config Concepts
- ✅ `deployment/docker_setup.md`
- ✅ `deployment/environment_configuration.md`
- ✅ `development/contributing_guide.md`
- ✅ `development/makefile_commands.md`
- ✅ `development/project_structure.md`

### Reference
- ✅ `reference/api_endpoints.md`
- ✅ `reference/configuration_options.md`
- ✅ `reference/dataset_citation.md`
- ✅ `reference/troubleshooting.md`

### Source Code - API
- ✅ `source/api/cv2_utils_py.md`
- ✅ `source/api/live_processing_py.md`
- ✅ `source/api/main_py.md`
- ✅ `source/api/run_py.md`
- ✅ `source/api/websocket_py.md`

### Source Code - Core
- ✅ `source/core/constants_py.md`
- ✅ `source/core/draw_kps_py.md`
- ✅ `source/core/mediapipe_utils_py.md`
- ✅ `source/core/utils_py.md`

### Source Code - Data
- ✅ `source/data/data_preparation_py.md`
- ✅ `source/data/dataloader_py.md`
- ✅ `source/data/lazy_dataset_py.md`
- ✅ `source/data/mmap_dataset_preprocessing_py.md`
- ✅ `source/data/mmap_dataset_py.md`
- ✅ `source/data/prepare_npz_kps_py.md`
- ✅ `source/data/shared_elements_py.md`
- ✅ `source/data/write_signs_to_json_py.md`
- ✅ `source/data/generate_mediapipe_face_symmetry_map_py.md`

### Source Code - Modelling
- ✅ `source/modelling/dashboard/app_py.md`
- ✅ `source/modelling/dashboard/loader_py.md`
- ✅ `source/modelling/dashboard/views_py.md`
- ✅ `source/modelling/dashboard/visualization_py.md`
- ✅ `source/modelling/export_py.md`
- ✅ `source/modelling/model_py.md`
- ✅ `source/modelling/onnx_benchmark_py.md`
- ✅ `source/modelling/parallel_train_py.md`
- ✅ `source/modelling/train_py.md`
- ✅ `source/modelling/visualize_model_performance_py.md`

### Source Code - Frontend
- ✅ `source/frontend/index_html.md`
- ✅ `source/frontend/live_signs_js.md`
- ✅ `source/frontend/styles_css.md`

### Configuration Files
- ✅ `source/config/docker_compose_yml.md`
- ✅ `source/config/dockerfile.md`
- ✅ `source/config/makefile.md`
- ✅ `source/config/pyproject_toml.md`

## ⏳ Pending Documentation

*None! All files documented.*

## 📝 Frontmatter Template

To update a file, add this at the very top:

```yaml
---
title: <File Title>
date: 2026-01-28
lastmod: 2026-01-28
---

## 🔧 How to Complete Documentation

### Option 1: Manual Creation

Follow the established patterns from completed files:

#### For Source Code Documentation:
1. **Header**: File path, tags, purpose
2. **Overview**: Brief description
3. **Classes/Functions**: Detailed documentation with:
   - Parameters and return values
   - "Called By" links (bidirectional)
   - "Calls" links (bidirectional)
   - Usage examples
4. **Related Documentation**: Links to conceptual docs
5. **File Location**: Link to actual source file

#### For Conceptual Documentation:
1. **Overview**: High-level explanation
2. **Key Concepts**: Main ideas
3. **Diagrams**: Mermaid diagrams where helpful
4. **Examples**: Code examples
5. **Related**: Links to source code and other concepts

### Option 2: Template-Based Generation

Use this template structure for new files:

```markdown
# [Title]

#tags #here

**File Path**: `path/to/file` (for source docs)

**Purpose**: Brief description

## Overview

Detailed explanation...

## [Sections as needed]

### [Subsections]

Content...

## Related Documentation

- [[link|Description]]

---

**File Location**: [filename](/path/to/file) (for source docs)
```

## 🎯 Priority Files to Create Next

1. **function-index.md** - Complete function cross-reference
2. **class-index.md** - Complete class cross-reference
3. **source/modelling/model_py.md** - Model architecture details
4. **models/architecture_design.md** - Conceptual model overview
5. **deployment/docker-setup.md** - Docker configuration
6. **deployment/environment-configuration.md** - Environment variables
7. **reference/troubleshooting.md** - Common issues and solutions

## 📝 Documentation Standards

### Obsidian Features Used

1. **Wiki Links**: `[[page_name|Display Text]]`
2. **Tags**: `#tag-name`
3. **Mermaid Diagrams**: ` ```mermaid ... ``` `
4. **Code Blocks**: ` ```python ... ``` `
5. **Tables**: Markdown tables
6. **Callouts**: `> [!NOTE]`, `> [!IMPORTANT]`, etc.

### Bidirectional Linking

Every function/class should document:
- **Called By**: Where it's used (with links)
- **Calls**: What it calls (with links)
- **Related**: Conceptual documentation

Example:
```markdown
**Called By**: [[source/api/websocket_py#ws_live_signs|ws_live_signs()]]
**Calls**: [[source/core/mediapipe_utils_py#extract_frame_keypoints|extract_frame_keypoints()]]
```

## � Mappings & Automation

### 📂 Directory Mappings
AI agents MUST follow these path translations when generating or updating documentation:

| Source Path | Documentation Path       | Pattern                                                 |
| :---------- | :----------------------- | :------------------------------------------------------ |
| `src/*`     | `docs/source/*`          | Code logic documentation                                |
| `static/*`  | `docs/source/frontend/*` | Client-side assets                                      |
| `*.py`      | `*_py.md`                | **Strict Rule**: Preserve `.py` type in filename suffix |
| `*.js`      | `*_js.md`                | Preserve `.js` type in filename suffix                  |

### ⚡ Automated Sync Triggers
The project uses automation to keep Documentation and Quartz configuration in sync.
- **Trigger**: `scripts/sync_docs_metadata.py`
- **Hook**: `.pre-commit-config.yaml`
- **Rule**: Documentation `lastmod` timestamps are automatically synced with Python file modification dates during pre-commit.
- **Trigger**: `data/sync_quartz_configs.py`
- **Hook**: `.git/hooks/pre-commit`
- **Rule**: DO NOT manually edit navigation or sidebars in `quartz.config.ts`. Update the documentation structure in Markdown, and the pre-commit hook will sync the changes.

### 📇 Indexing Compliance
When a new file is documented:
1.  Add the file to the appropriate category in `docs/README.md`.
2.  Add all public functions to `docs/indexes/function_index.md`.
3.  Add all public classes to `docs/indexes/class_index.md`.

## �🚀 Next Steps

1. Review completed documentation for quality
2. Run generator script for remaining files
3. Manually create priority files
4. Add cross-references between files
5. Test all wiki links in Obsidian
6. Add diagrams where helpful
7. Include code examples

## 📚 Resources

- **Obsidian**: https://obsidian.md/
- **Mermaid Diagrams**: https://mermaid.js.org/
- **Markdown Guide**: https://www.markdownguide.org/
