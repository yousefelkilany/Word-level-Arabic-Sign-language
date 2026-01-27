# Documentation Status & Generation Guide

This document tracks the documentation generation progress and provides instructions for completing the remaining files.

## ✅ Completed Documentation

### Main Documentation (3/3)
- ✅ `index.md` - Main documentation index with navigation
- ✅ `getting-started.md` - Installation and setup guide
- ✅ `architecture-overview.md` - System architecture with diagrams

### Source Code Documentation - API (5/5)
- ✅ `source/api/main-py.md` - FastAPI application setup
- ✅ `source/api/websocket-py.md` - WebSocket handler (comprehensive)
- ✅ `source/api/live-processing-py.md` - Frame buffer and processing
- ✅ `source/api/cv2-utils-py.md` - Motion detection
- ✅ `source/api/run-py.md` - Entry point

### Source Code Documentation - Core (1/4)
- ✅ `source/core/constants-py.md` - System constants
- ⏳ `source/core/mediapipe-utils-py.md` - MediaPipe integration
- ⏳ `source/core/utils-py.md` - Utility functions
- ⏳ `source/core/draw-kps-py.md` - Keypoint visualization

## 📋 Remaining Documentation Files

### Conceptual Documentation (19 files)

#### API Concepts (3 files)
- ⏳ `api/fastapi-application.md`
- ⏳ `api/websocket-communication.md`
- ⏳ `api/live-processing-pipeline.md`

#### Core Concepts (2 files)
- ⏳ `core/mediapipe-integration.md`
- ⏳ `core/keypoint-visualization.md`

#### Data Concepts (3 files)
- ⏳ `data/dataset-overview.md`
- ⏳ `data/data-preparation-pipeline.md`
- ⏳ `data/memory-mapped-datasets.md`

#### Model Concepts (3 files)
- ⏳ `models/architecture-design.md`
- ⏳ `models/training-process.md`
- ⏳ `models/onnx-export-process.md`

#### Frontend Concepts (2 files)
- ⏳ `frontend/web-interface-design.md`
- ⏳ `frontend/websocket-client-implementation.md`

#### Deployment (2 files)
- ⏳ `deployment/docker-setup.md`
- ⏳ `deployment/environment-configuration.md`

#### Development (3 files)
- ⏳ `development/project-structure.md`
- ⏳ `development/contributing-guide.md`
- ⏳ `development/makefile-commands.md`

#### Reference (4 files)
- ⏳ `reference/api-endpoints.md`
- ⏳ `reference/configuration-options.md`
- ⏳ `reference/dataset-citation.md`
- ⏳ `reference/troubleshooting.md`

### Source Code Documentation (35+ files)

#### Core (3 files)
- ⏳ `source/core/mediapipe-utils-py.md`
- ⏳ `source/core/utils-py.md`
- ⏳ `source/core/draw-kps-py.md`

#### Data (9 files)
- ⏳ `source/data/data-preparation-py.md`
- ⏳ `source/data/dataloader-py.md`
- ⏳ `source/data/lazy-dataset-py.md`
- ⏳ `source/data/mmap-dataset-py.md`
- ⏳ `source/data/mmap-dataset-preprocessing-py.md`
- ⏳ `source/data/prepare-npz-kps-py.md`
- ⏳ `source/data/shared-elements-py.md`
- ⏳ `source/data/write-signs-to-json-py.md`
- ⏳ `source/data/generate-mediapipe-face-symmetry-map-py.md`

#### Modelling (10 files)
- ⏳ `source/modelling/model-py.md`
- ⏳ `source/modelling/train-py.md`
- ⏳ `source/modelling/parallel-train-py.md`
- ⏳ `source/modelling/export-py.md`
- ⏳ `source/modelling/onnx-benchmark-py.md`
- ⏳ `source/modelling/visualize-model-performance-py.md`
- ⏳ `source/modelling/dashboard/app-py.md`
- ⏳ `source/modelling/dashboard/loader-py.md`
- ⏳ `source/modelling/dashboard/views-py.md`
- ⏳ `source/modelling/dashboard/visualization-py.md`

#### Frontend (3 files)
- ⏳ `source/frontend/live-signs-js.md`
- ⏳ `source/frontend/index-html.md`
- ⏳ `source/frontend/styles-css.md`

#### Configuration (4 files)
- ⏳ `source/config/dockerfile.md`
- ⏳ `source/config/docker-compose-yml.md`
- ⏳ `source/config/makefile.md`
- ⏳ `source/config/pyproject-toml.md`

### Cross-Reference Indexes (2 files)
- ⏳ `function-index.md`
- ⏳ `class-index.md`

## 📊 Progress Summary

- **Completed**: 9 files
- **Remaining**: 56 files
- **Total**: 65 files
- **Progress**: 14%

## 🔧 How to Complete Documentation

### Option 1: Use the Generator Script

A Python script `generate_remaining_docs.py` has been created with templates for key files. Run it to generate additional documentation:

```bash
python generate_remaining_docs.py
```

### Option 2: Manual Creation

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

### Option 3: Template-Based Generation

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

**File Location**: [filename](file:///path) (for source docs)
```

## 🎯 Priority Files to Create Next

1. **function-index.md** - Complete function cross-reference
2. **class-index.md** - Complete class cross-reference
3. **source/modelling/model-py.md** - Model architecture details
4. **models/architecture-design.md** - Conceptual model overview
5. **deployment/docker-setup.md** - Docker configuration
6. **deployment/environment-configuration.md** - Environment variables
7. **reference/troubleshooting.md** - Common issues and solutions

## 📝 Documentation Standards

### Obsidian Features Used

1. **Wiki Links**: `[[page-name|Display Text]]`
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
**Called By**: [[source/api/websocket-py#ws_live_signs|ws_live_signs()]]
**Calls**: [[source/core/mediapipe-utils-py#extract_frame_keypoints|extract_frame_keypoints()]]
```

## 🚀 Next Steps

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

---

**Last Updated**: 2026-01-27

**Status**: Foundation Complete (14%), In Progress
