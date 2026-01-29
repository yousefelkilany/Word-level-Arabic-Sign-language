---
title: Documentation Overview
date: 2026-01-28
lastmod: 2026-01-29
---

# KArSL-502 Documentation

Welcome to the official documentation for the **KArSL-502 Word-Level Arabic Sign Language Recognition** project. This documentation provides a comprehensive guide to the system arcitecture, data processing pipelines, machine learning models, and deployment configurations.

## 📖 Table of Contents

### 🚀 Getting Started
- [Installation & Setup](getting_started.md): How to set up the environment and run the project locally.
- [Architecture Overview](architecture_overview.md): High-level system design and component interactions.

### 🧩 System Modules
- **Core Engine**: [Constants](source/core/constants_py.md), [MediaPipe Integration](core/mediapipe_integration.md), and [Visualization Utilities](core/keypoint_visualization.md).
- **Data Pipeline**: [Dataset Overview](data/dataset_overview.md), [Preprocessing](data/data_preparation_pipeline.md), and [Memory Mapping](data/memory_mapped_datasets.md).
- **Modelling**: [Architecture Design](models/architecture_design.md), [Training Process](models/training_process.md), and [ONNX Export](models/onnx_export_process.md).
- **API & Backend**: [FastAPI Application](api/fastapi_application.md) and [Real-time WebSocket Communication](api/websocket_communication.md).
- **Frontend**: [Web Interface Design](frontend/web_interface_design.md) and [Client Implementation](frontend/websocket_client_implementation.md).

### 🛠️ Developer Reference
- [Project Structure](development/project_structure.md): Directory layout and file organization.
- [Makefile Commands](development/makefile_commands.md): Automation scripts for common tasks.
- [Contributing Guide](development/contributing_guide.md): Standards and workflow for contributors.
- [Function Index](indexes/function_index.md): Searchable index of all implemented functions.
- [Class Index](indexes/class_index.md): Searchable index of all implemented classes.

### � Deployment
- [Docker Setup](deployment/docker_setup.md): Containerization and orchestration details.
- [Environment Configuration](deployment/environment_configuration.md): Managing secrets and application settings.
- [Troubleshooting](reference/troubleshooting.md): Common issues and their solutions.

---

## � How to Navigate
This documentation is designed to be viewed in **Obsidian** or a standard Markdown viewer.
- Use `Ctrl+Click` (or `Cmd+Click`) on the links above to jump to specific modules.
- Every source file documentation includes bidirectional "Called By" and "Calls" sections for deep code exploration.

## ✍️ Documentation Standards
If you are contributing new documentation, please ensure:
1. Every file includes valid **YAML Frontmatter** (title, date).
2. Use **Wiki-links** (`[[page_name]]`) for internal referencing.
3. Include **Mermaid Diagrams** where high-level logic is involved.
4. Keep the bidirectional linking updated in the [Source Code](source/) section.

---

**Project Info**:
- **Author**: Yousef Elkilany
- **Repository**: [Arabic Sign Language KArSL](https://github.com/yousefelkilany/Word-level-Arabic-Sign-language)
- **Status**: Stable v1.0
