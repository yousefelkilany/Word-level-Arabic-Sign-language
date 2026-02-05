---
title: Getting Started
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Installation Guide", "Quick Start Setup"]
---

# Getting Started

#getting-started #installation #setup

This guide will help you set up and run the Arabic Sign Language Recognition system on your local machine or using Docker.

## Prerequisites

Choose one of the following setups:

### Option 1: Docker (Recommended)
- **Docker** and **Docker Compose** installed
- No other dependencies required

### Option 2: Local Development
- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package installer
- **Webcam** (for live recognition)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yousefelkilany/word-level-arabic-sign-language.git
cd word-level-arabic-sign-language
```

### 2. Download Models (Git LFS)
The landmarker models and latest ONNX checkpoints are stored using Git LFS. Run the following to ensure they are downloaded:

```bash
make download_lfs_files
```

> [!NOTE]
> Ensure you have `git-lfs` installed and initialized on your system.

### 2. Configuration Setup

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Model Configuration
ONNX_CHECKPOINT_FILENAME=last-checkpoint-signs_502.pth.onnx

# CORS Configuration
DOMAIN_NAME=http://localhost:8000

# Development Mode (1 = local, 0 = Kaggle paths)
LOCAL_DEV=1

# Force CPU execution (1 = CPU only, 0 = use GPU if available)
USE_CPU=1
```

> [!IMPORTANT]
> Set `LOCAL_DEV=1` to use local `data/` and `models/` directories instead of Kaggle paths.

See [[deployment/environment_configuration|Environment Configuration]] for detailed variable descriptions.

### 3. Choose Your Setup Method

#### Option A: Docker Setup (Recommended)

Build and start the services:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

**Features:**
- ✅ All dependencies included
- ✅ Consistent environment
- ✅ Easy deployment
- ✅ Hot reload enabled (code changes reflected immediately)

See [[deployment/docker_setup|Docker Setup]] for advanced configuration.

#### Option B: Local Development Setup

1. **Install Dependencies:**

```bash
uv sync
```

2. **Run the Backend:**

```bash
# Direct Python execution
python src/api/run.py

# OR using Make
make local_setup && python src/api/run.py
```

The API will be available at `http://localhost:8000`.

## First Run

### 1. Access the Web Interface

Navigate to [http://localhost:8000/live-signs](http://localhost:8000/live-signs) in your web browser.

### 2. Grant Camera Permissions

When prompted, allow the browser to access your webcam.

### 3. Start Signing

- Position yourself in front of the camera.
- Perform Arabic sign language gestures.
- **New**: Use the "Settings" menu to toggle skeletal visualization for Face, Pose, and Hands.
- **New**: View session history and past detections in the "Archive" and "History Log".
- The system will detect and display recognized signs in real-time.

## Project Structure

```
arabic-sign-language-karsl/
├── src/
│   ├── api/          # FastAPI application and WebSocket handlers
│   ├── core/         # Core utilities (MediaPipe, constants)
│   ├── data/         # Dataset processing and loading
│   └── modelling/    # Model architecture and training
├── static/           # Frontend (HTML, CSS, JavaScript)
├── models/           # ONNX models for inference
├── checkpoints/      # Training checkpoints
├── data/             # Dataset and labels
├── docs/             # This documentation
├── Dockerfile        # Container image configuration
├── docker-compose.yml
├── pyproject.toml    # Python dependencies
└── makefile          # Build automation
```

See [[development/project_structure|Project Structure]] for detailed organization.

## Available Commands

### Using Make

```bash
# Setup
make download_lfs_files  # Pull latest models/landmarkers via Git LFS

# Training
make train              # Train model with default settings
make parallel_train     # Multi-GPU training
make cpu_train          # Train on CPU only

# Model Export & Analysis
make export_onnx        # Export PyTorch checkpoint to ONNX
make onnx_benchmark     # Benchmark ONNX inference speed
make visualize_metrics  # Generate performance plots
```

See [[development/makefile_commands|Makefile Commands]] for all available commands.

### Using Docker

```bash
# Start services
docker-compose up

# Rebuild and start
docker-compose up --build

# Force recreate containers
docker-compose up --build --force-recreate

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

## Verification

### Test the API

```bash
curl http://localhost:8000/
```

Expected response: HTML content from the live signs interface.

### Test WebSocket Connection

Open the browser console at `http://localhost:8000/live-signs` and check for:
```
WebSocket connection established
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Change port in docker-compose.yml or when running locally
uvicorn api.main:app --port 8001
```

#### Model Not Found
- Ensure ONNX model exists in `models/` directory
- Check `ONNX_CHECKPOINT_FILENAME` in `.env`
- Download pre-trained model if needed

#### Camera Not Detected
- Grant browser camera permissions
- Check if another application is using the camera
- Try a different browser (Chrome/Edge recommended)

#### CORS Errors
- Verify `DOMAIN_NAME` in `.env` matches your frontend URL
- Check browser console for specific CORS errors

See [[reference/troubleshooting|Troubleshooting]] for more solutions.

## Next Steps

- **Understand the Architecture**: [[architecture_overview|Architecture Overview]]
- **Explore the API**: [[api/fastapi_application|FastAPI Application]]
- **Learn About the Model**: [[models/architecture_design|Model Architecture]]
- **Train Your Own Model**: [[models/training_process|Training Process]]
- **Customize the Frontend**: [[frontend/web_interface_design|Web Interface Design]]

## Resources

- **Official KArSL Website**: [hamzah-luqman.github.io/KArSL](https://hamzah-luqman.github.io/KArSL/)
- **Kaggle Dataset**: [kaggle.com/datasets/yousefdotpy/karsl-502](https://www.kaggle.com/datasets/yousefdotpy/karsl-502)
- **Google Drive**: [Dataset and Labels](https://drive.google.com/drive/folders/1LI6L7MSXOIwSgbVL0zmjnw7wryZ6aYl-)
- **GitHub Repository**: [github.com/yousefelkilany/word-level-arabic-sign-language](https://github.com/yousefelkilany/word-level-arabic-sign-language)

---

**Related Pages:**
- [[architecture_overview|Architecture Overview]]
- [[deployment/docker_setup|Docker Setup]]
- [[deployment/environment_configuration|Environment Configuration]]
- [[reference/troubleshooting|Troubleshooting]]
