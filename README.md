# Pediatric Radiology AI

Deep learning system for predicting skeletal bone age from hand X‑ray images using an Xception-based convolutional neural network.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 👤 Author

- **Name**: Mohamad AlJasem, MD MPH MSc  
- **Email**: [mohamad@aljasem.eu.org](mailto:mohamad@aljasem.eu.org)  
- **GitHub**: [github.com/m-aljasem](https://github.com/m-aljasem)  
- **Website**: [aljasem.eu.org](https://aljasem.eu.org)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Exported Weights](#-exported-weights)
- [Limitations](#-limitations)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

This project implements a **regression model** that predicts a child's bone age (in months and years) from a single hand X‑ray image.  
It provides a **fully modular Python package** with:

- A **training pipeline** (data loading, augmentation, training, evaluation)
- A **Streamlit web app** for interactive predictions
- **Exportable weights** for easy deployment to other environments

---

## ✨ Features

### 🔬 Modeling
- **Xception Transfer Learning** (ImageNet weights)
- Regression output predicting bone age in **months**
- Custom **MAE metric** computed in real age units (denormalized)

### 🧪 Data & Augmentation
- Real-time augmentation with `ImageDataGenerator`:
  - Rotation (±180°)
  - Zoom
  - Brightness jitter
  - Shifts & flips
- Train/validation/test split with reproducible seeds

### 🌐 App & Deployment
- Streamlit web UI for:
  - Uploading X‑ray images
  - Getting bone age predictions (months + years)
- Model weights saved under `models/bone_age_model.h5`
- Stats (mean/std) pluggable for portable normalization

---

## 🛠 Tech Stack

- **Language**: Python 3.8+
- **DL Framework**: TensorFlow / Keras
- **Model**: Xception (transfer learning)
- **UI**: Streamlit
- **Data Handling**: pandas, NumPy

---

## 📦 Installation

```bash
# Pediatric Radiology AI
pip install -r requirements.txt
```

Optional (dev tools, tests, formatting):

```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Quick Start

### 1️⃣ Download Dataset (RSNA Bone Age)

From the repo root:

```bash
python setup_data.py --project pediatric-radiology-ai
```

This will download and unpack the RSNA Bone Age dataset into `pediatric-radiology-ai/data/`.

### 2️⃣ Train the Model & Export Weights

```bash
cd pediatric-radiology-ai
python src/train.py
```

This will:
- Train the Xception model
- Save best weights to `models/bone_age_model.h5`
- (Optionally) save normalization stats for portable inference

### 3️⃣ Run the Streamlit App

```bash
cd pediatric-radiology-ai
streamlit run app.py
```

Upload a hand X‑ray and get a bone age prediction.

---

## 🧑‍💻 Usage

### 🌐 Web Application

```bash
cd pediatric-radiology-ai
streamlit run app.py
```

1. Open the app in your browser.  
2. Switch to **“Prediction”** mode in the sidebar.  
3. Upload a hand X‑ray image (PNG/JPG).  
4. The app will:
   - Load `models/bone_age_model.h5` if present
   - Run inference
   - Display predicted age in **months** and **years**  

If no weights are found, the app warns you and uses randomly initialized weights (for dev/demo only).

### 🧬 Programmatic Usage

```python
from src.data_loader import BoneAgeDataLoader
from src.model import build_xception_model

# Initialize data loader
loader = BoneAgeDataLoader(
    csv_path="data/boneage-training-dataset.csv",
    images_dir="data/images"
)

# Load and split data
df = loader.load_data()
df_train, df_val, df_test = loader.split_data()

# Generators
train_gen, val_gen, test_gen = loader.create_generators(
    df_train, df_val, df_test
)

# Build model with normalization stats
model = build_xception_model(
    boneage_mean=loader.boneage_mean,
    boneage_std=loader.boneage_std
)

# Train
model.fit(train_gen, validation_data=val_gen, epochs=10)
```

---

## 🗂 Project Structure

```text
pediatric-radiology-ai/
├── app.py                   # Streamlit app
├── config/
│   └── config.yaml          # Training/config options
├── data/                    # RSNA dataset
├── docs/                    # Research, architecture, benchmarks, guides
├── experiments/             # Experiment logs
├── models/                  # Exported weights (bone_age_model.h5, stats)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── scripts/
│   ├── preprocess_data.py
│   └── evaluate.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # BoneAgeDataLoader
│   ├── model.py             # build_xception_model, MAE metric
│   ├── train.py             # Training pipeline (saves weights)
│   ├── config_loader.py
│   ├── utils.py
│   └── metrics.py
└── tests/                   # Unit & integration tests
```

---

## 🧬 Dataset

- **Name**: RSNA Bone Age  
- **Source**: RSNA Bone Age Challenge Dataset  
- **Task**: Regression (bone age in months)  

Expected columns in CSV:

- `id`: Image ID (`<id>.png` in image folder)
- `boneage`: Bone age in months
- `male`: 1 for male, 0 for female (if used)

---

## 🧱 Model Architecture

- **Base**: `tf.keras.applications.Xception` (ImageNet weights, `include_top=False`)
- **Head**:
  - `GlobalMaxPooling2D`
  - `Dense(dense_units, activation="relu")`
  - `Dense(1, activation="linear")` → normalized age
- **Loss**: MSE
- **Metric**: Custom MAE after **denormalization** to real age units

Weights are exported as:

```text
models/bone_age_model.h5
```

---

## 📦 Exported Weights

- Training script (`src/train.py`) saves best weights to:
  - `../models/bone_age_model.h5`
- Streamlit app (`app.py`) loads from:
  - `models/bone_age_model.h5`

This makes the model **portable**:

1. Train once.
2. Copy `models/bone_age_model.h5` to any deployment.
3. Run the same app or a different client that loads these weights.

---

## ⚠️ Limitations

- Uses a single X‑ray per prediction (no clinical context).
- Performance depends heavily on dataset quality and preprocessing.
- Not calibrated for clinical decision-making.

---

## 📄 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

## 🏥 Disclaimer

> This project is for **research and educational purposes only**.  
> It must **not** be used for clinical diagnosis, treatment decisions, or any form of patient care.  
> Always consult qualified healthcare professionals for medical decisions.

---

**Made with ❤️ by Mohamad AlJasem for medical AI research and education.**


## 🌐 RESTful API

The project includes a FastAPI server for programmatic access to the model.

### Starting the API Server

```bash
python api.py
# or
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `GET /model/info` - Get model information
- `POST /predict` - Make a prediction

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction (example for image-based models)
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

## 🔌 MCP Server

The project includes a Model Context Protocol (MCP) server for integration with AI assistants.

### Starting the MCP Server

```bash
python mcp_server.py
```

### MCP Tools

The server exposes the following tools:

- `predict` - Make a prediction using the model
- `model_info` - Get information about the loaded model
- `health_check` - Check if the model is loaded and ready

### MCP Client Integration

To use with an MCP client:

```python
from mcp import ClientSession, StdioServerParameters
import asyncio

async def main():
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        # List tools
        tools = await session.list_tools()
        print(tools)
        
        # Call tool
        result = await session.call_tool(
            "health_check",
            {}
        )
        print(result)

asyncio.run(main())
```

