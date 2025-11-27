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

The project includes a FastAPI server for programmatic access to the model. This allows you to integrate predictions into your own applications, web services, or scripts.

### Installation

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

### Starting the API Server

Start the API server using one of these methods:

**Method 1: Direct Python execution**
```bash
python api.py
```

**Method 2: Using uvicorn directly**
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Method 3: Production mode (no auto-reload)**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API information |
| `/health` | GET | Health check endpoint (checks if model is loaded) |
| `/model/info` | GET | Get detailed model information |
| `/predict` | POST | Make a prediction |

### Interactive API Documentation

Once the server is running, you can access interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API explorer with "Try it out" feature
- **ReDoc**: `http://localhost:8000/redoc` - Beautiful, responsive API documentation

### Using the API

#### Health Check

```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
# Output: {"status": "healthy", "model_loaded": true}
```

#### Get Model Information

```python
import requests

response = requests.get("http://localhost:8000/model/info")
print(response.json())
# Output: Model type, input shape, classes, etc.
```

#### Make Predictions

# Example: Bone Age Prediction
import requests

# Upload X-ray image
with open("hand_xray.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()

print(f"Predicted Bone Age: {result['prediction']:.1f} months")
print(f"Age in years: {result['prediction'] / 12:.1f} years")

### Using cURL

You can also use cURL to interact with the API:

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Model Info:**
```bash
curl http://localhost:8000/model/info
```

**Make Prediction (for image-based models):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_image.jpg"
```

**Make Prediction (for JSON-based models like Alzheimer's):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender": 1.0, "age": 75.0, "education": 14.0, "ses": 2.0, "mmse": 27.0, "etiv": 1490.0, "nwbv": 0.73, "asf": 1.20}'
```

### Error Handling

The API returns appropriate HTTP status codes:

- `200 OK` - Successful request
- `400 Bad Request` - Invalid input data
- `503 Service Unavailable` - Model not loaded (train the model first)
- `500 Internal Server Error` - Server error during prediction

Example error handling:

```python
import requests

try:
    response = requests.post("http://localhost:8000/predict", json=data)
    response.raise_for_status()  # Raises exception for bad status codes
    result = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
```

### API Response Format

**Successful Prediction Response:**
```json
{
    "prediction": "Demented",
    "confidence": 0.85,
    "probabilities": {
        "Nondemented": 0.15,
        "Demented": 0.85
    }
}
```

**Error Response:**
```json
{
    "detail": "Model not loaded. Please train the model first."
}
```

### Deployment

For production deployment, consider:

1. **Using a production ASGI server**: Use `uvicorn` with multiple workers or `gunicorn` with uvicorn workers
2. **Adding authentication**: Implement API keys or OAuth2
3. **Rate limiting**: Add rate limiting to prevent abuse
4. **Logging**: Configure proper logging for monitoring
5. **HTTPS**: Use HTTPS in production with SSL certificates

Example production command:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

## 🔌 MCP Server

The project includes a Model Context Protocol (MCP) server that exposes the model as tools for AI assistants and other MCP-compatible clients. This allows AI assistants like Claude, ChatGPT, or custom MCP clients to interact with your model.

### What is MCP?

Model Context Protocol (MCP) is a standardized protocol for AI assistants to interact with external tools and services. It enables AI assistants to:
- Call your model for predictions
- Get model information
- Check model health status

### Installation

The MCP server requires the MCP SDK:

```bash
pip install mcp
```

### Starting the MCP Server

Start the MCP server:

```bash
python mcp_server.py
```

The server runs as a stdio-based server, communicating via standard input/output. It's designed to be used with MCP clients.

### MCP Tools

The server exposes the following tools:

| Tool | Description | Parameters |
|------|-------------|------------|
| `predict` | Make a prediction using the model | `input` (string): Input data as JSON string or file path |
| `model_info` | Get information about the loaded model | None |
| `health_check` | Check if the model is loaded and ready | None |

### Using MCP with Python Client

```python
from mcp import ClientSession, StdioServerParameters
import asyncio
import json

async def main():
    # Connect to MCP server
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
            env=None
        )
    ) as session:
        # Initialize the session
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        print("Available tools:", [tool.name for tool in tools])
        
        # Health check
        health_result = await session.call_tool(
            "health_check",
            {}
        )
        print("Health:", health_result.content[0].text)
        
        # Get model info
        model_info = await session.call_tool(
            "model_info",
            {}
        )
        print("Model Info:", model_info.content[0].text)
        
        # Make prediction
        # For image-based models, provide base64 encoded image or file path
        prediction_input = json.dumps({
            "file_path": "test_image.jpg"
        })
        
        prediction_result = await session.call_tool(
            "predict",
            {"input": prediction_input}
        )
        print("Prediction:", prediction_result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using MCP with Claude Desktop

To use with Claude Desktop, add this to your MCP configuration file:

```json
{
  "mcpServers": {
    "bone-age": {
      "command": "python",
      "args": ["/home/m-aljasem/projects/ai-projects/bone-age/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/home/m-aljasem/projects/ai-projects/bone-age"
      }
    }
  }
}
```

### MCP Tool Responses

**Health Check Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Model Info Response:**
```json
{
  "model_type": "TensorFlow/Keras",
  "model_path": "models/model.h5",
  "classes": ["Class1", "Class2"],
  "description": "Model description"
}
```

**Prediction Response:**
```json
{
  "prediction": "Class1",
  "confidence": 0.95,
  "probabilities": {
    "Class1": 0.95,
    "Class2": 0.05
  }
}
```

### Error Handling

The MCP server returns error messages in JSON format:

```json
{
  "error": "Model not loaded. Please train the model first."
}
```

### Integration Examples

**Example 1: Batch Predictions**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters

async def batch_predict(file_paths):
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        await session.initialize()
        
        results = []
        for file_path in file_paths:
            result = await session.call_tool(
                "predict",
                {"input": json.dumps({"file_path": file_path})}
            )
            results.append(json.loads(result.content[0].text))
        
        return results

# Usage
predictions = asyncio.run(batch_predict([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]))
```

**Example 2: Model Monitoring**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
import time

async def monitor_model():
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        await session.initialize()
        
        while True:
            health = await session.call_tool("health_check", {})
            print(f"[{time.strftime('%H:%M:%S')}] Health: {health.content[0].text}")
            await asyncio.sleep(60)  # Check every minute

# Run monitoring
asyncio.run(monitor_model())
```

### Troubleshooting

**Issue: MCP server not starting**
- Ensure `mcp` package is installed: `pip install mcp`
- Check that the model file exists in the `models/` directory
- Verify Python path and dependencies

**Issue: Tool calls failing**
- Check that the model is trained and weights are saved
- Verify input format matches expected format
- Check server logs for detailed error messages

**Issue: Connection errors**
- Ensure the MCP server process is running
- Check that stdio communication is working
- Verify environment variables if needed

