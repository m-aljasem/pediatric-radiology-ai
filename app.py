"""
Streamlit App for Bone Age Prediction from X-ray Images
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.explainability import ModelExplainer
from src.model import build_xception_model
from src.data_loader import BoneAgeDataLoader

st.set_page_config(
    page_title="Bone Age Prediction",
    page_icon="🦴",
    layout="wide"
)

st.title("🦴 Bone Age Prediction from X-ray Images")
st.markdown("Predict bone age using deep learning with Xception architecture")

# Sidebar
st.sidebar.title("Navigation")
mode = st.sidebar.selectbox("Mode", ["Prediction", "Explainability", "About"])

# Paths for saved model/artifacts
MODELS_DIR = Path("models")
WEIGHTS_PATH = MODELS_DIR / "bone_age_model.h5"
STATS_PATH = MODELS_DIR / "bone_age_stats.json"

if mode == "Prediction":
    st.header("Upload X-ray Image")
    
    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)
        
        # Preprocess
        img_array = np.array(image.resize((256, 256)))
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        img_array = preprocess_input(img_array.astype('float32'))
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Predict Bone Age"):
            with st.spinner("Loading model and making prediction..."):
                try:
                    # Ensure models directory exists
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)

                    # Load or build model
                    if WEIGHTS_PATH.exists():
                        # Rebuild architecture and load trained weights
                        st.info(f"Loading trained model weights from `{WEIGHTS_PATH}`")
                        model = build_xception_model()
                        model.load_weights(str(WEIGHTS_PATH))
                    else:
                        st.warning(
                            "⚠️ Trained model weights not found. "
                            "Using randomly initialized model (predictions will be unreliable)."
                        )
                        model = build_xception_model()

                    # Load normalization stats if available
                    if STATS_PATH.exists():
                        import json

                        with open(STATS_PATH, "r") as f:
                            stats = json.load(f)
                        boneage_mean = stats.get("boneage_mean", 127.0)
                        boneage_std = stats.get("boneage_std", 41.0)
                    else:
                        # Fallback defaults (should be replaced by real stats after training)
                        boneage_mean = 127.0
                        boneage_std = 41.0
                        st.info(
                            "ℹ️ Normalization statistics not found. "
                            "Using default values from typical RSNA Bone Age distribution."
                        )

                    # Predict (model outputs normalized age)
                    prediction = model.predict(img_array, verbose=0)[0][0]

                    # Denormalize to months/years
                    bone_age_months = prediction * boneage_std + boneage_mean
                    bone_age_years = bone_age_months / 12

                    st.success(
                        f"**Predicted Bone Age: {bone_age_months:.1f} months "
                        f"({bone_age_years:.1f} years)**"
                    )

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Note: Train the model first using the training script to generate weights.")


    elif mode == "Explainability":
        explainability_page()
    elif mode == "About":
    st.header("About Bone Age Prediction")
    st.markdown("""
    This application predicts bone age from X-ray images of the hand and wrist.
    
    **Features:**
    - Uses Xception transfer learning architecture
    - Trained on RSNA Bone Age dataset
    - Predicts age in months
    
    **Usage:**
    1. Upload an X-ray image of a hand/wrist
    2. Click "Predict Bone Age"
    3. View the predicted age
    
    **Note:** This is for research/educational purposes only.
    """)

