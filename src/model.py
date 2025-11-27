"""
Xception-based CNN Model for Bone Age Prediction

Implements a transfer learning approach using Xception architecture.
"""

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    GlobalMaxPooling2D, Flatten, Dense, Activation, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.metrics import MeanAbsoluteError
from typing import Tuple, Callable


def create_mae_metric(boneage_mean: float, boneage_std: float) -> Callable:
    """
    Create custom MAE metric that denormalizes predictions.
    
    Args:
        boneage_mean: Mean bone age for denormalization
        boneage_std: Standard deviation for denormalization
        
    Returns:
        Custom MAE metric function
    """
    def MAE(y_true, y_pred):
        return MeanAbsoluteError()(
            y_true * boneage_std + boneage_mean,
            y_pred * boneage_std + boneage_mean
        )
    return MAE


def build_xception_model(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    boneage_mean: float = 0.0,
    boneage_std: float = 1.0,
    dense_units: int = 10,
    learning_rate: float = 0.001
) -> Model:
    """
    Build Xception-based model for bone age prediction.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        boneage_mean: Mean bone age for normalization
        boneage_std: Standard deviation for normalization
        dense_units: Number of units in dense layer
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained Xception base
    base_model = Xception(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    
    # Build model
    model = Sequential([
        base_model,
        GlobalMaxPooling2D(),
        Flatten(),
        Dense(dense_units),
        Activation("relu"),
        Dense(1, activation='linear')  # Regression output
    ])
    
    # Compile with custom MAE metric
    mae_metric = create_mae_metric(boneage_mean, boneage_std)
    
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[mae_metric]
    )
    
    return model

