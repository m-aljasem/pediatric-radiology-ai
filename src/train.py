"""
Training script for bone age prediction model.

This script handles the complete training pipeline including data loading,
model building, training, and evaluation.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.model_selection import train_test_split

from model import build_xception_model


def load_and_prepare_data(data_path: str, csv_file: str):
    """
    Load and prepare the dataset.
    
    Args:
        data_path: Path to data directory
        csv_file: Name of CSV file with metadata
        
    Returns:
        Tuple of (train_df, val_df, test_df, boneage_mean, boneage_std)
    """
    data_path = Path(data_path)
    df = pd.read_csv(data_path / csv_file)
    
    # Calculate normalization parameters
    boneage_mean = df['boneage'].mean()
    boneage_std = df['boneage'].std()
    
    # Normalize target
    df['norm_age'] = (df['boneage'] - boneage_mean) / boneage_std
    
    # Create image paths
    image_dir = data_path / 'images'
    df['path'] = df['id'].apply(lambda x: str(image_dir / f'{x}.png'))
    
    # Split data
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, shuffle=True)
    
    return df_train, df_val, df_test, boneage_mean, boneage_std


def create_generators(df_train, df_val, df_test, image_size=(256, 256), batch_size=32):
    """
    Create data generators with augmentation.
    
    Args:
        df_train: Training dataframe
        df_val: Validation dataframe
        df_test: Test dataframe
        image_size: Target image size
        batch_size: Batch size
        
    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        preprocessing_function=preprocess_input,
        rotation_range=180,
        zoom_range=0.25,
        brightness_range=[0.2, 0.5],
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        shear_range=0.05,
        fill_mode='nearest'
    )
    
    # Validation/test generator (no augmentation)
    val_test_datagen = ImageDataGenerator(
        rescale=1/255.0,
        preprocessing_function=preprocess_input
    )
    
    # Create generators
    train_gen = train_datagen.flow_from_dataframe(
        df_train,
        x_col='path',
        y_col='norm_age',
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode='raw',
        color_mode='rgb',
        target_size=image_size
    )
    
    val_gen = val_test_datagen.flow_from_dataframe(
        df_val,
        x_col='path',
        y_col='norm_age',
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode='raw',
        color_mode='rgb',
        target_size=image_size
    )
    
    test_gen = val_test_datagen.flow_from_dataframe(
        df_test,
        x_col='path',
        y_col='norm_age',
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode='raw',
        color_mode='rgb',
        target_size=image_size
    )
    
    return train_gen, val_gen, test_gen


def train_model(
    data_path: str = '../data',
    csv_file: str = 'boneage-training-dataset.csv',
    image_size: tuple = (256, 256),
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    dense_units: int = 10,
    model_save_path: str = '../models/bone_age_model.h5'
):
    """
    Main training function.
    
    Args:
        data_path: Path to data directory
        csv_file: Name of CSV file
        image_size: Target image size
        batch_size: Batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        dense_units: Units in dense layer
        model_save_path: Path to save model
    """
    print("=" * 60)
    print("BONE AGE PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading and preparing data...")
    df_train, df_val, df_test, boneage_mean, boneage_std = load_and_prepare_data(data_path, csv_file)
    print(f"   Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    print(f"   Bone age - Mean: {boneage_mean:.2f}, Std: {boneage_std:.2f} months")
    
    # Create generators
    print("\n2. Creating data generators...")
    train_gen, val_gen, test_gen = create_generators(
        df_train, df_val, df_test, image_size, batch_size
    )
    print(f"   Train batches: {len(train_gen)}")
    print(f"   Val batches: {len(val_gen)}")
    print(f"   Test batches: {len(test_gen)}")
    
    # Build model
    print("\n3. Building model...")
    model = build_xception_model(
        input_shape=(*image_size, 3),
        boneage_mean=boneage_mean,
        boneage_std=boneage_std,
        dense_units=dense_units,
        learning_rate=learning_rate
    )
    print("   Model built successfully!")
    
    # Setup callbacks
    os.makedirs(Path(model_save_path).parent, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n4. Training model...")
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n5. Evaluating model...")
    test_results = model.evaluate(test_gen, verbose=1)
    print(f"\n   Test Loss (MSE): {test_results[0]:.4f}")
    print(f"   Test MAE: {test_results[1]:.2f} months")
    
    # Make predictions for detailed metrics
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=0)
    pred_ages = predictions.flatten() * boneage_std + boneage_mean
    actual_ages = df_test['boneage'].values[:len(pred_ages)]
    
    mae = np.mean(np.abs(pred_ages - actual_ages))
    rmse = np.sqrt(np.mean((pred_ages - actual_ages)**2))
    
    print(f"\n   Final Metrics:")
    print(f"     MAE: {mae:.2f} months")
    print(f"     RMSE: {rmse:.2f} months")
    
    print(f"\n✓ Model saved to: {model_save_path}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train bone age prediction model')
    parser.add_argument('--data_path', type=str, default='../data',
                       help='Path to data directory')
    parser.add_argument('--csv_file', type=str, default='boneage-training-dataset.csv',
                       help='CSV file name')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        csv_file=args.csv_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

