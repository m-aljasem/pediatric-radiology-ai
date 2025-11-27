"""
Data Loading and Preprocessing for Bone Age Prediction

Handles loading X-ray images, data augmentation, and dataset preparation.
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class BoneAgeDataLoader:
    """
    Handles loading and preprocessing of bone age X-ray images.
    """
    
    def __init__(self, csv_path: str, image_dir: str):
        """
        Initialize the data loader.
        
        Args:
            csv_path: Path to CSV file with image metadata
            image_dir: Directory containing X-ray images
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.df = None
        self.boneage_mean = None
        self.boneage_std = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV."""
        self.df = pd.read_csv(self.csv_path)
        
        # Create image paths
        if 'id' in self.df.columns:
            self.df['path'] = self.df['id'].map(
                lambda x: os.path.join(self.image_dir, f'{x}.png')
            )
        
        # Calculate normalization parameters
        if 'boneage' in self.df.columns:
            self.boneage_std = 2 * self.df['boneage'].std()
            self.boneage_mean = self.df['boneage'].mean()
            self.df['norm_age'] = (
                (self.df['boneage'] - self.boneage_mean) / self.boneage_std
            )
        
        return self.df
    
    def split_data(
        self,
        train_size: float = 0.8,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            self.load_data()
        
        df_train, df_temp = train_test_split(
            self.df, train_size=train_size, random_state=random_state, shuffle=True
        )
        
        test_size = 1 - train_size - val_size
        df_val, df_test = train_test_split(
            df_temp, test_size=test_size/(1-train_size),
            random_state=random_state, shuffle=True
        )
        
        return df_train, df_val, df_test
    
    def create_generators(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 32
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
        """
        Create data generators with augmentation.
        
        Returns:
            Tuple of (train_gen, val_gen, test_gen)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1/255,
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
        
        # No augmentation for validation/test
        val_test_datagen = ImageDataGenerator(
            rescale=1/255,
            preprocessing_function=preprocess_input
        )
        
        # Create generators
        train_gen = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            x_col='path',
            y_col='boneage',
            batch_size=batch_size,
            seed=42,
            shuffle=False,
            class_mode='raw',
            color_mode='rgb',
            target_size=image_size
        )
        
        val_gen = val_test_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col='path',
            y_col='boneage',
            batch_size=batch_size,
            seed=42,
            shuffle=False,
            class_mode='raw',
            color_mode='rgb',
            target_size=image_size
        )
        
        test_gen = val_test_datagen.flow_from_dataframe(
            dataframe=df_test,
            x_col='path',
            y_col='boneage',
            batch_size=batch_size,
            seed=42,
            shuffle=False,
            class_mode='raw',
            color_mode='rgb',
            target_size=image_size
        )
        
        return train_gen, val_gen, test_gen
    
    def denormalize_age(self, normalized_age: float) -> float:
        """Convert normalized age back to months."""
        if self.boneage_mean is None or self.boneage_std is None:
            raise ValueError("Data must be loaded first")
        return normalized_age * self.boneage_std + self.boneage_mean

