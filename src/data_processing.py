"""
Data processing module for Medical MNIST.
Handles downloading, preprocessing, and tf.data pipeline creation.
"""

import os
import tensorflow as tf
import opendatasets as od
from typing import Tuple

def load_medical_mnist(batch_size: int = 64, image_size: Tuple[int, int] = (64, 64)) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Downloads the Medical MNIST dataset and returns train/test tf.data.Datasets.
    
    Args:
        batch_size (int): The batch size for the datasets.
        image_size (Tuple[int, int]): Target size to resize images.
        
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets.
    """
    dataset_url = "https://www.kaggle.com/datasets/andrewmvd/medical-mnist"
    od.download(dataset_url, data_dir="./data/raw")
    
    data_dir = "./data/raw/medical-mnist" # Actual path depends on extracted zip
    
    # Check if a nested folder exists
    if os.path.exists(os.path.join(data_dir, "medical-mnist")):
         data_dir = os.path.join(data_dir, "medical-mnist")

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None, # Unsupervised learning, we don't need labels for standard AE
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        validation_split=0.2,
        subset="both",
        seed=42
    )
    
    train_ds, val_ds = dataset

    # Normalize images to [0, 1] and map to (input, target) pairs for autoencoders
    def process_step(image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32) / 255.0
        return image, image

    train_ds = train_ds.map(process_step, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(process_step, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

def add_noise(image: tf.Tensor, factor: float = 0.2) -> tf.Tensor:
    """Adds random Gaussian noise to an image tensor."""
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=factor)
    noisy_image = image + noise
    return tf.clip_by_value(noisy_image, 0.0, 1.0)
