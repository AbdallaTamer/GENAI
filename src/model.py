"""
Model module containing Autoencoder (AE) and Variational Autoencoder (VAE) architectures.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple

class MedicalAE(Model):
    """Standard Convolutional Autoencoder."""

    def __init__(self, latent_dim: int):
        super(MedicalAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(64, 64, 1)),
            layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ], name="encoder")

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(16 * 16 * 64, activation='relu'),
            layers.Reshape((16, 16, 64)),
            layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2D(1, 3, activation='sigmoid', padding='same')
        ], name="decoder")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class MedicalVAE(Model):
    """Variational Autoencoder with probabilistic latent space."""

    def __init__(self, latent_dim: int):
        super(MedicalVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder outputs both mean and log variance
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(64, 64, 1)),
            layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim) # mean and logvar
        ], name="vae_encoder")

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(16 * 16 * 64, activation='relu'),
            layers.Reshape((16, 16, 64)),
            layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2D(1, 3, activation='sigmoid', padding='same')
        ], name="vae_decoder")

    def encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def call(self, x: tf.Tensor) -> tf.Tensor:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decoder(z)

        # KL Divergence Loss
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
        self.add_loss(kl_loss)
        # Removed: self.add_metric(kl_loss, name="kl_loss", aggregation="mean") # This line caused the error

        return reconstructed
