import os
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from keras import ops, layers
from keras.saving import register_keras_serializable
import cv2
import tensorflow.image as tf_img
from utils.view.visualizacao import plot_autoencoder_2, plot_heat_map


os.environ["KERAS_BACKEND"] = "tensorflow"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def preprocess_images(target_shape, csv):
    imagens = []
    df = pd.read_csv(csv)
    for path in df["caminho_imagem"]:
        img = cv2.imread(path)  # carrega como BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converte para RGB se quiser
        imagens.append(img)

    X_processed = []
    for img in imagens:
        img_resized = cv2.resize(img, target_shape)
        X_processed.append(img_resized)
    #X_processed = np.expand_dims(X_processed, -1)  # (N, 28, 28, 1)
    return np.array(X_processed).astype("float32") / 255.0

treino = preprocess_images((64,64), "CSV/CNR/CNR_autoencoder_treino.csv")
teste = preprocess_images((64,64), "CSV/CNR/CNR_autoencoder_teste.csv")

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(42)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

latent_dim = 32

encoder_inputs = keras.Input(shape=(64, 64, 3))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((16, 16, 64))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

@keras.saving.register_keras_serializable()
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': keras.saving.serialize_keras_object(self.encoder),
            'decoder': keras.saving.serialize_keras_object(self.decoder),
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = keras.saving.deserialize_keras_object(config.pop('encoder'))
        decoder = keras.saving.deserialize_keras_object(config.pop('decoder'))
        return cls(encoder=encoder, decoder=decoder, **config)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

dataset = np.concatenate([treino, teste], axis=0)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(dataset, epochs=20, batch_size=16)
vae.build((None, 64, 64, 3))  # Aqui você especifica o formato da entrada.

vae.save_weights('Modelos/Modelo_VAE-0/Modelo-Base/Pesos/Modelo_VAE-0_Base-CNR.weights.h5')
vae.save('Modelos/Modelo_VAE-0/Modelo-Base/Estrutura/Modelo_VAE-0.keras')

vae = tf.keras.models.load_model('Modelos/Modelo_VAE-0/Modelo-Base/Estrutura/Modelo_VAE-0.keras')
vae.load_weights('Modelos/Modelo_VAE-0/Modelo-Base/Pesos/Modelo_VAE-0_Base-CNR.weights.h5')

def calcular_ssim(image1, image2):
    return tf_img.ssim(image1, image2, max_val=1.0).numpy()

plot_autoencoder_2(teste, vae, caminho_para_salvar='Modelos/Modelo_VAE-0/Plots')

plot_heat_map(teste, encoder, decoder)

from Modelos import *

cria_classificadores(1, 'Modelo_VAE', 'CNR', None, None, None, (64,64,3))
