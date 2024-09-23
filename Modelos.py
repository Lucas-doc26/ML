import tensorflow as tf
import numpy as np
import keras
from keras.layers import Input, Flatten, Dense, Reshape
from keras.models import Sequential

class Autoencoder:
    def __init__(self, input_shape=(64, 64, 3)):
        self.input_shape = input_shape
        self.model = self.construir_modelo()
        
    def encoder(self):
        return keras.models.Sequential([
            keras.layers.Input(shape=self.input_shape),
            keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
            keras.layers.MaxPool2D(pool_size=2),
            keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            keras.layers.MaxPool2D(pool_size=2),
            keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            keras.layers.MaxPool2D(pool_size=2),
            keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
            keras.layers.MaxPool2D(pool_size=2),  
        ], name='encoder')
        
    def decoder(self):
        return keras.models.Sequential([
            keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=2, padding="same", activation="relu", input_shape=(4, 4, 128)),
            keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding="same", activation="relu"),
            keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, padding="same", activation="relu"),
            keras.layers.Conv2DTranspose(16, kernel_size=(3, 3), strides=2, padding="same", activation="relu"),
            keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), padding="same", activation="sigmoid"),
        ], name='decoder')

    def construir_modelo(self):
        return keras.models.Sequential([self.encoder(), self.decoder()])

#autoencoder = Autoencoder()
#modelo = autoencoder.model
#encoder_model = autoencoder.encoder()
#decoder_model = autoencoder.decoder()

class GeradorRedeNeural:
    def __init__(self, input_shape=(64, 64, 3), min_layers=3, max_layers=10):
        self.input_shape = input_shape
        self.min_layers = min_layers
        self.max_layers = max_layers

    def gerar_modelo(self):
        model = tf.keras.Sequential()
        
        # Adiciona a camada de entrada
        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        
        # Número aleatório de camadas
        num_layers = np.random.randint(self.min_layers, self.max_layers + 1)
        
        for _ in range(num_layers):
            self._adicionar_camada_aleatoria(model)
        
        # Adiciona uma camada de saída densa com 10 unidades (pode ser ajustado conforme necessário)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        return model

    def _adicionar_camada_aleatoria(self, model):
        # Escolhe aleatoriamente o tipo de camada
        layer_type = np.random.choice(['conv', 'dense', 'maxpool'])
        
        if layer_type == 'conv':
            self._adicionar_camada_conv(model)
        elif layer_type == 'dense':
            self._adicionar_camada_densa(model)
        else:  # maxpool
            self._adicionar_camada_maxpool(model)
        
        # Adiciona Dropout aleatoriamente
        if np.random.rand() > 0.5:
            model.add(tf.keras.layers.Dropout(rate=np.random.uniform(0.1, 0.5)))

    def _adicionar_camada_conv(self, model):
        filters = np.random.choice([32, 64, 128, 256])
        kernel_size = np.random.choice([3, 5])
        model.add(tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu'))

    def _adicionar_camada_densa(self, model):
        units = np.random.choice([64, 128, 256, 512])
        model.add(tf.keras.layers.Dense(units, activation='relu'))

    def _adicionar_camada_maxpool(self, model):
        pool_size = np.random.choice([2, 3])
        model.add(tf.keras.layers.MaxPooling2D((pool_size, pool_size)))

#gerador = GeradorRedeNeural()
#modelo = gerador.gerar_modelo()
#modelo.summary()

