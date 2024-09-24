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

class GeradorAutoencoder:
    def __init__(self, input_shape=(64, 64, 3), min_layers=2, max_layers=5):
        self.input_shape = input_shape
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.encoder = None
        self.decoder = None

    def gerar_modelo(self):
        num_layers = np.random.randint(self.min_layers, self.max_layers + 1)
        encoder_layers, decoder_layers = self._gerar_camadas_espelhadas(num_layers)
        
        self.encoder = self._criar_encoder(encoder_layers)
        self.decoder = self._criar_decoder(decoder_layers)
        
        autoencoder = tf.keras.Sequential([self.encoder, self.decoder])
        return autoencoder

    def _gerar_camadas_espelhadas(self, num_layers):
        encoder_layers = []
        decoder_layers = []
        output_shape = self.input_shape
        
        for _ in range(num_layers):
            filters = np.random.choice([16, 32, 64, 128])
            kernel_size = np.random.choice([3, 5])
            
            encoder_layers.append(('conv', filters, kernel_size))
            output_shape = self._calcular_saida_conv(output_shape, kernel_size)
            encoder_layers.append(('maxpool', 2))
            output_shape = self._calcular_saida_maxpool(output_shape)
            
            decoder_layers.insert(0, ('conv_transpose', filters, kernel_size, 2))
        
        # Adiciona a camada final do decoder para reconstruir a imagem
        decoder_layers.append(('conv_transpose', self.input_shape[-1], 3, 1))
        
        return encoder_layers, decoder_layers

    def _calcular_saida_conv(self, input_shape, kernel_size):
        height, width, channels = input_shape
        new_height = height
        new_width = width
        # Considerando padding 'same'
        return (new_height, new_width, channels)

    def _calcular_saida_maxpool(self, input_shape):
        height, width, channels = input_shape
        new_height = height // 2
        new_width = width // 2
        return (new_height, new_width, channels)

    def _criar_encoder(self, layers):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        
        for layer in layers:
            if layer[0] == 'conv':
                model.add(tf.keras.layers.Conv2D(layer[1], kernel_size=(layer[2], layer[2]), padding="same", activation="relu"))
            elif layer[0] == 'maxpool':
                model.add(tf.keras.layers.MaxPool2D(pool_size=layer[1]))
        
        model.name = "Encoder"
        return model

    def _criar_decoder(self, layers):
        # A forma de entrada do decoder deve ser a forma de saída do encoder
        encoder_output_shape = self.encoder.output_shape[1:]
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=encoder_output_shape))
        
        for i, layer in enumerate(layers):
            if layer[0] == 'conv_transpose':
                activation = "relu" if i < len(layers) - 1 else "sigmoid"
                model.add(tf.keras.layers.Conv2DTranspose(layer[1], kernel_size=(layer[2], layer[2]), 
                                                          strides=layer[3], padding="same", activation=activation))
        
        # Adiciona a camada final do decoder para reconstruir a imagem
        model.add(tf.keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), padding="same", activation="sigmoid"))
        
        # A camada de Reshape pode não ser necessária se a forma estiver correta
        model.add(tf.keras.layers.Reshape(self.input_shape))
        
        model.name = "Decoder"
        return model

    def compilar_modelo(self, autoencoder):
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

# Exemplo de uso
gerador = GeradorAutoencoder(input_shape=(64, 64, 3))
autoencoder = gerador.gerar_modelo()
autoencoder = gerador.compilar_modelo(autoencoder)

