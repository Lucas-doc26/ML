import tensorflow as tf
import numpy as np
import keras
import pandas as pd

from keras.layers import Input, Flatten, Dense, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model

from datetime import datetime

from visualizacao import *


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

#Modo de uso:    
#autoencoder = Autoencoder()
#modelo = autoencoder.model
#encoder_model = autoencoder.encoder()
#decoder_model = autoencoder.decoder()

class Classificador:
    def __init__(self, pesos):
        self.model = self.modelo()
        self.compile()
        self.carrega_pesos(pesos)

    def modelo(self):
        encoder = Autoencoder().encoder()
        
        for layer in encoder.layers:
            layer.trainable = False

        classificador = keras.models.Sequential([
                encoder, 
                keras.layers.Flatten(),  
                keras.layers.Dropout(0.3),  
                keras.layers.Dense(128, activation='relu'),  
                keras.layers.Dense(2, activation='softmax')  
            ], name='Classificador1')
        
        return classificador
    
    def compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    def carrega_pesos(self, peso):
        try:
            self.model.load_weights(peso, skip_mismatch=True)
            print("Pesos carregados com sucesso")
        except Exception as e:
            print(f"Erro ao carregar os pesos: {e}")

#Exemplo de uso:
#classificador = Classificador(peso)
#classificador = classificador.model

class Gerador:
    """
    Dados de entrada: input_shape=(64, 64, 3), min_layers=2, max_layers=6\n
    Caso queira aumentar as camadas, deve aumentar o tamanho do input
    """
    def __init__(self, input_shape=(64, 64, 3), min_layers=2, max_layers=8):
        self.input_shape = input_shape
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.treino = None
        self.validacao = None
        self.teste = None

    def calcular_camadas(self):
        num_layers = np.random.randint(self.min_layers, self.max_layers + 1)
        
        encoder_layers = []
        decoder_layers = []
        
        shape_atual = self.input_shape
        
        # Encoder
        for i in range(num_layers):
            filters = np.random.randint(16, 512)
            encoder_layers.append(Conv2D(filters, (3, 3), activation='relu', padding='same'))
            encoder_layers.append(MaxPooling2D((2, 2), padding='same'))
            shape_atual = (shape_atual[0] // 2, shape_atual[1] // 2, filters) #att por conta da divisão do maxpool
        encoder_layers.append(Flatten())
        

        print(f"Current shape after encoder: {shape_atual}")  # Debugging
        latent_dim = np.prod(shape_atual)
        print(f"Latent dimension: {latent_dim}")

        # Decoder
        decoder_layers.append(Dense(latent_dim, activation='relu'))
        decoder_layers.append(Reshape(shape_atual))
        
        for i in range(num_layers):
            filters = np.random.randint(16, 512)
            decoder_layers.append(Conv2D(filters, (3, 3), activation='relu', padding='same'))
            decoder_layers.append(UpSampling2D((2, 2))) #maxpool ao contrário, aumenta a resolução 
        decoder_layers.append(Conv2D(self.input_shape[2], (3, 3), activation='sigmoid', padding='same'))
        
        return encoder_layers, decoder_layers, latent_dim

    def construir_modelo(self):
        encoder_layers, decoder_layers, latent_dim = self.calcular_camadas()
        
        # Construir encoder
        inputs = Input(shape=self.input_shape)
        x = inputs
        for layer in encoder_layers:
            x = layer(x) #as camadas vão sequencialmente sendo adicionadas 
        self.encoder = Model(inputs, x, name='encoder')
        
        # Construir decoder
        latent_inputs = Input(shape=(latent_dim,))
        x = latent_inputs
        for layer in decoder_layers:
            x = layer(x)
        self.decoder = Model(latent_inputs, x, name='decoder')
        
        # Construir autoencoder
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(inputs, decoded, name='autoencoder')
        
        return self.autoencoder

    def compilar_modelo(self, optimizer='adam', loss='mse'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def Dataset(self, treino, validacao, teste):
        self.treino = treino
        self.validacao = validacao
        self.teste = teste

    def treinar_autoencoder(self, salvar=False, epocas=10, batch_size=16):
        history = self.autoencoder.fit(self.treino, epochs=epocas, batch_size=batch_size, validation_data=(self.validacao))
        pd.DataFrame(history.history).plot()

        if salvar == True:
            data = datetime.now()
            nome = f"Autoencoder_Modelo_salvo_em__{data.day}_{data.month}_{data.year}_{data.hour}_{data.minute}"

            save_dir_models = "Modelos_keras/Autoencoders_Gerados"
            save_dir_weights = "weights_finais/Autoencoders_Gerados"

            if not os.path.exists(save_dir_models):
                os.makedirs(save_dir_models)

            if not os.path.exists(save_dir_weights):
                os.makedirs(save_dir_weights)

            self.autoencoder.save(f"{save_dir_models}/{nome}.keras")
            self.autoencoder.save_weights(f"{save_dir_weights}/{nome}.weights.h5")

        x, y = next(self.treino)
        plot_autoencoder(x, self.autoencoder)


class GeradorClassificador:
    def __init__(self, encoder, pesos):
        self.encoder = encoder
        self.model = self.modelo(self.encoder)
        self.compila()
        self.carrega_pesos(pesos)
        self.treino = None
        self.validacao = None
        self.teste = None

    def modelo(self, encoder):
        for layer in self.encoder.layers:
            layer.trainable = False

        classificador = keras.models.Sequential([
                self.encoder, 
                keras.layers.Flatten(),  
                keras.layers.Dropout(0.3),  
                keras.layers.Dense(128, activation='relu'),  
                keras.layers.Dense(2, activation='softmax')  
            ], name='classificador')
        
        return classificador
    
    def compila(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def carrega_pesos(self, peso):
        try:
            self.model.load_weights(peso, skip_mismatch=True)
            print("Pesos carregados com sucesso")
        except Exception as e:
            print(f"Erro ao carregar os pesos: {e}")

    def treinamento(self, epocas=10):
        history = self.model.fit(self.treino, epochs=epocas, batch_size=32 ,validation_data=self.validacao)
        pd.DataFrame(history.history).plot()

    def Dataset(self, treino, validacao, teste):
        self.treino = treino
        self.validacao = validacao
        self.teste = teste

    def predicao(self, teste_csv):
        predicoes = self.model.predict(self.teste)
        predicoes = np.argmax(predicoes, axis=1)

        y_verdadeiro = mapear_rotulos_binarios(teste_csv['classe'])

        plot_confusion_matrix(y_verdadeiro, predicoes, ['Empty', 'Occupied'], 'PUC')