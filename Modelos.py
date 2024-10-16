import tensorflow as tf
import numpy as np
import keras
import pandas as pd

from keras.layers import Input, Flatten, Dense, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model
from datetime import datetime

from sklearn.base import BaseEstimator, ClassifierMixin

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

#Exemplo de uso:    
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
    def __init__(self, input_shape=(64, 64, 3), min_layers=2, max_layers=6, nomeModelo:str=None):
        self.input_shape = input_shape
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.treino = None
        self.validacao = None
        self.teste = None
        self.nomeModelo = nomeModelo

    def setNome(self, nome):
        self.nomeModelo = nome

    def getPesos(self):
        dir = f"weights_finais/Autoencoders_Gerados/{self.nomeModelo}.weights.h5"
        return str(dir)
    
    def calcular_camadas(self):
        num_layers = np.random.randint(self.min_layers, self.max_layers + 1)#+1 por conta do randint ser somente de (min - max-1)
        
        encoder_layers = []
        decoder_layers = []
        
        shape_atual = self.input_shape
        filter_sizes = [] 
        
        # Encoder
        for i in range(num_layers):
            filters = np.random.randint(16, 256)
            filter_sizes.append(filters)
            encoder_layers.append(Conv2D(filters, (3, 3), activation='relu', padding='same'))
            encoder_layers.append(MaxPooling2D((2, 2), padding='same'))
            shape_atual = (shape_atual[0] // 2, shape_atual[1] // 2, filters) #att por conta da divisão do maxpool
        latent_dim = np.prod(shape_atual)
        encoder_layers.append(Flatten())

        # Decoder
        decoder_layers = [
            Dense(np.prod(shape_atual), activation='relu'), #calcula o vetor latente 
            Reshape(shape_atual)
        ]
        
        for i in range(num_layers):
            filters = filter_sizes[-(i+1)]
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

        if salvar == True and self.nomeModelo !=None:
            save_dir_models = "Modelos_keras/Autoencoders_Gerados"
            save_dir_weights = "weights_finais/Autoencoders_Gerados"

            if not os.path.exists(save_dir_models):
                os.makedirs(save_dir_models)

            if not os.path.exists(save_dir_weights):
                os.makedirs(save_dir_weights)

            self.autoencoder.save(f"{save_dir_models}/{self.nomeModelo}.keras")
            self.autoencoder.save_weights(f"{save_dir_weights}/{self.nomeModelo}.weights.h5")

        x, y = next(self.treino)
        plot_autoencoder(x, self.autoencoder, self.input_shape[0], self.input_shape[1])

    def carrega_modelo(self, modelo:str, pesos:str):
        self.autoencoder = tf.keras.models.load_model(modelo)
        self.autoencoder.load_weights(pesos)

        self.decoder = self.autoencoder.get_layer('decoder')
        self.encoder = self.autoencoder.get_layer('encoder')

        return self.autoencoder, self.encoder, self.decoder

#Exemplo de uso:
#gerador = Gerador(min_layers=2, max_layers=6) -> deve ser proporcional ao input_shape
#modelo = gerador.construir_modelo()
#encoder = gerador.encoder
#decoder = gerador.decoder
#gerador.Dataset(treino, validacao, teste)
#gerador.treinar_autoencoder(epocas=30, salvar=True) -> treina o autoencoder, plota já a reconstrução 

class GeradorClassificador:
    def __init__(self, encoder, pesos, nomeModelo:str=None):
        self.encoder = encoder
        self.nomeModelo = nomeModelo
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
            ], name=f'classificador{self.nomeModelo}')
        
        return classificador
    
    def setNome(self, nome):
        self.nomeModelo = nome
    
    def compila(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def carrega_pesos(self, peso):
        try:
            self.model.load_weights(peso, skip_mismatch=True)
            print("Pesos carregados com sucesso")
        except Exception as e:
            print(f"Erro ao carregar os pesos: {e}")

    def treinamento(self, salvar=False, epocas=10, batch_size=32):
        history = self.model.fit(self.treino, epochs=epocas, batch_size=batch_size ,validation_data=self.validacao)
        pd.DataFrame(history.history).plot()

        if salvar == True and self.nomeModelo !=None:
            save_dir_models = "Modelos_keras/Classificador_Gerados"
            save_dir_weights = "weights_finais/Classificador_Gerados"

            if not os.path.exists(save_dir_models):
                os.makedirs(save_dir_models)

            if not os.path.exists(save_dir_weights):
                os.makedirs(save_dir_weights)

            self.model.save(f"{save_dir_models}/Classificador-{self.nomeModelo}.keras")
            self.model.save_weights(f"{save_dir_weights}/Classificador-{self.nomeModelo}.weights.h5")


    def Dataset(self, treino, validacao, teste):
        self.treino = treino
        self.validacao = validacao
        self.teste = teste

    def predicao(self, teste_csv):
        predicoes = self.model.predict(self.teste)
        predicoes = np.argmax(predicoes, axis=1)

        y_verdadeiro = mapear_rotulos_binarios(teste_csv['classe'])

        plot_confusion_matrix(y_verdadeiro, predicoes, ['Empty', 'Occupied'], f'{self.nomeModelo}')

        return predicoes
    
    def carrega_modelo(self, modelo:str, pesos:str):
        self.model = tf.keras.models.load_model(modelo)
        self.model.load_weights(pesos)

        return self.model

    def predicao_diferente_dataset(self, teste, teste_csv):
        predicoes = self.model.predict(teste)
        predicoes = np.argmax(predicoes, axis=1)

        return predicoes

#Exemplo de uso:
#classificador = GeradorClassificador(encoder=encoder, pesos="pesos.weights.h5") -> crio o classificador encima do encodere seus pesos
#classificador.Dataset(treino, validacao, teste)
#classificador.treinamento()
#classificador.predicao(teste_df) -> cria a matriz de confusão


#testes:
"""class GeradorClassificador(BaseEstimator, ClassifierMixin):

    def __init__(self, encoder, pesos=None, nomeModelo:str=None):
        self.encoder = encoder
        self.nomeModelo = nomeModelo
        self.model = self.modelo(self.encoder)
        self.compila()
        self.pesos = pesos
        self.treino = None
        self.validacao = None
        self.teste = None

        if self.pesos is not None:
            self.carrega_pesos(self.pesos)

    def modelo(self, encoder):
        for layer in self.encoder.layers:
            layer.trainable = False

        classificador = keras.models.Sequential([
                self.encoder, 
                keras.layers.Flatten(),  
                keras.layers.Dropout(0.3),  
                keras.layers.Dense(128, activation='relu'),  
                keras.layers.Dense(2, activation='softmax')  
            ], name=f'classificador{self.nomeModelo}')
        
        return classificador
    
    def setNome(self, nome):
        self.nomeModelo = nome
    
    def compila(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def carrega_pesos(self, peso):
        try:
            self.model.load_weights(peso, skip_mismatch=True)
            print("Pesos carregados com sucesso")
        except Exception as e:
            print(f"Erro ao carregar os pesos: {e}")

    def fit(self, train_generator, validation_generator, epochs=10, verbose=1, salvar=False):
        history = self.model.fit(train_generator, 
                       epochs=epochs, 
                       validation_data=validation_generator, 
                       verbose=verbose)
        
        pd.DataFrame(history.history).plot()

        if salvar == True and self.nomeModelo !=None:
            save_dir_models = "Modelos_keras/Classificador_Gerados"
            save_dir_weights = "weights_finais/Classificador_Gerados"

            if not os.path.exists(save_dir_models):
                os.makedirs(save_dir_models)

            if not os.path.exists(save_dir_weights):
                os.makedirs(save_dir_weights)

            self.model.save(f"{save_dir_models}/Classificador-{self.nomeModelo}.keras")
            self.model.save_weights(f"{save_dir_weights}/Classificador-{self.nomeModelo}.weights.h5")

        return self

    def predict(self, X):
        predicoes = self.model.predict(X)
        return np.argmax(predicoes, axis=1)

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1] 


    def Dataset(self, treino, validacao, teste):
        self.treino = treino
        self.validacao = validacao
        self.teste = teste

    def predicao(self, teste_csv):
        predicoes = self.model.predict(self.teste)
        predicoes = np.argmax(predicoes, axis=1)

        y_verdadeiro = mapear_rotulos_binarios(teste_csv['classe'])

        plot_confusion_matrix(y_verdadeiro, predicoes, ['Empty', 'Occupied'], f'{self.nomeModelo}')

        return predicoes
    
    def carrega_modelo(self, modelo:str, pesos:str):
        self.model = tf.keras.models.load_model(modelo)
        self.model.load_weights(pesos)

        return self.model

"""