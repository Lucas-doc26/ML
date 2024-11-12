import tensorflow as tf
import numpy as np
import keras
import pandas as pd
from keras.layers import Input, Flatten, Dense, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from visualizacao import *
import keras.backend as k  
import gc
import re
import shutil

"""----------------------Funções Auxiliares-----------------------"""
def limpa_memoria():
    k.clear_session()  
    gc.collect() 

def encontrar_modelos_e_pesos(diretorio_base="Modelos", nome=None):
    dir_estruturas = os.path.join(diretorio_base, "Estruturas")
    dir_pesos = os.path.join(diretorio_base, "Pesos")
    
    modelos = os.listdir(dir_estruturas)
    pesos = os.listdir(dir_pesos)
    
    def extrair_numero(nome_arquivo):
        numeros = re.findall(r'\d+', nome_arquivo)
        return int(numeros[0]) if numeros else None
    
    modelos_dict = {}
    pesos_dict = {}
    
    for modelo in modelos:
        if nome in modelo and nome != None:
            numero = extrair_numero(modelo)
            if numero is not None:
                modelo_path = os.path.join(dir_estruturas, modelo)
                modelos_dict[numero] = modelo_path
    
    for peso in pesos:
        if nome in peso and nome != None:
            numero = extrair_numero(peso)
            if numero is not None:
                peso_path = os.path.join(dir_pesos, peso)
                pesos_dict[numero] = peso_path
    
    pares_modelo_peso = []
    for numero in sorted(modelos_dict.keys()):
        if numero in pesos_dict:
            pares_modelo_peso.append((modelos_dict[numero], pesos_dict[numero]))
    
    return pares_modelo_peso

def retorna_nome(caminho):
    nome = caminho.split('/')[-1]
    nome = nome.rsplit('.keras', 1)[0]
    return nome

def cria_pasta_modelos():
    if not os.path.isdir("Modelos"):
        os.makedirs("Modelos")
    
    """
    if not os.path.isdir("Modelos/Pesos"):
        os.makedirs("Modelos/Pesos")

    if not os.path.isdir("Modelos/Estruturas"):
        os.makedirs("Modelos/Estruturas")"""

    if not os.path.isdir("Pesos_parciais"):
        os.makedirs("Pesos_parciais")

def criar_diretorio_novo(caminho):
    if os.path.exists(caminho):
        shutil.rmtree(caminho)
    os.makedirs(caminho)


def mapear(classes):
    return np.array([1 if classe == '1' else 0 for classe in classes])

"""------------------Gerador de Autoencoders----------------------"""

class Gerador:
    """
    Dados de entrada: input_shape=(64, 64, 3), min_layers=2, max_layers=6\n
    Caso queira aumentar as camadas, deve aumentar o tamanho do input
    """

    def __init__(self, input_shape=(64, 64, 3), min_layers=2, max_layers=6, nome_modelo:str=None):
        self.input_shape = input_shape
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.treino = None
        self.validacao = None
        self.teste = None
        self.nome_modelo = nome_modelo
        cria_pasta_modelos()

    def setNome(self, nome):
        self.nome_modelo = nome

    def getNome(self):
        print(self.nome)

    def getPesos(self):
        dir = f"weights_finais/Autoencoders_Gerados/{self.nome_modelo}.weights.h5"
        return str(dir)
    
    def calcular_camadas(self, filters_list=[8,16,32,64,128]):
        num_layers = np.random.randint(self.min_layers, self.max_layers + 1) #+1 por conta do randint ser somente de (min - max-1)
        
        encoder_layers = []
        decoder_layers = []
        maxpoll_layers = []
        
        shape_atual = self.input_shape
        filter_sizes = [] 

        layers = {}
        
        # Encoder
        for i in range(num_layers):
            filters = np.random.choice(filters_list)
            filter_sizes.append(filters)
            encoder_layers.append(Conv2D(filters, (3, 3), activation='relu', padding='same'))
            
            maxpoll_layers.append(np.random.choice([0, 1, 1, 1]))

            if maxpoll_layers[i] == 1:
                encoder_layers.append(MaxPooling2D((2, 2), padding='same'))
                shape_atual = (shape_atual[0] // 2, shape_atual[1] // 2, filters) #att por conta da divisão do maxpool
            else:
                shape_atual = (shape_atual[0], shape_atual[1], filters) 

        latent_dim = np.random.randint(128,2048)
        encoder_layers.append(Flatten())
        encoder_layers.append(Dense(latent_dim, activation='relu')) #transformo o meu shape_atual no meu latent_dim 

        # Decoder
        decoder_layers = [
            Dense(np.prod(shape_atual), activation='relu'), #calcula o vetor latente 
            Reshape(shape_atual)
        ]
        
        for i in range(num_layers):
            filters = filter_sizes[-(i+1)]
            decoder_layers.append(Conv2D(filters, (3, 3), activation='relu', padding='same'))
            if maxpoll_layers[-(i+1)] == 1:
                decoder_layers.append(UpSampling2D((2, 2))) #maxpool ao contrário, aumenta a resolução 
            
        decoder_layers.append(Conv2D(self.input_shape[2], (3, 3), activation='sigmoid', padding='same'))

        gc.collect()
        k.clear_session()
        
        return encoder_layers, decoder_layers, latent_dim

    def construir_modelo(self, salvar=False, filters_list=[8,16,32,64,128]):
        encoder_layers, decoder_layers, latent_dim = self.calcular_camadas(filters_list)
        
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
            print(x)
        self.decoder = Model(latent_inputs, x, name='decoder')
        
        # Construir autoencoder
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(inputs, decoded, name=f'autoencoder')

        if salvar == True and self.nome_modelo !=None:
            save_dir = os.path.join("Modelos", self.nome_modelo)
            dir_raiz = os.path.join(save_dir, "Modelo-Base")
            dir_modelo = os.path.join(dir_raiz, "Estrutura")
            dir_pesos = os.path.join(dir_raiz, "Pesos")

            criar_diretorio_novo(save_dir)
            criar_diretorio_novo(dir_raiz)
            criar_diretorio_novo(dir_modelo)
            criar_diretorio_novo(dir_pesos)

            self.autoencoder.save(f"{dir_modelo}/{self.nome_modelo}.keras")

        return self.autoencoder

    def compilar_modelo(self, optimizer='adam', loss='mse'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def Dataset(self, treino, validacao, teste):
        self.treino = treino
        self.validacao = validacao
        self.teste = teste

    def treinar_autoencoder(self, salvar=False,nome_da_base='', epocas=10, batch_size=16):
        print("Treinando o modelo: ", self.nome_modelo)
        checkpoint_path = 'Pesos_parciais/weights-improvement-{epoch:02d}-{val_loss:.2f}.weights.h5'
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                        save_weights_only=True, 
                                        monitor='val_loss', 
                                        mode='max', 
                                        save_best_only=True, 
                                        verbose=1)

        history = self.autoencoder.fit(self.treino, epochs=epocas,callbacks=[cp_callback],batch_size=batch_size, validation_data=(self.validacao))
        pd.DataFrame(history.history).plot()

        shutil.rmtree('Pesos_parciais')

        if salvar == True and self.nome_modelo != None:
            save_dir = os.path.join("Modelos", self.nome_modelo)
            dir_raiz = os.path.join(save_dir, "Modelo-Base")
            dir_modelo = os.path.join(dir_raiz, "Estrutura")
            dir_pesos = os.path.join(dir_raiz, "Pesos")


            if os.listdir(dir_modelo) == []:
                criar_diretorio_novo(dir_modelo)
                self.autoencoder.save(f"{dir_modelo}/{self.nome_modelo}.keras")

            criar_diretorio_novo(dir_pesos)

            self.autoencoder.save_weights(f"{dir_pesos}/{self.nome_modelo}_Base:{nome_da_base}.weights.h5")

        x, y = next(self.treino)
        plot_autoencoder(x, self.autoencoder, self.input_shape[0], self.input_shape[1])

    def carrega_modelo(self, modelo:str, pesos:str=None):
        self.autoencoder = tf.keras.models.load_model(modelo)

        self.nome_modelo = retorna_nome(modelo)

        if pesos !=None:  
            self.autoencoder.load_weights(pesos)

        self.decoder = self.autoencoder.get_layer('decoder')
        self.encoder = self.autoencoder.get_layer('encoder')

        self.autoencoder.summary()

        return self.autoencoder, self.encoder, self.decoder

    def fineTuning(self, treino, validacao, teste, epocas=10, batch_size=32, nome=None, nome_da_base=None, n_camadas=3, salvar=False):

        for layer in self.encoder.layers[:n_camadas]:  
            layer.trainable = False

        for layer in self.decoder.layers[-n_camadas:]: 
            layer.trainable = False

        self.autoencoder.compile(optimizer=Adam(learning_rate=1e-5), loss='mse')  

        checkpoint_path = 'Pesos_parciais/weights-improvement-{epoch:02d}-{val_loss:.2f}.weights.h5'
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                        save_weights_only=True, 
                                        monitor='val_loss', 
                                        mode='max', 
                                        save_best_only=True, 
                                        verbose=1)

        history = self.autoencoder.fit(treino, epochs=epocas,callbacks=[cp_callback],batch_size=batch_size, validation_data=(validacao))
        pd.DataFrame(history.history).plot()

        if salvar == True and nome != None:
            save_dir = os.path.join("Modelos", self.nome_modelo)
            dir_raiz = os.path.join(save_dir, "Modelo-FineTuning")
            dir_modelo = os.path.join(dir_raiz, "Estrutura")
            dir_pesos = os.path.join(dir_raiz, "Pesos")

            criar_diretorio_novo(dir_raiz)
            criar_diretorio_novo(dir_modelo)
            criar_diretorio_novo(dir_pesos)

            self.autoencoder.save(f"{dir_modelo}/{nome}_FineTuning_{nome_da_base}.keras")
            self.autoencoder.save_weights(f"{dir_pesos}/{nome}_FineTuning_{nome_da_base}.weights.h5")

        x, y = next(treino)
        plot_autoencoder(x, self.autoencoder, self.input_shape[0], self.input_shape[1])

#Exemplo de uso:
#gerador = Gerador(min_layers=2, max_layers=6) -> deve ser proporcional ao input_shape
#modelo = gerador.construir_modelo()
#encoder = gerador.encoder
#decoder = gerador.decoder
#gerador.Dataset(treino, validacao, teste)
#gerador.treinar_autoencoder(epocas=30, salvar=True) -> treina o autoencoder, plota já a reconstrução 


"""------------------Funções para usar diversos autoencoders----------------------"""
def cria_modelos(n_modelos=10, nome_modelo=None):
    for i in range(n_modelos):  
        limpa_memoria()

        Modelo = Gerador(input_shape=(64, 64, 3))
        Modelo.setNome(f'{nome_modelo}-{i}')
        modelo = Modelo.construir_modelo(salvar=True)

        modelo.summary()

        encoder = Modelo.encoder  
        decoder = Modelo.decoder  

        encoder.summary()
        decoder.summary()

        del Modelo, modelo, encoder, decoder
    
def treina_modelos(treino, validacao, teste, nome_modelo=None, nome_base=None, n_epocas=10, batch_size=8):

    modelos = os.listdir("Modelos")
    modelos_para_treinar = []
    for modelo in modelos:
        if os.path.exists(os.path.join('Modelos', modelo)):
            if nome_modelo in modelo:
                modelo_base = os.path.join('Modelos', modelo, 'Modelo-Base')
                peso, estrutura = os.listdir(modelo_base)
                m = os.listdir(os.path.join(modelo_base , estrutura))
                dir_modelo = os.path.join(modelo_base, estrutura, m[0])
                modelos_para_treinar.append(dir_modelo)
            else:
                pass
        else:
            print(f"O diretório {modelo} não existe.")

    for i, m in enumerate(sorted(modelos_para_treinar)):
        limpa_memoria()

        Modelo = Gerador()

        AutoencoderModelo = Modelo.carrega_modelo(m)

        Modelo.Dataset(treino, validacao, teste)

        Modelo.compilar_modelo()

        Modelo.setNome(f"{nome_modelo}-{i}")

        limpa_memoria()

        Modelo.treinar_autoencoder(epocas=n_epocas, salvar=True, nome_da_base=nome_base ,batch_size=batch_size)
    
        del AutoencoderModelo, Modelo
 
def fine_tuning_modelos(treino, validacao, teste, nome_modelo=None, nome_base=None, n_epocas=10, camadas=3, salvar=False, batch_size=32):

    modelos = os.listdir("Modelos")

    pares_modelo_peso = []

    for modelo in sorted(modelos):
        modelo_base = os.path.join("Modelos", modelo, "Modelo-Base")
        peso, estrutura = os.listdir(modelo_base)
        p = os.listdir(os.path.join(modelo_base, peso))
        e = os.listdir(os.path.join(modelo_base, estrutura))

        peso = os.path.join(modelo_base, peso, p[0])
        estrutura = os.path.join(modelo_base, estrutura ,e[0])

        par = [estrutura, peso]

        pares_modelo_peso.append(par)

    for m, p in pares_modelo_peso:

        limpa_memoria()

        Modelo = Gerador()
        Modelo.carrega_modelo(m,p)
        Modelo.fineTuning(treino, validacao, teste, epocas=n_epocas, nome=retorna_nome(m), nome_da_base=nome_base, n_camadas=camadas, salvar=salvar, batch_size=batch_size)

        del Modelo


"""------------------Gerador de Classificador----------------------"""

class GeradorClassificador:
    def __init__(self, encoder, pesos, nome_modelo:str=None):
        self.encoder = encoder
        self.nome_modelo = nome_modelo
        self.model = self.modelo(self.encoder)
        self.carrega_pesos(pesos)
        self.compila()
        self.treino = None
        self.validacao = None
        self.teste = None

    def modelo(self, encoder):
        for layer in self.encoder.layers:
            layer.trainable = False

        classificador = keras.models.Sequential([
                self.encoder,  
                keras.layers.Dropout(0.2),  
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(128,activation='relu'),  
                keras.layers.Dense(2, activation='softmax')  
            ], name=f'classificador{self.nome_modelo}')
        
        return classificador
    
    def setNome(self, nome):
        self.nome_modelo = nome
    
    def compila(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def carrega_pesos(self, peso):
        try:
            self.model.load_weights(peso, skip_mismatch=True)
            print("Pesos carregados com sucesso")
        except Exception as e:
            print(f"Erro ao carregar os pesos: {e}")

    def treinamento(self, salvar=False, epocas=10, batch_size=32):
        checkpoint_path = 'Pesos_parciais/weights-improvement-{epoch:02d}-{val_loss:.2f}.weights.h5'
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                        save_weights_only=True, 
                                        monitor='val_loss', 
                                        mode='max', 
                                        save_best_only=True, 
                                        verbose=1)


        history = self.model.fit(self.treino, epochs=epocas, callbacks=[cp_callback], batch_size=batch_size ,validation_data=self.validacao)
        pd.DataFrame(history.history).plot()

        """if salvar == True and self.nome_modelo !=None:
            save_dir_models = "Modelos_keras/Classificador_Gerados"
            save_dir_weights = "weights_finais/Classificador_Gerados"

            if not os.path.exists(save_dir_models):
                os.makedirs(save_dir_models)

            if not os.path.exists(save_dir_weights):
                os.makedirs(save_dir_weights)

            self.model.save(f"{save_dir_models}/Classificador-{self.nome_modelo}.keras")
            self.model.save_weights(f"{save_dir_weights}/Classificador-{self.nome_modelo}.weights.h5")"""

    def Dataset(self, treino, validacao, teste):
        self.treino = treino
        self.validacao = validacao
        self.teste = teste

    def predicao(self, teste_csv):
        predicoes = self.model.predict(self.teste)
        predicoes = np.argmax(predicoes, axis=1)

        print(predicoes)

        y_verdadeiro = mapear(teste_csv['classe'])

        plot_confusion_matrix(y_verdadeiro, predicoes, ['Empty', 'Occupied'], f'{self.nome_modelo}')

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

"""------------------Funções para usar diversos classificadores----------------------"""
def cria_classificadores(n_modelos=10, nome_modelo=None, base_usada=None, treino=None, validacao=None, teste=None, teste_csv=None):
    gerador = Gerador()
    for i in range(n_modelos):  
        limpa_memoria()

        gerador.carrega_modelo(f'Modelos/{nome_modelo}-{i}/Modelo-Base/Estrutura/{nome_modelo}-{i}.keras')
        encoder = gerador.encoder

        classificador = GeradorClassificador(encoder=encoder, pesos=f'Modelos/{nome_modelo}-{i}/Modelo-Base/Pesos/{nome_modelo}-{i}_Base:{base_usada}.weights.h5')
        classificador.Dataset(treino, validacao, teste)
        classificador.compila()
        classificador.treinamento(epocas=10)
        classificador.predicao(teste_csv)

        limpa_memoria()



"""_________________Primeiro Autoencoder/Gerador______________________"""

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