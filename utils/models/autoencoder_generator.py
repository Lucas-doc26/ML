import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Reshape, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import mixed_precision
import tensorflow.keras.backend as k


# Importações para manipulação de dados e operações básicas
import numpy as np
import pandas as pd
import os
import gc
import re
import shutil
import datetime
from pathlib import Path

path = Path('/home/lucas/PIBIC')

# Importações locais do projeto
from utils.path_manager import *
from utils.preprocessing import map_classes_to_binary
from utils.gpu import clear_session
from utils.config import *
from utils.view import *


os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"


"""------------------Gerador de Autoencoders----------------------"""

class AutoencoderGenerator:
    """
    Dados de entrada: input_shape=(64, 64, 3), min_layers=2, max_layers=6\n
    """

    def __init__(self, input_shape=(64, 64, 3), min_layers=5, max_layers=8, model_name:str=None):
        self.input_shape = input_shape
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.train = None
        self.validation = None
        self.test = None
        self.model_name = model_name

    def set_model_name(self, name):
        self.model_name = name

    def get_model_name(self):
        print(self.model_name)
    
    def get_encoder(self):
        return self.encoder
    
    def calculate_layers(self, filters_list=[8,16,32,64,128]):
        num_layers = np.random.randint(self.min_layers, self.max_layers + 1) #+1 por conta do randint ser somente de (min - max-1)
        
        encoder_layers = []
        decoder_layers = []
        maxpoll_layers = []
        batch = False
        leaky = False


        shape_atual = self.input_shape
        filter_sizes = [] 
        
        # Encoder
        for i in range(num_layers):
            filters = np.random.choice(filters_list)
            filter_sizes.append(filters)
            encoder_layers.append(Conv2D(filters, (3, 3), activation='relu', padding='same'))
            maxpoll_layers.append(np.random.choice([0, 0, 0, 1, 1, 1]))

            if len(maxpoll_layers) > 4:
                maxpoll_layers[i] = 0
            elif maxpoll_layers[i] == 1: #limitando o valor para 4, por conta de ser 64x64 
                encoder_layers.append(MaxPooling2D((2, 2), padding='same'))
                shape_atual = (shape_atual[0] // 2, shape_atual[1] // 2, filters) #att por conta da divisão do maxpool
            else:
                shape_atual = (shape_atual[0], shape_atual[1], filters) 

        latent_dim = np.random.randint(256,512) #256,512 o padrão

        #Hiperparâmetros
        if np.random.choice(([0,0,1])):
            encoder_layers.append(BatchNormalization()) 
            batch = True
        if np.random.choice(([0,0,1])):
            encoder_layers.append(LeakyReLU(alpha=0.5)) #Função de ativação 
            leaky = True
        if np.random.choice(([0,0,1])):
            encoder_layers.append(Dropout(np.random.choice(([0.4 , 0.3 , 0.2]))))

        #encoder_layers.append(Dropout(0.4))
        encoder_layers.append(Flatten())

        encoder_layers.append(Dense(latent_dim, activation='relu')) #transformo o meu shape_atual no meu latent_dim 

        # Decoder
        decoder_layers = [
            Dense(np.prod(shape_atual), activation='relu'), #calcula o vetor latente 
            Reshape(shape_atual)
        ]
        
        for i in range(num_layers):
            filters = filter_sizes[-(i+1)]
            if maxpoll_layers[-(i+1)] == 1:
                decoder_layers.append(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu'))
            else:
                decoder_layers.append(Conv2DTranspose(filters, (3, 3), strides=(1, 1), padding='same', activation='relu'))

        if batch:
            decoder_layers.append(BatchNormalization())  
        if leaky:
            decoder_layers.append(LeakyReLU(alpha=0.5))
    
        decoder_layers.append(Conv2D(self.input_shape[2], (3, 3), activation='sigmoid', padding='same'))

        gc.collect()
        k.clear_session()
        
        return encoder_layers, decoder_layers, latent_dim

    def build_model(self, save=True, filters_list=[8,16,32,64,128]):
        # Limpar antigas referências 
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        encoder_layers, decoder_layers, latent_dim = self.calculate_layers(filters_list)
        
        # Construir encoder
        inputs = Input(shape=self.input_shape)
        x = inputs
        for layer in encoder_layers:
            x = layer(x) #as camadas vão sequencialmente sendo adicionadas, passo o input como argumento para cada camada
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
        self.autoencoder = Model(inputs, decoded, name=f'autoencoder')

        if save == True and self.model_name != None:
            save_dir = os.path.join(path, "Modelos", self.model_name)
            dir_root = os.path.join( save_dir, "Modelo-Base")
            dir_model = os.path.join(dir_root, "Estrutura")
            dir_weights = os.path.join(dir_root, "Pesos")

            recreate_folder_force(save_dir) #Modelos/Modelo_Kyoto-0
            recreate_folder_force(dir_root) #Modelos/Modelo_Kyoto-0/Modelo-Base
            recreate_folder_force(dir_model) #Modelos/Modelo_Kyoto-0/Modelo-Base/Estrutura
            recreate_folder_force(dir_weights) #Modelos/Modelo_Kyoto-0/Modelo-Base/Pesos

            self.autoencoder.save(f"{dir_model}/{self.model_name}.keras")
            #salvo a estrutura do modelo

        return self.autoencoder

    def model_compile(self, optimizer='adam', loss='mse'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss, jit_compile=False)

    def dataset(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test

    def train_autoencoder(self, save=False ,autoencoder_base='', epochs=10, batch_size=4):
        print("Treinando o modelo: ", self.model_name)
        checkpoint_path = os.path.join(path, 'Pesos/Pesos_parciais/weights-improvement-{epoch:02d}-{val_loss:.2f}.weights.h5')

        now = datetime.datetime.now().strftime("%d%m%y-%H%M")
        log_dir = f"logs/fit/{self.model_name}-Autoencoder-{now}"
        log_dir = os.path.join(path, f'Modelos/{self.model_name}/Modelo-Base', log_dir)

        recreate_folder_force(log_dir)

        #Criando o callback do tensorboard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        #Criando o callback do checkpoint
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                        save_weights_only=True, 
                                        monitor='val_loss', 
                                        mode='min', 
                                        save_best_only=True, 
                                        verbose=1)

        #Criando o callback do early stopping
        early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=150,  #interrompe se não melhorar
                               restore_best_weights=True, 
                               verbose=1)

        # Para caso quiser acompanhar o treinamento
        history = self.autoencoder.fit(self.train, epochs=epochs,
                                       callbacks=[cp_callback, early_stopping, tensorboard_callback],
                                       batch_size=batch_size, 
                                       validation_data=(self.validation))
        
        #Criando o dataframe com o histórico
        df = pd.DataFrame(history.history) 

        del history

        #Removendo o diretório de pesos parciais
        dir_weights_save = os.path.join(path, 'Pesos/Pesos_parciais')
        if os.path.isdir(dir_weights_save):
            shutil.rmtree(dir_weights_save)

        #Salvando o modelo
        if save == True and self.model_name != None:
            save_dir = os.path.join(path, "Modelos", self.model_name)
            dir_root = os.path.join(save_dir, "Modelo-Base")
            dir_model = os.path.join(dir_root, "Estrutura")
            dir_weights = os.path.join(dir_root, "Pesos")
            dir_imagens = os.path.join(save_dir, "Plots")
            

            if os.listdir(dir_model) == []:
                recreate_folder_force(dir_model)
                self.autoencoder.save(f"{dir_model}/{self.model_name}.keras")

            if not os.path.isdir(dir_weights):
                os.makedirs(dir_weights)

            self.autoencoder.save_weights(f"{dir_weights}/{self.model_name}_Base-{autoencoder_base}.weights.h5")
            #Modelos/Modelo_Kyoto-0/Modelo-Base/Pesos/Modelo_Kyoto-0_Base-CNR.weights.h5

            if not os.path.isdir(dir_imagens):
                os.makedirs(dir_imagens)

        x, y = next(self.test)

        print(self.input_shape[0])
        print(self.input_shape[1])

        plot_history_autoencoder(df, 
                     save_dir==dir_imagens, 
                     model_name=self.model_name, 
                     autoencoder_base=autoencoder_base)
        
        plot_autoencoder(x, self.autoencoder, self.input_shape[0], self.input_shape[1], dir_imagens, autoencoder_base)

    def load_model(self, modelo:Path, weights:Path=None):
        self.autoencoder = tf.keras.models.load_model(modelo)

        self.model_name = return_model_name(modelo)

        if weights == False:
            print("Carregado somente a estrutura do modelo!")
        elif weights !=None:  
            self.autoencoder.load_weights(weights)
        

        self.decoder = self.autoencoder.get_layer('decoder')
        self.encoder = self.autoencoder.get_layer('encoder')

        self.autoencoder.summary()

        return self.autoencoder, self.encoder, self.decoder

    def predict(self):
        x,y = next(self.test)
        plot_autoencoder(x, self.autoencoder, self.input_shape[0],self.input_shape[1])
        pred = self.autoencoder.predict(x[0].reshape((1,self.input_shape[0], self.input_shape[1],3)))
        pred_img = normalize_image(pred[0])

        return x[0], pred_img

def generate_models(n_models=10, model_name=None, filters_list=[8,16,32,64,128], input=(64,64,3), min_layers=3, max_layers=5):
    """
    Gera nº modelos de autoencoders.    
    """

    for i in range(n_models):  
        clear_session()

        autoencoder = AutoencoderGenerator(input_shape=input, max_layers=max_layers, min_layers=min_layers)
        autoencoder.set_model_name(f'{model_name}-{i}')
        model = autoencoder.build_model(save=True, filters_list=filters_list)

        model.summary()

        encoder = autoencoder.encoder  
        decoder = autoencoder.decoder  

        encoder.summary()
        decoder.summary()

        del autoencoder, encoder, decoder
        clear_session()
    
def generate_models2(n_models=10, model_name=None, filters_list=[8,16,32,64,128], input=(64,64,3), min_layers=3, max_layers=5):
    for i in range(n_models):  
        clear_session()

        autoencoder = AutoencoderGenerator(input_shape=input, max_layers=max_layers, min_layers=min_layers)
        autoencoder.set_model_name(f'{model_name}-{i}')
        model = autoencoder.build_model(save=True, filters_list=filters_list)

        model.summary()

        encoder = autoencoder.encoder  
        decoder = autoencoder.decoder  

        encoder.summary()
        decoder.summary()

        yield autoencoder, encoder, decoder
        clear_session()
    

def train_models(train, validation, test, model_name=None, autoencoder_base=None, n_epochs=10, batch_size=4, input_shape=(64,64,3)):
    """
    Treina nº modelos de autoencoders.
    """
    models = os.listdir(os.path.join(path,"Modelos"))
    pattern = re.compile(f"{re.escape(model_name)}-\\d+$") #crio um padrão para pegar os modelos: nome_modelo-Decimal
    models = [m for m in models if pattern.fullmatch(m)]#filtro pelo padrão que eu criei

    models_to_train = []
    for model in models:
        if os.path.exists(os.path.join(path, 'Modelos', model)) and model_name in model and "Fusoes" not in model:
            model_base = os.path.join(path, 'Modelos', model, 'Modelo-Base')
            
            # Mapeamento explícito dos arquivos esperados
            # Fazendo isso, pois SOs diferentes retornam arquivos em ordens diferentes
            files = os.listdir(model_base)
            structure = next((f for f in files if 'estrutura' in f.lower()), None)
            weights = next((f for f in files if 'peso' in f.lower()), None)
            log = next((f for f in files if 'log' in f.lower()), None)
            
            if not structure or not weights:
                print(f"Aviso: Arquivos necessários não encontrados em {model_base}")
                continue
                
            print(f"Peso encontrado: {weights}")
            m = os.listdir(os.path.join(model_base, structure))
            print(f"Arquivos na estrutura: {m}")
            dir_model = os.path.join(model_base, structure, m[0])
            models_to_train.append(dir_model)

    for i, m in enumerate(sorted(models_to_train)):
        clear_session()

        autoencoder = AutoencoderGenerator(input_shape=input_shape)
        autoencoder.load_model(m)
        autoencoder.dataset(train, validation, test)
        autoencoder.model_compile()
        autoencoder.set_model_name(f"{model_name}-{i}")
        clear_session()

        autoencoder.train_autoencoder(epochs=n_epochs, save=True, autoencoder_base=autoencoder_base, batch_size=batch_size)

        del autoencoder
        gc.collect()  
        tf.keras.backend.clear_session() 
