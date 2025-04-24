import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Flatten, Dense, Reshape, Conv2D, MaxPooling2D, 
                                     UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout, 
                                     LeakyReLU)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, plot_model
import keras
import keras.backend as k  
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import gc
import re
import shutil
from visualizacao import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0' = tudo, '1' = warnings, '2' = info, '3' = apenas erros graves

def retorna_nome_base(caminho):
    nome = caminho.split('/')[-1]
    nome = nome.rsplit('.csv', 1)[0]
    nome = nome.split('_')
    return (nome[0], nome[2])

from Preprocessamento import preprocessamento_dataframe

#Definindo reprotudibilidade 
SEED = 42
tf.random.set_seed(SEED)

# NumPy
np.random.seed(SEED)

# Random (Python)
random.seed(SEED)

# Definir variável de ambiente para operações determinísticas
os.environ['PYTHONHASHSEED'] = str(SEED)

#Usando ele como path para usar o espaço sobrando no segundo hd
path = r'/home/lucas/PIBIC'

#Configurações da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
"""if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Evita alocação total de memória
    except RuntimeError as e:
        print(e)"""

"""----------------------Funções Auxiliares-----------------------"""
def limpa_memoria():
    k.clear_session()  
    gc.collect() 
    tf.config.experimental.reset_memory_stats('GPU:0') #limpa memória da gpu

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
    if not os.path.isdir(os.path.join(path, "Modelos")):
        os.makedirs(os.path.join(path, "Modelos"))

    if not os.path.isdir(os.path.join(path, "Pesos/Pesos_parciais")):
        os.makedirs(os.path.join(path, "Pesos/Pesos_parciais"))

    if not os.path.isdir(os.path.join(path, "Modelos/Plots")):
        os.makedirs(os.path.join(path, "Modelos/Plots"))

def recria_diretorio(caminho):
    if os.path.exists(caminho):
        shutil.rmtree(caminho)
    os.makedirs(caminho)

def mapear(classes):
    return np.array([1 if classe == '1' else 0 for classe in classes])

def retorna_nome_df(df):
    faculdades = ['PUC', 'UFPR04', 'UFPR05']
    mask = df['caminho_imagem'].str.contains('|'.join(faculdades), regex=True)
    #/datasets/PKLot/PKLotSegmented/PUC/Cloudy/2012-10-16/Empty/2012-10-16_05_56_42#081.jpg,1
    if mask.any():  
        nome = df.loc[mask, 'caminho_imagem'].iloc[0].split('/')[4]  # Pega a faculdade
    else:
        nome = df['caminho_imagem'].iloc[0].split('/')[6]  
        #/datasets/CNR-EXT-Patches-150x150/PATCHES/SUNNY/2016-01-15/camera9/S_2016-01-15_11.40_C09_326.jpg,0

    return nome

"""caminho = "CSV/PUC/PUC_Segmentado_Teste.csv"
df = pd.read_csv(caminho)
nome_base_teste = retorna_nome_df(df)
print(nome_base_teste)"""


def extrair_nome_modelo1(nome):
    #Modelo_Kyoto-0.keras
    partes = nome.replace("-", "_").split("_")
    numero = partes[3].split('.')

    nome_modelo = "_".join([partes[1], partes[2]])
    nome_modelo = "-".join([nome_modelo, numero[0]])

    return nome_modelo

def normalize(image):
        image = np.clip(image, 0, 1)  # Garante que a imagem esteja no intervalo [0, 1]
        if isinstance(image, tf.Tensor): #Caso seja um tensor, ele transforma em np para evitar uso de vram
            image = image.numpy()
        return (image - image.min()) / (image.max() - image.min()) if image.max() != image.min() else image

"""------------------Gerador de Autoencoders----------------------"""

class Gerador:
    """
    Dados de entrada: input_shape=(64, 64, 3), min_layers=2, max_layers=6\n
    """

    def __init__(self, input_shape=(64, 64, 3), min_layers=5, max_layers=8, nome_modelo:str=None):
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
        #aqui eu crio o /Modelos e dependências 

    def setNome(self, nome):
        self.nome_modelo = nome

    def getNome(self):
        print(self.nome)
    
    def calcular_camadas(self, filters_list=[8,16,32,64,128]):
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

        latent_dim = np.random.randint(256,512)

        if np.random.choice(([0,0,1])):
            encoder_layers.append(BatchNormalization()) 
            batch = True
        if np.random.choice(([0,0,1])):
            encoder_layers.append(LeakyReLU(alpha=0.5)) #Função de ativação 
            leaky = True
        if np.random.choice(([0,0,1])):
            encoder_layers.append(Dropout(np.random.choice(([0.4 , 0.3 , 0.2])))) 

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

    def construir_modelo(self, salvar=True, filters_list=[8,16,32,64,128]):
        # Limpar antigas referências 
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

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
            save_dir = os.path.join(path, "Modelos", self.nome_modelo)
            dir_raiz = os.path.join( save_dir, "Modelo-Base")
            dir_modelo = os.path.join(dir_raiz, "Estrutura")
            dir_pesos = os.path.join(dir_raiz, "Pesos")

            recria_diretorio(save_dir) #Modelos/Modelo_Kyoto-0
            recria_diretorio(dir_raiz) #Modelos/Modelo_Kyoto-0/Modelo-Base
            recria_diretorio(dir_modelo) #Modelos/Modelo_Kyoto-0/Modelo-Base/Estrutura
            recria_diretorio(dir_pesos) #Modelos/Modelo_Kyoto-0/Modelo-Base/Pesos

            self.autoencoder.save(f"{dir_modelo}/{self.nome_modelo}.keras")
            #salvo a estrutura do modelo

        return self.autoencoder

    def compilar_modelo(self, optimizer='adam', loss='mse'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def Dataset(self, treino, validacao, teste):
        self.treino = treino
        self.validacao = validacao
        self.teste = teste

    def treinar_autoencoder(self, salvar=False ,nome_da_base='', epocas=10, batch_size=64):
        print("Treinando o modelo: ", self.nome_modelo)
        checkpoint_path = os.path.join(path, 'Pesos/Pesos_parciais/weights-improvement-{epoch:02d}-{val_loss:.2f}.weights.h5')
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                        save_weights_only=True, 
                                        monitor='val_loss', 
                                        mode='min', 
                                        save_best_only=True, 
                                        verbose=1)

        early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=150,  #interrompe se não melhorar
                               restore_best_weights=True, 
                               verbose=1)

        # Para caso quiser acompanhar o treinamento
        history = self.autoencoder.fit(self.treino, epochs=epocas,
                                       callbacks=[cp_callback, early_stopping],
                                       batch_size=batch_size, 
                                       validation_data=(self.validacao))
        df = pd.DataFrame(history.history) # crio um df com o histórico

        del history

        dir_pesos_save = os.path.join(path, 'Pesos/Pesos_parciais')
        if os.path.isdir(dir_pesos_save):
            shutil.rmtree(dir_pesos_save)

        if salvar == True and self.nome_modelo != None:
            save_dir = os.path.join(path, "Modelos", self.nome_modelo)
            dir_raiz = os.path.join(save_dir, "Modelo-Base")
            dir_modelo = os.path.join(dir_raiz, "Estrutura")
            dir_pesos = os.path.join(dir_raiz, "Pesos")
            dir_imagens = os.path.join(save_dir, "Plots")
            

            if os.listdir(dir_modelo) == []:
                recria_diretorio(dir_modelo)
                self.autoencoder.save(f"{dir_modelo}/{self.nome_modelo}.keras")

            if not os.path.isdir(dir_pesos):
                os.makedirs(dir_pesos)

            self.autoencoder.save_weights(f"{dir_pesos}/{self.nome_modelo}_Base-{nome_da_base}.weights.h5")
            #Modelos/Modelo_Kyoto-0/Modelo-Base/Pesos/Modelo_Kyoto-0_Base-CNR.weights.h5

            if not os.path.isdir(dir_imagens):
                os.makedirs(dir_imagens)

            

        x, y = next(self.teste)


        plot_history(df, 
                     save_dir=dir_imagens, 
                     modelo=self.nome_modelo, 
                     base_do_autoencoder=nome_da_base)
        
        plot_autoencoder(x, self.autoencoder, self.input_shape[0], self.input_shape[1],caminho_para_salvar=dir_imagens)

    def carrega_modelo(self, modelo:str, pesos:str=None):
        self.autoencoder = tf.keras.models.load_model(modelo)

        self.nome_modelo = retorna_nome(modelo)

        if pesos == False:
            print("Carregado somente a estrutura do modelo!")
        elif pesos !=None:  
            self.autoencoder.load_weights(pesos)
        

        self.decoder = self.autoencoder.get_layer('decoder')
        self.encoder = self.autoencoder.get_layer('encoder')

        self.autoencoder.summary()

        return self.autoencoder, self.encoder, self.decoder

    def predicao(self):
        x,y = next(self.teste)
        plot_autoencoder(x, self.autoencoder, self.input_shape[0],self.input_shape[1])
        pred = self.autoencoder.predict(x[0].reshape((1,self.input_shape[0], self.input_shape[1],3)))
        pred_img = normalize(pred[0])

        return x[0], pred_img


#Exemplo de uso:
#gerador = Gerador(min_layers=2, max_layers=6) -> deve ser proporcional ao input_shape
#modelo = gerador.construir_modelo()
#encoder = gerador.encoder
#decoder = gerador.decoder
#gerador.Dataset(treino, validacao, teste)
#gerador.treinar_autoencoder(epocas=30, salvar=True) -> treina o autoencoder, plota já a reconstrução 
#preci

"""------------------Funções para usar diversos autoencoders----------------------"""
def cria_modelos(n_modelos=10, nome_modelo=None, filters_list=[8,16,32,64,128]):
    for i in range(n_modelos):  
        limpa_memoria()

        Modelo = Gerador(input_shape=(64, 64, 3))
        Modelo.setNome(f'{nome_modelo}-{i}')
        modelo = Modelo.construir_modelo(salvar=True, filters_list=filters_list)

        modelo.summary()

        encoder = Modelo.encoder  
        decoder = Modelo.decoder  

        encoder.summary()
        decoder.summary()

        del Modelo, modelo, encoder, decoder
        limpa_memoria()
    
def treina_modelos(treino, validacao, teste, nome_modelo=None, nome_base=None, n_epocas=10, batch_size=16):
    modelos = os.listdir(os.path.join(path,"Modelos"))
    modelos_para_treinar = []
    
    for modelo in modelos:
        if os.path.exists(os.path.join(path, 'Modelos', modelo)) and nome_modelo in modelo and "Fusoes" not in modelo:
            modelo_base = os.path.join(path, 'Modelos', modelo, 'Modelo-Base')
            estrutura, peso  = sorted(os.listdir(modelo_base)) # no computador da puc ele retorna (p,e) / no meu pc ele retorna (e,p) -> ver uma maneira de arrumar isso
            print(peso)
            m = os.listdir(os.path.join(modelo_base, estrutura))
            print(m)
            dir_modelo = os.path.join(modelo_base, estrutura, m[0])
            modelos_para_treinar.append(dir_modelo)

    for i, m in enumerate(sorted(modelos_para_treinar)):
        limpa_memoria()

        Modelo = Gerador()
        Modelo.carrega_modelo(m)
        Modelo.Dataset(treino, validacao, teste)
        Modelo.compilar_modelo()
        Modelo.setNome(f"{nome_modelo}-{i}")
        limpa_memoria()

        Modelo.treinar_autoencoder(epocas=n_epocas, salvar=True, nome_da_base=nome_base, batch_size=batch_size)

        del Modelo
        gc.collect()  
        tf.keras.backend.clear_session() 

"""------------------Gerador de Classificador----------------------"""

class GeradorClassificador:
    def __init__(self, encoder=None, pesos=None, nome_modelo:str=None, nome_autoencoder:str='Sem-Peso'):
        self.encoder = encoder
        self.nome_modelo = nome_modelo

        self.model = self.modelo(self.encoder)
        self.carrega_pesos(pesos)

        if self.encoder != None:
            self.compila()

        self.treino = None
        self.validacao = None
        self.teste = None
        self.nome_autoencoder = nome_autoencoder
        
        if nome_modelo is None:
            raise ValueError("O nome do modelo não foi definido.")
        else:
            self.verifica_dirs()

    def modelo(self, encoder):
        if encoder != None:
            #eu congelo as camadas do encoder, não faço fine-tunning
            for layer in self.encoder.layers:
                layer.trainable = False
            encoder.trainable = False

            #crio o classificador com o enconder
            classificador = keras.models.Sequential([
                    self.encoder,  
                    keras.layers.Dropout(0.2),  
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(128,activation='relu'),  
                    keras.layers.Dense(2, activation='softmax')  
                ], name=f'classificador{self.nome_modelo}')
            
        else:
            classificador = keras.models.Sequential([ 
                    keras.layers.Dropout(0.2),  
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(128,activation='relu'),  
                    keras.layers.Dense(2, activation='softmax')  
                ], name=f'classificador{self.nome_modelo}')
            
        return classificador
    
    def verifica_dirs(self):
        save_dir = os.path.join(path, "Modelos", self.nome_modelo)
        dir_raiz = os.path.join(save_dir, f"Classificador-{self.nome_autoencoder}")
        dir_modelo = os.path.join(dir_raiz, "Estrutura")
        dir_pesos = os.path.join(dir_raiz, "Pesos")

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if not os.path.isdir(dir_raiz):
            os.mkdir(dir_raiz)
        if not os.path.isdir(dir_modelo):
            os.mkdir(dir_modelo)
        if not os.path.isdir(dir_pesos):
            os.mkdir(dir_pesos)

    def salva_modelo(self, save_dir=''):
        if save_dir == '':
            path_save_modelo = os.path.join(path, f'Modelos/{self.nome_modelo}/Classificador-{self.nome_autoencoder}/Estrutura/Classificador_{self.nome_modelo}.keras')
        else:
            path_save_modelo = os.path.join(path, save_dir, f'Modelos/{self.nome_modelo}/Classificador-{self.nome_autoencoder}/Estrutura/Classificador_{self.nome_modelo}.keras' )
        self.model.save(path_save_modelo)
    
    def setNome(self, nome):
        self.nome_modelo = nome
    
    def compila(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def carrega_pesos(self, peso):
        if peso == None:
            print("Criação do modelo de classificação sem pesos")
        elif peso == False:
            print("Criação do modelo sem carregar os pesos")
        else:
            try:
                self.model.load_weights(peso, skip_mismatch=True)
                print("Pesos carregados com sucesso")
            except Exception as e:
                print(f"Erro ao carregar os pesos: {e}")
        limpa_memoria()

    def treinamento(self, salvar=False, epocas=10, batch_size=64, n_batchs=None, nome_base=None, pesos=True):

        #Indico se no meu treina em batch, estou carregando ou não os pesos do encoder
        #para poder salvar de forma correta, ou seja, muda o diretório de onde irá salvar
        if pesos:
            path_slv = path
        else:
            path_slv = os.path.join(path)

        checkpoint_path = os.path.join(path_slv, 'Pesos/Pesos_parciais/weights-improvement-{epoch:02d}-{val_loss:.2f}.weights.h5')

        cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                        save_weights_only=True, 
                                        monitor='accuracy', 
                                        mode='max', 
                                        save_best_only=True, 
                                        verbose=1)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(path_slv, f'Modelos/{self.nome_modelo}/Classificador-{self.nome_autoencoder}', log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        #Se quiser acompanhar
        history = self.model.fit(self.treino, epochs=epocas, callbacks=[cp_callback, tensorboard_callback], batch_size=batch_size ,validation_data=self.validacao)
        

        if os.path.isdir(os.path.join(path_slv,"Pesos/Pesos_parciais")):
            shutil.rmtree(os.path.join(path_slv,"Pesos/Pesos_parciais"), ignore_errors=True)
            os.makedirs(os.path.join(path_slv,"Pesos/Pesos_parciais"), exist_ok=True)

        if salvar == True:
            save_dir = os.path.join(path_slv, "Modelos", self.nome_modelo)
            dir_raiz = os.path.join(save_dir, f"Classificador-{self.nome_autoencoder}")
            dir_modelo = os.path.join(dir_raiz, "Estrutura")
            dir_pesos = os.path.join(dir_raiz, "Pesos")
            dir_pesos_base = os.path.join(dir_pesos, f'Treinado_em_{nome_base}')
            dir_imagens = os.path.join(save_dir, 'Plots')

            # criando o ../classificador 
            if not os.path.isdir(dir_raiz):
                os.makedirs(dir_raiz)

            # criando o ../Estrutura e ../Pesos
            if not os.path.isdir(dir_modelo) and not os.path.isdir(dir_pesos):
                os.makedirs(dir_modelo)
                os.makedirs(dir_pesos)

            # salvo o modelo
            self.model.save(os.path.join(dir_modelo, f'Classificador_{self.nome_modelo}.keras'))

            if n_batchs != None:
                #crio o ../Pesos/Treinado_em_PUC/
                if not os.path.isdir(dir_pesos_base):
                    os.makedirs(dir_pesos_base)

                # salvo em ../Pesos/Treinado_em_PUC/Classificador-NomeModelo-batchs-64
                #print("aqui")
                self.model.save_weights(f"{dir_pesos_base}/Classificador_{self.nome_modelo}_batchs-{n_batchs}.weights.h5")
            else:
                self.model.save_weights(f"{dir_pesos_base}/Classificador_{self.nome_modelo}.weights.h5")  

            if not os.path.isdir(dir_imagens):
                os.makedirs(dir_imagens)   

            save_history = os.path.join(dir_imagens, 'History')
            if not os.path.isdir(save_history):
                os.makedirs(save_history)
            plot_history_batch(history, save_history, self.nome_modelo, nome_base, n_batchs) 
            del history                  

    def Dataset(self, treino, validacao, teste):
        self.treino = treino
        self.validacao = validacao
        self.teste = teste

    def setTreino(self, treino):
        self.treino = treino

    def setTeste(self, teste):
        self.teste = teste

    def predicao(self, teste_csv):
        predicoes_np = self.model.predict(self.teste)
        predicoes = np.argmax(predicoes_np, axis=1)

        print(predicoes)

        y_verdadeiro = mapear(teste_csv['classe'])

        #plot_confusion_matrix(y_verdadeiro, predicoes, ['Empty', 'Occupied'], title=f'{self.nome_modelo}')

        accuracia = accuracy_score(y_verdadeiro, predicoes)

        return predicoes_np, accuracia
    
    def carrega_modelo(self, modelo:str, pesos:str):
        modelo_carregado = tf.keras.models.load_model(modelo)
        self.model = modelo_carregado
        self.carrega_pesos(pesos)

        self.model.summary(show_trainable=True)

        return self.model


#Exemplo de uso:
#classificador = GeradorClassificador(encoder=encoder, pesos="pesos.weights.h5") -> crio o classificador encima do encoder e seus pesos
#classificador.Dataset(treino, validacao, teste)
#classificador.treinamento() 
#classificador.predicao(teste_df) -> cria a matriz de confusão

"""------------------Funções para usar diversos classificadores----------------------"""
def cria_classificadores(n_modelos=10, nome_modelo=None, base_autoencoder='Sem-Peso', treino=None, validacao=None, teste=None):
    gerador = Gerador()
    for i in range(n_modelos):  
        limpa_memoria()

        gerador.carrega_modelo(os.path.join(path,f'Modelos/{nome_modelo}-{i}/Modelo-Base/Estrutura/{nome_modelo}-{i}.keras'))
        encoder = gerador.encoder

        pesos= os.path.join(path,f'Modelos/{nome_modelo}-{i}/Modelo-Base/Pesos/{nome_modelo}-{i}_Base-{base_autoencoder}.weights.h5')
        classificador = GeradorClassificador(encoder=encoder, pesos=pesos, nome_autoencoder=base_autoencoder, nome_modelo=f'{nome_modelo}-{i}')
        classificador.Dataset(treino, validacao, teste)
        #classificador.compila()
        classificador.salva_modelo()
        #classificador.treinamento(epocas=10)
        #classificador.predicao(teste_csv)

        limpa_memoria()

def treinamento_em_batch(nome_modelo, base_usada, base_autoencoder, treino_csv, validacao, teste, teste_csv ,salvar=True, n_epocas=10, pesos=True):
    path_slv = path

    #nome da base de treino do classificador
    nome, _ = retorna_nome_base(treino_csv)
    nome_base_teste = retorna_nome_df(teste_csv)

    batch_dir = f"CSV/{nome}/batches"
    batchs = sorted(os.listdir(batch_dir), key=lambda x: int(x.split("batch-")[1].split(".")[0]))
    print(batchs)
    precisoes = []
    n_batchs = [64,128,256,512,1024] 

    if not pesos:
        base_autoencoder = 'Sem-Peso'


    #modelo = classificador.model

    # crio o Modelo/Classificador/Resultados
    if not os.path.isdir(os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{base_autoencoder}/Resultados')):
        os.makedirs(os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{base_autoencoder}/Resultados'))

    # crio o Modelo/Classificador/Resultados/Treinados_em_PUC
    dir_resultados_base = os.path.join(path_slv, f'Modelos/{nome_modelo}/Classificador-{base_autoencoder}/Resultados/Treinados_em_{base_usada}')
    if not os.path.isdir(dir_resultados_base):
        os.makedirs(dir_resultados_base)
    
    #para cada um dos meus batches sizes, eu vou pegar seu correspondendo em csv 
    for batch, batch_size in zip(batchs, n_batchs):
        limpa_memoria()

        gerador = Gerador()
        gerador.carrega_modelo(os.path.join(path,f'Modelos/{nome_modelo}/Modelo-Base/Estrutura/{nome_modelo}.keras'), pesos=False)
        encoder = gerador.encoder

        #caso eu queira carregar o pesos do encoder
        if pesos:
            classificador = GeradorClassificador(encoder=encoder, 
                                                pesos=os.path.join(path,f'Modelos/{nome_modelo}/Modelo-Base/Pesos/{nome_modelo}_Base-{base_autoencoder}.weights.h5'), 
                                                nome_modelo=nome_modelo, 
                                                nome_autoencoder=base_autoencoder)
        else:
            #se eu não quiser carregar os pesos do encoder, o meu gerador irá mudar o nome para nome_autoencoder='Sem-Peso'
            classificador = GeradorClassificador(encoder=encoder, pesos=False, nome_modelo=nome_modelo, nome_autoencoder=base_autoencoder)

        classificador.compila()
        classificador.Dataset(treino=None, validacao=validacao, teste=teste)

        treino, _ = preprocessamento_dataframe(os.path.join(batch_dir, batch), autoencoder=False)
        classificador.setTreino(treino)
        classificador.treinamento(epocas=n_epocas, salvar=salvar ,n_batchs=batch_size, nome_base=base_usada, pesos=pesos)
        predicoes_np, acuracia = classificador.predicao(teste_csv)
        precisoes.append(acuracia)

        #Modelo-Kyoto-1/Classificador/Resultados/Treinados_em_PUC/PUC
        resultados_dir = os.path.join(dir_resultados_base, nome_base_teste)
        if not os.path.isdir(resultados_dir):
            os.makedirs(resultados_dir)

        #Salvo o resultado npy
        arquivo = os.path.join(path_slv, resultados_dir, f"batches-{batch_size}.npy")
        np.save(arquivo, predicoes_np)

        #Salvar a precisão 
        dir_prec = os.path.join(path_slv, f"Modelos/{nome_modelo}/Classificador-{base_autoencoder}/Precisao")
        if not os.path.isdir(dir_prec):
            os.makedirs(dir_prec)

        #cria ../precisao/Treinado_em_UFPR04
        dir_prec_base = os.path.join(dir_prec, f'Treinado_em_{base_usada}')
        recria_diretorio(dir_prec_base)#Apaga e cria a pasta nova se já tiver

        #Salva a precisão 
        print(nome_base_teste)
        caminho_arquivo = os.path.join(dir_prec_base, f'precisao-{nome_base_teste}.txt')
        with open(caminho_arquivo, 'w') as f:
            for prec in precisoes:
                f.write(f"{prec}\n")
        
        limpa_memoria() 

    dir_graf = os.path.join(path_slv,f'Modelos/{nome_modelo}/Plots/Graficos')
    if not os.path.isdir(dir_graf):
        os.makedirs(dir_graf)

    dir_graf_facul = os.path.join(dir_graf, f'Treinado_em_{base_usada}')
    if not os.path.isdir(dir_graf_facul):
        os.makedirs(dir_graf_facul)

    grafico_batchs(n_batchs=n_batchs, precisoes=precisoes, nome_modelo=nome_modelo, 
                   nome_base_treino=base_usada, 
                   base_usada_teste=base_usada, 
                   nome_autoencoder=base_autoencoder, 
                   caminho_para_salvar=dir_graf_facul)

    #plot_model(encoder, show_shapes=True,show_layer_names=True,to_file=os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{base_autoencoder}/encoder-{nome_modelo}.png'))
    #plot_model(modelo, show_shapes=True,show_layer_names=True,to_file=os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{base_autoencoder}/classificador-{nome_modelo}.png'))
    
    print(precisoes)
    #return (n_batchs, precisoes, nome_modelo)

def treina_modelos_em_batch(nome_modelo, base_usada, base_autoencoder, treino_csv, validacao, teste, teste_csv, salvar=True, n_epocas=10):
    path_modelos = os.path.join(path, "Modelos")
    modelos = os.listdir(path_modelos)
    modelos_para_treinar = []
    for modelo in modelos:
        if os.path.exists(os.path.join(path_modelos, modelo)):
            if nome_modelo in modelo and 'Fusoes' not in modelo:
                modelo_base = os.path.join(path_modelos, modelo, 'Modelo-Base')
                peso, estrutura = os.listdir(modelo_base)
                m = os.listdir(os.path.join(modelo_base , estrutura))
                dir_modelo = os.path.join(modelo_base, estrutura, m[0])
                modelos_para_treinar.append(dir_modelo)
            else:
                pass
        else:
            print(f"O diretório {modelo} não existe.")

    for i, m in enumerate(sorted(modelos_para_treinar)):
        nome = nome_modelo + f"-{i}"
        #treino cada um dos modelos em batches 
        treinamento_em_batch(nome, base_usada, base_autoencoder, treino_csv, validacao, teste, teste_csv, salvar, n_epocas, True)

    #faço o plot comparando os modelos
    comparacao(caminho_para_salvar=os.path.join(path_modelos, "Plots"), 
               nome_modelo=nome_modelo, 
               base_usada=base_usada, 
               base_autoencoder=base_autoencoder,
               base_de_teste=base_usada)

def testa_modelos_em_batch(nome_modelo, teste, teste_df, base_do_classificador, pesos=True, nome_autoencder='Kyoto'):
    path_slv = path

    if not pesos:
        nome_autoencder='Sem-Peso'

    classificador = GeradorClassificador(nome_modelo=nome_modelo, nome_autoencoder=nome_autoencder)

    nome_base = retorna_nome_df(teste_df)

    classificador.setTeste(teste)

    if not os.path.isdir(os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Resultados')):
        recria_diretorio(os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Resultados'))
        
    acuracias = []
    batchs = [64,128,256,512,1024]

    estrutura = os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Estrutura/Classificador_{nome_modelo}.keras')
    for batch_size in [64,128,256,512,1024]:
        dir_peso = f'Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Pesos/Treinado_em_{base_do_classificador}/Classificador_{nome_modelo}_batchs-{batch_size}.weights.h5'
        peso = os.path.join(path_slv,dir_peso)
        classificador.carrega_modelo(estrutura, peso)
        predicoes_np, acuracia = classificador.predicao(teste_df)

        #Modelo-Kyoto-1/Classificador-CNR/Resultados/Treinados_em_PUC/UFPR04/batchs-64-npy
        dir_base = os.path.join(path_slv, f"Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Resultados/Treinados_em_{base_do_classificador}/{nome_base}")
        
        if not os.path.isdir(dir_base):
            os.makedirs(dir_base)
            print(f"Diretório criado: {dir_base}")
        else:
            print(f"Diretório {dir_base} já existia!")

        arquivo = os.path.join(dir_base, f'batchs-{batch_size}.npy')
        np.save(arquivo, predicoes_np)
        limpa_memoria()
        acuracias.append(acuracia)
        del peso, predicoes_np, acuracia


    print(acuracias)

    #Salvar as precisões no arquivo
    #print(nome_base_teste)
    print(nome_base)
    caminho_arquivo = os.path.join(path_slv, f'Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Precisao/Treinado_em_{base_do_classificador}', f'precisao-{nome_base}.txt')
    with open(caminho_arquivo, 'w') as f:
        for prec in acuracias:
            f.write(f"{prec}\n")

    dir_graf_facul = os.path.join(path_slv,f'Modelos/{nome_modelo}/Plots/Graficos/Treinado_em_{base_do_classificador}')
    grafico_batchs(n_batchs=batchs, precisoes=acuracias, nome_modelo=nome_modelo, 
                   nome_base_treino=base_do_classificador, 
                   base_usada_teste=nome_base, 
                   nome_autoencoder=nome_autoencder ,
                   caminho_para_salvar=dir_graf_facul) 

def testa_modelos(nome_modelo, teste, teste_df, base_do_classificador, nome_autoencoder):
    #classificador = GeradorClassificador()
    caminho_modelos = os.path.join(path, "Modelos")
    modelos = os.listdir(caminho_modelos) #todos as pastas no dir Modelos
    modelos_usados = []
    print("Os modelos em pastas são:", modelos)
    for modelo in modelos:
        if os.path.exists(os.path.join(caminho_modelos, modelo)):
            if nome_modelo in modelo and 'Fusoes' not in modelo: #Se nome modelo tiver em modelo e fusão não
                modelo_base = os.path.join(caminho_modelos, modelo, 'Classificador') #Modelo_Kyoto-0/Classificador_Modelo
                print("A estrutura disponível é:" , os.listdir(os.path.join(modelo_base, 'Estrutura')))
                estrutura = os.listdir(os.path.join(modelo_base, 'Estrutura'))[0]
                modelos_usados.append(estrutura)
            else:
                pass
        else:
            print(f"O diretório {modelo} não existe.")
    
    print(modelos_usados)
    for modelo in modelos_usados:
        nome = extrair_nome_modelo1(modelo)
        print(nome)
        testa_modelos_em_batch(
            nome_modelo=nome, teste=teste, teste_df=teste_df, 
            base_do_classificador=base_do_classificador, 
            pesos=True, 
            nome_autoencder=nome_autoencoder)
        limpa_memoria()

    base_testada = retorna_nome_df(teste_df)

    comparacao(caminho_para_salvar=os.path.join(path, 'Modelos/Plots'), 
               nome_modelo=nome_modelo, 
               base_usada=base_do_classificador, 
               base_de_teste=base_testada, 
               base_autoencoder=nome_autoencoder)


#Classe usada para combinar dois geradores, estava usando quandos os batches
#ainda eram 16*64, ai ia combinando eles de forma sequencial
class CombinarGeradores(Sequence):
    def __init__(self, gerador1, gerador2):
        self.gerador1 = gerador1
        self.gerador2 = gerador2
        self.batch_size = gerador1.batch_size
        self.n_imagens = len(gerador1) * gerador1.batch_size + len(gerador2) * gerador2.batch_size
        print("Total de imagens:", self.get_total_images())
    
    def __len__(self):
        #n total de batchs considerando os 2 geradores
        return int(np.ceil(self.n_imagens / self.batch_size))
    
    def __getitem__(self, posicao):
        #calcula em qual dos geradores o batch está
        if posicao < len(self.gerador1):
            return self.gerador1[posicao]
        else:
            posicao -= len(self.gerador1)
            return self.gerador2[posicao]
    
    def get_total_images(self):
        return self.n_imagens
