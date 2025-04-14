import pandas as pd
import albumentations as A
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocessamento_dataframe(caminho_csv: str, autoencoder: bool = False, data_algumentantation:bool = True, input_shape:int=(64,64)):
    """
    Ao passar um dataFrame .csv, ele irá retornar o gerador e dataframe
    
    Parâmetros:
        caminho (str): Caminho para o arquivo CSV.
        autoencoder (bool): Se True, prepara os dados para um autoencoder (class_mode='input').
                            Se False, prepara os dados para classificação binária (class_mode='binary').
        data_algumentation (bool): Se True, faz o aumento dos dados .

    Retorna:
        Gerador, dataframe
    """

    dataframe = pd.read_csv(caminho_csv)
    batch_size = 64

    datagen = ImageDataGenerator(preprocessing_function=albumentations if data_algumentantation else normalize_image)

    if len(dataframe.columns) > 1:
        dataframe['classe'] = dataframe['classe'].astype(str)

    #Embaralho o dataframe aqui e não no shuffle, para garantir o mesmo csv sempre 
    #dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    class_mode = 'input' if autoencoder else 'sparse'

    Gerador = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='caminho_imagem',
        y_col='caminho_imagem' if autoencoder else 'classe',
        target_size=(input_shape),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )

    print("Imagens totais:", Gerador.samples)

    dataframe.to_csv(caminho_csv, index=False)

    return Gerador, dataframe

#Preprocessamento que estava usando para separar em treino, val, teste 
def preprocessamento(caminho: str, proporcao_treino: float = 0.6, proporcao_teste: float = 0.2, proporcao_validacao: float = 0.2, autoencoder: bool = True, data_algumentantation = True, input_shape:int=(64,64)):
    """
    Ao passar um dataFrame .csv, ele irá retornar geradores de dados para treino, teste e validação + os 3 .csv dividos igualmente os geradores.
    
    Parâmetros:
        caminho (str): Caminho para o arquivo CSV.
        proporcao_treino (float): Proporção de dados de treino.
        proporcao_teste (float): Proporção de dados de teste.
        proporcao_validacao (float): Proporção de dados de validação.
        autoencoder (bool): Se True, prepara os dados para um autoencoder (class_mode='input').
                            Se False, prepara os dados para classificação binária (class_mode='binary').
        data_algumentation (bool): Se True, faz o aumento dos dados .

    
    Retorna:
        treino_gerador, validacao_gerador, teste_gerador, treino, validacao, teste
    """
    dataframe = pd.read_csv(caminho)

    treino, teste = train_test_split(dataframe, test_size=proporcao_teste, random_state=42)
    treino, validacao = train_test_split(treino, test_size=proporcao_validacao / (1 - proporcao_teste), random_state=42)

    batch_size = 32

    treino_datagen = ImageDataGenerator(preprocessing_function=albumentations if data_algumentantation else normalize_image)
    validacao_datagen = ImageDataGenerator(preprocessing_function=albumentations if data_algumentantation else normalize_image)
    teste_datagen = ImageDataGenerator(preprocessing_function=normalize_image)

    if autoencoder:
        class_mode = 'input'
        y_col = 'caminho_imagem'
    else:
        class_mode = 'binary'
        y_col = 'classe'

    treino_gerador = treino_datagen.flow_from_dataframe(
        dataframe=treino,
        x_col='caminho_imagem',
        y_col=y_col, #Usar a imagem como saída se for autoencoder
        target_size=(input_shape),
        batch_size=batch_size,
        class_mode=class_mode,  #Class mode baseado no parâmetro autoencoder
        shuffle=False
    )

    validacao_gerador = validacao_datagen.flow_from_dataframe(
        dataframe=validacao,
        x_col='caminho_imagem',
        y_col=y_col,  
        target_size=(input_shape),
        batch_size=batch_size,
        class_mode=class_mode, 
        shuffle=False
    )

    teste_gerador = teste_datagen.flow_from_dataframe(
        dataframe=teste,
        x_col='caminho_imagem',
        y_col='classe',  
        target_size=(input_shape),
        batch_size=batch_size,
        class_mode=class_mode, 
        shuffle=False
    )

    return treino_gerador, validacao_gerador, teste_gerador, treino, validacao, teste

#DataAugmentation com Albumentations 
#Transformações escolhidas: 
transform = A.Compose([
            A.RandomRain(
                drop_length=8, drop_width=1,
                drop_color=(180, 180, 180),  blur_value=5,brightness_coefficient=0.8, p=0.15
            ),
            A.GaussNoise(var_limit=(0.0, 0.0007), mean=0, p=0.15),
            A.ChannelShuffle(p=0.15),
            A.Rotate(limit=40, p=0.15),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True, p=0.15),
            A.AdvancedBlur(blur_limit=(7,9), noise_limit=(0.75, 1.25), p=0.15),
            #A.Resize(height=256, width=256)
    ])

#Funções auxiliares ao preprocessamento
def normalize_image(img):
    """
    Retorna a imagem normalizada 
    """
    return img / 255.0

def albumentations(img):
        """
        Faz a transformação da imagem a partir do transform definido
        """
        data = {"image": normalize_image(img)}
        augmented = transform(**data)  #** para expandir o dicionário
        return augmented['image']

#Funções auxiliares aos plots
def carregar_e_preprocessar_imagens(caminhos_imagens, target_size=(256, 256)):
    """
    Carrega e processa imagens a partir de caminhos fornecidos. Retornando um array numpy contendo todas as imgs. 

    Como usar:
    - caminhos_imagens = dataset_df['caminho_imagem'].tolist() 
    - passa a variável como argumento 
    """

    imagens = []
    for caminho in caminhos_imagens:
        img = load_img(caminho, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        imagens.append(img_array)
    return np.vstack(imagens)

# Exemplo de uso:
# caminhos_imagens = dataset_df['caminho_imagem'].tolist() 
# imagens = carregar_e_preprocessar_imagens(caminhos_imagens)
# modelo.predict(imagens).argmax(axis=1) -> assim que faz a previsão 

def mapear_rotulos_binarios(classes):
    """
    Converte as classes em binários: 
    - Occupied vira 1 
    - Empty vira 0
    """
    return np.array([1 if classe == 'Occupied' else 0 for classe in classes])





        