import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from typing import List, Tuple
import os

def preprocessamento(caminho: str, 
                     proporcao_treino: float = 0.6, proporcao_teste: float = 0.2, proporcao_validacao: float = 0.2, 
                     shape = (64, 64)):
    """
    Prepara e retorna geradores de dados para treino, teste e validação, além dos dados em formato de arrays numpy.
    """
    dataframe = pd.read_csv(caminho)
    dataframe['classe'] = dataframe['classe'].astype(str)
    
    treino, teste = train_test_split(dataframe, test_size=proporcao_teste, random_state=42)
    treino, validacao = train_test_split(treino, test_size=proporcao_validacao / (1 - proporcao_teste), random_state=42)

    batch_size = 32

    def normalize_image(img):
        return img / 255.0

    treino_datagen = ImageDataGenerator(preprocessing_function=normalize_image)
    validacao_datagen = ImageDataGenerator(preprocessing_function=normalize_image)
    teste_datagen = ImageDataGenerator(preprocessing_function=normalize_image)

    treino_gerador = treino_datagen.flow_from_dataframe(
        dataframe=treino,
        x_col='caminho_imagem',
        y_col='classe',
        target_size=(shape),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    validacao_gerador = validacao_datagen.flow_from_dataframe(
        dataframe=validacao,
        x_col='caminho_imagem',
        y_col='classe',
        target_size=(shape),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    teste_gerador = teste_datagen.flow_from_dataframe(
        dataframe=teste,
        x_col='caminho_imagem',
        y_col='classe',
        target_size=(shape),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    # Extração das imagens do gerador
    def extrair_dados(gerador):
        x = []
        y = []
        for img_batch, label_batch in gerador:
            x.append(img_batch)
            y.append(label_batch)
            if len(x) * gerador.batch_size >= gerador.samples:
                break
        return np.concatenate(x), np.concatenate(y)

    x_treino, y_treino = extrair_dados(treino_gerador)
    x_validacao, y_validacao = extrair_dados(validacao_gerador)
    x_teste, y_teste = extrair_dados(teste_gerador)

    return treino_gerador, validacao_gerador, teste_gerador, x_treino, y_treino, x_teste, y_teste, x_validacao, y_validacao

def preprocessamento_dataframe(caminho:str):
    """
    Retorna o dataFrame gerado pelo ImageDataGenerator e o dataFrame com os caminhos e classe
    """
    
    img_width, img_height = 256, 256
    batch_size = 32
    dataframe = pd.read_csv(caminho)

    def normalize_image(img):
                return img / 255.0


    #preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    datagen = ImageDataGenerator(preprocessing_function=normalize_image)
    dataframe_gerador = datagen.flow_from_dataframe(
            dataframe=dataframe,
            x_col='caminho_imagem',
            y_col='classe',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary'
        )
    
    return dataframe, dataframe_gerador

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

# Exemplo de uso:

def preprocessamento_dataframe_unico(caminho_csv: str, autoencoder: bool = False):
    """
    Processa um arquivo CSV para criar um fluxo de imagens para treinamento usando ImageDataGenerator.

    :param caminho_csv: Caminho para o arquivo CSV contendo informações das imagens.
    :param autoencoder: Se True, configura o modo autoencoder; caso contrário, configuração de classificação binária.
    :return: Um objeto DataFrameIterator do ImageDataGenerator.
    """
    dataframe = pd.read_csv(caminho_csv)

    img_width, img_height = 256, 256
    batch_size = 32

    #preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    def normalize_image(img):
            return img / 255.0


    datagen = ImageDataGenerator(preprocessing_function=normalize_image)

    # Define o class_mode
    class_mode = 'input' if autoencoder else 'binary'

    Dataframe_preprocessado = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='caminho_imagem',
        y_col='caminho_imagem' if autoencoder else 'classe',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True
    )

    return Dataframe_preprocessado, dataframe

def todos_df(autoencoder: bool = False):
    pasta = 'Datasets_csv'
    for csv in os.listdir(pasta):
        caminho_csv = os.path.join(pasta, csv)
        if os.path.isfile(caminho_csv) and csv.endswith('.csv'):
            print(f'Processando o arquivo: {csv}')
            return preprocessamento_dataframe_unico(caminho_csv, autoencoder)

