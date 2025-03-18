from Modelos import *
from Preprocessamento import *
from segmentandoDatasets import *
from Fusoes import *
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import matplotlib
import shutil
from tensorflow.keras.mixed_precision import set_global_policy
import tensorflow as tf
import os
from tensorflow.keras import mixed_precision
import os
import time
import sys
import argparse
import pdb


parser = argparse.ArgumentParser()

# Argumentos
parser.add_argument("nome", type=str, help="O nome do modelo")
parser.add_argument("numeros", type=int, help="Número de modelos")
parser.add_argument("classificador", type=str, help="Nome da base para o classificador")
parser.add_argument("epocas_classificador", type=int, help="Número de épocas para o classificador")
parser.add_argument("base_teste1", type=str, help="Nome da base de teste")
parser.add_argument("base_teste2", type=str, help="Nome da base de teste")


# Parse dos argumentos passados na linha de comando
args = parser.parse_args()

# Exibindo os valores
print(f'Nome: {args.nome}')
print(f'Número de Modelos: {args.numeros}')
print(f'Base usada para o classificador: {args.classificador}')
print(f'Épocas Classificador: {args.epocas_classificador}')
print(f'Bases de teste: {args.base_teste1} e {args.base_teste2}')

limpa_memoria()

# Ativa a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Desabilita os plots gerais 
matplotlib.use('Agg')

#Segemnta pklot
PKLot()

# Preprocessamento imagens dos classificadores
treino, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv', autoencoder=True)
validacao, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Validacao.csv', autoencoder=True)
teste, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv', autoencoder=True, data_algumentantation=False)

# Criação dos classificadores
cria_classificadores(n_modelos=args.numeros, nome_modelo=args.nome, base_usada=f'{args.classificador}', 
                treino=None, validacao=None, teste=None, teste_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv')

# Dados para treino em batches
val, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Validacao.csv', autoencoder=False, data_algumentantation=False)
teste, teste_df = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv', autoencoder=False, data_algumentantation=False)

# Treina em batches
treina_modelos_em_batch(args.nome, f'{args.classificador}', f'CSV/{args.classificador}/{args.classificador}_Segmentado_Treino.csv', val, teste, teste_df, True, args.epocas_classificador)

# Testa nas demais bases 
base1, df_base1 = preprocessamento_dataframe(caminho_csv=f'CSV/{args.base_teste1}/{args.base_teste1}.csv', autoencoder=False, data_algumentantation=False)
testa_modelos(args.nome, base1, df_base1, args.classificador)

base2, df_base2 = preprocessamento_dataframe(caminho_csv=f'CSV/{args.base_teste2}/{args.base_teste2}.csv', autoencoder=False, data_algumentantation=False)
testa_modelos(args.nome, base2, df_base2, args.classificador)

#Teste na cnr
cameras = ['camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']
for camera in cameras:
    cnr, df_cnr = preprocessamento_dataframe(caminho_csv=f'CSV/CNR/CNR_{camera}.csv', autoencoder=False, data_algumentantation=False)
    testa_modelos(args.nome, cnr, df_cnr, args.classificador)
    del cnr, df_cnr
