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

# Parse dos argumentos passados na linha de comando
args = parser.parse_args()

# Exibindo os valores
print(f'Nome: {args.nome}')
print(f'Número de Modelos: {args.numeros}')
print(f'Base usada para o classificador: {args.classificador}')
print(f'Épocas Classificador: {args.epocas_classificador}')

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

#
segmentacao_PKLot(imagens_treino=1024, dias_treino=2, imagens_validacao=64, dias_validaco=1, imagens_teste=None, dias_teste=None, faculdades=[f"{args.classificador}"])

# Preprocessamento imagens dos classificadores
#treino, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv', autoencoder=True)
#validacao, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Validacao.csv', autoencoder=True)
#teste, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv', autoencoder=True, data_algumentantation=False)

# Criação dos classificadores
cria_classificadores(n_modelos=args.numeros, nome_modelo=args.nome, base_usada=f'{args.classificador}', 
                treino=None, validacao=None, teste=None, teste_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv')

# Dados para treino em batches
segmentacao_PKLot(imagens_treino=1024, dias_treino=2, imagens_validacao=64, dias_validaco=1, imagens_teste=None, dias_teste=None, faculdades=["PUC"])
val, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Validacao.csv', autoencoder=False, data_algumentantation=False)
teste_puc, teste_df_puc = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv', autoencoder=False, data_algumentantation=False)

# Treina em batches
dividir_em_batchs(f'CSV/{args.classificador}/{args.classificador}_Segmentado_Treino.csv')
treina_modelos_em_batch(args.nome, f'{args.classificador}', f'CSV/{args.classificador}/{args.classificador}_Segmentado_Treino.csv', val, teste_puc, teste_df_puc, True, args.epocas_classificador)

# Testa nas demais bases 
#UFPR04
#segmentacao_PKLot(imagens_treino=2, dias_treino=1, imagens_validacao=2, dias_validaco=1, imagens_teste=None, dias_teste=None, faculdades=["UFPR04"])
teste_UFPR04, teste_df_UFPR04 = preprocessamento_dataframe(caminho_csv='CSV/UFPR04/UFPR04.csv', autoencoder=False, data_algumentantation=False)
testa_modelos(args.nome, teste_UFPR04, teste_df_UFPR04)

#UFPR05
#segmentacao_PKLot(imagens_treino=2, dias_treino=1, imagens_validacao=2, dias_validaco=1, imagens_teste=None, dias_teste=None, faculdades=["UFPR05"])
teste_UFPR05, teste_df_UFPR05 = preprocessamento_dataframe(caminho_csv='CSV/UFPR05/UFPR05.csv', autoencoder=False, data_algumentantation=False)
testa_modelos(args.nome, teste_UFPR05, teste_df_UFPR05)
