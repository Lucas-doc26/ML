from Modelos import *
from Preprocessamento import *
from segmentandoDatasets import *
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
parser.add_argument("numeros", type=int, help="Número de modelos para serem criados")
parser.add_argument("autoencoder", type=str, help="Nome da base para o autoencoder")
parser.add_argument("epocas_autoencoder", type=int, help="Número de épocas para o autoencoder")
parser.add_argument("classificador", type=str, help="Nome da base para o classificador")
parser.add_argument("epocas_classificador", type=int, help="Número de épocas para o classificador")
parser.add_argument("base_teste1", type=str, help="Nome da base de teste")
parser.add_argument("base_teste2", type=str, help="Nome da base de teste")

# Parse dos argumentos passados na linha de comando
args = parser.parse_args()

# Exibindo os valores
print(f'Nome: {args.nome}')
print(f'Número de Modelos: {args.numeros}')
print(f'Base usada para o autoencoder: {args.autoencoder}')
print(f'Épocas Autoencoder: {args.epocas_autoencoder}')
print(f'Base usada para o classificador: {args.classificador}')
print(f'Épocas Classificador: {args.epocas_classificador}')
print(f'Bases usadas para teste: {args.base_teste1} e {args.base_teste2}')


# Economia de memória - diminuir a precisão dos cálculos
#politica = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(politica)
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

# Preparando de todos os dados autoencoder
if args.autoencoder == "Kyoto":
    segmentacao_Kyoto()
elif args.autoencoder != "Kyoto":
    PKLot()

# Preprocessamento imagens autoencoder
treino_autoencoder, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.autoencoder}/{args.autoencoder}_Segmentado_Treino.csv', autoencoder=True, data_algumentantation=False)
validacao_autoencoder, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.autoencoder}/{args.autoencoder}_Segmentado_Validacao.csv', autoencoder=True, data_algumentantation=False)
teste_autoencoder, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.autoencoder}/{args.autoencoder}_Segmentado_Teste.csv', autoencoder=True, data_algumentantation=False)

# Criação dos autoencoders
#cria_modelos(n_modelos=args.numeros, nome_modelo=args.nome, filters_list=[32,64,128])

# Treinamento dos autoencoders
treina_modelos(treino_autoencoder, validacao_autoencoder, teste_autoencoder, 
                nome_modelo=args.nome, nome_base=args.autoencoder, n_epocas=args.epocas_autoencoder, batch_size=4)

# Apaga a memória cache do tensorflow 
limpa_memoria()

# Preprocessamento imagens dos classificadores
treino, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv', autoencoder=True)
val, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Validacao.csv', autoencoder=True)
teste, teste_df = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv', autoencoder=True, data_algumentantation=False)

# Criação dos classificadores
cria_classificadores(n_modelos=args.numeros, nome_modelo=args.nome, base_usada=f'{args.classificador}', 
                treino=treino, validacao=val, teste=teste, teste_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv')

# Treina em batches
treina_modelos_em_batch(args.nome, f'{args.autoencoder}', f'CSV/{args.classificador}/{args.classificador}_Segmentado_Treino.csv', 
                        val, teste, teste_df, True, args.epocas_classificador)

# Testa nas demais bases 
base1, df_base1 = preprocessamento_dataframe(caminho_csv=f'CSV/{args.base_teste1}/{args.base_teste1}.csv', autoencoder=False, data_algumentantation=False)
testa_modelos(args.nome, base1, df_base1, args.classificador)

base2, df_base2 = preprocessamento_dataframe(caminho_csv=f'CSV/{args.base_teste2}/{args.base_teste2}.csv', autoencoder=False, data_algumentantation=False)
testa_modelos(args.nome, base2, df_base2, args.classificador)

limpa_memoria()

#Teste na cnr
cameras = ['camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8','camera9']
for camera in cameras:
    cnr, df_cnr = preprocessamento_dataframe(caminho_csv=f'CSV/CNR/CNR_{camera}.csv', autoencoder=False, data_algumentantation=False)
    testa_modelos(args.nome, cnr, df_cnr, args.classificador)
    del cnr, df_cnr

# Testa as fusões na PUCPR
#soma = regra_soma(args.nome, 'PUC', 'CSV/PUC/PUC_Segmentado_Teste.csv', args.numeros)
#mult = regra_multiplicacao(args.nome, 'PUC', 'CSV/PUC/PUC_Segmentado_Teste.csv', args.numeros)
#voto = regra_votacao(args.nome, 'PUC', 'CSV/PUC/PUC_Segmentado_Teste.csv', args.numeros)
#comparacao((soma, voto, mult), nome_modelo='Fusões do Modelo Kyoto na PUC')

# Teste nas demais bases
#UFPR04
#soma = regra_soma(args.nome, 'UFPR04', 'CSV/UFPR04/UFPR04_Segmentado_Teste.csv', args.numeros)
#mult = regra_multiplicacao(args.nome, 'UFPR04', 'CSV/UFPR04/UFPR04_Segmentado_Teste.csv', args.numeros)
#voto = regra_votacao(args.nome, 'UFPR04', 'CSV/UFPR04/UFPR04_Segmentado_Teste.csv', args.numeros)
#comparacao((soma, voto, mult), nome_modelo='Fusões do Modelo Kyoto na UFPR04')

#UFPR05
#soma = regra_soma(args.nome, 'UFPR05', 'CSV/UFPR05/UFPR05_Segmentado_Teste.csv', args.numeros)
#mult = regra_multiplicacao(args.nome, 'UFPR05', 'CSV/UFPR05/UFPR05_Segmentado_Teste.csv', args.numeros)
#voto = regra_votacao(args.nome, 'UFPR05', 'CSV/UFPR05/UFPR05_Segmentado_Teste.csv', args.numeros)
#comparacao((soma, voto, mult), nome_modelo='Fusões do Modelo Kyoto na UFPR05')
