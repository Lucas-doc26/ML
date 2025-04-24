from Modelos import *
from Preprocessamento import *
from skimage.metrics import structural_similarity as ssim
import matplotlib
import tensorflow as tf
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()

# Argumentos
parser.add_argument("nome", type=str, help="O nome do modelo")
parser.add_argument("numeros", type=int, help="Número de modelos")
parser.add_argument("autoencoder", type=str, help="Nome da base de autoencoder utilizada")
parser.add_argument("classificador", type=str, help="Nome da base para o classificador")
parser.add_argument("epocas_classificador", type=int, help="Número de épocas para o classificador")
parser.add_argument("base_teste1", type=str, help="Nome da base de teste")
parser.add_argument("base_teste2", type=str, help="Nome da base de teste")


# Parse dos argumentos passados na linha de comando
args = parser.parse_args()

# Exibindo os valores
print(f'Nome: {args.nome}')
print(f'Modelo autoencoder, treinado na base: {args.autoencoder}')
print(f'Número de Modelos: {args.numeros}')
print(f'Base usada para o classificador: {args.classificador}')
print(f'Épocas Classificador: {args.epocas_classificador}')
print(f'Bases de teste: {args.base_teste1} e {args.base_teste2}')

limpa_memoria()

# Ativa a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# Desabilita os plots gerais 
matplotlib.use('Agg')

#Segemnta pklot
#PKLot()

# Preprocessamento imagens dos classificadores

# Criação dos classificadores
cria_classificadores(n_modelos=args.numeros, nome_modelo=args.nome, base_autoencoder=args.autoencoder, treino=None, validacao=None, teste=None)

# Dados para treino em batches
val, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Validacao.csv', autoencoder=False, data_algumentantation=False)

teste, teste_df = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv', autoencoder=False, data_algumentantation=False)

# Treina em batches

treina_modelos_em_batch(
    nome_modelo=args.nome, 
    base_usada=f'{args.classificador}', 
    base_autoencoder=f'{args.autoencoder}', 
    treino_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Treino.csv', 
    validacao=val, teste=teste, teste_csv=teste_df, 
    salvar=True, 
    n_epocas=args.epocas_classificador)


# Testa nas demais bases 
base1, df_base1 = preprocessamento_dataframe(caminho_csv=f'CSV/{args.base_teste1}/{args.base_teste1}.csv', autoencoder=False, data_algumentantation=False)
testa_modelos(args.nome, base1, df_base1, args.classificador, args.autoencoder)

base2, df_base2 = preprocessamento_dataframe(caminho_csv=f'CSV/{args.base_teste2}/{args.base_teste2}.csv', autoencoder=False, data_algumentantation=False)
testa_modelos(args.nome, base2, df_base2, args.classificador, args.autoencoder)

#Teste na cnr
"""
cameras = ['camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8','camera9']
for camera in cameras:
    cnr, df_cnr = preprocessamento_dataframe(caminho_csv=f'CSV/CNR/CNR_{camera}.csv', autoencoder=False, data_algumentantation=False)
    testa_modelos(args.nome, cnr, df_cnr, args.classificador)
    del cnr, df_cnr"""