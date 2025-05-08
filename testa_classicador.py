from Modelos import *
from Preprocessamento import *
from skimage.metrics import structural_similarity as ssim
import matplotlib
import tensorflow as tf
import argparse
from multiprocessing import Pool
import ast

parser = argparse.ArgumentParser()

# Argumentos
parser.add_argument("nome", type=str, help="O nome do modelo")
parser.add_argument("autoencoder", type=str, help="Nome da base de autoencoder utilizada")
parser.add_argument("input", type=str, help="Input Size das imagens do autoencoder")
parser.add_argument("classificador", type=str, help="Nome da base para o classificador")
parser.add_argument("bases_de_teste", type=str, help="Nome da base de teste : '[base1, base2, base3]'")

# Parse dos argumentos passados na linha de comando
args = parser.parse_args()

# Exibindo os valores
print(f'Nome: {args.nome}')
print(f'Modelo autoencoder, treinado na base: {args.autoencoder}')
print(f'Tamanho da imagem: {args.input}')
print(f'Base usada para o classificador: {args.classificador}')
print(f'Bases de teste: {args.bases_de_teste}')

limpa_memoria()

# Ativa a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# Desabilita os plots gerais 
matplotlib.use('Agg')

# Preprocessamento imagens dos classificadores
input_size = ast.literal_eval(args.input)
input_shape = (*input_size, 3)

bases_de_teste = args.bases_de_teste.split(', ')

for base_de_teste in bases_de_teste:
    if 'camera' in base_de_teste:
        csv = f'CSV/CNR/CNR_{base_de_teste}.csv'
    else:
        csv = f'CSV/{base_de_teste}/{base_de_teste}.csv'

    base, df_base = preprocessamento_dataframe(caminho_csv=csv, autoencoder=False, data_algumentantation=True, input_shape=input_size)

    testa_modelos(args.nome, base, df_base, args.classificador, args.autoencoder)
    
    del base, df_base, csv