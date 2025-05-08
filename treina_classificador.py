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
parser.add_argument("numeros", type=int, help="Número de modelos")
parser.add_argument("autoencoder", type=str, help="Nome da base de autoencoder utilizada")
parser.add_argument("input", type=str, help="Input Size das imagens do autoencoder")
parser.add_argument("classificador", type=str, help="Nome da base para o classificador")
parser.add_argument("epocas_classificador", type=int, help="Número de épocas para o classificador")


# Parse dos argumentos passados na linha de comando
args = parser.parse_args()

# Exibindo os valores
print(f'Nome: {args.nome}')
print(f'Modelo autoencoder, treinado na base: {args.autoencoder}')
print(f'Tamanho da imagem: {args.input}')
print(f'Número de Modelos: {args.numeros}')
print(f'Base usada para o classificador: {args.classificador}')
print(f'Épocas Classificador: {args.epocas_classificador}')

limpa_memoria()

# Ativa a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# Desabilita os plots gerais 
matplotlib.use('Agg')

# Preprocessamento imagens dos classificadores
input_size = ast.literal_eval(args.input)
input_shape = (*input_size, 3)

# Criação dos classificadores
cria_classificadores(n_modelos=args.numeros, nome_modelo=args.nome, base_autoencoder=args.autoencoder, treino=None, validacao=None, teste=None, input_shape=input_shape)

# Dados para treino em batches
val, _ = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Validacao.csv', autoencoder=False, data_algumentantation=False, input_shape=input_size)

teste, teste_df = preprocessamento_dataframe(caminho_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Teste.csv', autoencoder=False, data_algumentantation=False, input_shape=input_size)

# Treina em batches
treina_modelos_em_batch(
    nome_modelo=args.nome, 
    base_usada=f'{args.classificador}', 
    base_autoencoder=f'{args.autoencoder}', 
    treino_csv=f'CSV/{args.classificador}/{args.classificador}_Segmentado_Treino.csv', 
    validacao=val, teste=teste, teste_csv=teste_df, 
    salvar=True, 
    n_epocas=args.epocas_classificador,
    input_shape=input_shape)
