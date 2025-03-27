from Modelos import *
from Preprocessamento import *
from segmentandoDatasets import *
import matplotlib
from tensorflow.keras.mixed_precision import set_global_policy
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()

# Argumentos
parser.add_argument("nome", type=str, help="O nome do modelo")
parser.add_argument("numeros", type=int, help="Número de modelos para serem criados")
parser.add_argument("autoencoder", type=str, help="Nome da base para o autoencoder")
parser.add_argument("epocas_autoencoder", type=int, help="Número de épocas para o autoencoder")

# Parse dos argumentos passados na linha de comando
args = parser.parse_args()

# Exibindo os valores
print(f'Nome: {args.nome}')
print(f'Número de Modelos: {args.numeros}')
print(f'Base usada para o autoencoder: {args.autoencoder}')
print(f'Épocas Autoencoder: {args.epocas_autoencoder}')

# Ativa a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with tf.device('/GPU:0'):
    limpa_memoria()

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
    cria_modelos(n_modelos=args.numeros, nome_modelo=args.nome, filters_list=[32,64,128,256])

    # Treinamento dos autoencoders
    treina_modelos(treino_autoencoder, validacao_autoencoder, teste_autoencoder, 
                    nome_modelo=args.nome, nome_base=args.autoencoder, n_epocas=args.epocas_autoencoder, batch_size=4)

    # Apaga a memória cache do tensorflow 
    limpa_memoria()