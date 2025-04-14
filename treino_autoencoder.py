from Modelos import *
import argparse

parser = argparse.ArgumentParser()

# Argumentos
parser.add_argument("nome", type=str, help="O nome do modelo")
parser.add_argument("numeros", type=int, help="Número de modelos")
parser.add_argument("autoencoder", type=str, help="Nome da base de autoencoder utilizada")

args = parser.parse_args()
if args.autoencoder == 'CNR':
    path_treino = 'CSV/CNR/CNR_autoencoder_treino.csv'
    path_teste = 'CSV/CNR/CNR_autoencoder_teste.csv'
    path_val = 'CSV/CNR/CNR_autoencoder_treino.csv'

# Preprocessamento imagens autoencoder
treino_autoencoder, _ = preprocessamento_dataframe(caminho_csv=path_treino, autoencoder=True, data_algumentantation=False)
validacao_autoencoder, _ = preprocessamento_dataframe(caminho_csv=path_val, autoencoder=True, data_algumentantation=False)
teste_autoencoder, _ = preprocessamento_dataframe(caminho_csv=path_teste, autoencoder=True, data_algumentantation=False)

# Carrega autoencoder
treina_modelos(treino_autoencoder, validacao_autoencoder, teste_autoencoder,args.nome, args.autoencoder,n_epocas=20, batch_size=8)



