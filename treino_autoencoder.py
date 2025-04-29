from Modelos import *
import argparse
import ast
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("nome", type=str, help="O nome do modelo")
parser.add_argument("numeros", type=int, help="Número de modelos")
parser.add_argument("autoencoder", type=str, help="Nome da base de autoencoder utilizada")
parser.add_argument("input", type=str, help="Input Size das imagens")
parser.add_argument("epocas_autoencoder", type=int, help="Número de épocas para o autoencoder")

args = parser.parse_args()    

# Converte e valida input
input_size = ast.literal_eval(args.input)
if not (isinstance(input_size, tuple) and len(input_size) == 2 and all(isinstance(x, int) for x in input_size)):
    raise ValueError("O parâmetro 'input' deve ser uma tupla de dois inteiros, como (256,256)")

if input_size[0] > 64:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

if args.autoencoder == 'CNR':
    path_treino = 'CSV/CNR/CNR_autoencoder_treino.csv'
    path_teste = 'CSV/CNR/CNR_autoencoder_teste.csv'
    path_val = 'CSV/CNR/CNR_autoencoder_val.csv'
elif args.autoencoder == 'Kyoto':
    path_treino = 'CSV/Kyoto/Kyoto_Segmentado_Treino.csv'
    path_teste = 'CSV/Kyoto/Kyoto_Segmentado_Teste.csv'
    path_val = 'CSV/Kyoto/Kyoto_Segmentado_Validacao.csv'

# Preprocessa
treino_autoencoder, _ = preprocessamento_dataframe(caminho_csv=path_treino, autoencoder=True, data_algumentantation=False, input_shape=input_size)
validacao_autoencoder, _ = preprocessamento_dataframe(caminho_csv=path_val, autoencoder=True, data_algumentantation=False, input_shape=input_size)
teste_autoencoder, _ = preprocessamento_dataframe(caminho_csv=path_teste, autoencoder=True, data_algumentantation=False, input_shape=input_size)

input_shape = (*input_size, 3)
print(input_shape)

# Treina
treina_modelos(
    treino_autoencoder,
    validacao_autoencoder,
    teste_autoencoder,
    args.nome,
    args.autoencoder,
    n_epocas=args.epocas_autoencoder,
    batch_size=1,
    input_shape=input_shape
)
