
from Modelos import *
import argparse
import ast
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("nome", type=str, help="O nome do modelo")
parser.add_argument("numeros", type=int, help="Número de modelos")
parser.add_argument("input", type=str, help="Input Size das imagens")
parser.add_argument("filtros", type=str)
parser.add_argument("min", type=int)
parser.add_argument("max", type=int)

args = parser.parse_args()

# Converte e valida input
input_size = ast.literal_eval(args.input)
if not (isinstance(input_size, tuple) and len(input_size) == 2 and all(isinstance(x, int) for x in input_size)):
    raise ValueError("O parâmetro 'input' deve ser uma tupla de dois inteiros, como (256,256)")

input_shape = (*input_size, 3)

# Converte os filtros
filtros = ast.literal_eval(args.filtros)

# Cria modelo
cria_modelos(
    n_modelos=args.numeros,
    nome_modelo=args.nome,
    filters_list=filtros,
    input=input_shape,
    min_camadas=args.min,
    max_camadas=args.max
)