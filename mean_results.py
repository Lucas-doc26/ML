from utils.view.tables.tabela_resultados import *
import argparse

parser = argparse.ArgumentParser(description="Descrição do seu script.")
parser.add_argument('--name_model', type=str, help='Nome do modelo')
parser.add_argument('--autoencoder_base', type=str, help='Base do autoencoder')

args = parser.parse_args()
print("Argumentos:", args)

resultados(args.name_model, args.autoencoder_base)

