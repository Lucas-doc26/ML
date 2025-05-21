from utils.view.tables.tabela_resultados import *
import argparse

parser = argparse.ArgumentParser(description="Descrição do seu script.")
parser.add_argument('--name_model', type=str, help='Nome do modelo')
parser.add_argument('--autoencoder_base', type=str, help='Lista de bases a serem treinadas pelo autoencoder')
parser.add_argument('--classifiers', type=str, nargs='+', help='Lista de épocas de cada um dos modelos')

args = parser.parse_args()
print("Argumentos:", args)

for classifier in args.classifiers:
    retorna_resultados_csv(autoencoder=args.autoencoder_base, classificador=classifier, modelo=args.name_model)

