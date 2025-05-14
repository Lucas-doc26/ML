from utils import *
import argparse

parser = argparse.ArgumentParser(description="Descrição do seu script.")
parser.add_argument('--name_model', type=str, help='Nome do modelo')
parser.add_argument('--n_models', type=int, help='Número de modelos a serem gerados')
parser.add_argument('--filters_list', type=int, nargs='+', help='Lista de filtros a serem usados')
parser.add_argument('--min_layers', type=int, help='Número mínimo de camadas')
parser.add_argument('--max_layers', type=int, help='Número máximo de camadas')

args = parser.parse_args()
print("Argumentos:", args)

#Configurações
set_seeds() #Reprodutibilidade  
config_gpu() #Usar GPU

#Crio o gerenciador de caminhos para o projeto
path_manager = PathManager('/home/lucas/PIBIC')

#Crio os csvs para os datasets
create_datasets_csv(path_manager, '/datasets') #-> fazer uma forma de fazer um shuffle nos csv
#|-> caso queira colocar os datasets em outra pasta(usar ssd por exemplo), só passar o caminho como argumento

#Gerando modelos de autoencoders -> já estão prontos
generate_models(n_models=args.n_models, model_name=args.name_model, filters_list=args.filters_list, input=(64,64,3), min_layers=args.min_layers, max_layers=args.max_layers)