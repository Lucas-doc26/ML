from utils import *
import argparse

parser = argparse.ArgumentParser(description="Descrição do seu script.")
parser.add_argument('--bases', type=str, nargs='+', help='Lista de cameras a serem processadas')
args = parser.parse_args()
print("Bases:", args)

#Configurações
set_seeds() #Reprodutibilidade  
config_gpu() #Usar GPU


#Crio o gerenciador de caminhos para o projeto
path_manager = PathManager('/home/lucas/PIBIC')

#Crio os csvs para os datasets
create_datasets_csv(path_manager, '/datasets') #-> fazer uma forma de fazer um shuffle nos csv
#|-> caso queira colocar os datasets em outra pasta(usar ssd por exemplo), só passar o caminho como argumento

#Plotando as imagens
#plot_images_from_csv(Path('CSV/PKLot/PKLot_autoencoder_test.csv'), 3, (10, 10), 'PKLot')    

#Gerando modelos de autoencoders
#generate_models(n_models=10, model_name='Modelo_Kyoto', filters_list=[8,16,32,64,128], input=(64,64,3), min_layers=3, max_layers=5)

#Preprocessando as bases de treino:
for autoencoder_base in args.bases:
    train, _ = preprocessamento_dataframe(path_csv=f'CSV/{autoencoder_base}/{autoencoder_base}_autoencoder_train.csv', autoencoder=True, data_algumentantation=False, input_shape=(64,64))
    validation, _ = preprocessamento_dataframe(path_csv=f'CSV/{autoencoder_base}/{autoencoder_base}_autoencoder_validation.csv', autoencoder=True, data_algumentantation=False, input_shape=(64,64))
    test, _ = preprocessamento_dataframe(path_csv=f'CSV/{autoencoder_base}/{autoencoder_base}_autoencoder_test.csv', autoencoder=True, data_algumentantation=False, input_shape=(64,64))

    #Treinando modelos de autoencoder
    train_models(train, validation, test, model_name='Modelo_Kyoto', autoencoder_base=autoencoder_base, n_epochs=20, batch_size=4, input_shape=(64,64,3))
