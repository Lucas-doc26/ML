from utils import *

#Configurações
set_seeds() #Reprodutibilidade  
config_gpu() #Usar GPU


#Crio o gerenciador de caminhos para o projeto
#path_manager = PathManager('/home/lucas/PIBIC-2024-2025')

#Crio os csvs para os datasets
#create_datasets_csv(path_manager, '/opt/datasets') 
#|-> caso queira colocar os datasets em outra pasta(usar ssd por exemplo), só passar o caminho como argumento

#Plotando as imagens
#plot_images_from_csv(Path('CSV/PKLot/PKLot_autoencoder_test.csv'), 3, (10, 10), 'PKLot')    

#Gerando modelos de autoencoders
generate_models(n_models=10, model_name='Modelo_Kyoto', filters_list=[8,16,32,64,128], input=(64,64,3), min_layers=3, max_layers=5)

#Treinando modelos de autoencoders
train_models(train, validation, test, model_name='Modelo_Kyoto', autoencoder_base='Sem-Peso', n_epochs=10, batch_size=16, input_shape=(64,64,3))

