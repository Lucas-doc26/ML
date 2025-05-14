from utils import *
import argparse

parser = argparse.ArgumentParser(description="Descrição do seu script.")
parser.add_argument('--name_model', type=str, help='Nome do modelo')
parser.add_argument('--autoencoder_bases', type=str, nargs='+', help='Lista de bases a serem treinadas pelo autoencoder')
parser.add_argument('--autoencoder_epocas', type=int, nargs='+', help='Lista de épocas de cada um dos modelos')

args = parser.parse_args()
print("Argumentos:", args)

#Configurações
set_seeds() #Reprodutibilidade  
config_gpu() #Usar GPU

#Preprocessando as bases de treino:
for i, autoencoder_base in enumerate(args.autoencoder_bases):
    train, _ = preprocessing_dataframe(path_csv=f'CSV/{autoencoder_base}/{autoencoder_base}_autoencoder_train.csv', autoencoder=True, data_algumentantation=False, input_shape=(64,64))
    validation, _ = preprocessing_dataframe(path_csv=f'CSV/{autoencoder_base}/{autoencoder_base}_autoencoder_validation.csv', autoencoder=True, data_algumentantation=False, input_shape=(64,64))
    test, _ = preprocessing_dataframe(path_csv=f'CSV/{autoencoder_base}/{autoencoder_base}_autoencoder_test.csv', autoencoder=True, data_algumentantation=False, input_shape=(64,64))

    #Treinando modelos de autoencoder
    train_models(train, validation, test, model_name=args.name_model, autoencoder_base=autoencoder_base, n_epochs=args.autoencoder_epocas[i], batch_size=4, input_shape=(64,64,3))
