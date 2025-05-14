from utils import *
import argparse

parser = argparse.ArgumentParser(description="Descrição do seu script.")
parser.add_argument('--name', type=str, help='Nome do modelo')
parser.add_argument('--classifier_base', type=str, nargs='+', help='Lista de bases a serem treinadas pelo classificador')
parser.add_argument('--classifier_epochs', type=int, nargs='+', help='Lista de épocas de cada um dos modelos')
parser.add_argument('--autoencoder_base', type=str, help='Base do autoencoder')

args = parser.parse_args()
print("Argumentos:", args)

#Configurações
set_seeds() #Reprodutibilidade  
config_gpu() #Usar GPU

#Gerando modelos dos classificadores 
create_classifiers(n_models=10, model_name=args.name, autoencoder_base=args.autoencoder_base)
# Não tem problema de ficar gerando eles, pois eles são sempre gerados com o msm classificador

#Preprocessando as bases de treino:
for i, classifier_base in enumerate(args.classifier_base):
    train, _ = preprocessing_dataframe(path_csv=f'CSV/{classifier_base}/{classifier_base}_train.csv', autoencoder=False, data_algumentantation=False, input_shape=(64,64))
    validation, _ = preprocessing_dataframe(path_csv=f'CSV/{classifier_base}/{classifier_base}_validation.csv', autoencoder=False, data_algumentantation=False, input_shape=(64,64))
    test, test_df = preprocessing_dataframe(path_csv=f'CSV/{classifier_base}/{classifier_base}_test.csv', autoencoder=False, data_algumentantation=False, input_shape=(64,64))

    print(test.dtype)

    #Treinando modelos de classificação
    train_all_models_per_batch(model_name=args.name, 
                                classifier_base=classifier_base,
                                autoencoder_base=args.autoencoder_base,
                                train_csv=f'CSV/{classifier_base}/{classifier_base}_train.csv',
                                validation=validation,
                                test=test,
                                test_df=test_df,
                                save=True,
                                epochs=args.classifier_epochs[i],
                                input_shape=(64,64,3))
