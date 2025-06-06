from utils import *
import argparse

parser = argparse.ArgumentParser(description="Descrição do seu script.")
parser.add_argument('--classifier_base', type=str, help='Lista de bases a serem treinadas pelo classificador')
args = parser.parse_args()
print(args)

validation, _ = preprocessing_dataframe(path_csv=f'CSV/{args.classifier_base}/{args.classifier_base}_validation.csv', autoencoder=False, data_algumentantation=False, input_shape=(64,64))
test, test_df = preprocessing_dataframe(path_csv=f'CSV/{args.classifier_base}/{args.classifier_base}_test.csv', autoencoder=False, data_algumentantation=False, input_shape=(64,64))

#train_per_batch('Modelo_Kyoto-1', f'{args.classifier_base}', 'Kyoto', f'CSV/{args.classifier_base}/{args.classifier_base}_train.csv', validation, test, test_df, True, 50, True, (64,64,3))

for i in range(1,10):
    test, test_df = preprocessing_dataframe(path_csv=f'CSV/camera{i}/camera{i}.csv', autoencoder=False, data_algumentantation=False, input_shape=(64,64))
    test_model_per_batch('Modelo_Kyoto-1', test, test_df, f'{args.classifier_base}', weights=True, autoencoder_base='Kyoto')

    
    