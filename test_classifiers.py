from utils import *
from utils.fusion_rules import *
import argparse

parser = argparse.ArgumentParser(description="Descrição do seu script.")
parser.add_argument('--name_model', type=str, help='Nome do modelo')
parser.add_argument('--autoencoder_base', type=str, help='Nome da base do autoencoder')
parser.add_argument('--classifier_base', type=str, help='Nome da base do classificador')
parser.add_argument('--test_bases', type=str, nargs='+', help='Lista de bases a serem testadas')

args = parser.parse_args()
print("Argumentos:", args)

path_manager = PathManager('/home/lucas/PIBIC')
sum_fusion = SumFusion(path_manager)
mult_fusion = MultFusion(path_manager)
vote_fusion = VoteFusion(path_manager)

for base in args.test_bases:
    test, test_df = preprocessing_dataframe(path_csv=f'CSV/{base}/{base}_test.csv', autoencoder=False, data_algumentantation=False, input_shape=(64,64))
    print(args.name_model)
    test_all_models_per_batch(model_name=args.name_model, 
                            test=test, test_df=test_df, 
                            classifier_base=args.classifier_base, 
                            autoencoder_base=args.autoencoder_base)
