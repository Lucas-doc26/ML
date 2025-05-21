from utils.fusion_rules import *
import argparse

parser = argparse.ArgumentParser(description="Descrição do seu script.")
parser.add_argument('--name_model', type=str, help='Nome do modelo')
parser.add_argument('--autoencoder_base', type=str, help='Base que o autoencoder foi treinado')
parser.add_argument('--classifier_bases_train', type=str, nargs='+', help='Lista bases de treino')
parser.add_argument('--classifier_bases_test', type=str, nargs='+', help='Lista bases de teste')

args = parser.parse_args()
print("Argumentos:", args)

path_manager = PathManager('/home/lucas/PIBIC/')
sum_fusion = SumFusion(path_manager)
mult_fusion = MultFusion(path_manager)
vote_fusion = VoteFusion(path_manager)

# Executar as fusões
fusion_process(model_name=args.name_model, train_bases=args.classifier_bases_train, test_bases=args.classifier_bases_test, 
               fusion_rule=sum_fusion, autoencoder_base=args.autoencoder_base, number_of_models=10)

fusion_process(model_name=args.name_model, train_bases=args.classifier_bases_train, test_bases=args.classifier_bases_test, 
               fusion_rule=mult_fusion, autoencoder_base=args.autoencoder_base, number_of_models=10)

fusion_process(model_name=args.name_model, train_bases=args.classifier_bases_train, test_bases=args.classifier_bases_test, 
               fusion_rule=vote_fusion, autoencoder_base=args.autoencoder_base, number_of_models=10)
