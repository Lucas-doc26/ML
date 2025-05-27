set -e 

python3 fusion_rules.py --name_model Modelo_Kyoto --autoencoder_base CNR --classifier_bases_train PUC UFPR04 UFPR05 --classifier_bases_test PUC UFPR04 UFPR05
