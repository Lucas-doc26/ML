set -e 

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base PUC --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 UFPR04 UFPR05

# Rodar para classificador UFPR04 e testar todas as câmeras
python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base UFPR04 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 PUC UFPR05

# Rodar para classificador UFPR05 e testar todas as câmeras
python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base UFPR05 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 PUC UFPR04

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera1 --test_bases camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 PUC UFPR04 UFPR05

