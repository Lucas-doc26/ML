set -e 


# Rodar para classificador PUC e testar todas as câmeras
#python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base PUC --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9

# Rodar para classificador UFPR04 e testar todas as câmeras
#python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base UFPR04 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9

# Rodar para classificador UFPR05 e testar todas as câmeras
#python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base UFPR05 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9

# Agora para cada câmera como classificador, teste nas outras câmeras
python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera1 --test_bases camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera2 --test_bases camera1 camera3 camera4 camera5 camera6 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera3 --test_bases camera1 camera2 camera4 camera5 camera6 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera4 --test_bases camera1 camera2 camera3 camera5 camera6 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera5 --test_bases camera1 camera2 camera3 camera4 camera6 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera6 --test_bases camera1 camera2 camera3 camera4 camera5 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera7 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera8 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera9 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8

# Agora o autoencoder_base é a PKLot
python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifier_base camera1 --test_bases camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifier_base camera2 --test_bases camera1 camera3 camera4 camera5 camera6 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifier_base camera3 --test_bases camera1 camera2 camera4 camera5 camera6 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifier_base camera4 --test_bases camera1 camera2 camera3 camera5 camera6 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifier_base camera5 --test_bases camera1 camera2 camera3 camera4 camera6 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifier_base camera6 --test_bases camera1 camera2 camera3 camera4 camera5 camera7 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifier_base camera7 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera8 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifier_base camera8 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera9

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifier_base camera9 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8