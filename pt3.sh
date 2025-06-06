set -e 

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera6 --test_bases camera1 camera2 camera3 camera4 camera5 camera7 camera8 camera9 PUC UFPR04 UFPR05

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera7 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera8 camera9 PUC UFPR04 UFPR05

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera8 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera9 PUC UFPR04 UFPR05

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_base camera9 --test_bases camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 PUC UFPR04 UFPR05
