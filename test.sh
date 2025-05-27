set -e 

# Agora a base é a CNR
python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base CNR --classifier_base PUC --test_bases UFPR04 UFPR05

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base CNR --classifier_base UFPR04 --test_bases PUC UFPR05

python3 test_classifiers.py --name_model Modelo_Kyoto --autoencoder_base CNR --classifier_base UFPR05 --test_bases PUC UFPR04