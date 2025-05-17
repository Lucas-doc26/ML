#!/bin/bash
#args: nome, numeros, autoencoder, classificador, epocas, teste1, teste2 
set -e

echo "Executando Script!"

python3 train_autoencoders.py --name_model Modelo_Kyoto --autoencoder_base Kyoto CNR PKLot --autoencoder_epocas 200 50 50 

python3 train_classifier.py --name Modelo_Kyoto --classifier_base PUC --classifier_epochs 20 --autoencoder_base Kyoto
python3 train_classifier.py --name Modelo_Kyoto --classifier_base UFPR05 --classifier_epochs 20 --autoencoder_base Kyoto
python3 train_classifier.py --name Modelo_Kyoto --classifier_base UFPR04 --classifier_epochs 20 --autoencoder_base Kyoto


python3 train_classifier.py --name Modelo_Kyoto --classifier_base PUC --classifier_epochs 20 --autoencoder_base CNR
python3 train_classifier.py --name Modelo_Kyoto --classifier_base UFPR04 --classifier_epochs 20 --autoencoder_base CNR
python3 train_classifier.py --name Modelo_Kyoto --classifier_base UFPR05 --classifier_epochs 20 --autoencoder_base CNR

python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera1 --classifier_epochs 20 --autoencoder_base PKLot
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera2 --classifier_epochs 20 --autoencoder_base PKLot
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera3 --classifier_epochs 20 --autoencoder_base PKLot
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera4 --classifier_epochs 20 --autoencoder_base PKLot
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera5 --classifier_epochs 20 --autoencoder_base PKLot
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera6 --classifier_epochs 20 --autoencoder_base PKLot
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera7 --classifier_epochs 20 --autoencoder_base PKLot
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera8 --classifier_epochs 20 --autoencoder_base PKLot
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera9 --classifier_epochs 20 --autoencoder_base PKLot

python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera1 --classifier_epochs 20 --autoencoder_base Kyoto
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera2 --classifier_epochs 20 --autoencoder_base Kyoto
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera3 --classifier_epochs 20 --autoencoder_base Kyoto
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera4 --classifier_epochs 20 --autoencoder_base Kyoto
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera5 --classifier_epochs 20 --autoencoder_base Kyoto
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera6 --classifier_epochs 20 --autoencoder_base Kyoto
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera7 --classifier_epochs 20 --autoencoder_base Kyoto
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera8 --classifier_epochs 20 --autoencoder_base Kyoto
python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera9 --classifier_epochs 20 --autoencoder_base Kyoto
