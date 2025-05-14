#!/bin/bash
#args: nome, numeros, autoencoder, classificador, epocas, teste1, teste2 
set -e

echo "Executando Script: Treinando classificadores"

python3 train_classifier.py --name Modelo_Kyoto --classifier_base PUC UFPR04 UFPR05 --classifier_epochs 20 20 20 --autoencoder_base Kyoto

python3 train_classifier.py --name Modelo_Kyoto --classifier_base PUC UFPR04 UFPR05 --classifier_epochs 20 20 20 --autoencoder_base CNR

python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --classifier_epochs 20 20 20 20 20 20 20 20 20 --autoencoder_base PKLot

python3 train_classifier.py --name Modelo_Kyoto --classifier_base camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --classifier_epochs 20 20 20 20 20 20 20 20 20 --autoencoder_base Kyoto
