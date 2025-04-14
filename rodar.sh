#!/bin/bash
#args: nome, numeros, autoencoder, classificador, epocas, teste1, teste2 

echo "Executando Script: classificador e fusões"

python3 segmentandoDatasets.py True

python3 datasets.py

python3 classificador.py Modelo_Kyoto 10 Kyoto PUC 20 UFPR04 UFPR05

python3 classificador.py Modelo_Kyoto 10 Kyoto UFPR04 20 PUC UFPR05

python3 classificador.py Modelo_Kyoto 10 Kyoto UFPR05 20 PUC UFPR04

python3 treino_autoencoder.py Modelo_Kyoto 10 CNR

python3 classificador.py Modelo_Kyoto 10 CNR PUC 20 UFPR04 UFPR05

python3 classificador.py Modelo_Kyoto 10 CNR UFPR04 20 PUC UFPR05

python3 classificador.py Modelo_Kyoto 10 CNR UFPR05 20 PUC UFPR04

python3 Fusoes.py 
