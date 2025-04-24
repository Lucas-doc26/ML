#!/bin/bash
#args: nome, numeros, autoencoder, classificador, epocas, teste1, teste2 

echo "Executando Script: classificador e fusões"

# Preparando os datasets 
python3 segmentandoDatasets.py True
python3 datasets.py

#Treinando os modelos que eu já tinha 

python3 classificador.py Modelo_Kyoto 10 Kyoto '(64,64)' PUC 20 UFPR04 UFPR05
python3 classificador.py Modelo_Kyoto 10 Kyoto '(64,64)' UFPR04 20 PUC UFPR05
python3 classificador.py Modelo_Kyoto 10 Kyoto '(64,64)' UFPR05 20 PUC UFPR04

#python3 treino_autoencoder.py Modelo_Kyoto 10 CNR

python3 classificador.py Modelo_Kyoto 10 CNR '(64,64)' PUC 20 UFPR04 UFPR05
python3 classificador.py Modelo_Kyoto 10 CNR '(64,64)' UFPR04 20 PUC UFPR05
python3 classificador.py Modelo_Kyoto 10 CNR '(64,64)' UFPR05 20 PUC UFPR04

python3 Fusoes.py 

#Criando um modelo novo de autoencoder
python3 autoencoders.py Modelo_64 10 '(64,64)' '[16,32,64,128,256]' 5 8
python3 treino_autoencoder.py Modelo_64 10 CNR '(64,64)' 20
python3 treino_autoencoder.py Modelo_64 10 Kyoto '(64,64)' 200

python3 classificador.py Modelo_64 10 CNR '(64,64)' PUC 20 UFPR04 UFPR05
python3 classificador.py Modelo_64 10 CNR '(64,64)' UFPR04 20 PUC UFPR05
python3 classificador.py Modelo_64 10 CNR '(64,64)' UFPR05 20 PUC UFPR04

python3 classificador.py Modelo_64 10 Kyoto '(64,64)' PUC 20 UFPR04 UFPR05
python3 classificador.py Modelo_64 10 Kyoto '(64,64)' UFPR04 20 PUC UFPR05
python3 classificador.py Modelo_64 10 Kyoto '(64,64)' UFPR05 20 PUC UFPR04