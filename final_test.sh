## Sh para rodar todos os testes finais
set -e

## Organizo os datasets 
python3 datasets.py 

## Treino todos os autoencoders 
python3 train_autoencoders.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --autoencoder_epochs 200

python3 train_autoencoders.py --name_model Modelo_Kyoto --autoencoder_base CNR --autoencoder_epochs 50

python3 train_autoencoders.py --name_model Modelo_Kyoto --autoencoder_base PKLot --autoencoder_epochs 50

## Treino os classificadores 
# Base autoencoder - Kyoto
python3 train_classifiers.py --name Modelo_Kyoto --classifier_base PUC --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base UFPR05 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base UFPR04 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera1 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera2 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera3 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera4 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera5 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera6 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera7 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera8 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera9 --classifier_epochs 20 --autoencoder_base Kyoto

# Base autoencoder - CNR
python3 train_classifiers.py --name Modelo_Kyoto --classifier_base PUC --classifier_epochs 20 --autoencoder_base CNR

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base UFPR04 --classifier_epochs 20 --autoencoder_base CNR

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base UFPR05 --classifier_epochs 20 --autoencoder_base CNR

# Base autoencoder - PKLot
python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera1 --classifier_epochs 20 --autoencoder_base PKLot

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera2 --classifier_epochs 20 --autoencoder_base PKLot

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera3 --classifier_epochs 20 --autoencoder_base PKLot

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera4 --classifier_epochs 20 --autoencoder_base PKLot

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera5 --classifier_epochs 20 --autoencoder_base PKLot

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera6 --classifier_epochs 20 --autoencoder_base PKLot

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera7 --classifier_epochs 20 --autoencoder_base PKLot

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera8 --classifier_epochs 20 --autoencoder_base PKLot

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera9 --classifier_epochs 20 --autoencoder_base PKLot

## Testando com as demais bases
python3 mean_results.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifiers camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 PUC UFPR04 UFPR05

python3 mean_results.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifiers camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9

python3 mean_results.py --name_model Modelo_Kyoto --autoencoder_base CNR --classifiers PUC UFPR04 UFPR05

## Fazendo as fusões 
python3 fusion_rules.py --name_model Modelo_Kyoto --autoencoder_base Kyoto --classifier_bases_train camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 PUC UFPR04 UFPR05 --classifier_bases_test camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 PUC UFPR04 UFPR05

python3 fusion_rules.py --name_model Modelo_Kyoto --autoencoder_base CNR --classifier_bases_train PUC UFPR04 UFPR05 --classifier_bases_test PUC UFPR04 UFPR05

python3 fusion_rules.py --name_model Modelo_Kyoto --autoencoder_base PKLot --classifier_bases_train camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --classifier_bases_test camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9
