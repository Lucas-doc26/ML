set -e


python3 train_classifiers.py --name Modelo_Kyoto --classifier_base PUC --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base UFPR05 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base UFPR04 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera2 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera3 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera4 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera5 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera8 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera9 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera1 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera6 --classifier_epochs 20 --autoencoder_base Kyoto

python3 train_classifiers.py --name Modelo_Kyoto --classifier_base camera7 --classifier_epochs 20 --autoencoder_base Kyoto

./pt2.sh &
./pt3.sh &
./pt4.sh 

python3 mean_results.py --name_model Modelo_Kyoto --autoencoder_base Kyoto 

python3 teste_fusoes.py

python3 tabela.py