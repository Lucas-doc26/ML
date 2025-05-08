#Treinando o autoencoder que faltava
python3 treino_autoencoder.py Modelo_Kyoto 10 PKLOT '(64,64)' 20

#Treinando os classificadores que faltavam
python3 treina_classificador.py Modelo_Kyoto 10 PKLOT '(64,64)' camera1 20 
python3 testa_classificador.py Modelo_Kyoto 10 Kyoto '(64,64)' PUC 20 'camera1, camera2, camera3, camera4, camera5, camera6, camera7, camera8, camera9'