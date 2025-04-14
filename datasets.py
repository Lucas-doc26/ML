import pandas as pd
import numpy as np
import os 

SEED = 42

cameras = [['camera1', 56], ['camera2', 56], ['camera3', 56], ['camera4', 56], ['camera5', 56], ['camera6', 55], ['camera7', 55], ['camera8', 55], ['camera9', 55]]

def CNR_autoencoder():
    df_final = pd.DataFrame()
    for camera, valor in cameras:
        df_camera = pd.read_csv(f'CSV/CNR/CNR_{camera}.csv')
        df_occ = df_camera[df_camera['classe'] == 1].sample(n=valor, random_state=SEED)
        df_empt = df_camera[df_camera['classe'] == 0].sample(n=valor,random_state=SEED)
        df_final = pd.concat([df_final, df_occ, df_empt], axis=0, ignore_index=True)

    df_val = pd.DataFrame()
    for camera, valor in cameras:
        df_camera = pd.read_csv(f'CSV/CNR/CNR_{camera}.csv')
        df_camera = df_camera[~df_camera['caminho_imagem'].isin(df_final['caminho_imagem'])]
        df_occ = df_camera[df_camera['classe'] == 1].sample(n=5, random_state=SEED)
        df_empt = df_camera[df_camera['classe'] == 0].sample(n=5,random_state=SEED)
        df_val = pd.concat([df_val, df_occ, df_empt], axis=0, ignore_index=True)

    df_teste = pd.DataFrame()
    for camera, valor in cameras:
        df_camera = pd.read_csv(f'CSV/CNR/CNR_{camera}.csv')
        df_camera = df_camera[~df_camera['caminho_imagem'].isin(df_final['caminho_imagem'])]
        df_camera = df_camera[~df_camera['caminho_imagem'].isin(df_val['caminho_imagem'])]
        df_occ = df_camera[df_camera['classe'] == 1].sample(n=5, random_state=SEED)
        df_empt = df_camera[df_camera['classe'] == 0].sample(n=5,random_state=SEED)
        df_teste = pd.concat([df_teste, df_occ, df_empt], axis=0, ignore_index=True)
    
    return df_final, df_val, df_teste
    
treino, val, teste = CNR_autoencoder()
treino.to_csv('CSV/CNR/CNR_autoencoder_treino.csv', index=False)
teste.to_csv('CSV/CNR/CNR_autoencoder_teste.csv', index=False)
val.to_csv('CSV/CNR/CNR_autoencoder_val.csv', index=False)