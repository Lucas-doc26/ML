import pandas as pd
import os

# Variáveis Globais
path_base = "PKLot/PKLotSegmented"
faculdades = ['PUC', 'UFPR04', 'UFPR05']
tempos = ['Cloudy', 'Rainy', 'Sunny']
classes = ['Empty', 'Occupied']

def PKLot():
    dados = []
    for faculdade in faculdades:
        for tempo in tempos:
            path_facul_tempo = os.path.join(path_base, faculdade, tempo)
            if not os.path.isdir(path_facul_tempo):
                continue

            dias = os.listdir(path_facul_tempo)
            for dia in dias:
                for classe in classes:
                    path_imagens = os.path.join(path_facul_tempo, dia, classe)
                    if not os.path.isdir(path_imagens):
                        continue

                    imagens = os.listdir(path_imagens)
                    for img in imagens:
                        caminho_img = os.path.join(path_imagens, img)
                        dados.append([faculdade, tempo, dia, classe, caminho_img])
    
    dataFrame = pd.DataFrame(dados, columns=["Faculdade", "Tempo", "Dia", "Classe", "Caminho"])
    
    dataFrame = dataFrame.sort_values(by="Dia", ascending=True)

    dataFrame.to_csv('CSV/PKLot/PKLot.csv')

    return dataFrame  

def segmenta():
    df = pd.read_csv("CSV/PKLot/PKLot.csv")
    df_dias_por_faculdade = df[["Faculdade", "Dia"]].drop_duplicates().sort_values(by=["Faculdade", "Dia"])
    df_dias_PUC = df[df["Faculdade"] == 'PUC']["Dia"].unique()
    #array np de todos os dias que a PUC tem
    df_teste = df[df["Faculdade"] == 'PUC', df['Dia'] == ]

    return df
    