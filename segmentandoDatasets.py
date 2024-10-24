import pandas as pd
import os
import random
import numpy as np
import urllib.request
import tarfile
from pathlib import PurePath
from typing import Tuple, Optional

#Função antiga - Sem balanceamento
def segmentando_datasets(quantidade_PUC: Optional[int] = None, quantidade_UFPR04: Optional[int] = None, quantidade_UFPR05: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Função para criar os datasets csv com uma divisão igual entre as classes 'Empty' e 'Occupied'.
    Retorna uma tupla com os datasets separados em ordem PUC, UFPR04, UFPR05.
    """
    faculdades = ['PUC', 'UFPR04', 'UFPR05']
    
    limites_padrao = {
        'PUC': quantidade_PUC,
        'UFPR04': quantidade_UFPR04,
        'UFPR05': quantidade_UFPR05
    }

    tempos = ['Cloudy', 'Rainy', 'Sunny']


    
    dataframes = [] 

    for local in faculdades:
        caminhos_empty = []
        caminhos_occupied = []
        
        for tempo in tempos:
            sample_dir = os.path.join(
                r"/home/lucas/Downloads/PKLot/PKLotSegmented/",
                local, tempo)

            if not os.path.exists(sample_dir):
                print(f'Diretório não encontrado: {sample_dir}')
                continue

            pastas = os.listdir(sample_dir)

            for pasta in pastas:
                for class_dir in ['Empty', 'Occupied']:
                    full_class_dir = os.path.join(sample_dir, pasta, class_dir)
                    if os.path.exists(full_class_dir):
                        for file in os.listdir(full_class_dir):
                            if file.endswith('.jpg'):
                                if class_dir == 'Empty':
                                    caminhos_empty.append(os.path.join(full_class_dir, file))
                                else:
                                    caminhos_occupied.append(os.path.join(full_class_dir, file))

        # Definir o limite de arquivos de acordo com a quantidade passada (metade para 'Empty', metade para 'Occupied')
        limite_arquivos = limites_padrao[local] if limites_padrao[local] is not None else float('inf')
        limite_por_classe = min(len(caminhos_empty), len(caminhos_occupied), limite_arquivos // 2)

        # Embaralhar as listas para garantir a aleatoriedade
        random.shuffle(caminhos_empty)
        random.shuffle(caminhos_occupied)

        # Garantir que a quantidade seja limitada pela menor lista
        caminhos_empty = caminhos_empty[:limite_por_classe]
        caminhos_occupied = caminhos_occupied[:limite_por_classe]

        # Combinar as duas classes
        caminhos_imagem = caminhos_empty + caminhos_occupied
        classes = ['Empty'] * len(caminhos_empty) + ['Occupied'] * len(caminhos_occupied)

        # Embaralhar novamente as imagens combinadas
        combined_data = list(zip(caminhos_imagem, classes))
        random.shuffle(combined_data)
        caminhos_imagem, classes = zip(*combined_data)

        # Criar o DataFrame
        df = pd.DataFrame({
            'caminho_imagem': caminhos_imagem,
            'classe': classes
        })

        # Salvar o DataFrame como arquivo CSV
        csv_path = f'Datasets_csv/df_{local}.csv'
        df.to_csv(csv_path, index=False)
        print(f'DataFrame do local {local} salvo como: {csv_path}')

        # Adicionando o DataFrame à lista
        dataframes.append(df)

        print(f'DataFrame do local {local}:')
        print(df.head())
        print('\n')

    return tuple(dataframes)  # Retornar a tupla dos DataFrames

#Função de segmentar o PKLot Balanceado
def segmentacaoPklot(imagens_treino:int=1000, dias_treino:int=5, imagens_validacao:int=300, dias_validaco:int=2, imagens_teste:int=2000, dias_teste=3):
    """
    A soma máxima do número de dias, deve ser igual a 8 caso queira dias distintos entre treino/validacao/teste
    """
    data_dir = 'PKLot'
    faculdades = ['PUC', 'UFPR04', 'UFPR05']
    tempos = ['Cloudy', 'Rainy', 'Sunny']
    classes = ['Empty', 'Occupied']
    path_base = 'PKLot/PKLotSegmented'

    def cria_PKLot():
        dfs = []
        for local in faculdades:
            caminhos_empty = []
            caminhos_occupied = []
            
            for tempo in tempos:
                sample_dir = os.path.join(path_base, local, tempo)
                if not os.path.exists(sample_dir):
                    print(f'Diretório não encontrado: {sample_dir}')
                    continue

                for pasta in os.listdir(sample_dir):
                    for class_dir in ['Empty', 'Occupied']:
                        full_class_dir = os.path.join(sample_dir, pasta, class_dir)
                        if os.path.exists(full_class_dir):
                            for file in os.listdir(full_class_dir):
                                if file.endswith('.jpg'):
                                    caminho = PurePath(os.path.join(full_class_dir, file))
                                    if class_dir == 'Empty':
                                        caminhos_empty.append(str(caminho))
                                    else:
                                        caminhos_occupied.append(str(caminho))

            df = pd.DataFrame({
                'caminho_imagem': caminhos_empty + caminhos_occupied,
                'classe': ['Empty'] * len(caminhos_empty) + ['Occupied'] * len(caminhos_occupied)
            })
            dfs.append(df)

        df_final = pd.concat(dfs, axis=0, ignore_index=True)
        df_final.to_csv("PKLot.csv", index=False)

    def contagem_imagens():
        dic = {}
        for faculdade in faculdades:
            dic[faculdade] = {}
            for tempo in tempos:
                dic[faculdade][tempo] = {}
                path_tempo = os.path.join(path_base, faculdade, tempo)

                dias = os.listdir(path_tempo)
                for dia in dias:
                    dic[faculdade][tempo][dia] = {}
                    for classe in classes:
                        path_classe = os.path.join(path_tempo, dia, classe)
                        if os.path.isdir(path_classe):
                            imagens = os.listdir(path_classe)
                            dic[faculdade][tempo][dia][classe] = len(imagens)

        print("Contagem de imagens em todo diretório:", dic ,"\n\n")         

    def imagens_distribuidas(n_imgs):
        n_faculdades = len(faculdades)
        n_tempos = len(tempos)
        n_classes = len(classes)


        imagens_por_faculdade = n_imgs // n_faculdades
        resto_faculdades = n_imgs % n_faculdades

        valores = {}
        total_por_classe = {classe: 0 for classe in classes}

        for i, faculdade in enumerate(faculdades):
            valores[faculdade] = {}
            
            #Calcula o total de imagens para cada faculdade, considerando o resto
            imagens_totais_faculdade = imagens_por_faculdade + (1 if i < resto_faculdades else 0)
            
            imagens_por_tempo = imagens_totais_faculdade // n_tempos
            resto_tempo = imagens_totais_faculdade % n_tempos

            for j, tempo in enumerate(tempos):
                valores[faculdade][tempo] = {}

                imagens_para_esse_tempo = imagens_por_tempo + (1 if j < resto_tempo else 0)
                
                imagens_por_classe = imagens_para_esse_tempo // n_classes
                resto_classes = imagens_para_esse_tempo % n_classes

                for k, classe in enumerate(classes):
                    # Distribui as imagens entre as classes
                    valores[faculdade][tempo][classe] = imagens_por_classe + (1 if k < resto_classes else 0)
                    total_por_classe[classe] += valores[faculdade][tempo][classe]

        total_imagens = sum(total_por_classe.values())
        imagens_por_classe_ideal = total_imagens // n_classes

        for classe in classes:
            diferenca = imagens_por_classe_ideal - total_por_classe[classe]
            if diferenca != 0:
                for faculdade in faculdades:
                    for tempo in tempos:
                        if diferenca > 0:
                            valores[faculdade][tempo][classe] += 1
                            diferenca -= 1
                        elif diferenca < 0:
                            if valores[faculdade][tempo][classe] > 0:
                                valores[faculdade][tempo][classe] -= 1
                                diferenca += 1
                        if diferenca == 0:
                            break
                    if diferenca == 0:
                        break

        print("Distribuição de imagens para :", valores)

        # Verificação final
        total_por_classe = {classe: sum(valores[f][t][classe] for f in faculdades for t in tempos) for classe in classes}
        print("Total por classe:", total_por_classe, "\n")

        return valores
    
    def criar_csv(n_dias, valores, nome:str=''):
        df = pd.read_csv('PKLot.csv')
        data = []
        df_final = pd.DataFrame()

        for faculdade in faculdades:
            df_facul = df[df['caminho_imagem'].str.contains(faculdade)]
            
            for tempo in tempos: 
                df_tempo = df_facul[df_facul['caminho_imagem'].str.contains(tempo)]

                dias_dir = sorted(os.listdir(os.path.join(path_base, faculdade, tempo)))
                total_dias = len(dias_dir)

                if nome.upper() == 'TREINO':
                    dias_selecionados = dias_dir[:n_dias]
                elif nome.upper() == 'VALIDACAO':
                    inicio = (total_dias - n_dias) // 2
                    dias_selecionados = dias_dir[inicio:inicio + n_dias]
                else:
                    dias_selecionados = dias_dir[-n_dias:]
                
                for classe in classes:
                    df_classe = df_tempo[df_tempo['classe'].str.contains(classe)]
                    valor = valores[faculdade][tempo][classe]
                    
                    while valor > 0:
                        imagens_disponiveis = df_classe.copy()
                        for dia in dias_selecionados:
                            df_dia = imagens_disponiveis[imagens_disponiveis['caminho_imagem'].str.contains(dia)]
                            
                            if not df_dia.empty:  
                                imagem_selecionada = df_dia.sample(1)
                                data.append(imagem_selecionada)
                                valor -= 1
                                
                                imagens_disponiveis = imagens_disponiveis.drop(imagem_selecionada.index)  
                                
                                if valor <= 0:
                                    break

                        if valor <= 0:  # Saia do loop enquanto se atingir a meta
                            break
                    
                    # Resetando o índice do DataFrame se necessário
                    df.reset_index(drop=True, inplace=True)

        df_final = pd.concat(data, ignore_index=True)

        df_final.to_csv(f'PKLot_Segmentado{nome}.csv', index=False)

    if os.path.isdir(data_dir):
        print("Começando Segmentação do PKLot")
        cria_PKLot()
    else:
        url = 'http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz'
        file_name = 'PKLot.tar.gz'
        
        print("Baixando arquivo...")
        urllib.request.urlretrieve(url, file_name)
        
        print("Extraindo arquivo...")
        with tarfile.open(file_name, 'r:gz') as tar:
            tar.extractall(path='')
        
        print("Arquivo extraído com sucesso.")
        os.remove('PKLot.tar.gz')
        cria_PKLot()

    contagem_imagens()
    
    criar_csv(n_dias=dias_treino, valores=imagens_distribuidas(imagens_treino), nome='Treino')
    criar_csv(n_dias=dias_validaco, valores=imagens_distribuidas(imagens_validacao), nome='Validacao')
    criar_csv(n_dias=dias_teste, valores=imagens_distribuidas(imagens_teste), nome ='Teste')
#Exemplo de uso:
#segmentacaoPklot(imagens_treino=1000, dias_treino=5, imagens_validacao=300, dias_validaco=1, imagens_teste=1000, dias_teste=2)

def segmentacaoCNR(imagens_treino:int=1000, dias_treino:int=5, imagens_validacao:int=300, dias_validaco:int=2, imagens_teste:int=2000, dias_teste:int=3):
    path_labels = 'CNR-EXT-Patches-150x150/LABELS/all.txt'
    path_imgs = 'CNR-EXT-Patches-150x150/PATCHES'
    tempos = ['OVERCAST','RAINY', 'SUNNY']
    classes = ['Empty', 'Occupied']
    cameras = ['camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']

    def cria_CNR():
        caminhos_imagens = []
        classes = []

        with open(path_labels, 'r') as file:
            for linha in file:
                partes = linha.strip().split(' ')

                if len(partes) == 2:
                    caminho_imagem = partes[0]
                    caminhos_imagens.append(caminho_imagem)
                    classe = partes[1]
                    if classe == '0':
                        classe = 'Empty'
                    else:
                        classe = 'Occupied'
                    classes.append(classe)

        df = pd.DataFrame({
            'caminho_imagem': caminhos_imagens,
            'classe': classes
        })

        df.to_csv("CNR.csv", index=False)

    def imagens_distribuidas_cnr(n_imgs):
        n_tempos = len(tempos)
        n_classes = len(classes)
        n_cameras = len(cameras)

        imagens_por_tempo = n_imgs // n_tempos
        resto_tempo = n_imgs % n_tempos

        valores = {}
        total_por_classe = {classe: 0 for classe in classes}

        for i, tempo in enumerate(tempos):
            valores[tempo] = {}

            imagens_totais_tempo = imagens_por_tempo + (1 if i < resto_tempo else 0)

            # Calcular a distribuição por câmera
            imagens_por_camera = imagens_totais_tempo // n_cameras
            resto_camera = imagens_totais_tempo % n_cameras

            for j, camera in enumerate(cameras):
                valores[tempo][camera] = {}

                # Distribuir a imagem extra se houver sobra
                imagens_para_essa_camera = imagens_por_camera + (1 if j < resto_camera else 0)

                # Distribuir as imagens por classe
                imagens_por_classe = imagens_para_essa_camera // n_classes
                resto_classe = imagens_para_essa_camera % n_classes

                for k, classe in enumerate(classes):
                    valores[tempo][camera][classe] = imagens_por_classe + (1 if k < resto_classe else 0)
                    total_por_classe[classe] += valores[tempo][camera][classe]

        # Ajustar a distribuição para garantir que todas as câmeras tenham uma quantidade mínima
        total_imagens = sum(total_por_classe.values())
        imagens_por_classe_ideal = total_imagens // n_classes

        for classe in classes:
            diferenca = imagens_por_classe_ideal - total_por_classe[classe]
            if diferenca != 0:
                for tempo in tempos:
                    for camera in cameras:
                        if diferenca > 0:
                            valores[tempo][camera][classe] += 1
                            diferenca -= 1
                        elif diferenca < 0:
                            if valores[tempo][camera][classe] > 0:
                                valores[tempo][camera][classe] -= 1
                                diferenca += 1
                        if diferenca == 0:
                            break
                    if diferenca == 0:
                        break

        print("Distribuição de imagens para :", valores)

        # Verificação final
        total_por_classe = {classe: sum(valores[t][c][classe] for t in tempos for c in cameras) for classe in classes}
        print("Total por classe:", total_por_classe, "\n")

        return valores 

    def criar_csv_cnr(n_dias, valores, nome: str = ''):
        print(nome)
        df = pd.read_csv('CNR.csv')
        data = [] 

        for tempo in tempos:
            df_tempo = df[df['caminho_imagem'].str.contains(tempo)]
            dias_dir = sorted(os.listdir(os.path.join(path_imgs, tempo)))
            total_dias = len(dias_dir)

            if nome.upper() == 'TREINO':
                dias_selecionados = dias_dir[:n_dias]
            elif nome.upper() == 'VALIDACAO':
                inicio = (total_dias - n_dias) // 2
                dias_selecionados = dias_dir[inicio:inicio + n_dias]
            else:
                dias_selecionados = dias_dir[-n_dias:]

            for camera in cameras:
                df_camera = df_tempo[df_tempo['caminho_imagem'].str.contains(camera)]

                for classe in classes:
                    df_classe = df_camera[df_camera['classe'].str.contains(classe)]
                    valor = valores[tempo][camera][classe]

                    while valor > 0:
                        imagens_disponiveis = df_classe.copy()
                        for dia in dias_selecionados:
                            df_dia = imagens_disponiveis[imagens_disponiveis['caminho_imagem'].str.contains(dia)]

                            if not df_dia.empty:
                                imagem_selecionada = df_dia.sample(1)
                                data.append(imagem_selecionada)
                                valor -= 1

                                imagens_disponiveis = imagens_disponiveis.drop(imagem_selecionada.index)

                            if imagens_disponiveis.empty:
                                break

                        if imagens_disponiveis.empty:
                            break

        # Concatena todos os DataFrames de uma vez só no final
        df_final = pd.concat(data, ignore_index=True)
        df_final.to_csv(f'CNR_Segmentado{nome}.csv', index=False)

    criar_csv_cnr(n_dias=dias_treino, valores=imagens_distribuidas_cnr(imagens_treino), nome='Treino')
    criar_csv_cnr(n_dias=dias_validaco, valores=imagens_distribuidas_cnr(imagens_validacao), nome='Validacao')
    criar_csv_cnr(n_dias=dias_teste, valores=imagens_distribuidas_cnr(imagens_teste), nome ='Teste')

segmentacaoCNR(imagens_treino=20, dias_treino=5, imagens_validacao=30, dias_validaco=1, imagens_teste=10, dias_teste=2)