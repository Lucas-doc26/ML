import pandas as pd
import os
import random
import numpy as np
import urllib.request
import tarfile
from pathlib import PurePath, Path
from typing import Tuple, Optional
import requests
import zipfile
import shutil
import shutil
from sklearn.model_selection import train_test_split

def cria_dirs():
    dirs = ['PKLot','PUC', 'UFPR04', 'UFPR05', 'CNR', 'Kyoto']
    if not os.path.isdir('CSV'):
        os.makedirs('CSV')
        for dr in dirs:
            os.makedirs(f'CSV/{dr}')
    
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

#Função de segmentar o PKLot Balanceado - rever ela, está desatualizada 
def segmentacao_PKLot(imagens_treino:int=1000, dias_treino:int=5, imagens_validacao:int=300, dias_validaco:int=2, imagens_teste:int=2000, dias_teste=3, faculdades = ['PUC', 'UFPR04', 'UFPR05']):
    """
    A soma máxima do número de dias, deve ser igual a 8 caso queira dias distintos entre treino/validacao/teste
    """

    if len(faculdades) == 1:
        nome_faculdade = faculdades[0]
    else:
        nome_faculdade = "PKLot"

    data_dir = 'PKLot'
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


        if not os.path.isdir("CSV"):
            os.makedirs("CSV")
            os.makedirs(f"CSV/{nome_faculdade}")
        
        if not os.path.isdir(f"CSV/{nome_faculdade}"):
            os.makedirs(f"CSV/{nome_faculdade}")

        df_final.to_csv(f"CSV/{nome_faculdade}/{nome_faculdade}.csv", index=False)

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
    
    def criar_csv(n_dias, valores, nome: str = ''):
        df = pd.read_csv(f'CSV/{nome_faculdade}/{nome_faculdade}.csv')
        data = []
        dias_usados = []

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
                    # 'valor' é a quantidade de imgs que eu quero pegar
                    valor = valores[faculdade][tempo][classe]
                    # cria uma cópia única do dataframe disponível para a classe
                    imagens_disponiveis = df_classe.copy()
                    
                    while valor > 0 and not imagens_disponiveis.empty:
                        progresso = False 
                        
                        for dia in dias_selecionados:
                            df_dia = imagens_disponiveis[imagens_disponiveis['caminho_imagem'].str.contains(dia, na=False)]
                            if not df_dia.empty:
                                imagem_selecionada = df_dia.sample(1)#pega uma img aleat. 
                                data.append(imagem_selecionada)
                                valor -= 1  
                                imagens_disponiveis = imagens_disponiveis.drop(imagem_selecionada.index)#drop do dataset a img selecionada
                                progresso = True
                                if valor <= 0:
                                    break
                            if dia not in dias_usados:
                                dias_usados.append(dia)
                        
                        #se não conseguir pegar uma img, ele sia do loop, para evitar de ficar infinito
                        if not progresso:
                            print(f"Aviso: Não há imagens suficientes para {faculdade} - {tempo} - {classe}")
                            break

        df_final = pd.concat(data, ignore_index=True)
        df_final['classe'] = df_final['classe'].replace({'Empty': 1, 'Occupied': 0})
        df_final.to_csv(f'CSV/{nome_faculdade}/{nome_faculdade}_Segmentado_{nome}.csv', index=False)

        return dias_usados

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

    def imagens_para_teste(treino, validacao):
        dias_usados = treino + validacao
        df = pd.read_csv(f'CSV/{nome_faculdade}/{nome_faculdade}.csv')
        data = []
        df_final = pd.DataFrame()
        for faculdade in faculdades:
            df_facul = df[df['caminho_imagem'].str.contains(faculdade)]
            
            for tempo in tempos: 
                df_tempo = df_facul[df_facul['caminho_imagem'].str.contains(tempo)]
                dias_selecionados = sorted(os.listdir(os.path.join(path_base, faculdade, tempo)))
                
                for classe in classes:
                    df_classe = df_tempo[df_tempo['classe'].str.contains(classe)]
                    
                    imagens_disponiveis = df_classe.copy()
                    for dia in dias_selecionados:
                        if dia in dias_usados:
                            continue
                        else:
                            df_dia = imagens_disponiveis[imagens_disponiveis['caminho_imagem'].str.contains(dia, na=False)]
                            
                            if not df_dia.empty:  
                                imagens = df_dia.sample(len(df_dia))
                                data.append(imagens)
                                imagens_disponiveis = imagens_disponiveis.drop(imagens.index)
                    
                    # Resetando o índice do DataFrame se necessário
                    df.reset_index(drop=True, inplace=True)

        df_final = pd.concat(data, ignore_index=True)

        df_final['classe'] = df_final['classe'].replace({'Empty': 1, 'Occupied': 0})

        df_final.to_csv(f'CSV/{nome_faculdade}/{nome_faculdade}_Segmentado_Teste.csv', index=False)

    contagem_imagens()
        
    treino = criar_csv(n_dias=dias_treino, valores=imagens_distribuidas(imagens_treino), nome='Treino')
    validacao = criar_csv(n_dias=dias_validaco, valores=imagens_distribuidas(imagens_validacao), nome='Validacao')

    if imagens_teste == None and dias_teste == None:
        imagens_para_teste(treino, validacao)
    else:
        criar_csv(n_dias=dias_teste, valores=imagens_distribuidas(imagens_teste), nome ='Teste')

    df = pd.read_csv(f'CSV/{nome_faculdade}/{nome_faculdade}.csv')

    df['classe'] = df['classe'].replace({'Empty': 1, 'Occupied': 0})
    df.to_csv(f'CSV/{nome_faculdade}/{nome_faculdade}.csv')

def segmentacao_CNR(imagens_treino:int=1000, dias_treino:int=5, imagens_validacao:int=300, dias_validaco:int=2, imagens_teste:int=2000, dias_teste:int=3):
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
                    caminho_imagem_completo = 'CNR-EXT-Patches-150x150/PATCHES/' + caminho_imagem
                    caminhos_imagens.append(caminho_imagem_completo)
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

        if not os.path.isdir("CSV"):
            os.makedirs("CSV")
            os.makedirs("CSV/CNR")
        if not os.path.isdir("CSV/CNR"):
            os.makedirs("CSV/CNR")

        df.to_csv("CSV/CNR/CNR.csv", index=False)

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
        df = pd.read_csv('CSV/CNR/CNR.csv')
        data = []
        df_final = pd.DataFrame()

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

            for i, camera in enumerate(cameras):
                df_camera = df_tempo[df_tempo['caminho_imagem'].str.contains(camera)]

                for classe in classes:
                    df_classe = df_camera[df_camera['classe'].str.contains(classe)]
                    valor = valores[tempo][camera][classe]
                    
                    if len(dias_selecionados) < 2:
                        dia = dias_selecionados[0]
                        imagens_disponiveis = df_classe.copy()
                        while valor > 0 and not imagens_disponiveis.empty:
                            df_dia = imagens_disponiveis[imagens_disponiveis['caminho_imagem'].str.contains(dia)]
                            
                            if not df_dia.empty:
                                imagem_selecionada = df_dia.sample(1)
                                data.append(imagem_selecionada)
                                valor -= 1

                                imagens_disponiveis = imagens_disponiveis.drop(imagem_selecionada.index)
                            else:
                                if valor > 0:
                                    # Transfere o valor restante para a próxima câmera, se houver
                                    if i + 1 < len(cameras):
                                        valores[tempo][cameras[i + 1]][classe] += valor
                                    valor = 0 
                                
                            if imagens_disponiveis.empty:
                                break

                    else:
                        imagens_disponiveis = df_classe.copy()
                        while valor > 0 and not imagens_disponiveis.empty:
                            for dia in dias_selecionados:
                                df_dia = imagens_disponiveis[imagens_disponiveis['caminho_imagem'].str.contains(dia)]

                                if not df_dia.empty:
                                    imagem_selecionada = df_dia.sample(1)
                                    data.append(imagem_selecionada)
                                    valor -= 1

                                    imagens_disponiveis = imagens_disponiveis.drop(imagem_selecionada.index)

                                if valor == 0 or imagens_disponiveis.empty:
                                    break

                            if imagens_disponiveis.empty:
                                break

            df_final = pd.concat(data, ignore_index=True)

        df_final.to_csv(f'CSV/CNR/CNR_Segmentado_{nome}.csv', index=False)

    cria_CNR()

    #criar_csv_cnr(n_dias=dias_treino, valores=imagens_distribuidas_cnr(imagens_treino), nome='Treino')
    #criar_csv_cnr(n_dias=dias_validaco, valores=imagens_distribuidas_cnr(imagens_validacao), nome='Validacao')
    #criar_csv_cnr(n_dias=dias_teste, valores=imagens_distribuidas_cnr(imagens_teste), nome ='Teste')

def segmentacao_Kyoto(treino=32, validacao=20, teste=8):
    def download_Kyoto():
        kyoto_path.mkdir(exist_ok=True)
        print(f"Pasta Kyoto criada em: {kyoto_path.absolute()}")
        
        # Baixar o arquivo ZIP
        url = "https://github.com/eizaburo-doi/kyoto_natim/archive/refs/heads/master.zip"
        print("Baixando arquivo ZIP...")
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Erro ao baixar o arquivo. Status code: {response.status_code}")
        
        # Salvar o arquivo ZIP temporariamente
        zip_path = "kyoto_temp.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print(f"Arquivo ZIP salvo em: {zip_path}")
        
        # Extrair o arquivo ZIP
        print("Extraindo arquivo ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("temp_extract")
        
        # Listar todos os arquivos extraídos para debug
        print("\nConteúdo extraído:")
        for root, dirs, files in os.walk("temp_extract"):
            print(f"\nDiretório: {root}")
            for name in files:
                print(f"- {name}")
        
        # Tentar diferentes possíveis caminhos para a pasta thumb
        possible_paths = [
            Path("temp_extract/kyoto_natim-master/kyoto_natim-master/thumb"),
            Path("temp_extract/kyoto_natim-master/thumb"),
            Path("temp_extract/thumb")
        ]
        
        thumb_path = None
        for path in possible_paths:
            if path.exists():
                thumb_path = path
                print(f"\nPasta thumb encontrada em: {thumb_path}")
                break
        
        if not thumb_path:
            raise Exception("Pasta thumb não encontrada nos caminhos esperados")
        
        # Copiar todas as imagens para a pasta Kyoto
        print("\nCopiando imagens...")
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
        images_copied = 0
        
        for file in thumb_path.glob("*"):
            print(f"Encontrado arquivo: {file.name}")
            if file.suffix.lower() in image_extensions:
                shutil.copy2(file, kyoto_path)
                print(f"Copiado: {file.name}")
                images_copied += 1
        
        # Limpar arquivos temporários
        print("\nLimpando arquivos temporários...")
        os.remove(zip_path)
        shutil.rmtree("temp_extract")
        
        if images_copied == 0:
            print("\nNenhuma imagem foi encontrada para copiar!")
        else:
            print(f"\nProcesso concluído! {images_copied} imagens foram copiadas para a pasta Kyoto")
    
    kyoto_path = Path("Kyoto")

    if not os.path.isdir(kyoto_path):
        download_Kyoto()

    caminho_imagens = []

    imagens = os.listdir(kyoto_path)

    for img in imagens:
        if not img == 'dataAug':
            full_path = os.path.join(kyoto_path, img)
            caminho_imagens.append(full_path)

    df = pd.DataFrame({
            'caminho_imagem': sorted(caminho_imagens)
    })

    print(df)

    df_treino = df['caminho_imagem'][:treino]
    df_validacao = df['caminho_imagem'][treino:treino+validacao]
    df_teste = df['caminho_imagem'][-teste:]

    if not os.path.isdir("CSV"):
        os.makedirs("CSV")
        os.makedirs("CSV/Kyoto")

    if not os.path.isdir("CSV/Kyoto"):
        os.makedirs("CSV/Kyoto")

    caminho_imagens_dataAug = []    

    imagens_dataAug = os.listdir(os.path.join(kyoto_path, 'dataAug'))

    for img_data in imagens_dataAug:
        full_path_data = os.path.join(kyoto_path, 'dataAug', img_data)
        caminho_imagens_dataAug.append(full_path_data)    

    df_dataAug = pd.DataFrame({
        'caminho_imagem':sorted(caminho_imagens_dataAug)
    })

    df_treino = pd.concat([df_treino, df_dataAug], ignore_index=False)

    df_treino.to_csv('CSV/Kyoto/Kyoto_Segmentado_Treino.csv', index=False)
    df_validacao.to_csv('CSV/Kyoto/Kyoto_Segmentado_Validacao.csv', index=False)
    df_teste.to_csv('CSV/Kyoto/Kyoto_Segmentado_Teste.csv', index=False)

def retorna_nome_base(caminho):
    nome = caminho.split('/')[-1]
    nome = nome.rsplit('.csv', 1)[0]
    nome = nome.split('_')
    return (nome[0], nome[2])

def download_datasets():
    if not os.path.isdir("Kyoto"):
        download_Kyoto
            
def download_Kyoto():
        kyoto_path = Path("Kyoto")

        if not os.path.isdir(kyoto_path):
            kyoto_path.mkdir(exist_ok=True)
            print(f"Pasta Kyoto criada em: {kyoto_path.absolute()}")
            
            # Baixar o arquivo ZIP
            url = "https://github.com/eizaburo-doi/kyoto_natim/archive/refs/heads/master.zip"
            print("Baixando arquivo ZIP...")
            response = requests.get(url)
            
            if response.status_code != 200:
                raise Exception(f"Erro ao baixar o arquivo. Status code: {response.status_code}")
            
            # Salvar o arquivo ZIP temporariamente
            zip_path = "kyoto_temp.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print(f"Arquivo ZIP salvo em: {zip_path}")
            
            # Extrair o arquivo ZIP
            print("Extraindo arquivo ZIP...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("temp_extract")
            
            # Listar todos os arquivos extraídos para debug
            print("\nConteúdo extraído:")
            for root, dirs, files in os.walk("temp_extract"):
                print(f"\nDiretório: {root}")
                for name in files:
                    print(f"- {name}")
            
            # Tentar diferentes possíveis caminhos para a pasta thumb
            possible_paths = [
                Path("temp_extract/kyoto_natim-master/kyoto_natim-master/thumb"),
                Path("temp_extract/kyoto_natim-master/thumb"),
                Path("temp_extract/thumb")
            ]
            
            thumb_path = None
            for path in possible_paths:
                if path.exists():
                    thumb_path = path
                    print(f"\nPasta thumb encontrada em: {thumb_path}")
                    break
            
            if not thumb_path:
                raise Exception("Pasta thumb não encontrada nos caminhos esperados")
            
            # Copiar todas as imagens para a pasta Kyoto
            print("\nCopiando imagens...")
            image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
            images_copied = 0
            
            for file in thumb_path.glob("*"):
                print(f"Encontrado arquivo: {file.name}")
                if file.suffix.lower() in image_extensions:
                    shutil.copy2(file, kyoto_path)
                    print(f"Copiado: {file.name}")
                    images_copied += 1
            
            # Limpar arquivos temporários
            print("\nLimpando arquivos temporários...")
            os.remove(zip_path)
            shutil.rmtree("temp_extract")
            
            if images_copied == 0:
                print("\nNenhuma imagem foi encontrada para copiar!")
            else:
                print(f"\nProcesso concluído! {images_copied} imagens foram copiadas para a pasta Kyoto")

def PKLot():
    cria_dirs()

    def cria_pklot():
        df_final = pd.DataFrame()  
        n_imgs = [102, 102, 102, 103, 103]
        faculdades = ['PUC', 'UFPR04', 'UFPR05']
        tempos = ['Cloudy', 'Rainy', 'Sunny']
        classes = ['Empty', 'Occupied']

        dados = []
        path_pklot = "PKLot/PKLotSegmented"
        for faculdade in faculdades:
            for tempo in tempos:
                path_facul_tempo = os.path.join(path_pklot, faculdade, tempo) #"PKLot/PKLotSegmented/PUC/Sunny"
                dias = os.listdir(path_facul_tempo)
                for dia in dias:
                    for classe in classes:
                        path_imgs = os.path.join(path_facul_tempo, dia, classe) #"PKLot/PKLotSegmented/PUC/Sunny/2012-09-12/Empty"
                        
                        if not os.path.isdir(path_imgs):
                            continue

                        imagens = os.listdir(path_imgs)
                        for img in imagens:
                            caminho_img = os.path.join(path_imgs, img)
                            dados.append([faculdade, tempo, dia, caminho_img, classe])

        df = pd.DataFrame(data=dados, columns=['Faculdade', 'Tempo', 'Dia', 'caminho_imagem' ,'classe'])
        df['classe'] = df['classe'].replace({'Empty': 1, 'Occupied': 0})
        df.to_csv("CSV/PKLot/PKLot.csv")

    cria_pklot()

    #Variveis de controle
    df_final = pd.DataFrame()  
    n_imgs = [102, 102, 102, 103, 103]
    faculdades = ['PUC', 'UFPR04', 'UFPR05']
    tempos = ['Cloudy', 'Rainy', 'Sunny']
    classes = ['Empty', 'Occupied']
    df = pd.read_csv('CSV/PKLot/PKLot.csv')

    dias_cada_facul = []
    for faculdade in faculdades:
        dias = df[df["Faculdade"] == f'{faculdade}']["Dia"].unique()
        dias_cada_facul.append(dias)

    # Df de cada uma das faculdades:
    #Treino  
    for i, faculdade in enumerate(faculdades):
        df_facul = df[(df['Faculdade'] == f'{faculdade}')] 

        file_treino = f"CSV/{faculdade}/{faculdade}.csv"
        df_facul_final = df_facul[['caminho_imagem', 'classe']]
        df_facul_final.to_csv(file_treino, index=False)

        primeiros_dias = dias_cada_facul[i][:5]  
        print("Os respectivos dias foram selecionados para treino: ", primeiros_dias)
        dias_cada_facul[i] = dias_cada_facul[i][5:] #Removendo os dias selecionadas

        while df_final.shape[0] < 1024:
            for j, dia in enumerate(primeiros_dias):
                for classe in [0, 1]:
                    df_dias = df_facul[(df_facul['classe'] == classe)]  
                    df_imgs = df_dias.sample(n=n_imgs[j], random_state=42)
                    df_final = pd.concat([df_final, df_imgs], axis=0, ignore_index=True) 

                    if df_final.shape[0] >= 1024:
                        break
                if df_final.shape[0] >= 1024:
                    break

        file_treino = f"CSV/{faculdade}/{faculdade}_Segmentado_Treino.csv"
        df_final = df_final[['caminho_imagem', 'classe']]
        df_final.to_csv(file_treino, index=False)
        print(f"DataFrame da faculdade {faculdade} salvo em {file_treino}")

        # Resetando o df_final para a próxima faculdade
        df_final = pd.DataFrame()

    #Validação
    for i, faculdade in enumerate(faculdades):
        df_facul = df[(df['Faculdade'] == f'{faculdade}')] 

        primeiros_dias = dias_cada_facul[i][:1]  
        print("O(s) respectivo(s) dia(s) foram selecionado(s) para validação: ", primeiros_dias)
        dias_cada_facul[i] = dias_cada_facul[i][1:] #Removendo os dias selecionadas

        while df_final.shape[0] < 64:
            for dia in primeiros_dias:
                for classe in [0, 1]:
                    df_dias = df_facul[(df_facul['classe'] == classe)]  
                    df_imgs = df_dias.sample(n=32, random_state=42)
                    df_final = pd.concat([df_final, df_imgs], axis=0, ignore_index=True) 

                    if df_final.shape[0] >= 64:
                        break
                if df_final.shape[0] >= 64:
                    break

        file_val = f"CSV/{faculdade}/{faculdade}_Segmentado_Validacao.csv"
        df_final = df_final[['caminho_imagem', 'classe']]
        df_final.to_csv(file_val, index=False)
        print(f"DataFrame da faculdade {faculdade} salvo em {file_val}")

        # Resetando o df_final para a próxima faculdade
        df_final = pd.DataFrame()

    #Teste
    for i, faculdade in enumerate(faculdades):
        df_facul = df[(df['Faculdade'] == f'{faculdade}')] 

        print("O(s) respectivo(s) dia(s) foram selecionado(s) para validação: ", dias_cada_facul[i])

        df_final  = df_facul[(df_facul['Dia'].isin(dias_cada_facul[i]))]

        file_teste = f"CSV/{faculdade}/{faculdade}_Segmentado_Teste.csv"
        df_final = df_final[['caminho_imagem', 'classe']]

        df_final.to_csv(file_teste, index=False)
        print(f"DataFrame da faculdade {faculdade} salvo em {file_teste}")

        # Resetando o df_final para a próxima faculdade
        df_final = pd.DataFrame()

def split_balanced(df, size, class_column='classe'):
    #dividimos o df em dois: classe 0 e classe 1
    df_class_0 = df[df[class_column] == 0]
    df_class_1 = df[df[class_column] == 1]
    
    #quantas amostras de cada classe 
    n_class_0 = size // 2
    n_class_1 = size // 2
    
    #verifica se é impar
    if size % 2 != 0:
        n_class_0 += 1
    
    #pega as amostras de forma balanceada
    df_class_0_sampled = df_class_0.sample(n=n_class_0, random_state=42)
    df_class_1_sampled = df_class_1.sample(n=n_class_1, random_state=42)
    
    #concatenamos 
    balanced_df = pd.concat([df_class_0_sampled, df_class_1_sampled])
    
    return balanced_df

def dividir_em_batchs(csv, nome):
    df = pd.read_csv(csv)

    print(df['classe'].value_counts())

    sizes = [64, 128, 256, 512, 1024]
    dfs = []
    df_64 = split_balanced(df, sizes[0])
    dfs.append(df_64)

    for i in range(1, len(sizes)):
        previous_size = sizes[i-1]
        current_size = sizes[i]
        
        additional_size = current_size - previous_size
        
        remaining_df = df.drop(dfs[i-1].index)
        
        df_additional = split_balanced(remaining_df, additional_size)
        
        df_current = pd.concat([dfs[i-1], df_additional])
        dfs.append(df_current)

    for i, size in enumerate(sizes):
        if not os.path.isdir(f'CSV/{nome}/batches'):
            os.makedirs(f'CSV/{nome}/batches')
        dfs[i].to_csv(f'CSV/{nome}/batches/batch-{size}.csv', index=False)

    print("Arquivos CSV criados com sucesso!")

def cria_CNR():
    path_labels = 'CNR-EXT-Patches-150x150/LABELS/all.txt'
    dados = []
    caminhos_imagens = []
    classes = []

    with open(path_labels, 'r') as file:
        for linha in file:
            partes = linha.strip().split(' ')

            if len(partes) == 2:
                caminho_imagem = partes[0]
                caminho_imagem_completo = 'CNR-EXT-Patches-150x150/PATCHES/' + caminho_imagem
                tempo, data, camera, _ = caminho_imagem.strip().split('/')
                caminhos_imagens.append(caminho_imagem_completo)
                classe = partes[1]
                if classe == '0':
                    classe = 'Empty'
                else:
                    classe = 'Occupied'
                classes.append(classe)
                dados.append([tempo, data, camera, caminho_imagem_completo, classe])

    df = pd.DataFrame(data=dados, columns=['Tempo', 'Data', 'Camera', 'caminho_imagem', 'classe'])

    if not os.path.isdir("CSV"):
        os.makedirs("CSV")
        os.makedirs("CSV/CNR")
    if not os.path.isdir("CSV/CNR"):
        os.makedirs("CSV/CNR")

    df.to_csv("CSV/CNR/CNR.csv", index=False)

def cria_csvs():
    PKLot()
    cria_CNR()
    segmentacao_Kyoto()
    nomes = ['PUC', 'UFPR04', 'UFPR05']
    for nome in nomes:
        dividir_em_batchs(f'CSV/{nome}/{nome}_Segmentado_Treino.csv', nome)
    
    cameras = ['camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']
    df_cnr = pd.read_csv('CSV/CNR/CNR.csv')
    for camera in cameras:
        df_camera = df_cnr[(df_cnr['Camera'] == camera)]
        df_camera_final = df_camera[['caminho_imagem', 'classe']]
        df_camera_final['classe'] = df_camera_final['classe'].replace({'Empty': 1, 'Occupied': 0})
        df_camera_final.to_csv(f'CSV/CNR/CNR_{camera}.csv', index=False)

#cria_csvs()