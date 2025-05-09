import pandas as pd
import numpy as np
import os 
import random
import albumentations as A
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
import cv2
import requests
import argparse

parser = argparse.ArgumentParser(description="Descrição do seu script.")
parser.add_argument('--path', type=str, default='/datasets', help='Caminho base para salvar os datasets')
datasets_path = parser.parse_args().path

try:
    os.path.open(datasets_path)
except Exception as e:
    print(f"O diretório {datasets_path} não existe!")

#Definindo a seed para que os resultados sejam os mesmos
SEED = 42
random.seed(SEED)
np.random.seed(SEED)    

#Download dos datasets
def download_Kyoto(path):
    # Criar pasta Kyoto dentro do caminho especificado
    kyoto_path = Path(path) / "Kyoto"
    kyoto_path.mkdir(exist_ok=True)
    print(f"Pasta Kyoto criada em: {kyoto_path.absolute()}")
    
    # Baixar o arquivo ZIP
    url = "https://github.com/eizaburo-doi/kyoto_natim/archive/refs/heads/master.zip"
    print("Baixando arquivo ZIP...")
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Erro ao baixar o arquivo. Status code: {response.status_code}")
    
    # Salvar o arquivo ZIP temporariamente no caminho especificado
    zip_path = Path(path) / "kyoto_temp.zip"
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    print(f"Arquivo ZIP salvo em: {zip_path}")
    
    # Extrair o arquivo ZIP
    print("Extraindo arquivo ZIP...")
    temp_extract_path = Path(path) / "temp_extract"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)
    
    # Listar todos os arquivos extraídos para debug
    print("\nConteúdo extraído:")
    for root, dirs, files in os.walk(temp_extract_path):
        print(f"\nDiretório: {root}")
        for name in files:
            print(f"- {name}")
    
    # Tentar diferentes possíveis caminhos para a pasta thumb
    possible_paths = [
        temp_extract_path / "kyoto_natim-master/kyoto_natim-master/thumb",
        temp_extract_path / "kyoto_natim-master/thumb",
        temp_extract_path / "thumb"
    ]
    
    thumb_path = None
    for p in possible_paths:
        if p.exists():
            thumb_path = p
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
    shutil.rmtree(temp_extract_path)
    
    if images_copied == 0:
        print("\nNenhuma imagem foi encontrada para copiar!")
    else:
        print(f"\nProcesso concluído! {images_copied} imagens foram copiadas para a pasta Kyoto")

def download_PKLot(path):
    url = 'http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz'
    file_name = Path(path) / 'PKLot.tar.gz'
    
    print("Baixando arquivo...")
    urllib.request.urlretrieve(url, file_name)
    
    print("Extraindo arquivo...")
    with tarfile.open(file_name, 'r:gz') as tar:
        tar.extractall(path=path)
    
    print("Arquivo extraído com sucesso.")
    os.remove(file_name)

def download_CNR(path):
    url = 'https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip'
    file_name = Path(path) / 'CNR-EXT-Patches-150x150.zip'
    
    print("Baixando arquivo...")
    urllib.request.urlretrieve(url, file_name)
    
    print("Extraindo arquivo...")
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(path=path)
    
    print("Arquivo extraído com sucesso.")
    os.remove(file_name)

def download_all_datasets(path):
    # Criar o diretório se não existir
    Path(path).mkdir(exist_ok=True)
    print(f"Diretório de destino: {Path(path).absolute()}")
    
    download_PKLot(path)
    download_CNR(path)
    download_Kyoto(path)

#Data augmentation na Kyoto:
def data_augmentation_kyoto(kyoto_path):
    transform1 = A.Compose([
            A.GaussNoise(var_limit=(0.0, 0.0007), mean=0, p=1),
            A.Rotate(limit=180, p=1),
    ])
    transform2 = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True, p=1),
                A.AdvancedBlur(blur_limit=(7,9), noise_limit=(0.75, 1.25), p=1),
        ])
    transform3 = A.Compose([
                A.ChannelShuffle(p=1),
    ])
    transform4 = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True, p=1),
                A.AdvancedBlur(blur_limit=(7,9), noise_limit=(0.75, 1.25), p=1),
    ])

    if not os.path.isdir(os.path.join(kyoto_path, 'dataAug')):
        os.makedirs(os.path.join(kyoto_path, 'dataAug'))

    for img_name in os.listdir(kyoto_path):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):  # Verifica se é uma imagem
            img_path = os.path.join(kyoto_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Não foi possível ler a imagem: {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img1 = transform1(image=img)['image']
            img2 = transform2(image=img)['image']
            img3 = transform3(image=img)['image']
            img4 = transform4(image=img)['image']

            # Converter de volta para BGR para salvar com cv2
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
            img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)

            # Salvar as imagens usando cv2.imwrite
            base_name = os.path.splitext(img_name)[0]
            cv2.imwrite(os.path.join(kyoto_path, 'dataAug', f'kyoto_{base_name}_1.jpg'), img1)
            cv2.imwrite(os.path.join(kyoto_path, 'dataAug', f'kyoto_{base_name}_2.jpg'), img2)
            cv2.imwrite(os.path.join(kyoto_path, 'dataAug', f'kyoto_{base_name}_3.jpg'), img3)
            cv2.imwrite(os.path.join(kyoto_path, 'dataAug', f'kyoto_{base_name}_4.jpg'), img4)
            
            print(f"Processada imagem: {img_name}")

#Kyoto - criando csvs
def cria_Kyoto(kyoto_path='/datasets/Kyoto'):

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

    df_treino = df['caminho_imagem'][:32]
    df_validacao = df['caminho_imagem'][32:32+20]
    df_teste = df['caminho_imagem'][-8:]

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

    df_treino.to_csv('CSV/Kyoto/Kyoto_treino.csv', index=False)
    df_validacao.to_csv('CSV/Kyoto/Kyoto_validacao.csv', index=False)
    df_teste.to_csv('CSV/Kyoto/Kyoto_teste.csv', index=False)

#CNR - criando csvs
cameras = [f'camera{i}' for i in range(1, 10)]

def cria_CNR(path_dataset='/datasets/CNR-EXT-Patches-150x150') -> None:
    """
    Cria um csv com todas as imagens da base de dados CNR-EXT-Patches-150x150.
    Parâmetros:
        path_dataset: caminho da pasta onde está a base de dados CNR-EXT-Patches-150x150.
    """
    path_labels = os.path.join(path_dataset, 'LABELS/all.txt')
    dados = []
    caminhos_imagens = []
    classes = []

    with open(path_labels, 'r') as file:
        for linha in file:
            partes = linha.strip().split(' ')

            if len(partes) == 2:
                caminho_imagem = partes[0]
                caminho_imagem_completo = os.path.join(path_dataset, 'PATCHES', caminho_imagem)
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

def CNR_cameras():
    df = pd.read_csv('CSV/CNR/CNR.csv')
    
    for camera in cameras:
        df_camera = df[df['Camera'] == camera]
        os.makedirs(f"CSV/{camera}", exist_ok=True)
        df_camera.to_csv(f'CSV/{camera}/{camera}.csv', index=False)

        dias = sorted(df_camera['Data'].unique())

        # ---------- Treino ----------
        df_treino = pd.DataFrame()
        total_imagens = 0
        dias_treino = []
        for dia in dias:
            if total_imagens >= 1024:
                break

            df_dia = df_camera[df_camera['Data'] == dia]
            empty = df_dia[df_dia['classe'] == 'Empty']
            occupied = df_dia[df_dia['classe'] == 'Occupied']

            n_balanco = min(len(empty), len(occupied))

            # Verifica quanto ainda falta para 1024
            falta = 1024 - total_imagens

            # Cada dia contribui com 2 * n_balanco imagens (Empty + Occupied)
            if 2 * n_balanco > falta:
                # Reduz proporcionalmente para não ultrapassar 1024
                n_balanco = falta // 2

            if n_balanco == 0:
                dias_treino.append(dia)
                dias.remove(dia)
                continue  # pula o dia se não houver imagens suficientes

            empty_sample = empty.sample(n=n_balanco, random_state=SEED)
            occupied_sample = occupied.sample(n=n_balanco, random_state=SEED)

            df_amostrado = pd.concat([empty_sample, occupied_sample], ignore_index=True)
            df_treino = pd.concat([df_treino, df_amostrado], ignore_index=True)

            total_imagens += len(df_amostrado)
            dias_treino.append(dia)
            dias.remove(dia)
            
        df_treino.to_csv(f'CSV/{camera}/{camera}_treino.csv', columns=['caminho_imagem', 'classe'], index=False)
        print(f"Câmera {camera} - Treino: {len(df_treino)} imagens (balanceado), dias utilizados: {dias_treino}")

        # ---------- Validação ----------
        df_validacao = pd.DataFrame()
        total_imagens = 0
        dias_validacao = []
        for dia in dias:
            if total_imagens >= 64:
                break

            df_dia = df_camera[df_camera['Data'] == dia]
            empty = df_dia[df_dia['classe'] == 'Empty']
            occupied = df_dia[df_dia['classe'] == 'Occupied']

            n_balanco = min(len(empty), len(occupied))

            # Verifica quanto ainda falta para 64
            falta = 64 - total_imagens

            # Cada dia contribui com 2 * n_balanco imagens (Empty + Occupied)
            if 2 * n_balanco > falta:
                # Reduz proporcionalmente para não ultrapassar 64
                n_balanco = falta // 2

            if n_balanco == 0:
                dias_validacao.append(dia)
                dias.remove(dia)
                continue  # pula o dia se não houver imagens suficientes

            empty_sample = empty.sample(n=n_balanco, random_state=SEED)
            occupied_sample = occupied.sample(n=n_balanco, random_state=SEED)

            df_amostrado = pd.concat([empty_sample, occupied_sample], ignore_index=True)
            df_validacao = pd.concat([df_validacao, df_amostrado], ignore_index=True)

            total_imagens += len(df_amostrado)
            dias_validacao.append(dia)
            dias.remove(dia)
            
        df_validacao.to_csv(f'CSV/{camera}/{camera}_validacao.csv', columns=['caminho_imagem', 'classe'], index=False)
        print(f"Câmera {camera} - Validacao: {len(df_validacao)} imagens (balanceado), dias utilizados: {dias_validacao}")

        # ---------- Teste ----------
        df_teste = pd.DataFrame()
        total_imagens = 0
        dias_teste = []
        for dia in dias:
            df_dia = df_camera[df_camera['Data'] == dia]
            df_teste = pd.concat([df_teste, df_dia], ignore_index=True)

            total_imagens += len(df_amostrado)
            dias_teste.append(dia)
            dias.remove(dia)
            
        df_teste.to_csv(f'CSV/{camera}/{camera}_teste.csv', columns=['caminho_imagem', 'classe'], index=False)
        print(f"Câmera {camera} - Teste: {len(df_teste)} imagens (balanceado), dias utilizados: {dias_teste}")

def CNR_autoencoder():
    cameras = [['camera1', 56], ['camera2', 56], ['camera3', 56], ['camera4', 56], ['camera5', 56], ['camera6', 55], ['camera7', 55], ['camera8', 55], ['camera9', 55]]
    
    df_treino = pd.DataFrame()
    for camera, valor in cameras:
        df = pd.read_csv(f'CSV/{camera}/{camera}_teste.csv')
        occupied = df[df['classe'] == 'Occupied'].sample(n=valor, random_state=SEED)
        empty = df[df['classe'] == 'Empty'].sample(n=valor, random_state=SEED)
        df = pd.concat([occupied, empty], axis=0, ignore_index=True)
        df_treino =pd.concat([df_treino, df], axis=0, ignore_index=True)

    df_treino.to_csv(f'CSV/CNR/CNR_autoencoder_treino.csv', columns=['caminho_imagem', 'classe'], index=False) 

    cameras = [['camera1', 8], ['camera2', 7], ['camera3', 7], ['camera4', 7], ['camera5', 7], ['camera6', 7], ['camera7', 7], ['camera8', 7], ['camera9', 7]]
    df_val = pd.DataFrame()
    for camera, valor in cameras:
        df = pd.read_csv(f'CSV/{camera}/{camera}_validacao.csv')
        occupied = df[df['classe'] == 'Occupied'].sample(n=valor, random_state=SEED)
        empty = df[df['classe'] == 'Empty'].sample(n=valor, random_state=SEED)
        df = pd.concat([occupied, empty], axis=0, ignore_index=True)
        df_val =pd.concat([df_val, df], axis=0, ignore_index=True)
        
    df_val.to_csv(f'CSV/CNR/CNR_autoencoder_validacao.csv', columns=['caminho_imagem', 'classe'], index=False)

    df_teste = pd.DataFrame()
    for camera, valor in cameras:
        df = pd.read_csv(f'CSV/{camera}/{camera}_teste.csv')
        df_teste =pd.concat([df_teste, df], axis=0, ignore_index=True)
        
    df_teste.to_csv(f'CSV/CNR/CNR_autoencoder_teste.csv', columns=['caminho_imagem', 'classe'], index=False) 

#PKLot - criando csvs
faculdades = ['PUC', 'UFPR04', 'UFPR05']

def cria_PKLot(path_pklot='/datasets/PKLot/PKLotSegmented'):
    df_final = pd.DataFrame()  
    faculdades = ['PUC', 'UFPR04', 'UFPR05']
    tempos = ['Cloudy', 'Rainy', 'Sunny']
    classes = ['Empty', 'Occupied']

    dados = []
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
    if not os.path.isdir("CSV/PKLot"):
        os.makedirs("CSV/PKLot")

    df.to_csv("CSV/PKLot/PKLot.csv")

def PKLot_faculdades():
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
        if not os.path.isdir(f"CSV/{faculdade}"):
            os.makedirs(f"CSV/{faculdade}")

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
                    df_imgs = df_dias.sample(n=n_imgs[j], random_state=SEED)
                    df_final = pd.concat([df_final, df_imgs], axis=0, ignore_index=True) 

                    if df_final.shape[0] >= 1024:
                        break
                if df_final.shape[0] >= 1024:
                    break

        file_treino = f"CSV/{faculdade}/{faculdade}_treino.csv"
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
                    df_imgs = df_dias.sample(n=32, random_state=SEED)
                    df_final = pd.concat([df_final, df_imgs], axis=0, ignore_index=True) 

                    if df_final.shape[0] >= 64:
                        break
                if df_final.shape[0] >= 64:
                    break

        file_val = f"CSV/{faculdade}/{faculdade}_validacao.csv"
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

        file_teste = f"CSV/{faculdade}/{faculdade}_teste.csv"
        df_final = df_final[['caminho_imagem', 'classe']]

        df_final.to_csv(file_teste, index=False)
        print(f"DataFrame da faculdade {faculdade} salvo em {file_teste}")

        # Resetando o df_final para a próxima faculdade
        df_final = pd.DataFrame()

def PKLot_autoencoder():
    faculdades = [['PUC', 171], ['UFPR04', 170], ['UFPR05', 171]]
    
    df_treino = pd.DataFrame()
    for faculdade, valor in faculdades:
        df = pd.read_csv(f'CSV/{faculdade}/{faculdade}_teste.csv')
        occupied = df[df['classe'] == 0].sample(n=valor, random_state=SEED)
        empty = df[df['classe'] == 1].sample(n=valor, random_state=SEED)
        df = pd.concat([occupied, empty], axis=0, ignore_index=True)
        df_treino =pd.concat([df_treino, df], axis=0, ignore_index=True)

    df_treino.to_csv(f'CSV/PKLot/PKLot_autoencoder_treino.csv', columns=['caminho_imagem', 'classe'], index=False) 

    faculdades = [['PUC', 11], ['UFPR04', 10], ['UFPR05', 11]]
    df_val = pd.DataFrame()
    for faculdade, valor in faculdades:
        df = pd.read_csv(f'CSV/{faculdade}/{faculdade}_validacao.csv')
        occupied = df[df['classe'] == 0].sample(n=valor, random_state=SEED)
        empty = df[df['classe'] == 1].sample(n=valor, random_state=SEED)
        df = pd.concat([occupied, empty], axis=0, ignore_index=True)
        df_val =pd.concat([df_val, df], axis=0, ignore_index=True)
        
    df_val.to_csv(f'CSV/PKLot/PKLot_autoencoder_validacao.csv', columns=['caminho_imagem', 'classe'], index=False)

    df_teste = pd.DataFrame()
    for faculdade, valor in faculdades:
        df = pd.read_csv(f'CSV/{faculdade}/{faculdade}_teste.csv')
        df_teste =pd.concat([df_teste, df], axis=0, ignore_index=True)
        
    df_teste.to_csv(f'CSV/PKLot/PKLot_autoencoder_teste.csv', columns=['caminho_imagem', 'classe'], index=False) 

if not os.path.isdir('CSV'):
    download_all_datasets(datasets_path)

data_augmentation_kyoto(os.path.join(datasets_path ,'Kyoto'))
cria_Kyoto(os.path.join(datasets_path ,'Kyoto'))

cria_CNR(os.path.join(datasets_path ,'CNR-EXT-Patches-150x150'))
CNR_cameras()
CNR_autoencoder()

cria_PKLot(os.path.join(datasets_path ,'PKLot/PKLotSegmented'))
PKLot_faculdades()
PKLot_autoencoder()

print("Datasets criados com sucesso!")