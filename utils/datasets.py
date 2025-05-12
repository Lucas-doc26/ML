import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

from .path_manager import PathManager, create_folder
from .download import download_all_datasets
from .preprocessing import data_augmentation_kyoto

from .config import SEED

#Kyoto - criando csvs
def create_Kyoto_csv(kyoto_path:Path):

    path_images = []

    images = os.listdir(kyoto_path)

    for img in images:
        if not img == 'dataAug':
            full_path = os.path.join(kyoto_path, img)
            path_images.append(full_path)

    df = pd.DataFrame({
            'path_image': sorted(path_images)
    })

    df_train = df['path_image'][:32]
    df_validation = df['path_image'][32:32+20]
    df_test = df['path_image'][-8:]

    path_images_dataAug = []    

    images_dataAug = os.listdir(os.path.join(kyoto_path, 'dataAug'))

    for img_data in images_dataAug:
        full_path_data = os.path.join(kyoto_path, 'dataAug', img_data)
        path_images_dataAug.append(full_path_data)    

    df_dataAug = pd.DataFrame({
        'path_image':sorted(path_images_dataAug)
    })

    df_train = pd.concat([df_train, df_dataAug], ignore_index=False)

    df_train.to_csv('CSV/Kyoto/Kyoto_autoencoder_train.csv', index=False)
    df_validation.to_csv('CSV/Kyoto/Kyoto_autoencoder_validation.csv', index=False)
    df_test.to_csv('CSV/Kyoto/Kyoto_autoencoder_test.csv', index=False)

#CNR - criando csvs
cameras = [f'camera{i}' for i in range(1, 10)]

def create_CNR_csv(path_dataset:Path) -> None:
    """
    Cria um csv com todas as images da base de dados CNR-EXT-Patches-150x150.
    Parâmetros:
        path_dataset: caminho da pasta onde está a base de dados CNR-EXT-Patches-150x150.
    """
    path_labels = os.path.join(path_dataset, 'LABELS/all.txt')
    data = []
    path_images = []
    classes = []

    with open(path_labels, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')

            if len(parts) == 2:
                path_image = parts[0]
                path_image_complete = os.path.join(path_dataset, 'PATCHES', path_image)
                wheather, day, camera, _ = path_image.strip().split('/')
                path_images.append(path_image_complete)
                class_image = parts[1]
                classes.append(class_image)
                data.append([wheather, day, camera, path_image_complete, class_image])

    df = pd.DataFrame(data=data, columns=['wheather', 'day', 'camera', 'path_image', 'class'])

    df.to_csv("CSV/CNR/CNR.csv", index=False)

def create_CNR_cameras():
    df = pd.read_csv('CSV/CNR/CNR.csv')
    
    for camera in cameras:
        df_camera = df[df['camera'] == camera]
        df_camera.to_csv(f'CSV/{camera}/{camera}.csv', index=False)

        days = sorted(df_camera['day'].unique())

        # ---------- Treino ----------
        df_train = pd.DataFrame()
        total_images = 0
        days_train = []
        for day in days:
            if total_images >= 1024:
                break

            df_day = df_camera[df_camera['day'] == day]
            empty = df_day[df_day['class'] == 1]
            occupied = df_day[df_day['class'] == 0]

            images_per_class = min(len(empty), len(occupied))

            # Verifica quanto ainda falta para 1024
            missing_images = 1024 - total_images

            # Cada dia contribui com 2 * images_per_class (Empty + Occupied)
            if 2 * images_per_class > missing_images:
                # Reduz proporcionalmente para não ultrapassar 1024
                images_per_class = missing_images // 2

            if images_per_class == 0:
                days_train.append(day)
                days.remove(day)
                continue  # pula o dia se não houver images suficientes

            empty_sample = empty.sample(n=images_per_class, random_state=SEED)
            occupied_sample = occupied.sample(n=images_per_class, random_state=SEED)

            df_classes = pd.concat([empty_sample, occupied_sample], ignore_index=True)
            df_train = pd.concat([df_train, df_classes], ignore_index=True)

            total_images += len(df_classes)
            days_train.append(day)
            days.remove(day)
            
        df_train.to_csv(f'CSV/{camera}/{camera}_train.csv', columns=['path_image', 'class'], index=False)
        print(f"Câmera {camera} - Treino: {len(df_train)} images (balanceado), dias utilizados: {days_train}")

        # ---------- Validação ----------
        df_validation = pd.DataFrame()
        total_images = 0
        days_validation = []
        for day in days:
            if total_images >= 64:
                break

            df_day = df_camera[df_camera['day'] == day]
            empty = df_day[df_day['class'] == 1]
            occupied = df_day[df_day['class'] == 0]

            n_balanco = min(len(empty), len(occupied))

            # Verifica quanto ainda falta para 64
            falta = 64 - total_images

            # Cada dia contribui com 2 * n_balanco images (Empty + Occupied)
            if 2 * n_balanco > falta:
                # Reduz proporcionalmente para não ultrapassar 64
                n_balanco = falta // 2

            if n_balanco == 0:
                days_validation.append(day)
                days.remove(day)
                continue  # pula o dia se não houver images suficientes

            empty_sample = empty.sample(n=n_balanco, random_state=SEED)
            occupied_sample = occupied.sample(n=n_balanco, random_state=SEED)

            df_classes = pd.concat([empty_sample, occupied_sample], ignore_index=True)
            df_validation = pd.concat([df_validation, df_classes], ignore_index=True)

            total_images += len(df_classes)
            days_validation.append(day)
            days.remove(day)
            
        df_validation.to_csv(f'CSV/{camera}/{camera}_validation.csv', columns=['path_image', 'class'], index=False)
        print(f"Câmera {camera} - Validation: {len(df_validation)} images (balanceado), dias utilizados: {days_validation}")

        # ---------- Teste ----------
        df_test = pd.DataFrame()
        total_images = 0
        days_test = []
        for day in days:
            df_day = df_camera[df_camera['day'] == day]
            df_test = pd.concat([df_test, df_day], ignore_index=True)

            total_images += len(df_classes)
            days_test.append(day)
            days.remove(day)
            
        df_test.to_csv(f'CSV/{camera}/{camera}_test.csv', columns=['path_image', 'class'], index=False)
        print(f"Câmera {camera} - Teste: {len(df_test)} images (balanceado), dias utilizados: {days_test}")

def create_CNR_autoencoder():
    cameras = [['camera1', 56], ['camera2', 56], ['camera3', 56], ['camera4', 56], ['camera5', 56], ['camera6', 55], ['camera7', 55], ['camera8', 55], ['camera9', 55]]
    
    df_train = pd.DataFrame()
    for camera, value in cameras:
        df = pd.read_csv(f'CSV/{camera}/{camera}_test.csv')
        occupied = df[df['class'] == 0].sample(n=value, random_state=SEED)
        empty = df[df['class'] == 1].sample(n=value, random_state=SEED)
        df = pd.concat([occupied, empty], axis=0, ignore_index=True)
        df_train =pd.concat([df_train, df], axis=0, ignore_index=True)

    df_train.to_csv(f'CSV/CNR/CNR_autoencoder_train.csv', columns=['path_image', 'class'], index=False) 

    cameras = [['camera1', 8], ['camera2', 7], ['camera3', 7], ['camera4', 7], ['camera5', 7], ['camera6', 7], ['camera7', 7], ['camera8', 7], ['camera9', 7]]
    df_val = pd.DataFrame()
    for camera, value in cameras:
        df = pd.read_csv(f'CSV/{camera}/{camera}_validation.csv')
        occupied = df[df['class'] == 0].sample(n=value, random_state=SEED)
        empty = df[df['class'] == 1].sample(n=value, random_state=SEED)
        df = pd.concat([occupied, empty], axis=0, ignore_index=True)
        df_val =pd.concat([df_val, df], axis=0, ignore_index=True)
        
    df_val.to_csv(f'CSV/CNR/CNR_autoencoder_validation.csv', columns=['path_image', 'class'], index=False)

    df_test = pd.DataFrame()
    for camera, value in cameras:
        df = pd.read_csv(f'CSV/{camera}/{camera}_test.csv')
        df_test =pd.concat([df_test, df], axis=0, ignore_index=True)
        
    df_test.to_csv(f'CSV/CNR/CNR_autoencoder_test.csv', columns=['path_image', 'class'], index=False) 

#PKLot - criando csvs
universities = ['PUC', 'UFPR04', 'UFPR05']

def create_PKLot_csv(path_pklot):
    universities = ['PUC', 'UFPR04', 'UFPR05']
    wheathers = ['Cloudy', 'Rainy', 'Sunny']
    classes = ['Empty', 'Occupied']

    data = []
    for university in universities:
        for wheather in wheathers:
            path_university_wheather = os.path.join(path_pklot, university, wheather) #"PKLot/PKLotSegmented/PUC/Sunny"
            days = os.listdir(path_university_wheather)
            for day in days:
                for class_image in classes:
                    path_imgs = os.path.join(path_university_wheather, day, class_image) #"PKLot/PKLotSegmented/PUC/Sunny/2012-09-12/Empty"
                    
                    if not os.path.isdir(path_imgs):
                        continue

                    images = os.listdir(path_imgs)
                    for img in images:
                        caminho_img = os.path.join(path_imgs, img)
                        data.append([university, wheather, day, caminho_img, class_image])

    df = pd.DataFrame(data=data, columns=['University', 'Wheather', 'Day', 'path_image' ,'class'])
    df['class'] = df['class'].replace({'Empty': 1, 'Occupied': 0})
    df.to_csv("CSV/PKLot/PKLot.csv", index=False)

def create_PKLot_universities():
    #Variveis de controle
    final_df = pd.DataFrame()  
    n_imgs = [102, 102, 102, 103, 103]
    universities = ['PUC', 'UFPR04', 'UFPR05']
    df = pd.read_csv('CSV/PKLot/PKLot.csv')

    days_per_university = []
    for university in universities:
        days = df[df["University"] == f'{university}']["Day"].unique()
        days_per_university.append(days)

    # Df de cada uma das universitys:
    #Treino  
    for i, university in enumerate(universities):
        if not os.path.isdir(f"CSV/{university}"):
            os.makedirs(f"CSV/{university}")

        df_university = df[(df['University'] == f'{university}')] 

        train_file = f"CSV/{university}/{university}.csv"
        final_df_university = df_university[['path_image', 'class']]
        final_df_university.to_csv(train_file, index=False)

        first_days = days_per_university[i][:5]  
        print("Os respectivos dias foram selecionados para treino: ", first_days)
        days_per_university[i] = days_per_university[i][5:] #Removendo os dias selecionadas

        while final_df.shape[0] < 1024:
            for j, day in enumerate(first_days):
                for class_image in [0, 1]:
                    df_days = df_university[(df_university['class'] == class_image)]  
                    df_imgs = df_days.sample(n=n_imgs[j], random_state=SEED)
                    final_df = pd.concat([final_df, df_imgs], axis=0, ignore_index=True) 

                    if final_df.shape[0] >= 1024:
                        break
                if final_df.shape[0] >= 1024:
                    break

        train_file = f"CSV/{university}/{university}_train.csv"
        final_df = final_df[['path_image', 'class']]
        final_df.to_csv(train_file, index=False)
        print(f"DataFrame da university {university} salvo em {train_file}")

        # Resetando o final_df para a próxima university
        final_df = pd.DataFrame()

    #Validação
    for i, university in enumerate(universities):

        df_university = df[(df['University'] == f'{university}')] 

        first_days = days_per_university[i][:1]  
        print("O(s) respectivo(s) dia(s) foram selecionado(s) para validação: ", first_days)
        days_per_university[i] = days_per_university[i][1:] #Removendo os dias selecionadas

        while final_df.shape[0] < 64:
            for day in first_days:
                for class_image in [0, 1]:
                    df_days = df_university[(df_university['class'] == class_image)]  
                    df_imgs = df_days.sample(n=32, random_state=SEED)
                    final_df = pd.concat([final_df, df_imgs], axis=0, ignore_index=True) 

                    if final_df.shape[0] >= 64:
                        break
                if final_df.shape[0] >= 64:
                    break

        validation_file = f"CSV/{university}/{university}_validation.csv"
        final_df = final_df[['path_image', 'class']]
        final_df.to_csv(validation_file, index=False)
        print(f"DataFrame da Faculdade {university} salvo em {validation_file}")

        # Resetando o final_df para a próxima university
        final_df = pd.DataFrame()

    #Teste
    for i, university in enumerate(universities):
        df_university = df[(df['University'] == f'{university}')] 

        print("O(s) respectivo(s) dia(s) foram selecionado(s) para validação: ", days_per_university[i])

        final_df  = df_university[(df_university['Day'].isin(days_per_university[i]))]

        test_file = f"CSV/{university}/{university}_test.csv"
        final_df = final_df[['path_image', 'class']]

        final_df.to_csv(test_file, index=False)
        print(f"DataFrame da Faculdade {university} salvo em {test_file}")

        # Resetando o final_df para a próxima university
        final_df = pd.DataFrame()

def create_PKLot_autoencoder():
    universitys = [['PUC', 171], ['UFPR04', 170], ['UFPR05', 171]]
    
    df_train = pd.DataFrame()
    for university, value in universitys:
        df = pd.read_csv(f'CSV/{university}/{university}_test.csv')
        occupied = df[df['class'] == 0].sample(n=value, random_state=SEED)
        empty = df[df['class'] == 1].sample(n=value, random_state=SEED)
        df = pd.concat([occupied, empty], axis=0, ignore_index=True)
        df_train =pd.concat([df_train, df], axis=0, ignore_index=True)

    df_train.to_csv(f'CSV/PKLot/PKLot_autoencoder_train.csv', columns=['path_image', 'class'], index=False) 

    universitys = [['PUC', 11], ['UFPR04', 10], ['UFPR05', 11]]
    df_val = pd.DataFrame()
    for university, value in universitys:
        df = pd.read_csv(f'CSV/{university}/{university}_validation.csv')
        occupied = df[df['class'] == 0].sample(n=value, random_state=SEED)
        empty = df[df['class'] == 1].sample(n=value, random_state=SEED)
        df = pd.concat([occupied, empty], axis=0, ignore_index=True)
        df_val =pd.concat([df_val, df], axis=0, ignore_index=True)
        
    df_val.to_csv(f'CSV/PKLot/PKLot_autoencoder_validation.csv', columns=['path_image', 'class'], index=False)

    df_test = pd.DataFrame()
    for university, value in universitys:
        df = pd.read_csv(f'CSV/{university}/{university}_test.csv')
        df_test =pd.concat([df_test, df], axis=0, ignore_index=True)
        
    df_test.to_csv(f'CSV/PKLot/PKLot_autoencoder_test.csv', columns=['path_image', 'class'], index=False) 

def print_csv(path_manager:PathManager):
    dirs = []
    for root, _, files in os.walk(os.path.join(path_manager.get_base_path(), 'CSV')):
        for file in files:
            dir = os.path.join(root, file)
            print(dir)
            dirs.append(dir)
    return dirs

def split_balanced(df, size, class_column='class'):
    df_class_0 = df[df[class_column] == 0]
    df_class_1 = df[df[class_column] == 1]
    
    n_class_0 = size // 2
    n_class_1 = size // 2

    if size % 2 != 0:
        n_class_0 += 1

    if not df_class_0.empty and n_class_0 > 0:
        n_class_0 = min(n_class_0, len(df_class_0))
        df_class_0_sampled = df_class_0.sample(n=n_class_0, random_state=SEED)
    else:
        df_class_0_sampled = pd.DataFrame()

    if not df_class_1.empty and n_class_1 > 0:
        n_class_1 = min(n_class_1, len(df_class_1))
        df_class_1_sampled = df_class_1.sample(n=n_class_1, random_state=SEED)
    else:
        df_class_1_sampled = pd.DataFrame()

    balanced_df = pd.concat([df_class_0_sampled, df_class_1_sampled])

    return balanced_df

def create_batches(datasets):
    for base_name in datasets:
        csv_path = f'CSV/{base_name}/{base_name}.csv'

        if not os.path.exists(csv_path):
            print(f'[ERRO] CSV não encontrado: {csv_path}')
            continue

        print(f'[INFO] Processando {base_name}...')

        df = pd.read_csv(csv_path)
        print(df['class'].value_counts())

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

        batches_dir = f'CSV/{base_name}/batches'
        os.makedirs(batches_dir, exist_ok=True)

        for i, size in enumerate(sizes):
            dfs[i].to_csv(f'{batches_dir}/batch-{size}.csv', index=False)

        print(f'[OK] Batches criados para {base_name} em {batches_dir}')

def format_datasets(datasets_dirs):
    for dir_data in datasets_dirs:
        df = pd.read_csv(dir_data)
        if 'Kyoto' in dir_data:
            df = df.sample(frac=1, random_state=SEED)
            df.to_csv(dir_data, index=False)
        else:
            df = df[['path_image', 'class']].sample(frac=1, random_state=SEED)
            df.to_csv(dir_data, index=False)

def create_datasets_csv(path_manager:PathManager=None, path_datasets_downloaded:Path=None):
    """
    Função para criar os csvs para os datasets
    """
    if not os.path.isdir(path_datasets_downloaded) or path_datasets_downloaded is None:
        create_folder(path_manager, 'datasets')
    
    datasets_path = download_all_datasets(path_datasets_downloaded)
    print(datasets_path)
    
    pass
    #Realizo o data augmentation no dataset Kyoto
    data_augmentation_kyoto(datasets_path[2])

    #Crio os csvs para os datasets
    create_Kyoto_csv(datasets_path[2])

    #CNR
    create_CNR_csv(datasets_path[1])
    create_CNR_cameras()
    create_CNR_autoencoder()

    #PKLot
    create_PKLot_csv(datasets_path[0])
    create_PKLot_universities()
    create_PKLot_autoencoder()

    #Criando os batches:
    datasets = universities + cameras
    print(datasets)
    create_batches(datasets)
    
    all_datas = print_csv(path_manager)

    format_datasets(datasets_dirs=all_datas)

    print("Datasets criados com sucesso!")

