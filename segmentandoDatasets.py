import pandas as pd
import os
import random
from typing import Tuple, Optional

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

# Exemplo de uso: 
# segmentando_datasets(1000, 1000, 1000)

def segmentando(quantidade, dias, faculdade):
    """
    Função para criar os datasets csv com uma divisão igual entre as classes 'Empty' e 'Occupied'.
    Retorna uma tupla com os datasets separados em ordem PUC, UFPR04, UFPR05.
    """
    tempos = ['Cloudy', 'Rainy', 'Sunny']
    dias = []

    caminho = r"/home/lucas/Downloads/PKLot/PKLotSegmented/"
    pastas = [nome for nome in os.listdir(caminho) if os.path.isdir(os.path.join(caminho, nome))]


    dataframes = [] 

    return tuple(dataframes)  # Retornar a tupla dos DataFrames


    
