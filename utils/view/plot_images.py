import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from ..preprocessing import normalize
from ..evaluation_metrics import *

def plot_images_from_csv(path_csv: Path, num_images_per_column: int = 3, figsize: Tuple[int, int] = (10, 10), title: str = None, save_path: Path = None ):
    """
    Plota as imagens de um arquivo .csv em uma grade de subplots.

    :param path_csv: caminho para o arquivo .csv contendo os caminhos de imagens
    :param num_images_per_column: número de imagens por coluna
    :param figsize: tamanho da figura (largura, altura) -> (opcional)
    :param title: título da figura (opcional)
    :param save_path: caminho para salvar a figura (opcional)
    """

    # Carregar e embaralhar o DataFrame
    df = pd.read_csv(path_csv)
    df = df.sample(frac=1).reset_index(drop=True)  # embaralhar linhas

    # Criar a grade de subplots
    fig, axs = plt.subplots(
        num_images_per_column,
        num_images_per_column,
        figsize=figsize
    )

    axs = axs.flatten()  # Facilita indexação linear dos subplots

    # Iterar sobre as imagens e plotar
    for idx, (i, row) in enumerate(df.iterrows()):
        if idx >= len(axs):
            break

        img_path = row['path_image']
        try:
            label = row['class']
        except:
            label = row['path_image'].split('/')[-1].split('.')[0]

        try:
            image = plt.imread(img_path)
            axs[idx].imshow(image)
            axs[idx].axis('off')
            axs[idx].set_title(str(label), fontsize=8)
        except Exception as e:
            axs[idx].axis('off')
            axs[idx].set_title("Erro", fontsize=8)
            print(f"Erro ao carregar imagem: {img_path} -> {e}")

    # Remover subplots extras (caso não preencham todos)
    for j in range(idx + 1, len(axs)):
        axs[j].axis('off')

    # Título geral e layout
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # deixa espaço para o suptitle

    # Salvar ou exibir
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
    plt.clf()

def plot_images_from_dataframe_generator(ImageDataGenerator: ImageDataGenerator, num_images_per_column: int = 3, figsize: Tuple[int, int] = (10, 10), title: str = None, save_path: Path = None):  
    """
    Plota as imagens de um DataFrame em uma grade de subplots.

    :param ImageDataGenerator: ImageDataGenerator contendo os caminhos de imagens e rótulos
    :param num_images_per_column: número de imagens por coluna
    :param figsize: tamanho da figura (largura, altura) -> (opcional)
    :param title: título da figura (opcional)
    :param save_path: caminho para salvar a figura (opcional)
    """

    classes = ['Empty', 'Occupied']

    # Define a figura com subplots
    fig, axs = plt.subplots(num_images_per_column, num_images_per_column, figsize=figsize)
    images_per_plot = num_images_per_column ** 2 
    
    # Gera as imagens e labels a partir do gerador
    images, labels = next(ImageDataGenerator)

    for i in range(images_per_plot):
        if i >= len(images):
            break
        
        image = images[i]

        # Normaliza a imagem para o intervalo [0, 1]
        image = (image - image.min()) / (image.max() - image.min())

        axs[i // num_images_per_column, i % num_images_per_column].imshow(image)
        axs[i // num_images_per_column, i % num_images_per_column].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

    """
    Função para plotar imagens de um arquivo .csv contendo os caminhos de imagens.
    
    :param caminho_csv: caminho para o arquivo .csv contendo os caminhos de imagens
    :param img_por_coluna: número de imagens por coluna
    """
    # Ler o CSV para um DataFrame
    dataframe = pd.read_csv(caminho_csv)
    
    _, axs = plt.subplots(img_por_coluna, img_por_coluna, figsize=(10, 10))

    # Iterar sobre as linhas do DataFrame
    for index, linha in dataframe.iterrows():
        if index >= img_por_coluna**2:
            break  
        image_path = linha['caminho_imagem']
        image = plt.imread(image_path)
        linha_idx = index // img_por_coluna
        coluna_idx = index % img_por_coluna
        axs[linha_idx, coluna_idx].imshow(image)
        axs[linha_idx, coluna_idx].axis('off')  # Desligar os eixos
        axs[linha_idx, coluna_idx].set_title(f"{linha['classe']}")  # Título da imagem

    plt.tight_layout()
    plt.show()
    plt.close()    

#arrumar
def plot_incorrect_images(y_true, y_pred, caminhos_imagens, modelo_nome:str, dataset_nome:str, n_imagens_por_grade:3):
    """
    Plota imagens cujas previsões foram incorretas.

    Parâmetros:
    - y_binario: Array numpy com os rótulos reais.
    - y_predicao: Array numpy com as previsões do modelo.
    - caminhos_imagens: Lista de caminhos para as imagens correspondentes.
    - modelo_nome: Nome do modelo para fins de salvamento de arquivo.
    - dataset_nome: Nome do dataset para fins de salvamento de arquivo.
    - n_imagens_por_grade: Número de imagens por grade na visualização (default: 3).

    A função identifica as imagens onde as previsões do modelo diferem dos rótulos reais,
    e plota estas imagens junto com suas previsões e rótulos reais. As imagens são organizadas
    em uma grade de subplots para visualização fácil. As imagens incorretamente previstas são
    salvas em um arquivo PNG com base nos nomes do modelo e do dataset fornecidos.
    """

    labels = ['Empty', 'Occupied']

    indices_incorretos = np.where(y_pred != y_true)[0]

    num_imagens_plotadas = min(len(indices_incorretos), n_imagens_por_grade**2)
    indices_plotados = indices_incorretos[:num_imagens_plotadas]

    fig, axes = plt.subplots(n_imagens_por_grade, n_imagens_por_grade, figsize=(15, 15))
    axes = axes.flatten()

    plt.suptitle(f'{modelo_nome} vs {dataset_nome} - Imagens incorretas', fontsize=16)
    for ax, indice in zip(axes, indices_plotados):
        img = load_img(caminhos_imagens[indice])
        ax.imshow(img)
        ax.set_title(f'Predição: {labels[y_predicao[indice]]}\nReal: {labels[y_binario[indice]]}')
        ax.axis('off')

    # Remover qualquer eixo vazio
    for i in range(num_imagens_plotadas, n_imagens_por_grade**2):
        axes[i].axis('off')

    plt.tight_layout()
    save_path_imgs = os.path.join('Resultados', 'Imagens_Incorretas', modelo_nome, f'{modelo_nome}_vs_{dataset_nome}_imagens_incorretas.png')
    os.makedirs(os.path.dirname(save_path_imgs), exist_ok=True)
    plt.savefig(save_path_imgs)
    plt.close()
