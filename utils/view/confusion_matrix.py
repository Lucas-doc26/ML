import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image

def plot_confusion_matrix(y_true, y_pred, labels=['Empty', 'Occupied'], title=None, save_path=None):
    """
    Plota uma matriz de confusão.

    Args:
        y_true: Array numpy com os rótulos verdadeiros
        y_pred: Array numpy com as previsões do modelo
        labels: Lista de rótulos das classes
        title: Título da figura (opcional)
        save_path: Caminho para salvar a figura (opcional)
    """
    # Calcular matriz de confusão e acurácia
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Criar o gráfico
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    # Adicionar título com a acurácia
    accuracy_text = f"Accuracy: {accuracy * 100:.2f}%"
    plt.title('' if title is None else f"{title}\n{accuracy_text}")
    
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Salvar ou exibir a figura
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_confusion_matrix_with_images(y_true, y_pred, image_paths, labels=['Empty', 'Occupied'], title=None, save_path=None):
    """
    Plota uma matriz de confusão junto com as imagens correspondentes.

    Args:
        y_true: Array numpy com os rótulos verdadeiros
        y_pred: Array numpy com as previsões do modelo
        image_paths: Lista de caminhos das imagens
        labels: Lista de rótulos das classes
        title: Título da figura (opcional)
        save_path: Caminho para salvar a figura (opcional)
    """
    # Primeiro plota a matriz de confusão
    plot_confusion_matrix(y_true, y_pred, labels, title, save_path)

    # Depois plota as imagens
    plt.figure(figsize=(12, 8))
    for i, path in enumerate(image_paths):
        plt.subplot(3, 3, i + 1)
        try:
            img = Image.open(path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Imagem {i + 1}')
        except Exception as e:
            print(f"Erro ao carregar imagem {path}: {e}")
            plt.axis('off')
            plt.title(f'Erro ao carregar imagem {i + 1}')

    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path = save_path.with_name(f"{save_path.stem}_images{save_path.suffix}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()