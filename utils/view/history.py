import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_history_autoencoder(history_dataframe:pd.DataFrame, save_dir:Path, model_name:str, autoencoder_base:str):
    """
    Plota o histórico de treinamento do autoencoder.

    Args:
        df: DataFrame com o histórico de treinamento.
        save_dir: Diretório para salvar a imagem.   
        model_name: Nome do modelo.
        autoencoder_base: Base do autoencoder.
    """

    plt.figure(figsize=(8, 6))
    df.plot()
    plt.title("Treinamento do Autoencoder")
    plt.xlabel("Épocas")
    plt.ylabel("Métricas")
    plt.grid()

    nome_img = f"History-Autoencoder-{model_name}-{autoencoder_base}.png"


    if save_dir != None:
        plt.savefig(os.path.join(save_dir, nome_img)) 
    else:
        plt.show()

    plt.clf()
    plt.close('all')  

def plot_history_batch(history_dataframe:pd.DataFrame, save_dir:Path, model_name:str, train_base:str, autoencoder_base:str, batch:int):
    """
    Plota o histórico de treinamento do classificador baseado em um batch.

    Args:
        history_dataframe: Histórico de treinamento.
        save_dir: Diretório para salvar a imagem.
        model_name: Nome do modelo.
        train_base: Base de treinamento.
        autoencoder_base: Base do autoencoder.
        batch: Batch. 
    """
    accuracy = history_dataframe.history['accuracy']
    val_accuracy = history_dataframe.history['val_accuracy']
    loss = history_dataframe.history['loss']
    val_loss = history_dataframe.history['val_loss']

    # Criando o gráfico
    plt.figure(figsize=(12, 6))

    # Subplot para Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label='Acurácia (Treinamento)')
    plt.plot(val_accuracy, label='Acurácia (Validação)')
    plt.title('Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    # Subplot para Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Loss (Treinamento)')
    plt.plot(val_loss, label='Loss (Validação)')
    plt.title('Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    # Exibir o gráfico
    plt.tight_layout()
    if save_dir != None:
        plt.savefig(os.path.join(save_dir, f"History-{model_name}-{autoencoder_base}-{train_base}-{batch}.png")) 
    else:
        plt.show()

    plt.clf()
    plt.close('all')  

