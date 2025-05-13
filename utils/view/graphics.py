import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from pathlib import Path
from typing import Optional, List, Union
import pandas as pd

def graphic_accuracy_per_batch( batches: List[int], accuracies: List[float], model_name: str, train_base: str, test_base: str, autoencoder_base: str, save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plota um gráfico de acurácia por cada um dos batches.

    Args:
        batches: Lista de números de lotes.
        accuracies: Lista de acurácias.
        model_name: Nome do modelo.
        train_base: Base de treinamento.
        test_base: Base de teste.
        autoencoder_base: Base do autoencoder.
        save_path: Caminho para salvar o gráfico.
    """
    plt.figure() 
    plt.title(f"Comparação de acurácia - {model_name}")
    plt.xlabel('Número de imagens')
    plt.ylabel('Acurácia')

    #Cria o label com as informações do modelo
    label = (
        f"Nome = {model_name}\n"
        f"Base do Autoencoder: {autoencoder_base}\n"
        f"Base do Classificador: {train_base}\n"
        f"Base sendo testada: {test_base}"
    )

    plt.plot(batches, accuracies, marker='o', linestyle='-', color='b', label=label)
    plt.xticks(batches)  
    for xi, yi in zip(batches, accuracies):
            plt.text(xi, yi, f"{yi:.3f}", fontsize=6, ha='left', va='top') 

    plt.legend(loc='lower right', fontsize=9, title="Informações do modelo", title_fontsize=10)

    try:
        model_name = model_name.split(' ')[0]
    except:
        pass
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path = save_path / f'Grafico-{model_name}-{autoencoder_base}-{train_base}-{test_base}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Salvando gráfico no caminho: {save_path}")

    plt.show()
    plt.close('all')
    plt.clf()

def models_comparison( path_save: Optional[Path]=None, model_name:str = None, classifier_base: str = None,test_base: str = None,autoencoder_base: str = None) -> None:
    """
    Plota um gráfico de comparação entre os modelos.

    Args:
        path_save: Caminho para salvar o gráfico.
        model_name: Nome do modelo.
        classifier_base: Base de treinamento do classificador.
        test_base: Base de teste.
        autoencoder_base: Base do autoencoder.
    """
    
    if test_base is None:
        test_base = classifier_base

    data = []
    table = pd.DataFrame(columns=['Nome Modelo', 'Batch'])

    dir_base = Path("Modelos")
    models = [model for model in os.listdir(dir_base) if (f'{model_name}' in model and "Fusoes" not in model)]
    print(models)
    x = [64,128,256,512,1024]
    plt.figure(figsize=(10, 6))
    plt.xticks(x)  

    for model in sorted(models):
        dir_results = dir_base / model / f'Classificador-{autoencoder_base}/Precisao/Treinado_em_{classifier_base}'

        if test_base != classifier_base:
            precisions = [r for r in os.listdir(dir_results) if f'{test_base}' in r]
        else:
            precisions = [r for r in os.listdir(dir_results) if f'{classifier_base}' in r]

        print(precisions)
        dir_precisions = dir_results / precisions[0]

        with open(dir_precisions, 'r') as f:
            precisions_read = f.readlines()

        precisions_read = [item.strip() for item in precisions_read]
        precisions_list = [round(float(item), 4) for item in precisions_read]

        plt.plot(x, precisions_list, label=f"{model}", marker='o')

        for xi, yi in zip(x, precisions_list):
            plt.text(xi, yi, f"{yi:.3f}", fontsize=6, ha='left', va='top') 

        data.append([model] + precisions_list)
            
    plt.title(f'Comparação entre os(as) diferentes {model_name} - Treinado na base: {classifier_base}')
    plt.xlabel('Número de imagens')
    plt.ylabel('Acurácia')

    plt.legend()

    columns = ['Modelo'] + [f'Batch {batch}' for batch in x]
    df = pd.DataFrame(data, columns=columns)

    if path_save is not None:
        path_save = Path(path_save)
        save_path = path_save / f'Grafico-Comparacao-{model_name}-{autoencoder_base}-{classifier_base}-{test_base}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

        csv_path = path_save / f'Tabela-Comparacao-{model_name}-{autoencoder_base}-{classifier_base}-{test_base}.csv'
        df.to_csv(csv_path, index=False)

    plt.show()
    plt.close()



