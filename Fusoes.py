import numpy as np
from visualizacao import *
from Preprocessamento import *
from sklearn.metrics import accuracy_score
from Modelos import criar_diretorio_novo


def verifica_dir(nome_modelo, nome_base):
    if not os.path.isdir(f'Modelos/Fusao-{nome_modelo}'):
        os.mkdir(f'Modelos/Fusao-{nome_modelo}')
    if not os.path.isdir(f'Modelos/Fusao-{nome_modelo}/{nome_base}'):
        os.mkdir(f'Modelos/Fusao-{nome_modelo}/{nome_base}')
    if not os.path.isdir(f'Modelos/Fusao-{nome_modelo}/{nome_base}/Matriz_confusao'):
        os.mkdir(f'Modelos/Fusao-{nome_modelo}/{nome_base}/Matriz_confusao')
    if not os.path.isdir(f'Modelos/Fusao-{nome_modelo}/{nome_base}/Grafico_batchs'):
        os.mkdir(f'Modelos/Fusao-{nome_modelo}/{nome_base}/Grafico_batchs')

def mapear(classes):
    return np.array([1 if classe == 1 else 0 for classe in classes])

def soma_previsoes(nome_modelo, nome_base, batch):
    base = np.load(f'Modelos/{nome_modelo}-0/Classificador/Resultados/{nome_modelo}-0-{nome_base}-batchs-{batch}.npy')
    resultado = np.zeros_like(base)
    for i in range(10):
        array = np.load(f'Modelos/{nome_modelo}-{i}/Classificador/Resultados/{nome_modelo}-{i}-{nome_base}-batchs-{batch}.npy')
        resultado = resultado + array 
    resultado = np.argmax(resultado, axis=1)
    return resultado 

def regra_soma(nome_modelo, nome_base, caminho_csv):
    df = pd.read_csv(caminho_csv)
    df = mapear(df['classe'])
    batchs = []
    resultados = []
    verifica_dir(nome_modelo, nome_base)
    for i in range(16):
        resultado = soma_previsoes(f'{nome_modelo}', f'{nome_base}', i+1)
        plot_confusion_matrix(df, resultado, title=f'Fusão dos diferentes {nome_modelo} - Batchs: {i+1}', 
            save_path=f'Modelos/Fusao-{nome_modelo}/{nome_base}/Matriz_confusao/Soma_{nome_base}_batchs-{i+1}')
        batchs.append(i+1)
        acuracia = accuracy_score(df, resultado)
        resultados.append(acuracia)
    grafico_batchs(batchs, resultados, nome_modelo=f'Soma-{nome_modelo}',caminho_para_salvar=f'Modelos/Fusao-{nome_modelo}/{nome_base}/Grafico_batchs/')
    
    return (batchs, resultados, 'Soma')

def multiplicacao_previsoes(nome_modelo, nome_base, batch):
    base = np.load(f'Modelos/{nome_modelo}-0/Classificador/Resultados/{nome_modelo}-0-{nome_base}-batchs-{batch}.npy')
    resultado = np.ones_like(base)
    for i in range(10):
        array = np.load(f'Modelos/{nome_modelo}-{i}/Classificador/Resultados/{nome_modelo}-{i}-{nome_base}-batchs-{batch}.npy')
        resultado = resultado * array 
    resultado = np.argmax(resultado, axis=1)
    
    return resultado

def regra_multiplicacao(nome_modelo, nome_base, caminho_csv):
    df = pd.read_csv(caminho_csv)
    df = mapear(df['classe'])
    batchs = []
    resultados = []
    verifica_dir(nome_modelo, nome_base)
    for i in range(16):
        resultado = multiplicacao_previsoes(f'{nome_modelo}', f'{nome_base}', i+1)
        plot_confusion_matrix(df, resultado, title=f'Fusão dos diferentes {nome_modelo} - Batchs: {i+1}', 
            save_path=f'Modelos/Fusao-{nome_modelo}/{nome_base}/Matriz_confusao/Multiplicacao_{nome_base}_batchs-{i+1}')
        batchs.append(i+1)
        acuracia = accuracy_score(df, resultado)
        resultados.append(acuracia)
    grafico_batchs(batchs, resultados, nome_modelo=f'Multiplicacao-{nome_modelo}', 
    caminho_para_salvar=f'Modelos/Fusao-{nome_modelo}/{nome_base}/Grafico_batchs/')

    return (batchs, resultados, 'Multiplicação')

def votacao_previsoes(nome_modelo, nome_base, batch):
    base = np.load(f'Modelos/{nome_modelo}-0/Classificador/Resultados/{nome_modelo}-0-{nome_base}-batchs-{batch}.npy')
    resultado = np.zeros_like(base)
    for i in range(10):
        array = np.load(f'Modelos/{nome_modelo}-{i}/Classificador/Resultados/{nome_modelo}-{i}-{nome_base}-batchs-{batch}.npy')
        for j in range(len(array)):
            valor = array[j][0]
            if valor > 0.5:
                array[j] = [1 , 0]
            else:
                array[j] = [0 , 1]

        resultado = resultado + array
    
    resultado = np.argmax(resultado, axis=1)

    return resultado
    
def regra_votacao(nome_modelo, nome_base, caminho_csv):
    df = pd.read_csv(caminho_csv)
    df = mapear(df['classe'])
    batchs = []
    resultados = []
    verifica_dir(nome_modelo, nome_base)
    for i in range(16):
        resultado = votacao_previsoes(f'{nome_modelo}', f'{nome_base}', i+1)
        plot_confusion_matrix(df, resultado, title=f'Fusão dos diferentes {nome_modelo} - Batchs: {i+1}', 
            save_path=f'Modelos/Fusao-{nome_modelo}/{nome_base}/Matriz_confusao/Votacao_{nome_base}_batchs-{i+1}')
        batchs.append(i+1)
        acuracia = accuracy_score(df, resultado)
        resultados.append(acuracia)
    grafico_batchs(batchs, resultados, nome_modelo=f'Votacao-{nome_modelo}', 
    caminho_para_salvar=f'Modelos/Fusao-{nome_modelo}/{nome_base}/Grafico_batchs/')

    return (batchs, resultados, 'Votação')