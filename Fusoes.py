import numpy as np
from visualizacao import *
from Preprocessamento import *
from sklearn.metrics import accuracy_score
from Modelos import recria_diretorio
import sys

path = '/media/hd/mnt/data/Lucas$'

def verifica_dir(nome_modelo, nome_base):
    if not os.path.isdir(os.path.join(path, f'Modelos/Fusoes-{nome_modelo}')):
        os.mkdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}'))
    if not os.path.isdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/{nome_base}')):
        os.mkdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/{nome_base}'))
    if not os.path.isdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/{nome_base}/Matriz_confusao')):
        os.mkdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/{nome_base}/Matriz_confusao'))
    if not os.path.isdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/{nome_base}/Grafico_batchs')):
        os.mkdir(os.path.join(path, f'Modelos/Fusoes-{nome_modelo}/{nome_base}/Grafico_batchs'))

def mapear(classes):
    return np.array([1 if classe == 1 else 0 for classe in classes])


bases_teste = ['PUC', 'UFPR04', 'UFPR05', 'camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']


def previsao(nome_modelo, base_de_treino, base_de_teste, batch_size, n_modelos):
    for i in range(n_modelos):
        npy = f'Modelos/{nome_modelo}-{i}/Classificador/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batchs-{batch_size}.npy' 
        array = np.load(os.path.join(path,npy))

def soma_previsoes(nome_modelo, batch_size, n_modelos, base_de_treino, base_de_teste):
    """
    Retorna um array com todos os resultados de todas as classes conforme o batch passado
    """
    base = np.load(os.path.join(path, 
        f'Modelos/{nome_modelo}-0/Classificador/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batchs-{batch_size}')
    )

    resultado = np.zeros_like(base)
    for i in range(n_modelos):
        npy = f'Modelos/{nome_modelo}-{i}/Classificador/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batchs-{batch_size}.npy' 
        array = np.load(os.path.join(path,npy))
        resultado = resultado + array
    resultado = np.argmax(resultado, axis=1)
    return resultado 


def soma(nome_modelo, bases_de_treino, n_modelos):
    batches = [64,128,256,512,1024]
    for base_treino in bases_de_treino:
        for base_teste in bases_teste:
            for batch_size in batches:  
                resultados = []
                resultado = soma_previsoes(nome_modelo, batch_size, n_modelos, base_treino, base_teste)

                df = pd.read_csv(f'CSV/{base_teste}/{base_teste}.csv')
                df = mapear(df['classe'])

                plot_confusion_matrix(df, resultado, title=f'Fusão {nome_modelo} - Batch: {batch_size}')
                acc = accuracy_score(df, resultado)
                resultados.append(acc)

        grafico_batchs(batchs, resultados, nome_modelo=f'Soma-{nome_modelo}-{base_treino}',
            caminho_para_salvar=os.path.join(path, f'Modelos/Fusao-{nome_modelo}/Treinados_em_{base_treino}/'), 
            nome_base_treino={base_treino}, base_usada_teste={base_teste})
            


def regra_soma(nome_modelo, nome_base, caminho_csv, n_modelos):
    df = pd.read_csv(caminho_csv)
    df = mapear(df['classe'])
    batchs = []
    resultados = []
    verifica_dir(nome_modelo, nome_base)

    for i in range(16):
        resultado = soma_previsoes(f'{nome_modelo}', f'{nome_base}', i+1, n_modelos)
        plot_confusion_matrix(df, resultado, title=f'Fusão dos diferentes {nome_modelo} - Batchs: {i+1}', 
            save_path= os.path.join(path,f'Modelos/Fusao-{nome_modelo}/{nome_base}/Matriz_confusao/Soma_{nome_base}_batchs-{i+1}'))
        batchs.append(i+1)
        acuracia = accuracy_score(df, resultado)
        resultados.append(acuracia)
    grafico_batchs(batchs, resultados, nome_modelo=f'Soma-{nome_modelo}',caminho_para_salvar=os.path.join(path, f'Modelos/Fusao-{nome_modelo}/{nome_base}/Grafico_batchs/'))
    
    return (batchs, resultados, 'Soma')

def multiplicacao_previsoes(nome_modelo, nome_base, batch, n_modelos):
    base = np.load(os.path.join(path, f'Modelos/{nome_modelo}-0/Classificador/Resultados/{nome_modelo}-0-{nome_base}-batchs-{batch}.npy'))
    resultado = np.ones_like(base)
    for i in range(n_modelos):
        array = np.load(os.path.join(path, f'Modelos/{nome_modelo}-{i}/Classificador/Resultados/{nome_modelo}-{i}-{nome_base}-batchs-{batch}.npy'))
        resultado = resultado * array 
    resultado = np.argmax(resultado, axis=1)
    
    return resultado

def regra_multiplicacao(nome_modelo, nome_base, caminho_csv, n_modelos):
    df = pd.read_csv(caminho_csv)
    df = mapear(df['classe'])
    batchs = []
    resultados = []
    verifica_dir(nome_modelo, nome_base)
    for i in range(16):
        resultado = multiplicacao_previsoes(f'{nome_modelo}', f'{nome_base}', i+1, n_modelos)
        plot_confusion_matrix(df, resultado, title=f'Fusão dos diferentes {nome_modelo} - Batchs: {i+1}', 
            save_path= os.path.join(path, f'Modelos/Fusao-{nome_modelo}/{nome_base}/Matriz_confusao/Multiplicacao_{nome_base}_batchs-{i+1}'))
        batchs.append(i+1)
        acuracia = accuracy_score(df, resultado)
        resultados.append(acuracia)
    grafico_batchs(batchs, resultados, nome_modelo=f'Multiplicacao-{nome_modelo}', 
    caminho_para_salvar= os.path.join(path, f'Modelos/Fusao-{nome_modelo}/{nome_base}/Grafico_batchs/'))

    return (batchs, resultados, 'Multiplicação')

def votacao_previsoes(nome_modelo, nome_base, batch, n_modelos):
    base = np.load(os.path.join(path, f'Modelos/{nome_modelo}-0/Classificador/Resultados/{nome_modelo}-0-{nome_base}-batchs-{batch}.npy'))
    resultado = np.zeros_like(base)
    for i in range(n_modelos):
        array = np.load(os.path.join(path, f'Modelos/{nome_modelo}-{i}/Classificador/Resultados/{nome_modelo}-{i}-{nome_base}-batchs-{batch}.npy'))
        for j in range(len(array)):
            valor = array[j][0]
            if valor > 0.5:
                array[j] = [1 , 0]
            else:
                array[j] = [0 , 1]

        resultado = resultado + array
    
    resultado = np.argmax(resultado, axis=1)

    return resultado
    
def regra_votacao(nome_modelo, nome_base, caminho_csv, n_modelos):
    df = pd.read_csv(caminho_csv)
    df = mapear(df['classe'])
    batchs = []
    resultados = []
    verifica_dir(nome_modelo, nome_base)
    for i in range(16):
        resultado = votacao_previsoes(f'{nome_modelo}', f'{nome_base}', i+1, n_modelos)
        plot_confusion_matrix(df, resultado, title=f'Fusão dos diferentes {nome_modelo} - Batchs: {i+1}', 
            save_path= os.path.join(path, f'Modelos/Fusao-{nome_modelo}/{nome_base}/Matriz_confusao/Votacao_{nome_base}_batchs-{i+1}'))
        batchs.append(i+1)
        acuracia = accuracy_score(df, resultado)
        resultados.append(acuracia)
    grafico_batchs(batchs, resultados, nome_modelo=f'Votacao-{nome_modelo}', 
    caminho_para_salvar= os.path.join(path, f'Modelos/Fusao-{nome_modelo}/{nome_base}/Grafico_batchs/'))

    return (batchs, resultados, 'Votação')

def max_previsoes(nome_modelo, nome_base, batch, n_modelos):
    caminho_base = os.path.join(path, f'Modelos/{nome_modelo}-0/Classificador/Resultados/{nome_modelo}-0-{nome_base}-batchs-{batch}.npy')
    base = np.load(caminho_base)
    
    resultado = base.copy()

    for i in range(n_modelos):
        caminho = os.path.join(path, f'Modelos/{nome_modelo}-{i}/Classificador/Resultados/{nome_modelo}-{i}-{nome_base}-batchs-{batch}.npy')
        array = np.load(caminho)
        resultado = np.maximum(resultado, array) 
    
    return np.argmax(resultado, axis=1) 

def maximo(nome_modelo, nome_base, caminho_csv, n_modelos):
    df = pd.read_csv(caminho_csv)
    df = mapear(df['classe'])  
    
    batchs = []
    resultados = []
    
    verifica_dir(nome_modelo, nome_base)  
    
    for i in range(16):
        resultado = max_previsoes(nome_modelo, nome_base, i+1, n_modelos)
        
        plot_confusion_matrix(df, resultado, title=f'Fusão dos diferentes {nome_modelo} - Batchs: {i+1}', 
            save_path=os.path.join(path, f'Modelos/Fusao-{nome_modelo}/{nome_base}/Matriz_confusao/Maximo_{nome_base}_batchs-{i+1}'))
        
        batchs.append(i+1)
        acuracia = accuracy_score(df, resultado)
        resultados.append(acuracia)

    grafico_batchs(batchs, resultados, nome_modelo=f'Maximo_-{nome_modelo}', 
        caminho_para_salvar=os.path.join(path, f'Modelos/Fusao-{nome_modelo}/{nome_base}/Grafico_batchs/'))
    
    return (batchs, resultados, 'Máximo')
