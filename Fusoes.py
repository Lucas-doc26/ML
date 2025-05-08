import numpy as np
from visualizacao import *
from Preprocessamento import *
from sklearn.metrics import accuracy_score

path = '/home/lucas/PIBIC/'

base_treino = ['PUC', 'UFPR04', 'UFPR05']
bases_teste = ['PUC', 'UFPR04', 'UFPR05']

def verifica_dir(nome_modelo, nome_base, nome_autoencoder):
    if not os.path.isdir(os.path.join(path, f'Modelos/Fusoes-{nome_modelo}')):
        os.mkdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}'))
    if not os.path.isdir(os.path.join(path, f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}')):
        os.mkdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}'))
    if not os.path.isdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}/Treinados_em_{nome_base}')):
        os.mkdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}/Treinados_em_{nome_base}'))
    if not os.path.isdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}/Treinados_em_{nome_base}/Matriz_confusao')):
        os.mkdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}/Treinados_em_{nome_base}/Matriz_confusao'))
    if not os.path.isdir(os.path.join(path,f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}/Treinados_em_{nome_base}/Grafico_batchs')):
        os.mkdir(os.path.join(path, f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}/Treinados_em_{nome_base}/Grafico_batchs'))

def mapear(classes):
    return np.array([1 if classe == 1 else 0 for classe in classes])

def soma_previsoes(nome_modelo, batch_size, n_modelos, base_de_treino, base_de_teste, nome_autoencoder=None):
    """
    Retorna um array com todos os resultados de todas as classes conforme o batch passado
    """
    if nome_autoencoder != None:
        classificador = f'Classificador-{nome_autoencoder}'
    else:
        classificador = 'Classificador'

    if base_de_teste == base_de_treino:
        dir_base = f'Modelos/{nome_modelo}-0/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batches-{batch_size}.npy' 
    else:
        dir_base = f'Modelos/{nome_modelo}-0/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batchs-{batch_size}.npy' 

    base = np.load(os.path.join(path, 
        dir_base)
    )

    resultado = np.zeros_like(base)
    for i in range(n_modelos):
        if base_de_teste == base_de_treino:
            npy = f'Modelos/{nome_modelo}-{i}/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batches-{batch_size}.npy' 
        else:
            npy = f'Modelos/{nome_modelo}-{i}/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batchs-{batch_size}.npy' 
        caminho = os.path.join(path,npy)
        array = np.load(caminho)
        resultado = resultado + array
        #print(resultado)
    resultado = np.argmax(resultado, axis=1)
    return resultado 

def soma(nome_modelo, bases_de_treino, nome_autoencoder=None, n_modelos=10):
    batches = [64,128,256,512,1024]
    resultados_csv = []
    for base_treino in bases_de_treino:
        for base_teste in bases_teste:
            verifica_dir(nome_modelo, base_treino, nome_autoencoder)
            resultados = []
            for batch_size in batches:  
                resultado = soma_previsoes(nome_modelo, batch_size, n_modelos, base_treino, base_teste, nome_autoencoder)
                if base_teste != base_treino and not 'camera' in base_teste:
                    df = pd.read_csv(f'CSV/{base_teste}/{base_teste}.csv')
                else:
                    if 'camera' in base_teste:
                        df = pd.read_csv(f'CSV/CNR/CNR_{base_teste}.csv')
                    else:
                        df = pd.read_csv(f'CSV/{base_teste}/{base_teste}_Segmentado_Teste.csv')
                df = mapear(df['classe'])

                #plot_confusion_matrix(df, resultado, title=f'Fusão {nome_modelo} - Batch: {batch_size}',
                #save_path=os.path.join(path, f'Modelos/Fusoes-{nome_modelo}/Treinados_em_{base_treino}/Matriz_confusao'))
                acc = accuracy_score(df, resultado)
                resultados.append(acc)

                resultados_csv.append({
                    'Base do Autoencoder':nome_autoencoder,
                    'Base de Treino':base_treino,
                    'Base de Teste':base_teste,
                    'Acuracia':format(acc, '.3f'),
                    'Batch':int(batch_size)
                })

            grafico_batchs(batches, resultados, nome_modelo=f'Soma entre os diferentes {nome_modelo}',
                caminho_para_salvar=os.path.join(path, f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}/Treinados_em_{base_treino}/Grafico_batchs'), 
                nome_autoencoder=nome_autoencoder,
                nome_base_treino=base_treino, base_usada_teste=base_teste)
            
    df_resultados = pd.DataFrame(resultados_csv)
    caminho_csv = f'resultados/{nome_modelo}/tabela_resultado-Sum-{nome_autoencoder}.csv'
    df_resultados.to_csv(caminho_csv, index=False)

def mult_previsoes(nome_modelo, batch_size, n_modelos, base_de_treino, base_de_teste, nome_autoencoder=None):
    """
    Retorna um array com todos os resultados de todas as classes conforme o batch passado
    """
    if nome_autoencoder != None:
        classificador = f'Classificador-{nome_autoencoder}'
    else:
        classificador = 'Classificador'

    if base_de_teste == base_de_treino:
        dir_base = f'Modelos/{nome_modelo}-0/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batches-{batch_size}.npy' 
    else:
        dir_base = f'Modelos/{nome_modelo}-0/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batchs-{batch_size}.npy' 

    base = np.load(os.path.join(path, 
        dir_base)
    )

    resultado = np.ones_like(base)
    for i in range(n_modelos):
        if base_de_teste == base_de_treino:
            npy = f'Modelos/{nome_modelo}-{i}/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batches-{batch_size}.npy' 
        else:
            npy = f'Modelos/{nome_modelo}-{i}/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batchs-{batch_size}.npy' 
        caminho = os.path.join(path,npy)
        array = np.load(caminho)
        resultado = resultado * array
        #print(resultado)
    resultado = np.argmax(resultado, axis=1)
    return resultado 

def mult(nome_modelo, bases_de_treino, nome_autoencoder=None, n_modelos=10):
    batches = [64,128,256,512,1024]
    resultados_csv = []
    for base_treino in bases_de_treino:
        for base_teste in bases_teste:
            verifica_dir(nome_modelo, base_treino, nome_autoencoder)
            resultados = []
            for batch_size in batches:  
                resultado = mult_previsoes(nome_modelo, batch_size, n_modelos, base_treino, base_teste, nome_autoencoder)
                if base_teste != base_treino and not 'camera' in base_teste:
                    df = pd.read_csv(f'CSV/{base_teste}/{base_teste}.csv')
                else:
                    if 'camera' in base_teste:
                        df = pd.read_csv(f'CSV/CNR/CNR_{base_teste}.csv')
                    else:
                        df = pd.read_csv(f'CSV/{base_teste}/{base_teste}_Segmentado_Teste.csv')
                df = mapear(df['classe'])
                
                acc = accuracy_score(df, resultado)
                resultados.append(acc)
                
                resultados_csv.append({
                    'Base do Autoencoder':nome_autoencoder,
                    'Base de Treino':base_treino,
                    'Base de Teste':base_teste,
                    'Acuracia':format(acc, '.3f'),
                    'Batch':int(batch_size)
                })

            grafico_batchs(batches, resultados, nome_modelo=f'Mult entre os diferentes {nome_modelo}',
                caminho_para_salvar=os.path.join(path, f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}/Treinados_em_{base_treino}/Grafico_batchs'), 
                nome_autoencoder=nome_autoencoder,
                nome_base_treino=base_treino, base_usada_teste=base_teste)
    
    df_resultados = pd.DataFrame(resultados_csv)
    caminho_csv = f'resultados/{nome_modelo}/tabela_resultado-Mult-{nome_autoencoder}.csv'
    df_resultados.to_csv(caminho_csv, index=False)
      
def votacao_previsoes(nome_modelo, batch_size, n_modelos, base_de_treino, base_de_teste, nome_autoencoder=None):
    """
    Retorna um array com todos os resultados de todas as classes conforme o batch passado
    """
    if nome_autoencoder != None:
        classificador = f'Classificador-{nome_autoencoder}'
    else:
        classificador = 'Classificador'

    if base_de_teste == base_de_treino:
        dir_base = f'Modelos/{nome_modelo}-0/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batches-{batch_size}.npy' 
    else:
        dir_base = f'Modelos/{nome_modelo}-0/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batchs-{batch_size}.npy' 

    base = np.load(os.path.join(path, 
        dir_base)
    )

    resultado = np.zeros_like(base)
    for i in range(n_modelos):
        if base_de_teste == base_de_treino:
            npy = f'Modelos/{nome_modelo}-{i}/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batches-{batch_size}.npy' 
        else:
            npy = f'Modelos/{nome_modelo}-{i}/{classificador}/Resultados/Treinados_em_{base_de_treino}/{base_de_teste}/batchs-{batch_size}.npy' 
        caminho = os.path.join(path,npy)
        array = np.load(caminho)
        for j in range(len(array)):
            valor = array[j][0]
            if valor > 0.5:
                array[j] = [1 , 0]
            elif valor == 0.5:
                array[j] = [1 , 1]
            else:
                array[j] = [0 , 1]

        resultado = resultado + array
        ##print(resultado)
    resultado = np.argmax(resultado, axis=1)
    return resultado 

def voto(nome_modelo, bases_de_treino, nome_autoencoder=None, n_modelos=10):
    batches = [64,128,256,512,1024]
    resultados_csv = []

    for base_treino in bases_de_treino:
        for base_teste in bases_teste:
            verifica_dir(nome_modelo, base_treino, nome_autoencoder)
            resultados = []
            for batch_size in batches:  
                resultado = votacao_previsoes(nome_modelo, batch_size, n_modelos, base_treino, base_teste, nome_autoencoder)
                if base_teste != base_treino and not 'camera' in base_teste:
                    df = pd.read_csv(f'CSV/{base_teste}/{base_teste}.csv')
                else:
                    if 'camera' in base_teste:
                        df = pd.read_csv(f'CSV/CNR/CNR_{base_teste}.csv')
                    else:
                        df = pd.read_csv(f'CSV/{base_teste}/{base_teste}_Segmentado_Teste.csv')
                df = mapear(df['classe'])

                #plot_confusion_matrix(df, resultado, title=f'Fusão {nome_modelo} - Batch: {batch_size}',
                 #save_path=os.path.join(path, f'Modelos/Fusoes-{nome_modelo}/Treinados_em_{base_treino}/Matriz_confusao'))
                acc = accuracy_score(df, resultado)
                resultados.append(acc)

                resultados_csv.append({
                    'Base do Autoencoder':nome_autoencoder,
                    'Base de Treino':base_treino,
                    'Base de Teste':base_teste,
                    'Acuracia': f"{acc:.3f}",
                    'Batch':int(batch_size)
                })

            #print(base_treino)
            grafico_batchs(batches, resultados, nome_modelo=f'Voto entre os diferentes {nome_modelo}',
                caminho_para_salvar=os.path.join(path, f'Modelos/Fusoes-{nome_modelo}/Autoencoder-{nome_autoencoder}/Treinados_em_{base_treino}/Grafico_batchs'), 
                nome_autoencoder=nome_autoencoder,
                nome_base_treino=base_treino, base_usada_teste=base_teste)
    
    df_resultados = pd.DataFrame(resultados_csv)
    caminho_csv = f'resultados/{nome_modelo}/tabela_resultado-Voto-{nome_autoencoder}.csv'
    df_resultados.to_csv(caminho_csv, index=False)

soma("Modelo_Kyoto", base_treino, 'Kyoto', n_modelos=10)
mult("Modelo_Kyoto", base_treino, 'Kyoto', n_modelos=10)
voto("Modelo_Kyoto", base_treino, 'Kyoto', n_modelos=10)

soma("Modelo_Kyoto", base_treino, 'CNR', 5)
mult("Modelo_Kyoto", base_treino, 'CNR', 5)
voto("Modelo_Kyoto", base_treino, 'CNR', 5)


