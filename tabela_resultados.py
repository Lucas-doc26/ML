import numpy as np
import pandas as pd
import os
from statistics import median
from math import isnan
from itertools import filterfalse
import csv

path = '/media/lucas/mnt/data/Lucas$/Modelos/Plots'

def desvio_padrao(valores):
        media = np.mean(valores)
        desvio = np.std(valores, ddof=0)  
        print(f"Média: {media:.4f}")
        print(f"Desvio Padrão: {desvio:.4f}")
        return media, desvio

def retorna_resultados_csv(nome, nome_modelo):
    tabelas = [t for t in os.listdir(path) if 'Tabela' in t and f'-{nome}-' in t] 
    tabelas_formatadas = [t for t in tabelas if f'{nome_modelo}' in t and 'Grafico' not in t]
    tabelas_final = [os.path.join(path, t) for t in tabelas_formatadas]
    for t in tabelas_final:
        print(t)

    batches = [64, 128, 256, 512, 1024]
    dados = []

    for tabela in tabelas_final:
        try:
            df = pd.read_csv(tabela, encoding='ISO-8859-1')
        except:
            print("Erro:", tabela)
        
        for batch in batches:
            coluna = f'Batch {batch}'
            
            if coluna in df.columns:
                valores = df[coluna].astype(float)
                media, desvio = desvio_padrao(valores)
                
                # pegando as infos pelo caminho do arquivo
                treino = (tabela.split('/')[8]).split('-')[3] 
                teste = ((tabela.split('/')[8]).split('-')[4]).split('.')[0]
                
                print(f'Base de treino {treino}, base de teste {teste}, media dos valores {media}, desvio padrao {desvio}, no batch {batch}')
                
                # add os dados na lista
                txt = "{m:.3f} ~ {d:.3f}"
                dados.append([treino, teste, txt.format(m=media, d=desvio), batch])
            else:
                print(f"Aviso: Coluna {coluna} não encontrada em {tabela}")

    print(dados)
    save_dir = f'resultados/{nome_modelo}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    arquivo_csv = os.path.join(save_dir, f'tabela_resultado-{nome}.csv')

    ordem_teste = ['PUC', 'UFPR04', 'UFPR05'] + [f'camera{i}' for i in range(1, 10)]
    dados_ordenados = sorted(dados, key=lambda x: ordem_teste.index(x[1]) if x[1] in ordem_teste else float('inf'))

    with open(arquivo_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Cabeçalho do CSV
        writer.writerow(['Base de Treino', 'Base de Teste', 'Média', 'Batch'])
        
        for linha in dados_ordenados:
            writer.writerow(linha)

    print(f'Arquivo {arquivo_csv} criado com sucesso!')



retorna_resultados_csv('PUC', 'Modelo_Kyoto')
retorna_resultados_csv('UFPR04', 'Modelo_Kyoto')
retorna_resultados_csv('UFPR05', 'Modelo_Kyoto')
