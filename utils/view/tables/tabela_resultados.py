import numpy as np
import pandas as pd
import os
import csv
import ezodf
from ezodf import Cell

def preencher_planilha(sheet, linha, col_inicio, valores):
    for i in range(min(5, len(valores))):
        cell = Cell()
        try:
            valor = float(valores[i])
        except ValueError:
            valor = str(valores[i])  # fallback para string
        cell.set_value(valor)
        sheet[linha, col_inicio + i] = cell

def extrair_valores(df, base_autoencoder, base_treino, base_teste, coluna):
    filtro = (
        (df['Base do Autoencoder'] == base_autoencoder) &
        (df['Base de Treino'] == base_treino) &
        (df['Base de Teste'] == base_teste)
    )
    return df.loc[filtro, coluna].values

def tabela_excel():
    doc = ezodf.opendoc("resultados/Modelo_Kyoto/resultados.ods")
    sheet = doc.sheets[0]

    df1 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-Kyoto.csv')
    df2 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-Sum-Kyoto.csv')
    df3 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-Voto-Kyoto.csv')
    df4 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-Mult-Kyoto.csv')

    df5 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-CNR.csv')
    df6 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-Sum-CNR.csv')
    df7 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-Voto-CNR.csv')
    df8 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-Mult-CNR.csv')

    combinacoes_kyoto = [
        ('PUC', 'PUC', 5), ('UFPR04', 'PUC', 6), ('UFPR05', 'PUC', 7),
        ('PUC', 'UFPR04', 34), ('UFPR04', 'UFPR04', 35), ('UFPR05', 'UFPR04', 36),
        ('PUC', 'UFPR05', 63), ('UFPR04', 'UFPR05', 64), ('UFPR05', 'UFPR05', 65),
    ]

    sum_kyoto = [
        ('PUC', 'PUC', 8), ('UFPR04', 'PUC', 9), ('UFPR05', 'PUC', 10),
        ('PUC', 'UFPR04', 37), ('UFPR04', 'UFPR04', 38), ('UFPR05', 'UFPR04', 39),
        ('PUC', 'UFPR05', 66), ('UFPR04', 'UFPR05', 67), ('UFPR05', 'UFPR05', 68),
    ]

    voto_kyoto = [
        ('PUC', 'PUC', 11), ('UFPR04', 'PUC', 12), ('UFPR05', 'PUC', 13),
        ('PUC', 'UFPR04', 40), ('UFPR04', 'UFPR04', 41), ('UFPR05', 'UFPR04', 42),
        ('PUC', 'UFPR05', 69), ('UFPR04', 'UFPR05', 70), ('UFPR05', 'UFPR05', 71),
    ]

    mul_kyoto = [
        ('PUC', 'PUC', 14), ('UFPR04', 'PUC', 15), ('UFPR05', 'PUC', 16),
        ('PUC', 'UFPR04', 43), ('UFPR04', 'UFPR04', 44), ('UFPR05', 'UFPR04', 45),
        ('PUC', 'UFPR05', 72), ('UFPR04', 'UFPR05', 73), ('UFPR05', 'UFPR05', 74),
    ]

    combinacoes_cnr = [
        ('PUC', 'PUC', 17), ('UFPR04', 'PUC', 18), ('UFPR05', 'PUC', 19),
        ('PUC', 'UFPR04', 46), ('UFPR04', 'UFPR04', 47), ('UFPR05', 'UFPR04', 48),
        ('PUC', 'UFPR05', 75), ('UFPR04', 'UFPR05', 76), ('UFPR05', 'UFPR05', 77),
    ]

    sum_cnr = [
        ('PUC', 'PUC', 20), ('UFPR04', 'PUC', 21), ('UFPR05', 'PUC', 22),
        ('PUC', 'UFPR04', 49), ('UFPR04', 'UFPR04', 50), ('UFPR05', 'UFPR04', 51),
        ('PUC', 'UFPR05', 78), ('UFPR04', 'UFPR05', 79), ('UFPR05', 'UFPR05', 80),
    ]

    voto_cnr = [
        ('PUC', 'PUC', 23), ('UFPR04', 'PUC', 24), ('UFPR05', 'PUC', 25),
        ('PUC', 'UFPR04', 52), ('UFPR04', 'UFPR04', 53), ('UFPR05', 'UFPR04', 54),
        ('PUC', 'UFPR05', 81), ('UFPR04', 'UFPR05', 82), ('UFPR05', 'UFPR05', 83),
    ]

    mul_cnr = [
        ('PUC', 'PUC', 26), ('UFPR04', 'PUC', 27), ('UFPR05', 'PUC', 28),
        ('PUC', 'UFPR04', 55), ('UFPR04', 'UFPR04', 56), ('UFPR05', 'UFPR04', 57),
        ('PUC', 'UFPR05', 84), ('UFPR04', 'UFPR05', 85), ('UFPR05', 'UFPR05', 86),
    ]



    for treino, teste, linha in combinacoes_kyoto:
        valores = extrair_valores(df1, 'Kyoto', treino, teste, 'Média')
        preencher_planilha(sheet, linha, 4, valores)

    for treino, teste, linha in sum_kyoto:
        valores = extrair_valores(df2, 'Kyoto', treino, teste, 'Acuracia')
        preencher_planilha(sheet, linha, 4, valores)
    
    for treino, teste, linha in voto_kyoto:
        valores = extrair_valores(df3, 'Kyoto', treino, teste, 'Acuracia')
        preencher_planilha(sheet, linha, 4, valores)
    
    for treino, teste, linha in mul_kyoto:
        valores = extrair_valores(df4, 'Kyoto', treino, teste, 'Acuracia')
        preencher_planilha(sheet, linha, 4, valores)

    for treino, teste, linha in combinacoes_cnr:
        valores = extrair_valores(df5, 'CNR', treino, teste, 'Média')
        preencher_planilha(sheet, linha, 4, valores)

    for treino, teste, linha in sum_cnr:
        valores = extrair_valores(df6, 'CNR', treino, teste, 'Acuracia')
        preencher_planilha(sheet, linha, 4, valores)
    
    for treino, teste, linha in voto_cnr:
        valores = extrair_valores(df7, 'CNR', treino, teste, 'Acuracia')
        preencher_planilha(sheet, linha, 4, valores)
    
    for treino, teste, linha in mul_cnr:
        valores = extrair_valores(df8, 'CNR', treino, teste, 'Acuracia')
        preencher_planilha(sheet, linha, 4, valores)

    doc.save()

tabela_excel()

path = 'Modelos/Plots'

def desvio_padrao(valores):
        media = np.mean(valores)
        desvio = np.std(valores, ddof=0)  
        return media, desvio

def retorna_resultados_csv(autoencoder, classificador, modelo):
    #Modelos/Plots/Tabela-Comparacao-Modelo_Kyoto-CNR-PUC-PUC.csv
    tabelas = [t for t in os.listdir(path) if f'Tabela-Comparacao-{modelo}' in t and f'-{classificador}-' in t] 
    tabelas_formatadas = [t for t in tabelas if f'{autoencoder}' in t and 'Grafico' not in t]
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
    save_dir = f'resultados/{modelo}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    arquivo_csv = os.path.join(save_dir, f'tabela_resultado-{classificador}.csv')

    ordem_teste = ['PUC', 'UFPR04', 'UFPR05'] + [f'camera{i}' for i in range(1, 10)]
    dados_ordenados = sorted(dados, key=lambda x: ordem_teste.index(x[1]) if x[1] in ordem_teste else float('inf'))

    with open(arquivo_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Cabeçalho do CSV
        writer.writerow(['Base de Treino', 'Base de Teste', 'Média', 'Batch'])
        
        for linha in dados_ordenados:
            writer.writerow(linha)

    print(f'Arquivo {arquivo_csv} criado com sucesso!')

def resultados(modelo, autoencoder, classificador=None):
    tabelas = [t for t in os.listdir(path) if f'Tabela' in t and 'Grafico' not in t] 
    tabelas_modelos = [t for t in tabelas if f'-{modelo}-' in t]
    tabelas_filtradas = [t for t in tabelas_modelos if f'-{autoencoder}' in t]

    batches = [64, 128, 256, 512, 1024]
    dados = []

    for tabela in tabelas_filtradas:
        try:
            df = pd.read_csv(os.path.join(path, tabela), encoding='ISO-8859-1')
        except:
            print("Erro:", tabela)
        
        for batch in batches:
            coluna = f'Batch {batch}'
            
            if coluna in df.columns:
                valores = df[coluna].astype(float)
                media, desvio = desvio_padrao(valores)
                
                # pegando as infos pelo caminho do arquivo
                #Modelos/Plots/Tabela-Comparacao-Modelo_Kyoto-CNR-PUC-PUC.csv
                treino = tabela.split('-')[4] 
                teste = (tabela.split('-')[5]).split('.')[0]
                
                print(f'Base de treino {treino}, base de teste {teste}, media dos valores {media}, desvio padrao {desvio}, no batch {batch}')
                
                # add os dados na lista
                txt = "{m:.3f} ~ {d:.3f}"
                dados.append([autoencoder, treino, teste, txt.format(m=media, d=desvio), batch])
            else:
                print(f"Aviso: Coluna {coluna} não encontrada em {tabela}")

    save_dir = f'resultados/{modelo}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    arquivo_csv = os.path.join(save_dir, f'tabela_resultado-{autoencoder}.csv')

    ordem_teste = ['PUC', 'UFPR04', 'UFPR05']
    dados_ordenados = sorted(dados, key=lambda x: ordem_teste.index(x[1]) if x[1] in ordem_teste else float('inf'))

    with open(arquivo_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Cabeçalho do CSV
        writer.writerow(['Base do Autoencoder','Base de Treino', 'Base de Teste', 'Média', 'Batch'])
        
        for linha in dados_ordenados:
            writer.writerow(linha)

    print(f'Arquivo {arquivo_csv} criado com sucesso!')

#resultados('Modelo_Kyoto', 'CNR')
#resultados('Modelo_Kyoto', 'Kyoto')
