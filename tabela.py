import numpy as np
import pandas as pd
import ezodf
from ezodf import Cell

ezodf.document.MIMETYPES['.ods'] = 'application/vnd.oasis.opendocument.spreadsheet'

def preencher_planilha_media_desvio(sheet, linha, col_inicio, valores):
    for i, valor in enumerate(valores):
        cell = Cell()
        try:
            cell.set_value(float(valor))
        except:
            cell.set_value(str(valor))
        
        # Garante que a linha e coluna existem
        while sheet.nrows() <= linha:
            sheet.append_rows(1)
        while sheet.ncols() <= col_inicio + i:
            sheet.append_columns(1)

        sheet[linha, col_inicio + i] = cell

def preencher_planilha_fusoes(sheet, linha, col_inicio, valores):
    for i in range(len(valores)):
        cell_valor = Cell()
        try:
            valor = float(valores[i])
        except ValueError:
            valor = str(valores[i])
        cell_valor.set_value(valor)

        # Garantir linhas e colunas
        while sheet.nrows() <= linha:
            sheet.append_rows(1)
        while sheet.ncols() <= col_inicio + (i * 2) + 1:
            sheet.append_columns(1)

        sheet[linha, col_inicio + (i * 2)] = cell_valor  # posição alternada

        cell_hifen = Cell()
        cell_hifen.set_value('-')
        sheet[linha, col_inicio + (i * 2) + 1] = cell_hifen

def extrair_valores(df, base_autoencoder, base_treino, base_teste, coluna):
    colunas_necessarias = ['Base do Autoencoder', 'Base de Treino', 'Base de Teste', coluna]
    for c in colunas_necessarias:
        if c not in df.columns:
            print(f"Coluna '{c}' não encontrada no DataFrame!")
            return []

    filtro = (
        (df['Base do Autoencoder'] == base_autoencoder) &
        (df['Base de Treino'] == base_treino) &
        (df['Base de Teste'] == base_teste)
    )
    return df.loc[filtro, coluna].values

def extrair_media_desvio(df, base_autoencoder, base_treino, base_teste):
    # Filtro pelas bases
    filtro = (
        (df['Base do Autoencoder'] == base_autoencoder) &
        (df['Base de Treino'] == base_treino) &
        (df['Base de Teste'] == base_teste)
    )

    if filtro.any():
        df_filtrado = df.loc[filtro, ['Média', 'Desvio Padrão']]
    else:
        print(f"Nenhum dado encontrado para: Base do Autoencoder={base_autoencoder}, Base de Treino={base_treino}, Base de Teste={base_teste}")
        df_filtrado = pd.DataFrame(columns=['Média', 'Desvio Padrão'])

    
    df_filtrado = df.loc[filtro, ['Média', 'Desvio Padrão']]

    valores = []

    for _, row in df_filtrado.iterrows():
        valores.append(row['Média'])
        valores.append(row['Desvio Padrão'])

    # Garante que tenha exatamente 10 valores (5 pares)
    while len(valores) < 10:
        valores.append('-')  # ou algum valor padrão, tipo None ou 'NA'
    
    return valores[:10]  # caso tenha mais, corta pra só os 10 necessários

def tabela_excel():
    path = 'resultados/Modelo_Kyoto/Resultados Experimentos finais - 29_05.ods'
    doc = ezodf.opendoc(path)
    print(f"Quantidade de planilhas: {len(doc.sheets)}")

    if len(doc.sheets) <= 2:
        print("Erro: O arquivo não tem a planilha índice 2.")
        return


    #Tabela Kyoto 
    sheet = doc.sheets[0]

    df1 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-Kyoto.csv')
    df2 = pd.read_csv('resultados/Modelo_Kyoto/tabela_SumFusion-Kyoto.csv')
    df3 = pd.read_csv('resultados/Modelo_Kyoto/tabela_MultFusion-Kyoto.csv')
    df4 = pd.read_csv('resultados/Modelo_Kyoto/tabela_VoteFusion-Kyoto.csv') 

    Kyoto = [
        # Para PUC
        ('PUC', 'PUC', 8), ('UFPR04', 'PUC', 9), ('UFPR05', 'PUC', 10),
        ('camera1', 'PUC', 11), ('camera2', 'PUC', 12), ('camera3', 'PUC', 13),
        ('camera4', 'PUC', 14), ('camera5', 'PUC', 15), ('camera6', 'PUC', 16),
        ('camera7', 'PUC', 17), ('camera8', 'PUC', 18), ('camera9', 'PUC', 19),

        # Para UFPR04
        ('PUC', 'UFPR04', 67), ('UFPR04', 'UFPR04', 68), ('UFPR05', 'UFPR04', 69),
        ('camera1', 'UFPR04', 70), ('camera2', 'UFPR04', 71), ('camera3', 'UFPR04', 72),
        ('camera4', 'UFPR04', 73), ('camera5', 'UFPR04', 74), ('camera6', 'UFPR04', 75),
        ('camera7', 'UFPR04', 76), ('camera8', 'UFPR04', 77), ('camera9', 'UFPR04', 78),

        # Para UFPR05
        ('PUC', 'UFPR05', 126), ('UFPR04', 'UFPR05', 127), ('UFPR05', 'UFPR05', 128),
        ('camera1', 'UFPR05', 129), ('camera2', 'UFPR05', 130), ('camera3', 'UFPR05', 131),
        ('camera4', 'UFPR05', 132), ('camera5', 'UFPR05', 133), ('camera6', 'UFPR05', 134),
        ('camera7', 'UFPR05', 135), ('camera8', 'UFPR05', 136), ('camera9', 'UFPR05', 137),

        # Para camera1
        ('PUC', 'camera1', 185), ('UFPR04', 'camera1', 186), ('UFPR05', 'camera1', 187),
        ('camera1', 'camera1', 188), ('camera2', 'camera1', 189), ('camera3', 'camera1', 190),
        ('camera4', 'camera1', 191), ('camera5', 'camera1', 192), ('camera6', 'camera1', 193),
        ('camera7', 'camera1', 194), ('camera8', 'camera1', 195), ('camera9', 'camera1', 196),

        # Para camera2
        ('PUC', 'camera2', 244), ('UFPR04', 'camera2', 245), ('UFPR05', 'camera2', 246),
        ('camera1', 'camera2', 247), ('camera2', 'camera2', 12), ('camera3', 'camera2', 13),
        ('camera4', 'camera2', 14), ('camera5', 'camera2', 15), ('camera6', 'camera2', 16),
        ('camera7', 'camera2', 17), ('camera8', 'camera2', 18), ('camera9', 'camera2', 19),

        # Para camera3
        ('PUC', 'camera3', 303), ('UFPR04', 'camera3', 9), ('UFPR05', 'camera3', 10),
        ('camera1', 'camera3', 11), ('camera2', 'camera3', 12), ('camera3', 'camera3', 13),
        ('camera4', 'camera3', 14), ('camera5', 'camera3', 15), ('camera6', 'camera3', 16),
        ('camera7', 'camera3', 17), ('camera8', 'camera3', 18), ('camera9', 'camera3', 19),

        # Para camera4
        ('PUC', 'camera4', 362), ('UFPR04', 'camera4', 9), ('UFPR05', 'camera4', 10),
        ('camera1', 'camera4', 11), ('camera2', 'camera4', 12), ('camera3', 'camera4', 13),
        ('camera4', 'camera4', 14), ('camera5', 'camera4', 15), ('camera6', 'camera4', 16),
        ('camera7', 'camera4', 17), ('camera8', 'camera4', 18), ('camera9', 'camera4', 19),

        # Para camera6 (tem duas entradas, conferi)
        ('PUC', 'camera6', 421), ('UFPR04', 'camera6', 9), ('UFPR05', 'camera6', 10),
        ('camera1', 'camera6', 11), ('camera2', 'camera6', 12), ('camera3', 'camera6', 13),
        ('camera4', 'camera6', 14), ('camera5', 'camera6', 15), ('camera6', 'camera6', 16),
        ('camera7', 'camera6', 17), ('camera8', 'camera6', 18), ('camera9', 'camera6', 19),

        ('PUC', 'camera6', 480), ('UFPR04', 'camera6', 9), ('UFPR05', 'camera6', 10),
        ('camera1', 'camera6', 11), ('camera2', 'camera6', 12), ('camera3', 'camera6', 13),
        ('camera4', 'camera6', 14), ('camera5', 'camera6', 15), ('camera6', 'camera6', 16),
        ('camera7', 'camera6', 17), ('camera8', 'camera6', 18), ('camera9', 'camera6', 19),

        # Para camera7
        ('PUC', 'camera7', 539), ('UFPR04', 'camera7', 9), ('UFPR05', 'camera7', 10),
        ('camera1', 'camera7', 11), ('camera2', 'camera7', 12), ('camera3', 'camera7', 13),
        ('camera4', 'camera7', 14), ('camera5', 'camera7', 15), ('camera6', 'camera7', 16),
        ('camera7', 'camera7', 17), ('camera8', 'camera7', 18), ('camera9', 'camera7', 19),

        # Para camera8
        ('PUC', 'camera8', 598), ('UFPR04', 'camera8', 9), ('UFPR05', 'camera8', 10),
        ('camera1', 'camera8', 11), ('camera2', 'camera8', 12), ('camera3', 'camera8', 13),
        ('camera4', 'camera8', 14), ('camera5', 'camera8', 15), ('camera6', 'camera8', 16),
        ('camera7', 'camera8', 17), ('camera8', 'camera8', 18), ('camera9', 'camera8', 19),

        # Para camera9
        ('PUC', 'camera9', 657), ('UFPR04', 'camera9', 9), ('UFPR05', 'camera9', 10),
        ('camera1', 'camera9', 11), ('camera2', 'camera9', 12), ('camera3', 'camera9', 13),
        ('camera4', 'camera9', 14), ('camera5', 'camera9', 15), ('camera6', 'camera9', 16),
        ('camera7', 'camera9', 17), ('camera8', 'camera9', 18), ('camera9', 'camera9', 19),
    
    ]


    Kyoto_sum = [
        ('PUC', 'PUC', 12), ('UFPR04', 'PUC', 13), ('UFPR05', 'PUC', 14),
        ('PUC', 'UFPR04', 35), ('UFPR04', 'UFPR04', 36), ('UFPR05', 'UFPR04', 37),
        ('PUC', 'UFPR05', 58), ('UFPR04', 'UFPR05', 59), ('UFPR05', 'UFPR05', 60),
    ]

    Kyoto_mult = [
        ('PUC', 'PUC', 16), ('UFPR04', 'PUC', 17), ('UFPR05', 'PUC', 18),
        ('PUC', 'UFPR04', 39), ('UFPR04', 'UFPR04', 40), ('UFPR05', 'UFPR04', 41),
        ('PUC', 'UFPR05', 62), ('UFPR04', 'UFPR05', 63), ('UFPR05', 'UFPR05', 64),
    ]

    Kyoto_voto = [
        ('PUC', 'PUC', 20), ('UFPR04', 'PUC', 21), ('UFPR05', 'PUC', 22),
        ('PUC', 'UFPR04', 59), ('UFPR04', 'UFPR04', 60), ('UFPR05', 'UFPR04', 61),
        ('PUC', 'UFPR05', 66), ('UFPR04', 'UFPR05', 67), ('UFPR05', 'UFPR05', 68),
    ]

    for treino, teste, linha in Kyoto:
        valores = extrair_media_desvio(df1, 'Kyoto', treino, teste)
        preencher_planilha_media_desvio(sheet, linha, 4, valores)

    for treino, teste, linha in Kyoto_sum:
        valores = extrair_valores(df2, 'Kyoto', treino, teste, 'Acuracia')  # retorna array
        preencher_planilha_fusoes(sheet, linha, 4, valores)

    for treino, teste, linha in Kyoto_mult:
        valores = extrair_valores(df3, 'Kyoto', treino, teste, 'Acuracia')  # retorna array
        preencher_planilha_fusoes(sheet, linha, 4, valores)
    
    for treino, teste, linha in Kyoto_voto:
        valores = extrair_valores(df4, 'Kyoto', treino, teste, 'Acuracia')  # retorna array
        preencher_planilha_fusoes(sheet, linha, 4, valores)

    #Tabela CNR
    sheet = doc.sheets[2]

    # Carrega os CSVs
    df1 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-CNR.csv')
    df2 = pd.read_csv('resultados/Modelo_Kyoto/tabela_SumFusion-CNR.csv')
    df3 = pd.read_csv('resultados/Modelo_Kyoto/tabela_MultFusion-CNR.csv')
    df4 = pd.read_csv('resultados/Modelo_Kyoto/tabela_VoteFusion-CNR.csv')

    cnr = [
        ('PUC', 'PUC', 8), ('UFPR04', 'PUC', 9), ('UFPR05', 'PUC', 10),
        ('PUC', 'UFPR04', 31), ('UFPR04', 'UFPR04', 32), ('UFPR05', 'UFPR04', 33),
        ('PUC', 'UFPR05', 54), ('UFPR04', 'UFPR05', 55), ('UFPR05', 'UFPR05', 56),
    ]

    cnr_sum = [
        ('PUC', 'PUC', 12), ('UFPR04', 'PUC', 13), ('UFPR05', 'PUC', 14),
        ('PUC', 'UFPR04', 35), ('UFPR04', 'UFPR04', 36), ('UFPR05', 'UFPR04', 37),
        ('PUC', 'UFPR05', 58), ('UFPR04', 'UFPR05', 59), ('UFPR05', 'UFPR05', 60),
    ]

    cnr_mult = [
        ('PUC', 'PUC', 16), ('UFPR04', 'PUC', 17), ('UFPR05', 'PUC', 18),
        ('PUC', 'UFPR04', 39), ('UFPR04', 'UFPR04', 40), ('UFPR05', 'UFPR04', 41),
        ('PUC', 'UFPR05', 62), ('UFPR04', 'UFPR05', 63), ('UFPR05', 'UFPR05', 64),
    ]

    cnr_voto = [
        ('PUC', 'PUC', 20), ('UFPR04', 'PUC', 21), ('UFPR05', 'PUC', 22),
        ('PUC', 'UFPR04', 59), ('UFPR04', 'UFPR04', 60), ('UFPR05', 'UFPR04', 61),
        ('PUC', 'UFPR05', 66), ('UFPR04', 'UFPR05', 67), ('UFPR05', 'UFPR05', 68),
    ]

    for treino, teste, linha in cnr:
        valores = extrair_media_desvio(df1, 'CNR', treino, teste)
        preencher_planilha_media_desvio(sheet, linha, 4, valores)

    for treino, teste, linha in cnr_sum:
        valores = extrair_valores(df2, 'CNR', treino, teste, 'Acuracia')  # retorna array
        preencher_planilha_fusoes(sheet, linha, 4, valores)

    for treino, teste, linha in cnr_mult:
        valores = extrair_valores(df3, 'CNR', treino, teste, 'Acuracia')  # retorna array
        preencher_planilha_fusoes(sheet, linha, 4, valores)
    
    for treino, teste, linha in cnr_voto:
        valores = extrair_valores(df4, 'CNR', treino, teste, 'Acuracia')  # retorna array
        preencher_planilha_fusoes(sheet, linha, 4, valores)

    # Salvar o arquivo (sobrescreve o original)
    doc.save()
    print("Arquivo salvo com sucesso!")

if __name__ == '__main__':
    tabela_excel()
