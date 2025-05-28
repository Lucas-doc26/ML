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
        ('camera1', 'camera2', 247), ('camera2', 'camera2', 248), ('camera3', 'camera2', 249),
        ('camera4', 'camera2', 250), ('camera5', 'camera2', 251), ('camera6', 'camera2', 252),
        ('camera7', 'camera2', 253), ('camera8', 'camera2', 254), ('camera9', 'camera2', 255),

        # Para camera3
        ('PUC', 'camera3', 303), ('UFPR04', 'camera3', 304), ('UFPR05', 'camera3', 305),
        ('camera1', 'camera3', 306), ('camera2', 'camera3', 307), ('camera3', 'camera3', 308),
        ('camera4', 'camera3', 309), ('camera5', 'camera3', 310), ('camera6', 'camera3', 311),
        ('camera7', 'camera3', 312), ('camera8', 'camera3', 313), ('camera9', 'camera3', 314),

        # Para camera4
        ('PUC', 'camera4', 362), ('UFPR04', 'camera4', 363), ('UFPR05', 'camera4', 364),
        ('camera1', 'camera4', 365), ('camera2', 'camera4', 366), ('camera3', 'camera4', 367),
        ('camera4', 'camera4', 368), ('camera5', 'camera4', 369), ('camera6', 'camera4', 370),
        ('camera7', 'camera4', 371), ('camera8', 'camera4', 372), ('camera9', 'camera4', 373),

        # Para camera5
        ('PUC', 'camera5', 421), ('UFPR04', 'camera5', 422), ('UFPR05', 'camera5', 423),
        ('camera1', 'camera5', 424), ('camera2', 'camera5', 425), ('camera3', 'camera5', 426),
        ('camera4', 'camera5', 427), ('camera5', 'camera5', 428), ('camera6', 'camera5', 429),
        ('camera7', 'camera5', 430), ('camera8', 'camera5', 431), ('camera9', 'camera5', 432),

        # Para camera6
        ('PUC', 'camera6', 480), ('UFPR04', 'camera6', 481), ('UFPR05', 'camera6', 482),
        ('camera1', 'camera6', 483), ('camera2', 'camera6', 484), ('camera3', 'camera6', 485),
        ('camera4', 'camera6', 486), ('camera5', 'camera6', 487), ('camera6', 'camera6', 488),
        ('camera7', 'camera6', 489), ('camera8', 'camera6', 490), ('camera9', 'camera6', 491),

        # Para camera7
        ('PUC', 'camera7', 539), ('UFPR04', 'camera7', 540), ('UFPR05', 'camera7', 541),
        ('camera1', 'camera7', 542), ('camera2', 'camera7', 543), ('camera3', 'camera7', 544),
        ('camera4', 'camera7', 545), ('camera5', 'camera7', 546), ('camera6', 'camera7', 547),
        ('camera7', 'camera7', 548), ('camera8', 'camera7', 549), ('camera9', 'camera7', 550),

        # Para camera8
        ('PUC', 'camera8', 598), ('UFPR04', 'camera8', 599), ('UFPR05', 'camera8', 600),
        ('camera1', 'camera8', 601), ('camera2', 'camera8', 602), ('camera3', 'camera8', 603),
        ('camera4', 'camera8', 604), ('camera5', 'camera8', 605), ('camera6', 'camera8', 606),
        ('camera7', 'camera8', 607), ('camera8', 'camera8', 608), ('camera9', 'camera8', 609),

        # Para camera9
        ('PUC', 'camera9', 657), ('UFPR04', 'camera9', 658), ('UFPR05', 'camera9', 659),
        ('camera1', 'camera9', 660), ('camera2', 'camera9', 661), ('camera3', 'camera9', 662),
        ('camera4', 'camera9', 663), ('camera5', 'camera9', 664), ('camera6', 'camera9', 665),
        ('camera7', 'camera9', 666), ('camera8', 'camera9', 667), ('camera9', 'camera9', 668)
    ]

    Kyoto_sum = [
        # Para PUC
        ('PUC', 'PUC', 21), ('UFPR04', 'PUC', 22), ('UFPR05', 'PUC', 23),
        ('camera1', 'PUC', 24), ('camera2', 'PUC', 25), ('camera3', 'PUC', 26),
        ('camera4', 'PUC', 27), ('camera5', 'PUC', 28), ('camera6', 'PUC', 29),
        ('camera7', 'PUC', 30), ('camera8', 'PUC', 31), ('camera9', 'PUC', 32),

        # Para UFPR04
        ('PUC', 'UFPR04', 80), ('UFPR04', 'UFPR04', 81), ('UFPR05', 'UFPR04', 82),
        ('camera1', 'UFPR04', 83), ('camera2', 'UFPR04', 84), ('camera3', 'UFPR04', 85),
        ('camera4', 'UFPR04', 86), ('camera5', 'UFPR04', 87), ('camera6', 'UFPR04', 88),
        ('camera7', 'UFPR04', 89), ('camera8', 'UFPR04', 90), ('camera9', 'UFPR04', 91),

        # Para UFPR05
        ('PUC', 'UFPR05', 139), ('UFPR04', 'UFPR05', 140), ('UFPR05', 'UFPR05', 141),
        ('camera1', 'UFPR05', 142), ('camera2', 'UFPR05', 143), ('camera3', 'UFPR05', 144),
        ('camera4', 'UFPR05', 145), ('camera5', 'UFPR05', 146), ('camera6', 'UFPR05', 147),
        ('camera7', 'UFPR05', 148), ('camera8', 'UFPR05', 149), ('camera9', 'UFPR05', 150),

        # Para camera1
        ('PUC', 'camera1', 198), ('UFPR04', 'camera1', 199), ('UFPR05', 'camera1', 200),
        ('camera1', 'camera1', 201), ('camera2', 'camera1', 202), ('camera3', 'camera1', 203),
        ('camera4', 'camera1', 204), ('camera5', 'camera1', 205), ('camera6', 'camera1', 206),
        ('camera7', 'camera1', 207), ('camera8', 'camera1', 208), ('camera9', 'camera1', 209),

        # Para camera2
        ('PUC', 'camera2', 257), ('UFPR04', 'camera2', 258), ('UFPR05', 'camera2', 259),
        ('camera1', 'camera2', 260), ('camera2', 'camera2', 261), ('camera3', 'camera2', 262),
        ('camera4', 'camera2', 263), ('camera5', 'camera2', 264), ('camera6', 'camera2', 265),
        ('camera7', 'camera2', 266), ('camera8', 'camera2', 267), ('camera9', 'camera2', 268),

        # Para camera3
        ('PUC', 'camera3', 316), ('UFPR04', 'camera3', 317), ('UFPR05', 'camera3', 318),
        ('camera1', 'camera3', 319), ('camera2', 'camera3', 320), ('camera3', 'camera3', 321),
        ('camera4', 'camera3', 322), ('camera5', 'camera3', 323), ('camera6', 'camera3', 324),
        ('camera7', 'camera3', 325), ('camera8', 'camera3', 326), ('camera9', 'camera3', 327),

        # Para camera4
        ('PUC', 'camera4', 375), ('UFPR04', 'camera4', 376), ('UFPR05', 'camera4', 377),
        ('camera1', 'camera4', 378), ('camera2', 'camera4', 379), ('camera3', 'camera4', 380),
        ('camera4', 'camera4', 381), ('camera5', 'camera4', 382), ('camera6', 'camera4', 383),
        ('camera7', 'camera4', 384), ('camera8', 'camera4', 385), ('camera9', 'camera4', 386),

        # Para camera5
        ('PUC', 'camera5', 434), ('UFPR04', 'camera5', 435), ('UFPR05', 'camera5', 436),
        ('camera1', 'camera5', 437), ('camera2', 'camera5', 438), ('camera3', 'camera5', 439),
        ('camera4', 'camera5', 440), ('camera5', 'camera5', 441), ('camera6', 'camera5', 442),
        ('camera7', 'camera5', 443), ('camera8', 'camera5', 444), ('camera9', 'camera5', 445),

        # Para camera6
        ('PUC', 'camera6', 493), ('UFPR04', 'camera6', 494), ('UFPR05', 'camera6', 495),
        ('camera1', 'camera6', 496), ('camera2', 'camera6', 497), ('camera3', 'camera6', 498),
        ('camera4', 'camera6', 499), ('camera5', 'camera6', 500), ('camera6', 'camera6', 501),
        ('camera7', 'camera6', 502), ('camera8', 'camera6', 503), ('camera9', 'camera6', 504),

        # Para camera7
        ('PUC', 'camera7', 552), ('UFPR04', 'camera7', 553), ('UFPR05', 'camera7', 554),
        ('camera1', 'camera7', 555), ('camera2', 'camera7', 556), ('camera3', 'camera7', 557),
        ('camera4', 'camera7', 558), ('camera5', 'camera7', 559), ('camera6', 'camera7', 560),
        ('camera7', 'camera7', 561), ('camera8', 'camera7', 562), ('camera9', 'camera7', 563),

        # Para camera8
        ('PUC', 'camera8', 611), ('UFPR04', 'camera8', 612), ('UFPR05', 'camera8', 613),
        ('camera1', 'camera8', 614), ('camera2', 'camera8', 615), ('camera3', 'camera8', 616),
        ('camera4', 'camera8', 617), ('camera5', 'camera8', 618), ('camera6', 'camera8', 619),
        ('camera7', 'camera8', 620), ('camera8', 'camera8', 621), ('camera9', 'camera8', 622),

        # Para camera9
        ('PUC', 'camera9', 670), ('UFPR04', 'camera9', 671), ('UFPR05', 'camera9', 672),
        ('camera1', 'camera9', 673), ('camera2', 'camera9', 674), ('camera3', 'camera9', 675),
        ('camera4', 'camera9', 676), ('camera5', 'camera9', 677), ('camera6', 'camera9', 678),
        ('camera7', 'camera9', 679), ('camera8', 'camera9', 680), ('camera9', 'camera9', 681)
    ]

    Kyoto_mult = [
        # Para PUC
        ('PUC', 'PUC', 34), ('UFPR04', 'PUC', 35), ('UFPR05', 'PUC', 36),
        ('camera1', 'PUC', 37), ('camera2', 'PUC', 38), ('camera3', 'PUC', 39),
        ('camera4', 'PUC', 40), ('camera5', 'PUC', 41), ('camera6', 'PUC', 42),
        ('camera7', 'PUC', 43), ('camera8', 'PUC', 44), ('camera9', 'PUC', 45),

        # Para UFPR04
        ('PUC', 'UFPR04', 93), ('UFPR04', 'UFPR04', 94), ('UFPR05', 'UFPR04', 95),
        ('camera1', 'UFPR04', 96), ('camera2', 'UFPR04', 97), ('camera3', 'UFPR04', 98),
        ('camera4', 'UFPR04', 99), ('camera5', 'UFPR04', 100), ('camera6', 'UFPR04', 101),
        ('camera7', 'UFPR04', 102), ('camera8', 'UFPR04', 103), ('camera9', 'UFPR04', 104),

        # Para UFPR05
        ('PUC', 'UFPR05', 152), ('UFPR04', 'UFPR05', 153), ('UFPR05', 'UFPR05', 154),
        ('camera1', 'UFPR05', 155), ('camera2', 'UFPR05', 156), ('camera3', 'UFPR05', 157),
        ('camera4', 'UFPR05', 158), ('camera5', 'UFPR05', 159), ('camera6', 'UFPR05', 160),
        ('camera7', 'UFPR05', 161), ('camera8', 'UFPR05', 162), ('camera9', 'UFPR05', 163),

        # Para camera1
        ('PUC', 'camera1', 211), ('UFPR04', 'camera1', 212), ('UFPR05', 'camera1', 213),
        ('camera1', 'camera1', 214), ('camera2', 'camera1', 215), ('camera3', 'camera1', 216),
        ('camera4', 'camera1', 217), ('camera5', 'camera1', 218), ('camera6', 'camera1', 219),
        ('camera7', 'camera1', 220), ('camera8', 'camera1', 221), ('camera9', 'camera1', 222),

        # Para camera2
        ('PUC', 'camera2', 270), ('UFPR04', 'camera2', 271), ('UFPR05', 'camera2', 272),
        ('camera1', 'camera2', 273), ('camera2', 'camera2', 274), ('camera3', 'camera2', 275),
        ('camera4', 'camera2', 276), ('camera5', 'camera2', 277), ('camera6', 'camera2', 278),
        ('camera7', 'camera2', 279), ('camera8', 'camera2', 280), ('camera9', 'camera2', 281),

        # Para camera3
        ('PUC', 'camera3', 329), ('UFPR04', 'camera3', 330), ('UFPR05', 'camera3', 331),
        ('camera1', 'camera3', 332), ('camera2', 'camera3', 333), ('camera3', 'camera3', 334),
        ('camera4', 'camera3', 335), ('camera5', 'camera3', 336), ('camera6', 'camera3', 337),
        ('camera7', 'camera3', 338), ('camera8', 'camera3', 339), ('camera9', 'camera3', 340),

        # Para camera4
        ('PUC', 'camera4', 388), ('UFPR04', 'camera4', 389), ('UFPR05', 'camera4', 390),
        ('camera1', 'camera4', 391), ('camera2', 'camera4', 392), ('camera3', 'camera4', 393),
        ('camera4', 'camera4', 394), ('camera5', 'camera4', 395), ('camera6', 'camera4', 396),
        ('camera7', 'camera4', 397), ('camera8', 'camera4', 398), ('camera9', 'camera4', 399),

        # Para camera5
        ('PUC', 'camera5', 447), ('UFPR04', 'camera5', 448), ('UFPR05', 'camera5', 449),
        ('camera1', 'camera5', 450), ('camera2', 'camera5', 451), ('camera3', 'camera5', 452),
        ('camera4', 'camera5', 453), ('camera5', 'camera5', 454), ('camera6', 'camera5', 455),
        ('camera7', 'camera5', 456), ('camera8', 'camera5', 457), ('camera9', 'camera5', 458),

        # Para camera6
        ('PUC', 'camera6', 506), ('UFPR04', 'camera6', 507), ('UFPR05', 'camera6', 508),
        ('camera1', 'camera6', 509), ('camera2', 'camera6', 510), ('camera3', 'camera6', 511),
        ('camera4', 'camera6', 512), ('camera5', 'camera6', 513), ('camera6', 'camera6', 514),
        ('camera7', 'camera6', 515), ('camera8', 'camera6', 516), ('camera9', 'camera6', 517),

        # Para camera7
        ('PUC', 'camera7', 565), ('UFPR04', 'camera7', 566), ('UFPR05', 'camera7', 567),
        ('camera1', 'camera7', 568), ('camera2', 'camera7', 569), ('camera3', 'camera7', 570),
        ('camera4', 'camera7', 571), ('camera5', 'camera7', 572), ('camera6', 'camera7', 573),
        ('camera7', 'camera7', 574), ('camera8', 'camera7', 575), ('camera9', 'camera7', 576),

        # Para camera8
        ('PUC', 'camera8', 624), ('UFPR04', 'camera8', 625), ('UFPR05', 'camera8', 626),
        ('camera1', 'camera8', 627), ('camera2', 'camera8', 628), ('camera3', 'camera8', 629),
        ('camera4', 'camera8', 630), ('camera5', 'camera8', 631), ('camera6', 'camera8', 632),
        ('camera7', 'camera8', 633), ('camera8', 'camera8', 634), ('camera9', 'camera8', 635),

        # Para camera9
        ('PUC', 'camera9', 683), ('UFPR04', 'camera9', 684), ('UFPR05', 'camera9', 685),
        ('camera1', 'camera9', 686), ('camera2', 'camera9', 687), ('camera3', 'camera9', 688),
        ('camera4', 'camera9', 689), ('camera5', 'camera9', 690), ('camera6', 'camera9', 691),
        ('camera7', 'camera9', 692), ('camera8', 'camera9', 693), ('camera9', 'camera9', 694)
    ]

    Kyoto_voto = [
        # Para PUC
        ('PUC', 'PUC', 47), ('UFPR04', 'PUC', 48), ('UFPR05', 'PUC', 49),
        ('camera1', 'PUC', 50), ('camera2', 'PUC', 51), ('camera3', 'PUC', 52),
        ('camera4', 'PUC', 53), ('camera5', 'PUC', 54), ('camera6', 'PUC', 55),
        ('camera7', 'PUC', 56), ('camera8', 'PUC', 57), ('camera9', 'PUC', 58),

        # Para UFPR04
        ('PUC', 'UFPR04', 105), ('UFPR04', 'UFPR04', 106), ('UFPR05', 'UFPR04', 107),
        ('camera1', 'UFPR04', 108), ('camera2', 'UFPR04', 109), ('camera3', 'UFPR04', 110),
        ('camera4', 'UFPR04', 111), ('camera5', 'UFPR04', 112), ('camera6', 'UFPR04', 113),
        ('camera7', 'UFPR04', 114), ('camera8', 'UFPR04', 115), ('camera9', 'UFPR04', 116),

        # Para UFPR05
        ('PUC', 'UFPR05', 164), ('UFPR04', 'UFPR05', 165), ('UFPR05', 'UFPR05', 166),
        ('camera1', 'UFPR05', 167), ('camera2', 'UFPR05', 168), ('camera3', 'UFPR05', 169),
        ('camera4', 'UFPR05', 170), ('camera5', 'UFPR05', 171), ('camera6', 'UFPR05', 172),
        ('camera7', 'UFPR05', 173), ('camera8', 'UFPR05', 174), ('camera9', 'UFPR05', 175),

        # Para camera1
        ('PUC', 'camera1', 224), ('UFPR04', 'camera1', 225), ('UFPR05', 'camera1', 226),
        ('camera1', 'camera1', 227), ('camera2', 'camera1', 228), ('camera3', 'camera1', 229),
        ('camera4', 'camera1', 230), ('camera5', 'camera1', 231), ('camera6', 'camera1', 232),
        ('camera7', 'camera1', 233), ('camera8', 'camera1', 234), ('camera9', 'camera1', 235),

        # Para camera2
        ('PUC', 'camera2', 283), ('UFPR04', 'camera2', 284), ('UFPR05', 'camera2', 285),
        ('camera1', 'camera2', 286), ('camera2', 'camera2', 287), ('camera3', 'camera2', 288),
        ('camera4', 'camera2', 289), ('camera5', 'camera2', 290), ('camera6', 'camera2', 291),
        ('camera7', 'camera2', 292), ('camera8', 'camera2', 293), ('camera9', 'camera2', 294),

        # Para camera3
        ('PUC', 'camera3', 342), ('UFPR04', 'camera3', 343), ('UFPR05', 'camera3', 344),
        ('camera1', 'camera3', 345), ('camera2', 'camera3', 346), ('camera3', 'camera3', 347),
        ('camera4', 'camera3', 348), ('camera5', 'camera3', 349), ('camera6', 'camera3', 350),
        ('camera7', 'camera3', 351), ('camera8', 'camera3', 352), ('camera9', 'camera3', 353),

        # Para camera4
        ('PUC', 'camera4', 401), ('UFPR04', 'camera4', 402), ('UFPR05', 'camera4', 403),
        ('camera1', 'camera4', 404), ('camera2', 'camera4', 405), ('camera3', 'camera4', 406),
        ('camera4', 'camera4', 407), ('camera5', 'camera4', 408), ('camera6', 'camera4', 409),
        ('camera7', 'camera4', 410), ('camera8', 'camera4', 411), ('camera9', 'camera4', 412),

        # Para camera5
        ('PUC', 'camera5', 460), ('UFPR04', 'camera5', 461), ('UFPR05', 'camera5', 462),
        ('camera1', 'camera5', 463), ('camera2', 'camera5', 464), ('camera3', 'camera5', 465),
        ('camera4', 'camera5', 466), ('camera5', 'camera5', 467), ('camera6', 'camera5', 468),
        ('camera7', 'camera5', 469), ('camera8', 'camera5', 470), ('camera9', 'camera5', 471),

        # Para camera6
        ('PUC', 'camera6', 519), ('UFPR04', 'camera6', 520), ('UFPR05', 'camera6', 521),
        ('camera1', 'camera6', 522), ('camera2', 'camera6', 523), ('camera3', 'camera6', 524),
        ('camera4', 'camera6', 525), ('camera5', 'camera6', 526), ('camera6', 'camera6', 527),
        ('camera7', 'camera6', 528), ('camera8', 'camera6', 529), ('camera9', 'camera6', 530),

        # Para camera7
        ('PUC', 'camera7', 578), ('UFPR04', 'camera7', 579), ('UFPR05', 'camera7', 580),
        ('camera1', 'camera7', 581), ('camera2', 'camera7', 582), ('camera3', 'camera7', 583),
        ('camera4', 'camera7', 584), ('camera5', 'camera7', 585), ('camera6', 'camera7', 586),
        ('camera7', 'camera7', 587), ('camera8', 'camera7', 588), ('camera9', 'camera7', 589),

        # Para camera8
        ('PUC', 'camera8', 637), ('UFPR04', 'camera8', 638), ('UFPR05', 'camera8', 639),
        ('camera1', 'camera8', 640), ('camera2', 'camera8', 641), ('camera3', 'camera8', 642),
        ('camera4', 'camera8', 643), ('camera5', 'camera8', 644), ('camera6', 'camera8', 645),
        ('camera7', 'camera8', 646), ('camera8', 'camera8', 647), ('camera9', 'camera8', 648),

        # Para camera9
        ('PUC', 'camera9', 696), ('UFPR04', 'camera9', 697), ('UFPR05', 'camera9', 698),
        ('camera1', 'camera9', 699), ('camera2', 'camera9', 700), ('camera3', 'camera9', 701),
        ('camera4', 'camera9', 702), ('camera5', 'camera9', 703), ('camera6', 'camera9', 704),
        ('camera7', 'camera9', 705), ('camera8', 'camera9', 706), ('camera9', 'camera9', 707)
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

    del df1, df2, df3, df4, sheet, Kyoto, Kyoto_sum, Kyoto_mult, Kyoto_voto

    #Tabela PKLot
    sheet = doc.sheets[1]

    df1 = pd.read_csv('resultados/Modelo_Kyoto/tabela_resultado-PKLot.csv')
    df2 = pd.read_csv('resultados/Modelo_Kyoto/tabela_SumFusion-PKLot.csv')
    df3 = pd.read_csv('resultados/Modelo_Kyoto/tabela_MultFusion-PKLot.csv')
    df4 = pd.read_csv('resultados/Modelo_Kyoto/tabela_VoteFusion-PKLot.csv') 

    pklot = [
        #Para camera1
        ('camera1', 'camera1', 8), ('camera2', 'camera1', 9), ('camera3', 'camera1', 10),
        ('camera4', 'camera1', 11), ('camera5', 'camera1', 12), ('camera6', 'camera1', 13),
        ('camera7', 'camera1', 14), ('camera8', 'camera1', 15), ('camera9', 'camera1', 16),

        #Para camera2
        ('camera1', 'camera2', 55), ('camera2', 'camera2', 56), ('camera3', 'camera2', 57),
        ('camera4', 'camera2', 58), ('camera5', 'camera2', 59), ('camera6', 'camera2', 60),
        ('camera7', 'camera2', 61), ('camera8', 'camera2', 62), ('camera9', 'camera2', 63),

        #Para camera3
        ('camera1', 'camera3', 102), ('camera2', 'camera3', 103), ('camera3', 'camera3', 104),
        ('camera4', 'camera3', 105), ('camera5', 'camera3', 106), ('camera6', 'camera3', 107),
        ('camera7', 'camera3', 108), ('camera8', 'camera3', 109), ('camera9', 'camera3', 110),

        #Para camera4
        ('camera1', 'camera4', 149), ('camera2', 'camera4', 150), ('camera3', 'camera4', 151),
        ('camera4', 'camera4', 152), ('camera5', 'camera4', 153), ('camera6', 'camera4', 154),
        ('camera7', 'camera4', 155), ('camera8', 'camera4', 156), ('camera9', 'camera4', 157),
        
        #Para camera5
        ('camera1', 'camera5', 196), ('camera2', 'camera5', 197), ('camera3', 'camera5', 198),
        ('camera4', 'camera5', 199), ('camera5', 'camera5', 200), ('camera6', 'camera5', 201),
        ('camera7', 'camera5', 202), ('camera8', 'camera5', 203), ('camera9', 'camera5', 204),

        #Para camera6
        ('camera1', 'camera6', 243), ('camera2', 'camera6', 244), ('camera3', 'camera6', 245),
        ('camera4', 'camera6', 246), ('camera5', 'camera6', 247), ('camera6', 'camera6', 248),
        ('camera7', 'camera6', 249), ('camera8', 'camera6', 250), ('camera9', 'camera6', 251),

        #Para camera7   
        ('camera1', 'camera7', 290), ('camera2', 'camera7', 291), ('camera3', 'camera7', 292),
        ('camera4', 'camera7', 293), ('camera5', 'camera7', 294), ('camera6', 'camera7', 295),
        ('camera7', 'camera7', 296), ('camera8', 'camera7', 297), ('camera9', 'camera7', 298),

        #Para camera8
        ('camera1', 'camera8', 337), ('camera2', 'camera8', 338), ('camera3', 'camera8', 339),  
        ('camera4', 'camera8', 340), ('camera5', 'camera8', 341), ('camera6', 'camera8', 342),
        ('camera7', 'camera8', 343), ('camera8', 'camera8', 344), ('camera9', 'camera8', 345),

        #Para camera9
        ('camera1', 'camera9', 384), ('camera2', 'camera9', 385), ('camera3', 'camera9', 386),
        ('camera4', 'camera9', 387), ('camera5', 'camera9', 388), ('camera6', 'camera9', 389),
        ('camera7', 'camera9', 390), ('camera8', 'camera9', 391), ('camera9', 'camera9', 392),        
    ]

    pklot_sum = [
        #Para camera1
        ('camera1', 'camera1', 18), ('camera2', 'camera1', 19), ('camera3', 'camera1', 20),
        ('camera4', 'camera1', 21), ('camera5', 'camera1', 22), ('camera6', 'camera1', 23),
        ('camera7', 'camera1', 24), ('camera8', 'camera1', 25), ('camera9', 'camera1', 26),

        #Para camera2
        ('camera1', 'camera2', 65), ('camera2', 'camera2', 66), ('camera3', 'camera2', 67),
        ('camera4', 'camera2', 68), ('camera5', 'camera2', 69), ('camera6', 'camera2', 70),
        ('camera7', 'camera2', 71), ('camera8', 'camera2', 72), ('camera9', 'camera2', 73),

        #Para camera3
        ('camera1', 'camera3', 112), ('camera2', 'camera3', 113), ('camera3', 'camera3', 114),
        ('camera4', 'camera3', 115), ('camera5', 'camera3', 116), ('camera6', 'camera3', 117),
        ('camera7', 'camera3', 118), ('camera8', 'camera3', 119), ('camera9', 'camera3', 120),

        #Para camera4
        ('camera1', 'camera4', 159), ('camera2', 'camera4', 160), ('camera3', 'camera4', 161),
        ('camera4', 'camera4', 162), ('camera5', 'camera4', 163), ('camera6', 'camera4', 164),
        ('camera7', 'camera4', 165), ('camera8', 'camera4', 166), ('camera9', 'camera4', 167),
        
        #Para camera5
        ('camera1', 'camera5', 206), ('camera2', 'camera5', 207), ('camera3', 'camera5', 208),
        ('camera4', 'camera5', 209), ('camera5', 'camera5', 210), ('camera6', 'camera5', 211),
        ('camera7', 'camera5', 212), ('camera8', 'camera5', 213), ('camera9', 'camera5', 214),

        #Para camera6
        ('camera1', 'camera6', 253), ('camera2', 'camera6', 254), ('camera3', 'camera6', 255),
        ('camera4', 'camera6', 256), ('camera5', 'camera6', 257), ('camera6', 'camera6', 258),
        ('camera7', 'camera6', 259), ('camera8', 'camera6', 260), ('camera9', 'camera6', 261),

        #Para camera7   
        ('camera1', 'camera7', 300), ('camera2', 'camera7', 301), ('camera3', 'camera7', 302),
        ('camera4', 'camera7', 303), ('camera5', 'camera7', 304), ('camera6', 'camera7', 305),
        ('camera7', 'camera7', 306), ('camera8', 'camera7', 307), ('camera9', 'camera7', 308),

        #Para camera8
        ('camera1', 'camera8', 347), ('camera2', 'camera8', 348), ('camera3', 'camera8', 349),  
        ('camera4', 'camera8', 350), ('camera5', 'camera8', 351), ('camera6', 'camera8', 352),
        ('camera7', 'camera8', 353), ('camera8', 'camera8', 354), ('camera9', 'camera8', 355),

        #Para camera9
        ('camera1', 'camera9', 394), ('camera2', 'camera9', 395), ('camera3', 'camera9', 396),
        ('camera4', 'camera9', 397), ('camera5', 'camera9', 398), ('camera6', 'camera9', 399),
        ('camera7', 'camera9', 400), ('camera8', 'camera9', 401), ('camera9', 'camera9', 402),        
    ]

    pklot_mult = [
        #Para camera1
        ('camera1', 'camera1', 28), ('camera2', 'camera1', 29), ('camera3', 'camera1', 30),
        ('camera4', 'camera1', 31), ('camera5', 'camera1', 32), ('camera6', 'camera1', 33),
        ('camera7', 'camera1', 34), ('camera8', 'camera1', 35), ('camera9', 'camera1', 36),

        #Para camera2
        ('camera1', 'camera2', 75), ('camera2', 'camera2', 76), ('camera3', 'camera2', 77),
        ('camera4', 'camera2', 78), ('camera5', 'camera2', 79), ('camera6', 'camera2', 80),
        ('camera7', 'camera2', 81), ('camera8', 'camera2', 82), ('camera9', 'camera2', 83),

        #Para camera3
        ('camera1', 'camera3', 122), ('camera2', 'camera3', 123), ('camera3', 'camera3', 124),
        ('camera4', 'camera3', 125), ('camera5', 'camera3', 126), ('camera6', 'camera3', 127),
        ('camera7', 'camera3', 128), ('camera8', 'camera3', 129), ('camera9', 'camera3', 130),

        #Para camera4
        ('camera1', 'camera4', 169), ('camera2', 'camera4', 170), ('camera3', 'camera4', 171),
        ('camera4', 'camera4', 172), ('camera5', 'camera4', 173), ('camera6', 'camera4', 174),
        ('camera7', 'camera4', 175), ('camera8', 'camera4', 176), ('camera9', 'camera4', 177),
        
        #Para camera5
        ('camera1', 'camera5', 216), ('camera2', 'camera5', 217), ('camera3', 'camera5', 218),
        ('camera4', 'camera5', 219), ('camera5', 'camera5', 220), ('camera6', 'camera5', 221),
        ('camera7', 'camera5', 222), ('camera8', 'camera5', 223), ('camera9', 'camera5', 224),

        #Para camera6
        ('camera1', 'camera6', 263), ('camera2', 'camera6', 264), ('camera3', 'camera6', 265),
        ('camera4', 'camera6', 266), ('camera5', 'camera6', 267), ('camera6', 'camera6', 268),
        ('camera7', 'camera6', 269), ('camera8', 'camera6', 270), ('camera9', 'camera6', 271),

        #Para camera7   
        ('camera1', 'camera7', 310), ('camera2', 'camera7', 311), ('camera3', 'camera7', 312),
        ('camera4', 'camera7', 313), ('camera5', 'camera7', 314), ('camera6', 'camera7', 315),
        ('camera7', 'camera7', 316), ('camera8', 'camera7', 317), ('camera9', 'camera7', 318),

        #Para camera8
        ('camera1', 'camera8', 357), ('camera2', 'camera8', 358), ('camera3', 'camera8', 359),  
        ('camera4', 'camera8', 360), ('camera5', 'camera8', 361), ('camera6', 'camera8', 362),
        ('camera7', 'camera8', 363), ('camera8', 'camera8', 364), ('camera9', 'camera8', 365),

        #Para camera9
        ('camera1', 'camera9', 404), ('camera2', 'camera9', 405), ('camera3', 'camera9', 406),
        ('camera4', 'camera9', 407), ('camera5', 'camera9', 408), ('camera6', 'camera9', 409),
        ('camera7', 'camera9', 410), ('camera8', 'camera9', 411), ('camera9', 'camera9', 412),        
    ]

    pklot_voto = [
        #Para camera1
        ('camera1', 'camera1', 38), ('camera2', 'camera1', 39), ('camera3', 'camera1', 40),
        ('camera4', 'camera1', 41), ('camera5', 'camera1', 42), ('camera6', 'camera1', 43),
        ('camera7', 'camera1', 44), ('camera8', 'camera1', 45), ('camera9', 'camera1', 46),

        #Para camera2
        ('camera1', 'camera2', 85), ('camera2', 'camera2', 86), ('camera3', 'camera2', 87),
        ('camera4', 'camera2', 88), ('camera5', 'camera2', 89), ('camera6', 'camera2', 90),
        ('camera7', 'camera2', 91), ('camera8', 'camera2', 92), ('camera9', 'camera2', 93),

        #Para camera3
        ('camera1', 'camera3', 132), ('camera2', 'camera3', 133), ('camera3', 'camera3', 134),
        ('camera4', 'camera3', 135), ('camera5', 'camera3', 136), ('camera6', 'camera3', 137),
        ('camera7', 'camera3', 138), ('camera8', 'camera3', 139), ('camera9', 'camera3', 140),

        #Para camera4
        ('camera1', 'camera4', 179), ('camera2', 'camera4', 180), ('camera3', 'camera4', 181),
        ('camera4', 'camera4', 182), ('camera5', 'camera4', 183), ('camera6', 'camera4', 184),
        ('camera7', 'camera4', 185), ('camera8', 'camera4', 186), ('camera9', 'camera4', 187),
        
        #Para camera5
        ('camera1', 'camera5', 226), ('camera2', 'camera5', 227), ('camera3', 'camera5', 228),
        ('camera4', 'camera5', 229), ('camera5', 'camera5', 230), ('camera6', 'camera5', 231),
        ('camera7', 'camera5', 232), ('camera8', 'camera5', 233), ('camera9', 'camera5', 234),

        #Para camera6
        ('camera1', 'camera6', 273), ('camera2', 'camera6', 274), ('camera3', 'camera6', 275),
        ('camera4', 'camera6', 276), ('camera5', 'camera6', 277), ('camera6', 'camera6', 278),
        ('camera7', 'camera6', 279), ('camera8', 'camera6', 280), ('camera9', 'camera6', 281),

        #Para camera7   
        ('camera1', 'camera7', 320), ('camera2', 'camera7', 321), ('camera3', 'camera7', 322),
        ('camera4', 'camera7', 323), ('camera5', 'camera7', 324), ('camera6', 'camera7', 325),
        ('camera7', 'camera7', 326), ('camera8', 'camera7', 327), ('camera9', 'camera7', 328),

        #Para camera8
        ('camera1', 'camera8', 367), ('camera2', 'camera8', 368), ('camera3', 'camera8', 369),  
        ('camera4', 'camera8', 370), ('camera5', 'camera8', 371), ('camera6', 'camera8', 372),
        ('camera7', 'camera8', 373), ('camera8', 'camera8', 374), ('camera9', 'camera8', 375),

        #Para camera9
        ('camera1', 'camera9', 414), ('camera2', 'camera9', 415), ('camera3', 'camera9', 416),
        ('camera4', 'camera9', 417), ('camera5', 'camera9', 418), ('camera6', 'camera9', 419),
        ('camera7', 'camera9', 420), ('camera8', 'camera9', 421), ('camera9', 'camera9', 422),        
    ]

    for treino, teste, linha in pklot:
        valores = extrair_media_desvio(df1, 'PKLot', treino, teste)
        preencher_planilha_media_desvio(sheet, linha, 4, valores)

    for treino, teste, linha in pklot_sum:
        valores = extrair_valores(df2, 'PKLot', treino, teste, 'Acuracia')  # retorna array
        preencher_planilha_fusoes(sheet, linha, 4, valores)

    for treino, teste, linha in pklot_mult:
        valores = extrair_valores(df3, 'PKLot', treino, teste, 'Acuracia')  # retorna array
        preencher_planilha_fusoes(sheet, linha, 4, valores)
    
    for treino, teste, linha in pklot_voto:
        valores = extrair_valores(df4, 'PKLot', treino, teste, 'Acuracia')  # retorna array
        preencher_planilha_fusoes(sheet, linha, 4, valores)

    del df1, df2, df3, df4, sheet, pklot, pklot_sum, pklot_mult, pklot_voto

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
        ('PUC', 'UFPR04', 43), ('UFPR04', 'UFPR04', 44), ('UFPR05', 'UFPR04', 45),
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
        if treino == 'UFPR04':
            print(treino, teste, len(valores))
        preencher_planilha_fusoes(sheet, linha, 4, valores)

    # Salvar o arquivo (sobrescreve o original)
    doc.save()
    print("Arquivo salvo com sucesso!")

if __name__ == '__main__':
    tabela_excel()
