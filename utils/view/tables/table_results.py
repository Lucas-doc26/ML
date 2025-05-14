import xlsxwriter

#Definindo a tabela de resultados:
workbook = xlsxwriter.Workbook("Resultados.xlsx")
worksheet = workbook.add_worksheet()

#Definindo estilos:
merge = workbook.add_format({"align": "center", "valign": "vcenter"})
bold = workbook.add_format({"bold": True})

worksheet.merge_range("B1:I2", "Modelos_Kyoto", merge)

autoencoders = ['PKLOT', 'CNR', 'Kyoto']
classifiers = ['PUC', 'UFPR04', 'UFPR05','camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']
tests = ['PUCPR', 'UFPR04', 'UFPR05', 'camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']
batches = ['64', '128', '256', '512', '1024']
fusion_rules = ['-', 'Sum', 'Voto', 'Mult']

#A-0 B-1 C-2 D-3 E-4 F-5 G-6 H-7 I-8 J-9 K-10 L-11 M-12 N-13 O-14 P-15 Q-16 R-17 S-18 T-19 U-20 V-21 W-22 X-23 Y-24 Z-25

def convert_cels(cel):
    map = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9, "K":10, "L":11, "M":12, 
           "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19, "U":20, "V":21, "W":22, "X":23, "Y":24, "Z":25}
    
    col = ''
    row = ''
    for c in cel:
        if c.isalpha():
            col += c
        elif c.isdigit():
            row += c
    
    col_num = map[col] if col in map else None
    row_num = int(row) if row else None
    
    return col_num, row_num - 1 if row_num is not None else None 

def write_test(cels_test, test):
    worksheet.write(cels_test, f"Teste {test}", merge)

def write_infos(cels_bases, cels_fusions, cels_batches):
    worksheet.merge_range(cels_bases[0], cels_bases[1], cels_bases[2], cels_bases[3], "Bases de Treino", merge)
    worksheet.merge_range(cels_fusions[0], cels_fusions[1], cels_fusions[2], cels_fusions[3], "Técnica de Fusão", merge)
    worksheet.merge_range(cels_batches[0], cels_batches[1], cels_batches[2], cels_batches[3], "Batches", merge)



write_infos()

workbook.close()