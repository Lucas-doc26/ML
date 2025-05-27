import os 
import pandas as pd

dados = []
for i in range(10):
    files = os.listdir(f"/home/lucas/PIBIC/Modelos/Modelo_Kyoto-{i}/Plots")
    files = [os.path.join(f'/home/lucas/PIBIC/Modelos/Modelo_Kyoto-{i}/Plots', f) for f in files if f.endswith(".txt")]
    for file in files:
        f = open(file, "r")
        base = file.split("/")[7]
        base = base.split('.txt')[0].split('-')[1]
        for line in f:
            parts = line.split(": ")
            metric = parts[0].split(' ')[1]
            value = parts[1]
            dados.append([f"Modelo_Kyoto-{i}", base, metric, f"{float(value):.3f}"])
    print(dados)

df = pd.DataFrame(dados, columns=['Modelo', 'Dataset', 'Metrica', 'Valor'])

df['Valor'] = df['Valor'].str.strip().astype(float)
df.to_csv('/home/lucas/PIBIC/resultados/Modelo_Kyoto/metricas.csv', index=False)