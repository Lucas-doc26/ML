from Modelos import *
from segmentandoDatasets import *

segmentacao_PKLot(1024, 2, 64, 1, None, None, ['UFPR05'])
dividir_em_batchs('CSV/UFPR04/UFPR04_Segmentado_Treino.csv')

#modelo = Gerador()
#modelo.carrega_modelo(modelo='/media/hd/mnt/data/Lucas$/Modelos/Modelo_exp-0/Modelo-Base/Estrutura/Modelo_exp-0.keras', pesos='/media/hd/mnt/data/Lucas$/Modelos/Modelo_exp-0/Modelo-Base/Pesos/Modelo_exp-0_Base-Kyoto.weights.h5')
#encoder = modelo.encoder
#encoder.summary()
#decoder = modelo.decoder
#decoder.summary()