from Modelos import *
from Preprocessamento import preprocessamento_dataframe
from segmentandoDatasets import *

segmentacao_PKLot(imagens_treino=1024, dias_treino=2, imagens_validacao=64, dias_validaco=1, imagens_teste=None, dias_teste=None, faculdades=["PUC"])
val, _ = preprocessamento_dataframe(caminho_csv='CSV/PUC/PUC_Segmentado_Validacao.csv', autoencoder=False, data_algumentantation=False)
teste, teste_df = preprocessamento_dataframe(caminho_csv='CSV/PUC/PUC_Segmentado_Teste.csv', autoencoder=False, data_algumentantation=False)

resultado = treina_modelos_em_batch('Modelo_Kyoto', 'Kyoto', 'CSV/PUC/PUC_Segmentado_Treino.csv', val, teste, teste_df, True, 10)