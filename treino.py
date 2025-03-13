from Modelos import *
from segmentandoDatasets import *
import matplotlib
from tensorflow.keras.optimizers import Adam
import sys
import os 

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

modelo_nome = sys.argv[1]

matplotlib.use('Agg')

path = '/media/hd/mnt/data/Lucas$'

#segmentacao_Kyoto()
#segmentacao_PKLot(imagens_treino=1024, dias_treino=2, imagens_validacao=64, dias_validaco=1, imagens_teste=None, dias_teste=None, faculdades=["PUC"])
#segmentacao_PKLot(imagens_treino=2, dias_treino=1, imagens_validacao=2, dias_validaco=1, imagens_teste=None, dias_teste=None, faculdades=["UFPR04"])
#segmentacao_PKLot(imagens_treino=2, dias_treino=1, imagens_validacao=2, dias_validaco=1, imagens_teste=None, dias_teste=None, faculdades=["UFPR05"])

# Preprocessamento imagens autoencoder
treino_autoencoder, _ = preprocessamento_dataframe(caminho_csv='CSV/Kyoto/Kyoto_Segmentado_Treino.csv', autoencoder=True, data_algumentantation=False)
validacao_autoencoder, _ = preprocessamento_dataframe(caminho_csv='CSV/Kyoto/Kyoto_Segmentado_Validacao.csv', autoencoder=True, data_algumentantation=False)
teste_autoencoder, _ = preprocessamento_dataframe(caminho_csv='CSV/Kyoto/Kyoto_Segmentado_Teste.csv', autoencoder=True, data_algumentantation=False)

# Carrega autoencoder
autoencoder = Gerador(min_layers=7, max_layers=8, nome_modelo=f"{modelo_nome}")
autoencoder.construir_modelo(True, [64,128,256])
encoder = autoencoder.encoder
decoder = autoencoder.decoder
encoder.summary()
decoder.summary()
autoencoder.setNome(f"{modelo_nome}")

# Treinando autoencoder
autoencoder.Dataset(treino=treino_autoencoder, validacao=validacao_autoencoder, teste=teste_autoencoder)
autoencoder.compilar_modelo()
autoencoder.treinar_autoencoder(salvar=True, epocas=1000, nome_da_base='Kyoto', batch_size=8)

#Criando classificador
encoder = autoencoder.encoder
gerador = GeradorClassificador(encoder=encoder, pesos=os.path.join(path ,f'Modelos/{modelo_nome}/Modelo-Base/Pesos/{modelo_nome}_Base-Kyoto.weights.h5'), 
                                nome_modelo=f'{modelo_nome}')

#Preprocessamento imagens dos classificadores
val, _ = preprocessamento_dataframe(caminho_csv='CSV/PUC/PUC_Segmentado_Validacao.csv', autoencoder=False, data_algumentantation=False)
teste_puc, teste_df_puc = preprocessamento_dataframe(caminho_csv='CSV/PUC/PUC_Segmentado_Teste.csv', autoencoder=False, data_algumentantation=False)

#PUCPR
treinamento_em_batch(f'{modelo_nome}', 'Kyoto', 'CSV/PUC/PUC_Segmentado_Treino.csv', val, teste_puc, teste_df_puc, salvar=True, n_epocas=10)

#UFPR04
teste_UFPR04, teste_df_UFPR04 = preprocessamento_dataframe(caminho_csv='CSV/UFPR04/UFPR04.csv', autoencoder=False, data_algumentantation=False)
testa_modelos(f'{modelo_nome}', teste_UFPR04, teste_df_UFPR04)

#UFPR05
teste_UFPR05, teste_df_UFPR05 = preprocessamento_dataframe(caminho_csv='CSV/UFPR05/UFPR05.csv', autoencoder=False, data_algumentantation=False)
testa_modelos(f'{modelo_nome}', teste_UFPR05, teste_df_UFPR05)

#Fusões 

