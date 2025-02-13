from Modelos import *
from Preprocessamento import *
from segmentandoDatasets import *
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import matplotlib
import shutil

matplotlib.use('Agg')

segmentacao_Kyoto()

treino, _ = preprocessamento_dataframe(caminho_csv='CSV/Kyoto/Kyoto_Segmentado_Treino.csv', autoencoder=True)
validacao, _ = preprocessamento_dataframe(caminho_csv='CSV/Kyoto/Kyoto_Segmentado_Validacao.csv', autoencoder=True)
teste, _ = preprocessamento_dataframe(caminho_csv='CSV/Kyoto/Kyoto_Segmentado_Teste.csv', autoencoder=True, data_algumentantation=False)

def SSIMLoss(true, pred):
    return 1 - tf.reduce_mean(tf.image.ssim(true, pred, 1.0))

def combined_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.mse(y_true, y_pred) + 0.5 * (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)))

optimizer = tf.keras.optimizers.Adam(0.001)

Modelo = Gerador(input_shape=(64, 64, 3))
Modelo.setNome('Modelo_Teste')
modelo = Modelo.construir_modelo(salvar=True)
modelo.summary()
Modelo.encoder.summary()
Modelo.decoder.summary()

Modelo.Dataset(treino, validacao, teste)
Modelo.compilar_modelo(optimizer=optimizer,loss=SSIMLoss)
Modelo.treinar_autoencoder(epocas=10000, salvar=True, nome_da_base='Kyoto' ,batch_size=42)

img1,img2 = Modelo.predicao()

ssim_score, _ = ssim(img1, img2, full=True, channel_axis=-1, data_range=1.0)  
print(f"SSIM Score: {ssim_score}")


shutil.rmtree('mnt/data/lucas/Pesos/Pesos_parciais')
matplotlib.use('TkAgg')
img1,img2 = Modelo.predicao()
