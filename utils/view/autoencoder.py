import numpy as np
from tensorflow.keras.models import Model
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from utils.evaluation_metrics import *

def plot_autoencoder_quality(x_test:np.ndarray, Autoencoder:Model, width:int=64, height:int=64, path_save:Path=None, autoencoder_name:str='Kyoto'):
    plt.figure(figsize=(16, 8))
    for i in range(8):
        # Imagem original
        plt.subplot(2, 8, i + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        plt.axis("off")

        # Predição e normalização
        pred = Autoencoder.predict(x_test[i].reshape((1,width, height,3)))
        pred_img = normalize(pred[0])

        plt.subplot(2, 8, i + 8 + 1)
        plt.imshow(pred_img)

        ssim = float(calcular_ssim(x_test[i], pred[0]))
        mse = float(calcular_mse(x_test[i], pred[0]))
        psnr = float(calcular_psnr(x_test[i], pred[0]))

        del pred_img, pred
        plt.title(f"SSIM: {ssim:.2f}\nMSE: {mse:.2f}\nPsnr: {psnr:.2f}")
        plt.axis("off")
    
    plt.show()

    evaluations = []
    for i in range(len(x_test)):
        pred = Autoencoder.predict(x_test[i].reshape((1,width, height,3)))
        ssim = float(calcular_ssim(x_test[i], pred[0]))
        mse = float(calcular_mse(x_test[i], pred[0]))
        psnr = float(calcular_psnr(x_test[i], pred[0]))
        evaluations.append([ssim, mse, psnr])

    df_evaluations = pd.DataFrame(evaluations, columns=["SSIM", "MSE", "PSNR"])
    media_ssim = np.mean(df_evaluations['SSIM'].values)
    media_mse = np.mean(df_evaluations['MSE'].values)
    media_psnr = np.mean(df_evaluations['PSNR'].values)
    
    if path_save != None:
        save_path = os.path.join(path_save, f'Autoencoder-{autoencoder_name}.png')
        plt.savefig(save_path)

        file = os.path.join(path_save,f'metrics-{autoencoder_name}.txt')
        with open(file, 'w') as f:
            f.write(f'Media SSIM: {media_ssim}\n')
            f.write(f'Media MSE: {media_mse}\n')
            f.write(f'Media PSNR: {media_psnr}')

    
    plt.close("all") 

def plot_vae_quality(x_test:np.ndarray, Autoencoder:Model, width:int=64, height:int=64, save_dir:Path=None):
    """
    Plota a qualidade de um modelo VAE.

    Args:
        x_test: Array numpy com as imagens de teste.
        Autoencoder: Modelo VAE.
        width: Largura das imagens.
        height: Altura das imagens.
        save_dir: Diretório para salvar a imagem.
    """

    plt.figure(figsize=(16, 8))

    evaluations = []
    for i in range(8):
        # Imagem original
        plt.subplot(2, 8, i + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        plt.axis("off")

        # Predição e normalização
        z_mean, z_log_var, z = Autoencoder.encoder(x_test[i].reshape((1, width, height, 3)))
        pred = Autoencoder.decoder(z)
        pred_img = normalize(pred[0])

        plt.subplot(2, 8, i + 8 + 1)
        plt.imshow(pred_img)

        ssim = float(calcular_ssim(x_test[i], pred))
        evaluations.append(ssim)

        del pred_img, pred
        plt.title(f"SSIM: {ssim:.2f}")
        plt.axis("off")

    plt.show()
    media_ssim = np.mean(evaluations)
    
    if save_dir != None:
        save_path = os.path.join(save_dir, 'Autoencoder.png')
        plt.savefig(save_path)

        arquivo = os.path.join(caminho_para_salvar,'media_ssim.txt')
        with open(file, 'w') as f:
            f.write(f'Media SSIM: {media_ssim}')
    
    plt.close("all") 

