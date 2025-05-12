import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils.evaluation_metrics import *


def plot_autoencoder(x_test, Autoencoder, width=64, height=64, caminho_para_salvar=None, nome_autoencoder='Kyoto'):
    def normalize(image):
        image = np.clip(image, 0, 1)  # Garante que a imagem esteja no intervalo [0, 1]
        return (image - image.min()) / (image.max() - image.min()) if image.max() != image.min() else image

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

        ssim = float(calculete_mse(x_test[i], pred[0]))
        mse = float(calculete_mse(x_test[i], pred[0]))
        psnr = float(calculete_psnr(x_test[i], pred[0]))

        del pred_img, pred
        plt.title(f"SSIM: {ssim:.2f}\nMSE: {mse:.2f}\nPsnr: {psnr:.2f}")
        plt.axis("off")
    
    plt.show()

    avaliacoes = []
    for i in range(len(x_test)):
        pred = Autoencoder.predict(x_test[i].reshape((1,width, height,3)))
        ssim = float(calculete_ssim(x_test[i], pred[0]))
        mse = float(calculete_mse(x_test[i], pred[0]))
        psnr = float(calculete_psnr(x_test[i], pred[0]))
        avaliacoes.append([ssim, mse, psnr])

    df_avaliacoes = pd.DataFrame(avaliacoes, columns=["SSIM", "MSE", "PSNR"])
    media_ssim = np.mean(df_avaliacoes['SSIM'].values)
    media_mse = np.mean(df_avaliacoes['MSE'].values)
    media_psnr = np.mean(df_avaliacoes['PSNR'].values)
    
    if caminho_para_salvar != None:
        save_path = os.path.join(caminho_para_salvar, f'Autoencoder-{nome_autoencoder}.png')
        plt.savefig(save_path)

        arquivo = os.path.join(caminho_para_salvar,f'avaliacoes-{nome_autoencoder}.txt')
        with open(arquivo, 'w') as f:
            f.write(f'Media SSIM: {media_ssim}\n')
            f.write(f'Media MSE: {media_mse}\n')
            f.write(f'Media PSNR: {media_psnr}')

    
    plt.close("all") 

