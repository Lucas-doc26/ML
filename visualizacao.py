import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os
import numpy as np
import math
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from Preprocessamento import mapear_rotulos_binarios, carregar_e_preprocessar_imagens
from avaliacoes import *
import keras

path = r'/home/lucas/PIBIC'

def plot_imagens_com_csv(caminho_csv, img_por_coluna):
    """
    Função para plotar imagens de um arquivo .csv contendo os caminhos de imagens.
    
    :param caminho_csv: caminho para o arquivo .csv contendo os caminhos de imagens
    :param img_por_coluna: número de imagens por coluna
    """
    # Ler o CSV para um DataFrame
    dataframe = pd.read_csv(caminho_csv)
    
    _, axs = plt.subplots(img_por_coluna, img_por_coluna, figsize=(10, 10))

    # Iterar sobre as linhas do DataFrame
    for index, linha in dataframe.iterrows():
        if index >= img_por_coluna**2:
            break  
        image_path = linha['caminho_imagem']
        image = plt.imread(image_path)
        linha_idx = index // img_por_coluna
        coluna_idx = index % img_por_coluna
        axs[linha_idx, coluna_idx].imshow(image)
        axs[linha_idx, coluna_idx].axis('off')  # Desligar os eixos
        axs[linha_idx, coluna_idx].set_title(f"{linha['classe']}")  # Título da imagem

    plt.tight_layout()
    plt.show()
    plt.close()
     
def plot_imagens_dataframe_gerador(dataframe_gerador, img_por_coluna=3):
    """
    Função para plotar imagens geradas pelo DataFrameIterator.
    :param dataframe_gerador: DataFrameIterator contendo as imagens geradas.
    :param Autoencoder: Se True, não mostra a classe, apenas a imagem (como no Autoencoder).
    :param img_por_coluna: Número de imagens por coluna e por linha a serem plotadas.
    """
    classes = ['Empty', 'Occupied']

    # Define a figura com subplots
    fig, axs = plt.subplots(img_por_coluna, img_por_coluna, figsize=(10, 10))
    imagens_por_plot = img_por_coluna ** 2 
    
    # Gera as imagens e labels a partir do gerador
    imagens, labels = next(dataframe_gerador)

    for i in range(imagens_por_plot):
        if i >= len(imagens):
            break
        
        image = imagens[i]

        # Normaliza a imagem para o intervalo [0, 1]
        image = (image - image.min()) / (image.max() - image.min())

        axs[i // img_por_coluna, i % img_por_coluna].imshow(image)
        axs[i // img_por_coluna, i % img_por_coluna].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()
        
def plot_confusion_matrix(y_true, y_pred, labels=['Empty', 'Occupied'], title=None, save_path=None):
    """
    Plota uma matriz de confusão.

    Args:
    - y_true (array): rótulos verdadeiros.
    - y_pred (array): rótulos previstos.
    - labels (list): lista de rótulos das classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calcular a acurácia
    accuracy = accuracy_score(y_true, y_pred)

    # Criar o gráfico
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    # Adicionar título com a acurácia
    accuracy_text = f"Accuracy: {accuracy * 100:.2f}%"
    plt.title('' if title is None else f"{title}\n{accuracy_text}")
    
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()

def exibir_primeiras_imagens(dataframe):
    # Pegar os caminhos das 9 primeiras imagens
    caminhos_imagens = dataframe['caminho_imagem'].iloc[:9].tolist()

    # Configurar a exibição das imagens
    plt.figure(figsize=(12, 8))
    for i, caminho in enumerate(caminhos_imagens):
        plt.subplot(3, 3, i + 1)
        img = Image.open(caminho)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Imagem {i + 1}')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_imagens_incorretas(y_binario, y_predicao, caminhos_imagens, modelo_nome:str, dataset_nome:str, n_imagens_por_grade:3):
    """
    Plota imagens cujas previsões foram incorretas.

    Parâmetros:
    - y_binario: Array numpy com os rótulos reais.
    - y_predicao: Array numpy com as previsões do modelo.
    - caminhos_imagens: Lista de caminhos para as imagens correspondentes.
    - modelo_nome: Nome do modelo para fins de salvamento de arquivo.
    - dataset_nome: Nome do dataset para fins de salvamento de arquivo.
    - n_imagens_por_grade: Número de imagens por grade na visualização (default: 3).

    A função identifica as imagens onde as previsões do modelo diferem dos rótulos reais,
    e plota estas imagens junto com suas previsões e rótulos reais. As imagens são organizadas
    em uma grade de subplots para visualização fácil. As imagens incorretamente previstas são
    salvas em um arquivo PNG com base nos nomes do modelo e do dataset fornecidos.
    """

    labels = ['Empty', 'Occupied']

    indices_incorretos = np.where(y_predicao != y_binario)[0]

    num_imagens_plotadas = min(len(indices_incorretos), n_imagens_por_grade**2)
    indices_plotados = indices_incorretos[:num_imagens_plotadas]

    fig, axes = plt.subplots(n_imagens_por_grade, n_imagens_por_grade, figsize=(15, 15))
    axes = axes.flatten()

    plt.suptitle(f'{modelo_nome} vs {dataset_nome} - Imagens incorretas', fontsize=16)
    for ax, indice in zip(axes, indices_plotados):
        img = load_img(caminhos_imagens[indice])
        ax.imshow(img)
        ax.set_title(f'Predição: {labels[y_predicao[indice]]}\nReal: {labels[y_binario[indice]]}')
        ax.axis('off')

    # Remover qualquer eixo vazio
    for i in range(num_imagens_plotadas, n_imagens_por_grade**2):
        axes[i].axis('off')

    plt.tight_layout()
    save_path_imgs = os.path.join('Resultados', 'Imagens_Incorretas', modelo_nome, f'{modelo_nome}_vs_{dataset_nome}_imagens_incorretas.png')
    os.makedirs(os.path.dirname(save_path_imgs), exist_ok=True)
    plt.savefig(save_path_imgs)
    plt.close()

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

        ssim = float(calcular_ssim(x_test[i], pred[0]))
        mse = float(calcular_mse(x_test[i], pred[0]))
        psnr = float(calcular_psnr(x_test[i], pred[0]))

        del pred_img, pred
        plt.title(f"SSIM: {ssim:.2f}\nMSE: {mse:.2f}\nPsnr: {psnr:.2f}")
        plt.axis("off")
    
    plt.show()

    avaliacoes = []
    for i in range(len(x_test)):
        pred = Autoencoder.predict(x_test[i].reshape((1,width, height,3)))
        ssim = float(calcular_ssim(x_test[i], pred[0]))
        mse = float(calcular_mse(x_test[i], pred[0]))
        psnr = float(calcular_psnr(x_test[i], pred[0]))
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

def plot_batch(batch, batch_size):

    img_por_coluna = int(math.sqrt(batch_size))

    classes = ['Empty', 'Occupied']

    # Define a figura com subplots
    fig, axs = plt.subplots(img_por_coluna, img_por_coluna, figsize=(10, 10))
    
    imagens, _ = next(batch)  # Obtém as imagens (e os rótulos, que não estamos usando)
    print(f'Total de imagens no lote: {len(imagens)}')  # Imprime a quantidade de imagens

    for i in range(batch_size):
        if i >= len(imagens):
            print(f'Número de imagens no lote é menor que o esperado! Apenas {len(imagens)} imagens foram retornadas.')
            break
        
        image = imagens[i]

        # Normaliza a imagem para o intervalo [0, 1]
        image = (image - image.min()) / (image.max() - image.min())

        axs[i // img_por_coluna, i % img_por_coluna].imshow(image)
        axs[i // img_por_coluna, i % img_por_coluna].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

def avaliar_modelo_em_datasets(modelo, datasets_info):
    """
    Avalia um modelo em múltiplos datasets e gera métricas e visualizações.

    :param modelo: O modelo a ser avaliado. Deve ter o método `predict`.
    :param datasets_info: Lista de tuplas contendo o nome do dataset, o dataset em si e o DataFrame associado.
    """
    for dataset_nome, dataset, dataset_df in datasets_info:
        y_verdadeiro = dataset_df['classe'].values
        y_binario = mapear_rotulos_binarios(y_verdadeiro) 

        caminhos_imagens = dataset_df['caminho_imagem'].tolist()
        imagens = carregar_e_preprocessar_imagens(caminhos_imagens)
        
        y_predicao = modelo.predict(imagens).argmax(axis=1)  

        labels = ['Empty', 'Occupied']  
        
        save_path_matriz = os.path.join('Resultados', 'Matriz_de_confusao', modelo.name, f'{modelo.name}_vs_{dataset_nome}_matriz_de_confusao.png')
        titulo_matriz = f'Matriz de Confusão - {modelo.name} vs {dataset_nome}'
        plot_confusion_matrix(y_binario, y_predicao, labels, save_path_matriz, titulo_matriz)
        plot_imagens_incorretas(y_binario, y_predicao, caminhos_imagens, modelo.name, dataset_nome, 3)

def grafico_batchs(n_batchs, precisoes, nome_modelo, nome_base_treino, base_usada_teste, nome_autoencoder, caminho_para_salvar=None):
    #print(f"Base_treino: {nome_base_treino}")
    #print(f"Base_teste: {base_usada_teste}")

    plt.clf()
    plt.close('all')
    plt.figure() 
    plt.close
    plt.title(f"Comparação de acurácia - {nome_modelo}")
    plt.xlabel('Número de imagens')
    plt.ylabel('Acurácia')

    label = (
        f"Nome = {nome_modelo}\n"
        f"Base do Autoencoder: {nome_autoencoder}\n"
        f"Base do Classificador: {nome_base_treino}\n"
        f"Base sendo testada: {base_usada_teste}"
    )

    plt.plot(n_batchs, precisoes, marker='o', linestyle='-', color='b', label=label)
    plt.xticks(n_batchs)  
    for xi, yi in zip(n_batchs, precisoes):
            plt.text(xi, yi, f"{yi:.3f}", fontsize=6, ha='left', va='top') 

    plt.legend(loc='lower right', fontsize=9, title="Informações do modelo", title_fontsize=10)

    try:
        nome = nome_modelo.split(' ')[0]
        nome_modelo = nome
    except:
        pass
    
    if caminho_para_salvar != None:
        save_path = os.path.join(caminho_para_salvar, f'Grafico-{nome_modelo}-{nome_autoencoder}-{nome_base_treino}-{base_usada_teste}')
        plt.savefig(save_path)
        print(f"Salvando gráfico no caminho: {save_path}.png")

    plt.show()
    plt.close()

#Compara os n modelos criados 
def comparacao(caminho_para_salvar=None, nome_modelo=None, base_usada=None, base_de_teste=None, base_autoencoder=None):
    
    if base_de_teste == None:
        base_de_teste = base_usada

    dados = []
    tabela = pd.DataFrame(columns=['Nome Modelo', 'Batch'])

    dir_base = os.path.join(path, "Modelos")
    modelos = [modelo for modelo in os.listdir(dir_base) if (f'{nome_modelo}' in modelo and "Fusoes" not in modelo)]
    print(modelos)
    x = [64,128,256,512,1024]
    plt.figure(figsize=(10, 6))
    plt.xticks(x)  

    for modelo in sorted(modelos):
        dir_resultados = os.path.join(dir_base, modelo, f'Classificador-{base_autoencoder}/Precisao/Treinado_em_{base_usada}')
        #..Precisao/
        if base_de_teste != base_usada:
            precisao = [r for r in os.listdir(dir_resultados) if f'{base_de_teste}' in r]
        else:
            precisao = [r for r in os.listdir(dir_resultados) if f'{base_usada}' in r]

        print(precisao)
        dir_precisao = os.path.join(dir_resultados, precisao[0])

        with open(dir_precisao, 'r') as f:
            lista_lida = f.readlines()

        lista_lida = [item.strip() for item in lista_lida]

        lista = [round(float(item), 4) for item in lista_lida]

        plt.plot(x, lista, label=f"{modelo}", marker='o')

        #x é o batch
        #y a precisão
        for xi, yi in zip(x, lista):
            plt.text(xi, yi, f"{yi:.3f}", fontsize=6, ha='left', va='top') 

        dados.append([modelo] + lista)
            
    plt.title(f'Comparação entre os(as) diferentes {nome_modelo} - Treinado na base: {base_usada}')
    plt.xlabel('Número de imagens')
    plt.ylabel('Acurácia')

    plt.legend()

    colunas = ['Modelo'] + [f'Batch {batch}' for batch in x]
    df = pd.DataFrame(dados, columns=colunas)

    if caminho_para_salvar != None:
        save_path = os.path.join(path, caminho_para_salvar, f'Grafico-Comparacao-{nome_modelo}-{base_autoencoder}-{base_usada}-{base_de_teste}.png')
        plt.savefig(save_path)

        csv_path = os.path.join(path, caminho_para_salvar, f'Tabela-Comparacao-{nome_modelo}-{base_autoencoder}-{base_usada}-{base_de_teste}.csv')
        df.to_csv(csv_path, index=False)

    plt.show()
    plt.close()

#Testes:
#comparacao('/media/lucas/mnt/data/Lucas$/Modelos/Plots', 'Modelo_Kyoto', 'PUC', 'UFPR05')

def plot_history(df, save_dir, modelo, base_do_autoencoder):
    plt.figure(figsize=(8, 6))
    df.plot()
    plt.title("Treinamento do Autoencoder")
    plt.xlabel("Épocas")
    plt.ylabel("Métricas")
    plt.grid()

    nome_img = f"History-Autoencoder-{modelo}-{base_do_autoencoder}.png"

    plt.savefig(os.path.join(save_dir, nome_img)) 

    plt.clf()
    plt.close('all')  

def plot_history_batch(history, save_dir, nome_modelo, nome_base_treino, nome_base_autoencoder, batch):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Criando o gráfico
    plt.figure(figsize=(12, 6))

    # Subplot para Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label='Acurácia (Treinamento)')
    plt.plot(val_accuracy, label='Acurácia (Validação)')
    plt.title('Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    # Subplot para Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Loss (Treinamento)')
    plt.plot(val_loss, label='Loss (Validação)')
    plt.title('Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    # Exibir o gráfico
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"History-{nome_modelo}-{nome_base_autoencoder}-{nome_base_treino}-{batch}.png")) 

def plot_heat_map(teste, encoder, decoder):
    # Pegue a camada de interesse (a última Conv2D do encoder)
    layer_name = [layer.name for layer in encoder.layers if isinstance(layer, keras.layers.Conv2D)][-1]
    print("Usando camada:", layer_name)

    # Cria um modelo auxiliar que vai até essa camada
    activation_model = Model(inputs=encoder.input,
                             outputs=encoder.get_layer(layer_name).output)

    # Função para processar uma imagem
    def process_image(input_img):
        # Redimensionar a imagem para garantir a forma correta
        if input_img.shape != (256, 256, 3):
            input_img = tf.image.resize(input_img, (256, 256))

        # Expandir a imagem para o formato batch
        input_img_batch = np.expand_dims(input_img, axis=0)  # shape: (1, 256, 256, 3)

        # Obter as ativações da última camada Conv2D
        activations = activation_model.predict(input_img_batch)  # shape: (1, H, W, filters)
        activation_map = np.mean(activations[0], axis=-1)  # média entre todos os filtros

        # Obter a codificação latente z e reconstrução da imagem
        _, _, z_occ = encoder.predict(input_img_batch)  # shape: (1, latent_dim)
        occ_reconstructed_img = decoder.predict(z_occ)  # reconstrução da imagem

        return input_img, occ_reconstructed_img[0], activation_map

    # Imagem 1
    input_img_1, occ_reconstructed_img_1, activation_map_1 = process_image(teste[0])

    # Imagem 2
    input_img_2, empty_reconstructed_img_2, activation_map_2 = process_image(teste[33])

    # Exibir gráfico
    plt.figure(figsize=(12, 8))

    # Exibir para a primeira imagem
    plt.subplot(2, 3, 1)
    plt.imshow(input_img_1)
    plt.title("Imagem Original 1")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(occ_reconstructed_img_1)
    plt.title("Imagem Reconstruída 1")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(activation_map_1, cmap='viridis')
    plt.title("Mapa de Ativação 1")
    plt.axis('off')

    # Exibir para a segunda imagem
    plt.subplot(2, 3, 4)
    plt.imshow(input_img_2)
    plt.title("Imagem Original 2")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(empty_reconstructed_img_2)
    plt.title("Imagem Reconstruída 2")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(activation_map_2, cmap='viridis')
    plt.title("Mapa de Ativação 2")
    plt.axis('off')

    # Ajuste o layout e exiba
    plt.tight_layout()
    plt.savefig("img.png")
    plt.show()

def plot_autoencoder_2(x_test, Autoencoder, width=64, height=64, caminho_para_salvar=None):
    def normalize(image):
        image = np.clip(image, 0, 1)  # Garante que a imagem esteja no intervalo [0, 1]
        return (image - image.min()) / (image.max() - image.min()) if image.max() != image.min() else image

    plt.figure(figsize=(16, 8))

    avaliacoes = []
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
        avaliacoes.append(ssim)

        del pred_img, pred
        plt.title(f"SSIM: {ssim:.2f}")
        plt.axis("off")

    plt.show()
    media_ssim = np.mean(avaliacoes)
    
    if caminho_para_salvar != None:
        save_path = os.path.join(caminho_para_salvar, 'Autoencoder.png')
        plt.savefig(save_path)

        arquivo = os.path.join(caminho_para_salvar,'media_ssim.txt')
        with open(arquivo, 'w') as f:
            for av in avaliacoes:
                f.write(f'{av}\n')
            f.write(f'Media geral: {media_ssim}')
    
    plt.close("all") 


#base_de_teste = ['PUC', 'UFPR04', 'UFPR05', 'camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8','camera9']

"""for faculdade in ['PUC', 'UFPR04', 'UFPR05']:
    for base in base_de_teste:
        comparacao('/media/lucas/mnt/data/Lucas$/Modelos/Plots', 'Modelo_Kyoto', faculdade, base)"""
