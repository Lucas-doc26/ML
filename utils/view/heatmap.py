import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Tuple, Optional, Union
from pathlib import Path
from utils.preprocessing import process_image_for_heatmap, process_image_for_heatmap_2
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

def plot_heat_map( test: np.ndarray, encoder: Model, decoder: Model, 
    input_img_shape: Tuple[int, int, int] = (64, 64, 3), 
    save_path: Optional[Path] = None) -> None:
    """
    Plota um mapa de calor das ativações do encoder junto com as imagens originais e reconstruídas.

    Args:
        test: Array numpy com as imagens de teste
        encoder: Modelo do encoder
        decoder: Modelo do decoder
        input_img_shape: Formato das imagens de entrada (altura, largura, canais)
        save_path: Caminho para salvar a figura (opcional)
    """
    # Pegar a última camada Conv2D do encoder
    layer_name = [layer.name for layer in encoder.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    print("Usando camada:", layer_name)

    # Criar modelo auxiliar para obter as ativações
    activation_model = Model(inputs=encoder.input,
                           outputs=encoder.get_layer(layer_name).output)

    

    # Processar duas imagens para comparação
    input_img_1, reconstructed_img_1, activation_map_1 = process_image_for_heatmap(test[0], input_img_shape, activation_model, encoder, decoder)
    input_img_2, reconstructed_img_2, activation_map_2 = process_image_for_heatmap(test[33], input_img_shape, activation_model, encoder, decoder)

    # Criar figura com subplots
    plt.figure(figsize=(12, 8))

    # Plotar primeira imagem
    plt.subplot(2, 3, 1)
    plt.imshow(input_img_1)
    plt.title("Imagem Original 1")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(reconstructed_img_1)
    plt.title("Imagem Reconstruída 1")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(activation_map_1, cmap='viridis')
    plt.title("Mapa de Ativação 1")
    plt.axis('off')

    # Plotar segunda imagem
    plt.subplot(2, 3, 4)
    plt.imshow(input_img_2)
    plt.title("Imagem Original 2")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(reconstructed_img_2)
    plt.title("Imagem Reconstruída 2")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(activation_map_2, cmap='viridis')
    plt.title("Mapa de Ativação 2")
    plt.axis('off')

    # Ajustar layout e salvar/exibir
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.savefig("img.png")
    plt.show()
    plt.close()

import cv2

# Função para encontrar automaticamente a última camada convolucional
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose)):
            return layer.name
    raise ValueError("Nenhuma camada convolucional encontrada no modelo.")

def make_gradcam_heatmap(img_array, model, pred_index=None):
    for layer in model.layers:
        
        layer_name = layer.name

        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        yield heatmap.numpy()

# Função para sobrepor heatmap na imagem
def superimpose_heatmap(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))  # Redimensiona heatmap
    heatmap = np.uint8(255 * heatmap)  # Normaliza para 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Aplica colormap

    # Sobrepõe heatmap na imagem original
    superimposed_img = heatmap * alpha + original_img
    return superimposed_img.astype(np.uint8)

# Função para plotar múltiplos heatmaps
def display_multiple_gradcams(original_imgs, heatmaps, alpha=0.4, save_path='/home/lucas/PIBIC/plot/multiple_gradcams.png'):
    num_imgs = len(original_imgs)
    cols = 4  # 4 colunas
    rows = 2  # 2 linhas para 8 imagens

    plt.figure(figsize=(16, 8))  # Ajusta tamanho da figura

    for i in range(num_imgs):
        superimposed_img = superimpose_heatmap(original_imgs[i], heatmaps[i], alpha=alpha)
        plt.subplot(rows, cols, i + 1)  # Define a posição no grid
        plt.imshow(superimposed_img)
        plt.axis('off')

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def return_gradcam_heatmap(img_array, model, pred_index=None):
    #transformar para batch
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)  

    heatmaps = []
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Conv2D):
            continue

        layer_name = layer.name

        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        heatmaps.append(heatmap.numpy())
    return heatmaps

def errors_heatmap(test_df, classifier, save_path=None):
    """
    Considerando 4 imagens: 
    - 1 True True; 
    - 1 True False; 
    - 1 False False; 
    - 1 False True. 
    """
    plt.figure(figsize=(16,8))

    filepaths = test_df['path_image'].values
    labels = test_df['class'].values

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    encoder = classifier.get_encoder()

    num_imgs = len(dataset)
    data = iter(dataset)
    corrected_imgs = []
    wrong_imgs = []
    corrected = 0
    wrong = 0

    for i in range(num_imgs):
        image, label = next(data)
        image_path = image.numpy().decode('utf-8')  # Decodifica caminho da imagem

        # Carrega a imagem e normaliza
        image = tf.keras.utils.load_img(image_path, target_size=(64, 64)) 
        image = tf.keras.utils.img_to_array(image)
        image = image / 255.0

        # Expande dimensão para batch
        image_with_batch = np.expand_dims(image, axis=0)

        # Predição
        predicted_probs = classifier.model.predict(image_with_batch, verbose=0)
        predicted_label = np.argmax(predicted_probs, axis=1)[0]  # Pega o label predito

        true_label = labels[i]
        predicted_label, true_label = int(predicted_label), int(true_label)
        # Verifica se é previsão correta
        if predicted_label == true_label:
            if true_label == 0 and len(corrected_imgs) == 0:
                corrected_imgs.append(image)
                corrected += 1
            elif true_label == 1 and len(corrected_imgs) == 1:
                corrected_imgs.append(image)
                corrected += 1
        else:
            if true_label == 0 and len(wrong_imgs) == 0:
                wrong_imgs.append(image)
                wrong += 1
            elif true_label == 1 and len(wrong_imgs) == 1:
                wrong_imgs.append(image)
                wrong += 1

        # Se já coletou 4 imagens no total, para
        if len(corrected_imgs) + len(wrong_imgs) >= 4:
            break
    
    rows = 4
    cols_name = [l for l in encoder.layers if isinstance(l, tf.keras.layers.Conv2D)]
    cols = len(cols_name) + 1

    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(20, 15))
    axs[0, 0].imshow(corrected_imgs[0])
    axs[1, 0].imshow(corrected_imgs[1])
    axs[2, 0].imshow(wrong_imgs[0])
    axs[3, 0].imshow(wrong_imgs[1])

    axs[0, 0].set_title('Previsão Correta')
    axs[1, 0].set_title('Previsão Correta')
    axs[2, 0].set_title('Previsão Incorreta')
    axs[3, 0].set_title('Previsão Incorreta')

    imgs = corrected_imgs + wrong_imgs
    
    for i, img in enumerate(imgs):
        heatmaps = return_gradcam_heatmap(img, encoder)
        print("Tamanho dos heatmaps:", len(heatmaps))
        for col in range(cols-1):
            axs[i, col+1].imshow(superimpose_heatmap(img, heatmaps[col], 0.4))
            #axs[i, col+1].set_title(cols_name[i].name)

    fig.show()
    
    if save_path is not None:
        fig.savefig(save_path)


