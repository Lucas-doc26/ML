import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Tuple, Optional, Union
from pathlib import Path
from utils.preprocessing import process_image_for_heatmap

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