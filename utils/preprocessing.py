import albumentations as A
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from typing import Tuple
from tensorflow.keras.models import Model


def data_augmentation_kyoto(kyoto_path):
    """
    Função para realizar data augmentation no dataset Kyoto
    """
    transform1 = A.Compose([
            A.GaussNoise(var_limit=(0.0, 0.0007), mean=0, p=1),
            A.Rotate(limit=180, p=1),
    ])
    transform2 = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True, p=1),
                A.AdvancedBlur(blur_limit=(7,9), noise_limit=(0.75, 1.25), p=1),
        ])
    transform3 = A.Compose([
                A.ChannelShuffle(p=1),
    ])
    transform4 = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True, p=1),
                A.AdvancedBlur(blur_limit=(7,9), noise_limit=(0.75, 1.25), p=1),
    ])

    if not os.path.isdir(os.path.join(kyoto_path, 'dataAug')):
        os.makedirs(os.path.join(kyoto_path, 'dataAug'))

    for img_name in os.listdir(kyoto_path):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):  # Verifica se é uma imagem
            img_path = os.path.join(kyoto_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Não foi possível ler a imagem: {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img1 = transform1(image=img)['image']
            img2 = transform2(image=img)['image']
            img3 = transform3(image=img)['image']
            img4 = transform4(image=img)['image']

            # Converter de volta para BGR para salvar com cv2
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
            img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)

            # Salvar as imagens usando cv2.imwrite
            base_name = os.path.splitext(img_name)[0]
            cv2.imwrite(os.path.join(kyoto_path, 'dataAug', f'kyoto_{base_name}_1.jpg'), img1)
            cv2.imwrite(os.path.join(kyoto_path, 'dataAug', f'kyoto_{base_name}_2.jpg'), img2)
            cv2.imwrite(os.path.join(kyoto_path, 'dataAug', f'kyoto_{base_name}_3.jpg'), img3)
            cv2.imwrite(os.path.join(kyoto_path, 'dataAug', f'kyoto_{base_name}_4.jpg'), img4)
            
            print(f"Processada imagem: {img_name}")

def normalize(image):
    """
    Função para normalizar as imagens
    """
    image = np.clip(image, 0, 1)  # Garante que a imagem esteja no intervalo [0, 1]
    if isinstance(image, tf.Tensor): #Caso seja um tensor, ele transforma em np para evitar uso de vram
        image = image.numpy()
        return (image - image.min()) / (image.max() - image.min()) if image.max() != image.min() else image

def preprocess_images(target_shape, csv):
    """
    Função para pré-processar as imagens de um arquivo .csv
    """
    images = []
    df = pd.read_csv(csv)
    for path in df["path_image"]:
        img = cv2.imread(path)  # carrega como BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converte para RGB se quiser
        img = normalize(img)
        images.append(img)

    X_processed = []
    for img in images:
        img_resized = cv2.resize(img, target_shape)
        X_processed.append(img_resized)
    #X_processed = n    p.expand_dims(X_processed, -1)  # (N, 28, 28, 1)
    return np.array(X_processed).astype("float32") / 255.0

def map_classes_to_binary(classes):
    return np.array([1 if classe == '1' else 0 for classe in classes])

def map_classes(classes):
    return np.array([1 if classe == 'Empty' else 0 for classe in classes])

def process_image_for_heatmap(input_img: np.ndarray, input_img_shape: Tuple[int, int, int], activation_model: Model, encoder: Model, decoder: Model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processa uma imagem para obter a imagem original, reconstruída e mapa de ativação.

        Args:
            input_img: Imagem de entrada
            input_img_shape: Shape da imagem de entrada
            activation_model: Modelo de ativação
            encoder: Modelo de encoder
            decoder: Modelo de decoder

        Returns:
            Tupla contendo (imagem original, imagem reconstruída, mapa de ativação)
        """

        # Redimensionar a imagem se necessário
        if input_img.shape != input_img_shape:
            input_img = tf.image.resize(input_img, (input_img_shape[0], input_img_shape[1]))

        # Expandir dimensões para batch
        input_img_batch = np.expand_dims(input_img, axis=0)

        # Obter ativações e mapa de calor
        activations = activation_model.predict(input_img_batch)
        activation_map = np.mean(activations[0], axis=-1)

        # Obter codificação latente e reconstrução
        _, _, z = encoder.predict(input_img_batch)
        reconstructed_img = decoder.predict(z)[0]

        return input_img, reconstructed_img, activation_map