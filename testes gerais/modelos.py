import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import keras
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV3Small
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models, losses

def modelo_mobileNetV3Small(trainable_conv:int):
    """
    Cria o modelo com base no MobileNetV3Small, usando os pesos da imagenet\n
    """
    
    img_width, img_height = 256, 256 
    base_modelo = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3), pooling='avg')
    for layer in base_modelo.layers:
        layer.trainable = False

    if trainable_conv != 0:
        conv_layers = [layer for layer in base_modelo.layers if isinstance(layer, Conv2D)]
        for layer in conv_layers[-trainable_conv:]:
            layer.trainable = True

    modelo = Sequential([
        base_modelo,
        Flatten(),
        Dense(256, activation='relu'),  
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])

    return modelo

def carrega_modelo(caminho:str, pesos:str):
    """
    Carrega o modelo passando o modelo e seus pesos
    """
    modelo = keras.models.load_model(caminho)
    modelo.load_weights(pesos)

    return modelo

