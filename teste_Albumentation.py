import tensorflow as tf
import numpy as np
import pandas as pd
import albumentations as A
import tensorflow_datasets as tfds

from tensorflow.keras import layers

from segmentandoDatasets import segmentando_datasets
from Preprocessamento import preprocessamento_dataframe

from tensorflow.keras.preprocessing.image import ImageDataGenerator


from Modelos import Autoencoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define as transformações de data augmentation
transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.ColorJitter(),
    A.Normalize()
])

def data_augmentation(image, label):
    image = transform(image=image)['image']
    return np.expand_dims(image, axis=0), np.expand_dims(image, axis=0)

# Carregue os dados de treinamento
train_dataset, dataframe = preprocessamento_dataframe('Datasets_csv/df_PUC.csv', autoencoder=True)

# Aplique o data augmentation ao conjunto de treinamento
train_dataset = tf.data.Dataset.from_generator(
    lambda: (data_augmentation(image, image) for image, image in train_dataset),
    output_types=(tf.float32, tf.float32),
    output_shapes=((1, 64, 64, 3), (1, 64, 64, 3))
)

autoencoder = Autoencoder()
model = autoencoder.model

model.compile(loss='mse', optimizer='adam')

# Treine o modelo de autoencoder
model.fit(train_dataset, epochs=10, batch_size=32)