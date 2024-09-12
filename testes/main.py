import keras
import tensorflow as tf

from segmentandoDatasets import *

segmentando_datasets(10000,10000,10000)

from preprocessamento import *

csv_file = 'Datasets_csv/df_PUC.csv'
_,_,_, x_treino, y_treino, x_teste, y_teste, x_val, y_val = preprocessamento(csv_file)

from keras.layers import Input, Flatten, Dense, Reshape
from keras.models import Sequential

encoder = keras.models.Sequential([
    keras.layers.Reshape([256,256,3], input_shape=[256,256,3]),
    keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),  
])

decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=2, padding="same", activation="relu", input_shape=[16, 16, 128]),
    keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding="same", activation="relu"),
    keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, padding="same", activation="relu"),
    keras.layers.Conv2DTranspose(16, kernel_size=(3, 3), strides=2, padding="same", activation="relu"),
    keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), padding="same", activation="sigmoid"),
])

autoencoder = keras.models.Sequential([encoder, decoder])
autoencoder.compile(loss="binary_crossentropy", optimizer='adam')

history = autoencoder.fit(x_teste,x_teste, epochs=10, batch_size=8, validation_data=(x_val,x_val))

pd.DataFrame(history.history).plot()

autoencoder.save("Modelos_keras/Autoencoder.keras")
autoencoder.save_weights("weights_finais/Autoencoder.weights.h5")


imagens_codificadas = encoder.predict(x_teste)
imagens_decodificadas = decoder.predict(imagens_codificadas)

shape_imagem_codificada = imagens_codificadas.reshape((imagens_codificadas.shape[0], -1))

classificador = Sequential([
    Dense(128, activation='relu', input_shape=(shape_imagem_codificada.shape[1],)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax') 
])

classificador.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classificador.fit(shape_imagem_codificada, y_teste, epochs=10, batch_size=8, validation_split=0.1)
teste_loss, teste_accuracy = classificador.evaluate(shape_imagem_codificada, y_teste)
print(f'Teste loss: {teste_loss}, Teste accuracy: {teste_accuracy}')

classificador.save("Modelos_keras/classificador.h5")
classificador.save_weights("weights_finais/classificador.weights.h5")
