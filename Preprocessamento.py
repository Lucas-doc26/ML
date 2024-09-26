import pandas as pd
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def preprocessamento_dataframe(caminho_csv: str, autoencoder: bool = False, data_algumentantation:bool = True):
    dataframe = pd.read_csv(caminho_csv)

    img_width, img_height = 64, 64
    batch_size = 32

    transform = A.Compose([
            A.ChannelDropout(channel_drop_range=(1, 2), fill_value=0, p=0.5),
            A.Rotate(limit=50, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True, p=1),
            A.AdvancedBlur(blur_limit=(7,9), noise_limit=(0.75, 1.25), p=1)
    ])

    def albumentations(img):
        data = {"image": normalize_image(img)}
        augmented = transform(**data)  #** para expandir o dicionário
        return augmented['image']
        
    datagen = ImageDataGenerator(preprocessing_function=albumentations if data_algumentantation else normalize_image)

    class_mode = 'input' if autoencoder else 'binary'

    Dataframe_preprocessado = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='caminho_imagem',
        y_col='caminho_imagem' if autoencoder else 'classe',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True
    )

    return Dataframe_preprocessado, dataframe

def preprocessamento(caminho: str, proporcao_treino: float = 0.6, proporcao_teste: float = 0.2, proporcao_validacao: float = 0.2, autoencoder: bool = True, data_algumentantation = True):
    """
    Ao passar um dataFrame .csv, ele irá retornar geradores de dados para treino, teste e validação + os 3 .csv dividos igualmente os geradores.
    
    Parâmetros:
        caminho (str): Caminho para o arquivo CSV.
        proporcao_treino (float): Proporção de dados de treino.
        proporcao_teste (float): Proporção de dados de teste.
        proporcao_validacao (float): Proporção de dados de validação.
        autoencoder (bool): Se True, prepara os dados para um autoencoder (class_mode='input').
                            Se False, prepara os dados para classificação binária (class_mode='binary').
    
    Retorna:
        treino_gerador, validacao_gerador, teste_gerador, treino, validacao, teste
    """
    dataframe = pd.read_csv(caminho)

    treino, teste = train_test_split(dataframe, test_size=proporcao_teste, random_state=42)
    treino, validacao = train_test_split(treino, test_size=proporcao_validacao / (1 - proporcao_teste), random_state=42)

    img_width, img_height = 256, 256
    batch_size = 32

    treino_datagen = ImageDataGenerator(preprocessing_function=albumentations if data_algumentantation else normalize_image)
    validacao_datagen = ImageDataGenerator(preprocessing_function=albumentations if data_algumentantation else normalize_image)
    teste_datagen = ImageDataGenerator(preprocessing_function=normalize_image)

    class_mode = 'input' if autoencoder else 'binary'

    treino_gerador = treino_datagen.flow_from_dataframe(
        dataframe=treino,
        x_col='caminho_imagem',
        y_col='caminho_imagem' if autoencoder else 'classe', #Usar a imagem como saída se for autoencoder
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode,  #Class mode baseado no parâmetro autoencoder
        shuffle=False
    )

    validacao_gerador = validacao_datagen.flow_from_dataframe(
        dataframe=validacao,
        x_col='caminho_imagem',
        y_col='caminho_imagem' if autoencoder else 'classe',  
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode, 
        shuffle=False
    )

    teste_gerador = teste_datagen.flow_from_dataframe(
        dataframe=teste,
        x_col='caminho_imagem',
        y_col='classe',  
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode, 
        shuffle=False
    )

    return treino_gerador, validacao_gerador, teste_gerador, treino, validacao, teste

#DataAugmentation com Albumentations 
transform = A.Compose([
            A.RandomRain(
                drop_length=8, drop_width=1,
                drop_color=(180, 180, 180),  blur_value=5,brightness_coefficient=0.8, p=1
            ),
            A.GaussNoise(var_limit=(0.0, 0.0007), mean=0, p=0.15),
            A.ChannelShuffle(p=0.15),
            A.Rotate(limit=40, p=0.15),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True, p=0.15),
            A.AdvancedBlur(blur_limit=(7,9), noise_limit=(0.75, 1.25), p=0.15),
            A.Resize(height=256, width=256)
    ])

#Funções auxiliares ao preprocessamento
def normalize_image(img):
    return img / 255.0

def albumentations(img):
            data = {"image": normalize_image(img)}
            augmented = transform(**data)  #** para expandir o dicionário
            return augmented['image']