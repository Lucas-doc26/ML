from utils.datasets import *
from utils.models import *
from pathlib import Path


autoencoder_base = ['Kyoto', 'CNR', 'PKLot']
epochs = [200, 50, 50]

for i, base in enumerate(autoencoder_base):
    autoencoder = AutoencoderGenerator()

    train, _ = preprocessing_dataframe(path_csv=f'CSV/{base}/{base}_autoencoder_train.csv', autoencoder=True, data_algumentantation=False, input_shape=(64,64))
    validation, _ = preprocessing_dataframe(path_csv=f'CSV/{base}/{base}_autoencoder_validation.csv', autoencoder=True, data_algumentantation=False, input_shape=(64,64))
    test, _ = preprocessing_dataframe(path_csv=f'CSV/{base}/{base}_autoencoder_test.csv', autoencoder=True, data_algumentantation=False, input_shape=(64,64))


    autoencoder.load_model(modelo='Modelos/Modelo_Kyoto-9/Modelo-Base/Estrutura/Modelo_Kyoto-9.keras')
    autoencoder.dataset(train, validation, test)
    autoencoder.model_compile()
    autoencoder.set_model_name(f'Modelo_Kyoto-9')
    autoencoder.train_autoencoder(save=True, autoencoder_base=base, epochs=epochs[i], batch_size=4)

    del train, validation, test, autoencoder
