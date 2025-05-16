import os
import shutil
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from utils.models.autoencoder_generator import AutoencoderGenerator
from utils.gpu import clear_session
from utils.preprocessing import map_classes_to_binary, map_classes, preprocessing_dataframe
from pathlib import Path
import utils
from utils.path_manager import *
from utils.view import *

tf.config.optimizer.set_jit(False)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"


class ClassifierGenerator:
    def __init__(self, encoder:AutoencoderGenerator=None, weights:Path=None, model_name:str=None, autoencoder_base:str='Sem-Peso', path:Path=None):
        self.encoder = encoder
        self.model_name = model_name

        self.model = self.build_model(self.encoder)
        self.load_weights(weights)

        if self.encoder != None:
            self.compile()

        self.train = None
        self.validation = None
        self.test = None
        self.autoencoder_base = autoencoder_base
        
        if model_name is None:
            raise ValueError("O name do modelo não foi definido.")
        else:
            self.verify_dirs()

    def build_model(self, encoder):
        """Build the model architecture"""
        if encoder != None:
            #eu congelo as camadas do encoder, não faço fine-tunning
            for layer in self.encoder.layers:
                layer.trainable = False
            encoder.trainable = False

            #crio o classificador com o enconder
            classifier = keras.models.Sequential([
                    self.encoder,  
                    keras.layers.Dropout(0.2),  
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(128,activation='relu'),  
                    keras.layers.Dense(2, activation='softmax')  
                ], name=f'classificador{self.model_name}')
            
        else:
            classifier = keras.models.Sequential([ 
                    keras.layers.Dropout(0.2),  
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(128,activation='relu'),  
                    keras.layers.Dense(2, activation='softmax')  
                ], name=f'classificador{self.model_name}')
            
        return classifier
    
    def verify_dirs(self):
        save_dir = os.path.join("Modelos", self.model_name)
        root_dir = os.path.join(save_dir, f"Classificador-{self.autoencoder_base}")
        dir_model = os.path.join(root_dir, "Estrutura")
        dir_weights = os.path.join(root_dir, "Pesos")

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)
        if not os.path.isdir(dir_model):
            os.mkdir(dir_model)
        if not os.path.isdir(dir_weights):
            os.mkdir(dir_weights)

    def save_model(self, save_dir=''):
        if save_dir == '':
            path_save_model = f'Modelos/{self.model_name}/Classificador-{self.autoencoder_base}/Estrutura/Classificador_{self.model_name}.keras'
        else:
            path_save_model = os.path.join(save_dir, f'Modelos/{self.model_name}/Classificador-{self.autoencoder_base}/Estrutura/Classificador_{self.model_name}.keras')
        self.model.save(path_save_model)
    
    def set_name(self, name):
        self.model_name = name
    
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False)

    def load_weights(self, weights):
        if weights == None:
            print("Criação do modelo de classificação sem pesos")
        elif weights == False:
            print("Criação do modelo sem carregar os pesos")
        else:
            try:
                self.model.load_weights(weights, skip_mismatch=True)
                print("Pesos carregados com sucesso")
            except Exception as e:
                print(f"Erro ao carregar os pesos: {e}")
        clear_session()

    def train_classifier(self, save=False, epochs=10, batch_size=64, n_batches=None, classifier_base=None, weights=True):
        checkpoint_path = 'Pesos/Pesos_parciais/weights-improvement-{epoch:02d}-{val_loss:.2f}.weights.h5'

        cp_callback = ModelCheckpoint(
            filepath=checkpoint_path, 
            save_weights_only=True, 
            monitor='accuracy', 
            mode='max', 
            save_best_only=True, 
            verbose=1
        )

        now = datetime.datetime.now().strftime("%d%m%y-%H%M")
        log_dir = os.path.join(
            'Modelos',
            self.model_name,
            f'Classificador-{self.autoencoder_base}',
            'logs',
            f'fit-{now}'
        )
        
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print(f"type(classifier): {type(self.model)}")

        history = self.model.fit(
            self.train, 
            epochs=epochs, 
            callbacks=[cp_callback, tensorboard_callback], 
            batch_size=8,
            validation_data=self.validation
        )

        if os.path.isdir("Pesos/Pesos_parciais"):
            shutil.rmtree("Pesos/Pesos_parciais", ignore_errors=True)
            os.makedirs("Pesos/Pesos_parciais", exist_ok=True)

        if save:
            save_dir = os.path.join("Modelos", self.model_name)
            root_dir = os.path.join(save_dir, f"Classificador-{self.autoencoder_base}")
            dir_model = os.path.join(root_dir, "Estrutura")
            dir_weights = os.path.join(root_dir, "Pesos")
            dir_weights_base = os.path.join(dir_weights, f'Treinado_em_{classifier_base}')
            dir_images = os.path.join(save_dir, 'Plots')

            os.makedirs(root_dir, exist_ok=True)
            os.makedirs(dir_model, exist_ok=True)
            os.makedirs(dir_weights, exist_ok=True)
            os.makedirs(dir_images, exist_ok=True)

            # salvo o modelo
            self.model.save(os.path.join(dir_model, f'Classificador_{self.model_name}.keras'))

            if n_batches is not None:
                os.makedirs(dir_weights_base, exist_ok=True)
                self.model.save_weights(f"{dir_weights_base}/Classificador_{self.model_name}_batches-{n_batches}.weights.h5")
            else:
                self.model.save_weights(f"{dir_weights_base}/Classificador_{self.model_name}.weights.h5")  

            save_history = os.path.join(dir_images, 'History')
            os.makedirs(save_history, exist_ok=True)
            plot_history_batch(history, save_history, self.model_name, classifier_base, self.autoencoder_base, n_batches) 
            
        return history

    def dataset(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test

    def set_train(self, train):
        self.train = train

    def set_test(self, test):
        self.test = test

    def predict(self, test_csv: pd.DataFrame):
        predict_np = self.model.predict(self.test)
        predicts = np.argmax(predict_np, axis=1)

        print(predicts)

        y_true = map_classes_to_binary(test_csv['class'])

        #plot_confusion_matrix(y_verdadeiro, predicoes, ['Empty', 'Occupied'], title=f'{self.model_name}')

        accuracy = accuracy_score(y_true, predicts)

        return predict_np, accuracy
    
    def load_model(self, model_path:str, weights_path:str):
        model_loaded = tf.keras.models.load_model(model_path)
        self.model = model_loaded
        self.load_weights(weights_path)

        self.model.summary(show_trainable=True)

        return self.model


#Exemplo de uso:
#classificador = GeradorClassificador(encoder=encoder, pesos="pesos.weights.h5") -> crio o classificador encima do encoder e seus pesos
#classificador.Dataset(train, validation, test)
#classificador.treinamento() 
#classificador.predicao(test_df) -> cria a matriz de confusão

"""------------------Funções para usar diversos classificadores----------------------"""
def create_classifiers(n_models=10, model_name=None, autoencoder_base='Sem-Peso', train=None, validation=None, test=None, 
                         input_shape=(64,64,3)):
    for i in range(n_models):  
        autoencoder = AutoencoderGenerator(input_shape=input_shape)

        clear_session()

        autoencoder.load_model(f'Modelos/{model_name}-{i}/Modelo-Base/Estrutura/{model_name}-{i}.keras')
        encoder = autoencoder.encoder

        weights = f'Modelos/{model_name}-{i}/Modelo-Base/Pesos/{model_name}-{i}_Base-{autoencoder_base}.weights.h5'
        classifier = ClassifierGenerator(encoder=encoder, weights=weights, autoencoder_base=autoencoder_base, model_name=f'{model_name}-{i}')
        classifier.save_model()

        clear_session()

        del autoencoder, encoder, classifier

def train_per_batch(model_name, classifier_base, autoencoder_base, train_csv, validation, test, test_df, save=True, epochs=10, weights=True, input_shape=(64,64,3)):
    #name da base de train do classificador
    name = return_name_csv(train_csv)
    test_base = return_name_df(test_df)

    batch_dir = f"CSV/{name}/batches"
    batches = sorted(os.listdir(batch_dir), key=lambda x: int(x.split("batch-")[1].split(".")[0]))
    print(batches)
    accuracies = []
    n_batches = [64,128,256,512,1024] 

    if not weights:
        autoencoder_base = 'Sem-Peso'

    # crio o Modelo/Classificador/Resultados
    if not os.path.isdir(f'Modelos/{model_name}/Classificador-{autoencoder_base}/Resultados'):
        os.makedirs(f'Modelos/{model_name}/Classificador-{autoencoder_base}/Resultados')

    # crio o Modelo/Classificador/Resultados/Treinados_em_PUC
    dir_resultados_base = f'Modelos/{model_name}/Classificador-{autoencoder_base}/Resultados/Treinados_em_{classifier_base}'
    if not os.path.isdir(dir_resultados_base):
        os.makedirs(dir_resultados_base)
    
    #para cada um dos meus batches sizes, eu vou pegar seu correspondendo em csv 
    for batch, batch_size in zip(batches, n_batches):
        clear_session()

        autoencoder = AutoencoderGenerator(input_shape=input_shape)
        autoencoder.load_model(f'Modelos/{model_name}/Modelo-Base/Estrutura/{model_name}.keras', weights=False)
        encoder = autoencoder.encoder

        #caso eu queira carregar o pesos do encoder
        if weights:
            classifier = ClassifierGenerator(
                encoder=encoder, 
                weights=f'Modelos/{model_name}/Modelo-Base/Pesos/{model_name}_Base-{autoencoder_base}.weights.h5',
                model_name=model_name, 
                autoencoder_base=autoencoder_base
            )
        else:
            classifier = ClassifierGenerator(
                encoder=encoder,
                weights=False,
                model_name=model_name,
                autoencoder_base=autoencoder_base
            )

        classifier.compile()
        classifier.dataset(train=None, validation=validation, test=test)

        train, _ = preprocessing_dataframe(os.path.join(batch_dir, batch), autoencoder=False)
        classifier.set_train(train)
        print(classifier)
        classifier.train_classifier(epochs=epochs, save=save ,n_batches=batch_size, classifier_base=classifier_base, weights=weights)
        predicts_np, accuracy = classifier.predict(test_df)
        accuracies.append(accuracy)

        results_dir = os.path.join(dir_resultados_base, test_base)
        os.makedirs(results_dir, exist_ok=True)

        file = os.path.join(results_dir, f"batches-{batch_size}.npy")
        np.save(file, predicts_np)

        dir_prec = f"Modelos/{model_name}/Classificador-{autoencoder_base}/Precisao"
        os.makedirs(dir_prec, exist_ok=True)

        dir_prec_base = os.path.join(dir_prec, f'Treinado_em_{classifier_base}')
        recreate_folder(dir_prec_base)

        print(f"Saving accuracy for {test_base}")
        dir_file_path = os.path.join(dir_prec_base, f'precisao-{test_base}.txt')
        with open(dir_file_path, 'w') as f:
            for acc in accuracies:
                f.write(f"{acc}\n")
        
        clear_session() 

    dir_graf = f'Modelos/{model_name}/Plots/Graficos'
    os.makedirs(dir_graf, exist_ok=True)

    dir_graph_university = os.path.join(dir_graf, f'Treinado_em_{classifier_base}')
    os.makedirs(dir_graph_university, exist_ok=True)

    graphic_accuracy_per_batch(
        batches=n_batches,
        accuracies=accuracies,
        model_name=model_name,
        train_base=classifier_base,
        test_base=test_base,
        autoencoder_base=autoencoder_base,
        save_path=dir_graph_university
    )

    print("Final accuracies:", accuracies)

def train_all_models_per_batch(model_name, classifier_base, autoencoder_base, train_csv, validation, test, test_df, save=True, epochs=10, input_shape=(64,64,3)):
    path_models = "Modelos"
    models = os.listdir(path_models)
    models_to_train = []
    for model in models:
        if os.path.exists(os.path.join(path_models, model)):
            if model_name in model and 'Fusoes' not in model:
                model_base = os.path.join(path_models, model, 'Modelo-Base')

                files = os.listdir(model_base)
                structure = next((f for f in files if 'estrutura' in f.lower()), None)
                weights = next((f for f in files if 'peso' in f.lower()), None)
                log = next((f for f in files if 'log' in f.lower()), None)
                
                if not structure or not weights:
                    print(f"Aviso: Arquivos necessários não encontrados em {model_base}")
                    continue
                    
                print(f"Peso encontrado: {weights}")
                m = os.listdir(os.path.join(model_base, structure))
                print(f"Arquivos na estrutura: {m}")
                dir_model = os.path.join(model_base, structure, m[0])
                models_to_train.append(dir_model)
            else:
                pass
        else:
            print(f"O diretório {model} não existe.")

    for i, model in enumerate(sorted(models_to_train)):
        name = model_name + f"-{i}"
        #train cada um dos models em batches 
        train_per_batch(name, classifier_base, autoencoder_base, train_csv, validation, test, test_df, save, epochs, True, input_shape)

    #faço o plot comparando os models
    utils.view.graphics.models_comparison(path_save=os.path.join(path_models, "Plots"), model_name=model_name, classifier_base=classifier_base, 
        test_base=classifier_base, autoencoder_base=autoencoder_base)

def test_model_per_batch(model_name, test, test_df, classifier_base, weights=True, autoencoder_base='Kyoto'):

    if not weights:
        autoencoder_base='Sem-Peso'

    classifier = ClassifierGenerator(model_name=model_name, autoencoder_base=autoencoder_base)

    test_base = return_name_df(test_df)

    classifier.set_test(test)

    if not os.path.isdir(f'Modelos/{model_name}/Classificador-{autoencoder_base}/Resultados'):
        recreate_folder(f'Modelos/{model_name}/Classificador-{autoencoder_base}/Resultados')
        
    accuracies = []
    batches = [64,128,256,512,1024]

    structure = f'Modelos/{model_name}/Classificador-{autoencoder_base}/Estrutura/Classificador_{model_name}.keras'
    for batch_size in [64,128,256,512,1024]:
        dir_peso = f'Modelos/{model_name}/Classificador-{autoencoder_base}/Pesos/Treinado_em_{classifier_base}/Classificador_{model_name}_batches-{batch_size}.weights.h5'
        weights = dir_peso
        classifier.load_model(structure, weights)
        predicts_np, accuracy = classifier.predict(test_df)

        #Modelo-Kyoto-1/Classificador-CNR/Resultados/Treinados_em_PUC/UFPR04/batches-64-npy
        dir_base =  f"Modelos/{model_name}/Classificador-{autoencoder_base}/Resultados/Treinados_em_{classifier_base}/{test_base}"
        
        if not os.path.isdir(dir_base):
            os.makedirs(dir_base)
            print(f"Diretório criado: {dir_base}")
        else:
            print(f"Diretório {dir_base} já existia!")

        file = os.path.join(dir_base, f'batches-{batch_size}.npy')
        np.save(file, predicts_np)
        clear_session()
        accuracies.append(accuracy)
        del weights, predicts_np, accuracy


    print(accuracies)

    #save as precisões no arquivo
    #print(test_base)
    print(test_base)
    file_path =  f'Modelos/{model_name}/Classificador-{autoencoder_base}/Precisao/Treinado_em_{classifier_base}', f'precisao-{test_base}.txt'
    with open(file_path, 'w') as f:
        for prec in accuracies:
            f.write(f"{prec}\n")

    dir_graf_facul = f'Modelos/{model_name}/Plots/Graficos/Treinado_em_{classifier_base}'

    graphic_accuracy_per_batch(batches=batches, accuracies=accuracies, model_name=model_name, train_base=classifier_base, test_base=test_base, autoencoder_base=autoencoder_base, save_path=dir_graf_facul)

def test_all_models_per_batch(model_name, test, test_df, classifier_base, autoencoder_base):
    #classificador = GeradorClassificador()
    path_models = "Modelos"
    models = os.listdir(path_models) #todos as pastas no dir Modelos
    models_to_test = []
    print("Os modelos em pastas são:", models)
    for model in models:
        if os.path.exists(os.path.join(path_models, model)):
            if model_name in model and 'Fusoes' not in model: #Se name modelo tiver em modelo e fusão não
                model_base = os.path.join(path_models, model, f'Classificador-{autoencoder_base}') #Modelo_Kyoto-0/Classificador_Modelo
                print("A estrutura disponível é:" , os.listdir(os.path.join(modelo_base, 'Estrutura')))
                structure = os.listdir(os.path.join(model_base, 'Estrutura'))[0]
                models_to_test.append(structure)
            else:
                pass
        else:
            print(f"O diretório {model} não existe.")
    
    print(models_to_test)
    for model in models_to_test:
        name = return_model_name(model)
        print(f"Testando o modelo: {name}")
        test_model_per_batch(name, test, test_df, classifier_base, weights=True, autoencoder_base=autoencoder_base)
        clear_session()

    test_base = return_name_df(test_df)

    models_comparison(
        path_save=os.path.join(path_models, "Plots"),
        model_name=model_name,
        classifier_base=classifier_base,
        test_base=test_base,
        autoencoder_base=autoencoder_base
    )


