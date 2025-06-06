import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from utils.view import graphic_accuracy_per_batch
from utils.path_manager import PathManager
path = '/home/lucas/PIBIC'

class FusionRule:
    def __init__(self, path_manager:PathManager):
        self.path_manager = path_manager

    def apply_fusion(self, model_name, batch_size, autoencoder_base, train_base, test_base):
        #me retorna o npy
        base = np.load(self.path_manager.get_prediction_path(model_name=model_name, batch_size=batch_size, model_index=0, train_base=train_base, test_base=test_base, autoencoder_base=autoencoder_base))
        
        if isinstance(self, SumFusion) or isinstance(self, VoteFusion):
            result = np.zeros_like(base)
        else:
            result = np.ones_like(base) 

        for i in range(10):
            npy = self.path_manager.get_prediction_path(model_name=model_name, batch_size=batch_size, model_index=i, train_base=train_base, test_base=test_base, autoencoder_base=autoencoder_base)
            array = np.load(npy)

            #Faz a fusão
            result = self._combine_predictions(result, array)
            del array

        result = np.argmax(result, axis=1)
        return result

    def run(self, model_name, autoencoder_base, train_bases, test_bases):
        path = self.path_manager.get_base_path()
        batches = [64, 128, 256, 512, 1024]
        results_csv = []
        fusion_name = self.__class__.__name__

        for train in train_bases:
            for test in test_bases:
                results = []
                for batch in batches:
                    result_fusion = self.apply_fusion(
                        model_name=model_name, 
                        batch_size=batch, 
                        autoencoder_base=autoencoder_base, 
                        train_base=train, 
                        test_base=test
                    ) 

                    if test == train:
                        path_csv = self.path_manager.get_csv_path(base=test, type='_test')
                    else:
                        path_csv = self.path_manager.get_csv_path(base=test, type=None)
                    
                    df = pd.read_csv(path_csv)
                    y_true = mapear(df['class'])  # Confirma que mapear está definida!

                    acc = accuracy_score(y_true, result_fusion)

                    results.append(acc)
                    results_csv.append({
                        'Base do Autoencoder': autoencoder_base,
                        'Base de Treino': train,
                        'Base de Teste': test,
                        'Acuracia': format(acc, '.3f'),
                        'Batch': int(batch)
                    })

                graphic_accuracy_per_batch(
                    batches=batches, 
                    accuracies=results, 
                    model_name=f'Soma entre os diferentes {model_name}',
                    save_path=os.path.join(
                        path, 
                        f'Modelos/Fusoes-{model_name}/Autoencoder-{autoencoder_base}/Treinados_em_{train}/Grafico_batches_{fusion_name}-{autoencoder_base}-{train}-{test}.png'
                    ), 
                    autoencoder_base=autoencoder_base,
                    train_base=train, 
                    test_base=test
                )

        df_results = pd.DataFrame(results_csv)  
        path_table = os.path.join(
            path,
            f'resultados/{model_name}/tabela_{fusion_name}-{autoencoder_base}.csv'
        )
        df_results.to_csv(path_table, index=False)

    def _combine_predictions(self, result, array):
        raise NotImplementedError("A implementação deve ser feita na classe filha")
    
class SumFusion(FusionRule):
    def _combine_predictions(self, result, array):
        return result + array

class MultFusion(FusionRule):
    def _combine_predictions(self, result, array):
        return result * array

class VoteFusion(FusionRule):
    def _combine_predictions(self, result, array):
        array_result = np.zeros_like(array)
        for j in range(len(array)):
            value = array[j][0]
            if value > 0.5:
                array_result[j] = [1, 0]  # Voto para classe 0
            else:
                array_result[j] = [0, 1]  # Voto para classe 1
        return result + array_result

def mapear(classes):
    return np.array([1 if classe == 1 else 0 for classe in classes])

path = PathManager('/home/lucas/PIBIC')
sum = SumFusion(path)
mult = MultFusion(path)
voto = VoteFusion(path)

#CNR
sum.run(model_name='Modelo_Kyoto',
        autoencoder_base='CNR', 
        train_bases=['PUC', 'UFPR04', 'UFPR05'], 
        test_bases=['PUC', 'UFPR04', 'UFPR05'])

mult.run(model_name='Modelo_Kyoto',
        autoencoder_base='CNR', 
        train_bases=['PUC', 'UFPR04', 'UFPR05'], 
        test_bases=['PUC', 'UFPR04', 'UFPR05'])

voto.run(model_name='Modelo_Kyoto',
        autoencoder_base='CNR', 
        train_bases=['PUC', 'UFPR04', 'UFPR05'], 
        test_bases=['PUC', 'UFPR04', 'UFPR05'])

#PKLot
sum.run(model_name='Modelo_Kyoto',
        autoencoder_base='PKLot', 
        train_bases=['camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9',], 
        test_bases=['camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9',])

mult.run(model_name='Modelo_Kyoto',
        autoencoder_base='PKLot', 
        train_bases=['camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9',], 
        test_bases=['camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9',])

voto.run(model_name='Modelo_Kyoto',
        autoencoder_base='PKLot', 
        train_bases=['camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9',], 
        test_bases=['camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9',])

#Kyoto
sum.run(model_name='Modelo_Kyoto',
        autoencoder_base='Kyoto', 
        train_bases=['PUC', 'UFPR04', 'UFPR05', 'camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9'], 
        test_bases=['PUC', 'UFPR04', 'UFPR05', 'camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9'])

mult.run(model_name='Modelo_Kyoto',
        autoencoder_base='Kyoto', 
        train_bases=['PUC', 'UFPR04', 'UFPR05', 'camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9'], 
        test_bases=['PUC', 'UFPR04', 'UFPR05', 'camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9'])

voto.run(model_name='Modelo_Kyoto',
        autoencoder_base='Kyoto', 
        train_bases=['PUC', 'UFPR04', 'UFPR05', 'camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9'], 
        test_bases=['PUC', 'UFPR04', 'UFPR05', 'camera1', 'camera2', 'camera3','camera4','camera5','camera6','camera7','camera8','camera9'])