import numpy as np
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from .path_manager import PathManager
from .models.autoencoder_generator import map_classes
from .utils.preprocessing import map_classes_to_binary
from .path_manager import PathManager
from .view.graphics import graphic_accuracy_per_batch

class FusionRule:
    def __init__(self, path_manager):
        self.path_manager = path_manager
        
    def apply_fusion(self, model_name, batch_size, n_models, train_base, test_base, autoencoder_base=None):
        """
        Aplica a fusão entre as predições
        """
        base = np.load(self.path_manager.get_prediction_path(model_name, batch_size, 0, train_base, test_base, autoencoder_base))
        result = np.zeros_like(base) # Cria um array com o mesmo tamanho da base de predições
        
        for i in range(n_models):
            #pega o array de predições do modelo i
            array = np.load(self.path_manager.get_prediction_path(model_name, batch_size, i, train_base, test_base, autoencoder_base))
            result = self._combine_predictions(result, array)
            
        return np.argmax(result, axis=1)
        #retorna o array com os resultados da fusão
    
    # Método abstrato das predições
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
                array_result[j] = [1, 0]
            elif value == 0.5:
                array_result[j] = [1, 1]
            else:
                array_result[j] = [0, 1]
        return result + array_result

def fusion_process(model_name, train_bases, test_bases, fusion_rule, autoencoder_base=None, number_of_models=10, path_manager='/home/lucas/PIBIC/'):
    """
    Processa a fusão entre as predições e cria os gráficos e tabelas de resultados
    """
    batches = [64, 128, 256, 512, 1024]
    results_csv = []
    path_manager = PathManager(path_manager)
    
    for train_base in train_bases:
        for test_base in test_bases:
            path_manager.verify_path(model_name, train_base, autoencoder_base)
            results = []
            
            for batch_size in batches:
                fusion_rule_result = fusion_rule.apply_fusion(model_name, batch_size, number_of_models, train_base, test_base, autoencoder_base)
                df = pd.read_csv(path_manager.get_csv_path(test_base, train_base))
                df = map_classes_to_binary(df['class']) # Converte a classe para 0 e 1
                
                acc = accuracy_score(df, fusion_rule_result)
                results.append(acc)
                
                results_csv.append({
                    'Base do Autoencoder': autoencoder_base,
                    'Base de Treino': train_base,
                    'Base de Teste': test_base,
                    'Acuracia': f"{acc:.3f}",
                    'Batch': int(batch_size)
                })
            
            graphic_accuracy_per_batch(
                batches, 
                results, 
                model_name=f'{fusion_rule.__class__.__name__} entre os diferentes {model_name}',
                train_base=train_base,
                test_base=test_base,
                autoencoder_base=autoencoder_base, 
                save_path=os.path.join(path_manager.get_base_path(), f'Modelos/Fusoes-{model_name}/Grafico_batchs/Grafico_batchs_{fusion_rule.__class__.__name__}-{autoencoder_base}-{train_base}-{test_base}.png')
            )
    
    df_results = pd.DataFrame(results_csv)
    df_results.to_csv(path_manager.get_results_path(model_name, autoencoder_base, fusion_rule.__class__.__name__), index=False)

"""# Exemplo de uso:
path_manager = PathManager('/home/lucas/PIBIC/')
sum_fusion = SumFusion(path_manager)
mult_fusion = MultFusion(path_manager)
vote_fusion = VoteFusion(path_manager)
train_bases = ['PUC', 'UFPR04', 'UFPR05']

# Executar as fusões
fusion_process("Modelo_Kyoto", train_bases, sum_fusion, 'Kyoto', n_modelos=10)
fusion_process("Modelo_Kyoto", train_bases, mult_fusion, 'Kyoto', n_modelos=10)
fusion_process("Modelo_Kyoto", train_bases, vote_fusion, 'Kyoto', n_modelos=10)

fusion_process("Modelo_Kyoto", train_bases, sum_fusion, 'CNR', n_modelos=5)
fusion_process("Modelo_Kyoto", train_bases, mult_fusion, 'CNR', n_modelos=5)
fusion_process("Modelo_Kyoto", train_bases, vote_fusion, 'CNR', n_modelos=5)"""