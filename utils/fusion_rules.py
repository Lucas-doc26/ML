import numpy as np
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from .preprocessing import map_classes_to_binary
from .path_manager import PathManager, verify_path
from .view.graphics import graphic_accuracy_per_batch

class FusionRule:
    def __init__(self, path_manager):
        self.path_manager = path_manager
        
    def apply_fusion(self, model_name, batch_size, n_models, train_base, test_base, autoencoder_base=None):
        """
        Aplica a fusão entre as predições
        """
        # Carrega a primeira predição para inicializar o formato
        base = np.load(self.path_manager.get_prediction_path(model_name, batch_size, 0, train_base, test_base, autoencoder_base))
        
        if isinstance(self, SumFusion):
            result = np.zeros_like(base)  # Para soma, começa com zeros
        elif isinstance(self, MultFusion):
            result = np.ones_like(base)   # Para multiplicação, começa com uns
        elif isinstance(self, VoteFusion):
            result = np.zeros_like(base)  # Para votação, começa com zeros
        else:
            raise ValueError("Tipo de fusão não reconhecido")
        
        # Combina as predições de todos os modelos
        for i in range(n_models):
            npy = self.path_manager.get_prediction_path(model_name, batch_size, i, train_base, test_base, autoencoder_base)
            print(f"Carregando predição do modelo {i}: {npy}")
            array = np.load(npy)
            result = self._combine_predictions(result, array)
        
        # Retorna a classe com maior probabilidade
        return np.argmax(result, axis=1)

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
                array_result[j] = [1, 0]  # Voto para classe 0
            else:
                array_result[j] = [0, 1]  # Voto para classe 1
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
            #verify_path(path_manager, model_name, train_base, autoencoder_base)
            results = []
            for batch_size in batches:
                fusion_rule_result = fusion_rule.apply_fusion(model_name, batch_size, number_of_models, train_base, test_base, autoencoder_base)
                if train_base == test_base:
                    df = pd.read_csv(f'CSV/{test_base}/{test_base}_test.csv')
                else:
                    df = pd.read_csv(f'CSV/{test_base}/{test_base}.csv')
                df = map_classes_to_binary(df['class']) # Converte a classe para 0 e 1
                
                print(f"Treino e teste: {train_base}, {test_base} - {batch_size}")

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
                save_path=os.path.join(path_manager.get_base_path(), f'Modelos/Fusoes-{model_name}/Autoencoder-{autoencoder_base}/Grafico_batchs/Treinado_em_{train_base}/{fusion_rule.__class__.__name__}/Grafico_batchs_{fusion_rule.__class__.__name__}-{autoencoder_base}-{train_base}-{test_base}.png')
            )

    df_results = pd.DataFrame(results_csv)
    print(path_manager.get_results_path(model_name, autoencoder_base, fusion_rule.__class__.__name__))
    df_results.to_csv(path_manager.get_results_path(model_name, autoencoder_base, fusion_rule.__class__.__name__), index=False)
    