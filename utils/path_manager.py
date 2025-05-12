import os
import shutil
from pathlib import Path


class PathManager:
    """
    Gerencia os caminhos do projeto
    """
    def __init__(self, base_path):
        # Caminho base para o projeto, bom para usar em lugares diferentes
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"O diretório base não existe: {base_path}")
        else:
            self.base_path = base_path
            self.create_import_folder()

    def get_base_path(self):
        return self.base_path
    
    def create_import_folder(self):
        """
        Cria as principais pastas do projeto
        """
        try:
            print('Criando pastas...')
            #os.makedirs(os.path.join(self.base_path, 'Modelos'), exist_ok=True)
            #os.makedirs(os.path.join(self.base_path, 'Modelos', 'Fusoes'), exist_ok=True)
            #os.makedirs(os.path.join(self.base_path, 'Modelos', 'Plots'), exist_ok=True)
            os.makedirs(os.path.join(self.base_path, 'Pesos_parciais'), exist_ok=True)
            os.makedirs(os.path.join(self.base_path, 'resultados'), exist_ok=True)

            csvs = ['PUC', 'UFPR04', 'UFPR05', 'CNR', 'PKLot', 'Kyoto', 'camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']
            
            os.makedirs(os.path.join(self.base_path, 'CSV'), exist_ok=True) 
            for csv in csvs:
                os.makedirs(os.path.join(self.base_path, 'CSV', csv), exist_ok=True)
        except PermissionError as e:
            print(f"Erro de permissão ao criar pastas: {e}")
            print("Verifique se você tem permissão de escrita no diretório:", self.base_path)
        except Exception as e:
            print(f"Erro inesperado ao criar pastas: {e}")

    def get_prediction_path(self, model_name, batch_size, model_index, train_base, test_base, autoencoder_base=None):
        """
        Me retorna o caminho para o arquivo de previsão
        """
        classifier = f'Classificador-{autoencoder_base}' if autoencoder_base else 'Classificador'
        batch_suffix = 'batches' if test_base == train_base else 'batchs'
        return os.path.join(
            self.base_path,
            f'Modelos/{model_name}-{model_index}/{classifier}/Resultados/Treinados_em_{train_base}/{test_base}/{batch_suffix}-{batch_size}.npy'
        )
    
    def get_csv_path(self, base_teste, base_treino):
        """
        Me retorna o caminho para o arquivo CSV
        """
        if base_teste != base_treino and not 'camera' in base_teste:
            return f'CSV/{base_teste}/{base_teste}.csv'
        elif 'camera' in base_teste:
            return f'CSV/CNR/CNR_{base_teste}.csv'
        else:
            return f'CSV/{base_teste}/{base_teste}_Segmentado_Teste.csv'
    
    def get_results_path(self, model_name, autoencoder_base, fusion_type):
        """
        Me retorna o caminho para o arquivo de resultados
        """
        return f'resultados/{model_name}/tabela_resultado-{fusion_type}-{autoencoder_base}.csv'

def recreate_folder(PathManager, path_folder):
    """
    Recria a pasta do modelo
    """
    if os.path.exists(os.path.join(PathManager.get_base_path(), path_folder)):
        shutil.rmtree(os.path.join(PathManager.get_base_path(), path_folder))
    os.makedirs(os.path.join(PathManager.get_base_path(), path_folder))

def create_folder(PathManager, *args):
    """
    Cria diretórios baseados nos argumentos passados
    
    Args:
        *args: Argumentos que formam o caminho do diretório
                Exemplo: create_folder('Modelos', 'Fusoes', 'Kyoto') 
                criará: /base_path/Modelos/Fusoes/Kyoto/
    """
    path = os.path.join(PathManager.get_base_path(), *args)
    os.makedirs(path, exist_ok=True)
    return path

def return_model_name(path):
    """
    Retorna o nome do modelo
    """
    return path.split('/')[-1].rsplit('.keras', 1)[0]

def recreate_folder_force(path_folder):
    """
    Recria a pasta do modelo
    """
    if os.path.exists(path_folder):
        shutil.rmtree(path_folder)
    os.makedirs(path_folder)