import tensorflow as tf
import numpy as np
import random
import os
import pandas as pd

# Configurações de reprodutibilidade
SEED = 42

def set_seeds():
    """
    Configura todas as seeds necessárias para reprodutibilidade
    """
    # TensorFlow
    tf.random.set_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    
    # NumPy
    np.random.seed(SEED)
    
    # Python random
    random.seed(SEED)
    
    # Variáveis de ambiente
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

    # Configurações do TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0' = tudo, '1' = warnings, '2' = info, '3' = apenas erros graves
    
    # Configurações do Pandas
    pd.set_option('mode.chained_assignment', None)

def config_gpu():
    """
    Configura a GPU para uso determinístico
    """
    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configurada para uso determinístico")
    else:
        print("Nenhuma GPU encontrada")

def config_tensorflow():
    """
    Configura o TensorFlow para uso determinístico e mixed precision
    """
    # Configurar logs do TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0' = tudo, '1' = warnings, '2' = info, '3' = apenas erros graves
    
    # Configurar mixed precision
    try:
        # Habilitar mixed precision independente da GPU
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision configurado com sucesso")
        
        # Configurar GPU se disponível
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs disponíveis: {len(gpus)}")
        else:
            print("Usando CPU com mixed precision")
            
    except Exception as e:
        print(f"Aviso: Erro ao configurar mixed precision: {e}")
        print("Continuando com precisão padrão...")

# Executa as configurações ao importar o módulo
set_seeds()
config_gpu() 
config_tensorflow()