import tensorflow as tf
import gc
import keras

def clear_session():
    keras.backend.clear_session()  
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.reset_memory_stats('GPU:0') #limpa memória da gpu
