import numpy as np
from tensorflow.keras.utils import Sequence

#Classe usada para combinar dois geradores, estava usando quandos os batches
#ainda eram 16*64, ai ia combinando eles de forma sequencial
class CombineGenerators(Sequence):
    """
    Cria um gerador que combina dois geradores, útil para juntar dois datasets
    """
    def __init__(self, generator1, generator2):
        self.generator1 = generator1
        self.generator2 = generator2
        self.batch_size = generator1.batch_size
        self.n_images = len(generator1) * generator1.batch_size + len(generator2) * generator2.batch_size
        print("Total de imagens:", self.get_total_images())
    
    def __len__(self):
        #n total de batchs considerando os 2 geradores
        return int(np.ceil(self.n_images / self.batch_size))
    
    def __getitem__(self, position):
        #calcula em qual dos geradores o batch está
        if position < len(self.generator1):
            return self.generator1[position]
        else:
            position -= len(self.generator1)
            return self.generator2[position]
    
    def get_total_images(self):
        return self.n_images