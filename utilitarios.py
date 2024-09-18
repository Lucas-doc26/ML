import os
import shutil
import numpy as np
import glob

def codificacao_em_batch(teste_gerador, encoder):
    """
    Caso o número de predições para ser feito seja muito extenso, pode usar essa função. \n
    Retorno: array numpy com todas as predições
    """
    pasta_resultados = 'Predicoes/predicoes.numpy'

    if os.path.exists(pasta_resultados):
        shutil.rmtree(pasta_resultados)
    os.makedirs(pasta_resultados)

    for i in range(len(teste_gerador)):
        batch = next(teste_gerador)
        encodings = encoder.predict(batch)
        
        # Salvar os encodings parciais
        np.save(os.path.join(pasta_resultados, f'encodings_batch_{i}.npy'), encodings)
        
        print(f'Processado lote {i+1}/{len(teste_gerador)}')

    encoding_files = sorted(glob.glob('/home/lucas/PIBIC (copy)/Predicoes/predicoes.numpy/encodings_batch_*.npy'))

    #Carregar e concatenar todos os arquivos
    all_encodings = []
    for file in encoding_files:
        encodings = np.load(file)
        all_encodings.append(encodings)

    #Concatenar todos os lotes em um único array
    final_encodings = np.concatenate(all_encodings, axis=0)

    print(f"Shape final das codificações: {final_encodings.shape}")

    #salvar o resultado final em um único arquivo
    os.makedirs('Predicoes/resultados_finais', exist_ok=True)
    np.save('Predicoes/resultados_finais/all_encodings.npy', final_encodings)

    return final_encodings

def mapear_classes_como_binario(y):
    class_mapping = {'Empty': 0, 'Occupied': 1}
    y = y.map(class_mapping)
    return y