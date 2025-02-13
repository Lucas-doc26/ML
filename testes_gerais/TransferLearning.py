import tensorflow as tf
import os
import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from visualizacao import plot_confusion_matrix, plot_imagens_incorretas
from preprocessamento import preprocessamento, preprocessamento_dataframe, mapear_rotulos_binarios, carregar_e_preprocessar_imagens
from modelos import carrega_modelo, modelo_mobileNetV3Small


def teste_modelos(caminho:str, num:int):
    """
    Compila e treina o modelo e faz evaluate com todos os datasets, salva os resultado em /Resultados/teste_{Nome do Modelo}
    """

    model = modelo_mobileNetV3Small(num)
    treino, validacao, teste, _, _, _ = preprocessamento(caminho)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    checkpoint_path = 'weights_parciais/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.weights.h5'
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                  save_weights_only=True, 
                                  monitor='val_accuracy', 
                                  mode='max', 
                                  save_best_only=True, 
                                  verbose=1)

    history = model.fit(treino,
                        epochs=10,
                        callbacks=[cp_callback],
                        validation_data=validacao)

    dataset_paths = {
        'PUC': 'Datasets_csv/df_PUC.csv',
        'UFPR04': 'Datasets_csv/df_UFPR04.csv',
        'UFPR05': 'Datasets_csv/df_UFPR05.csv'
    }

    nome_Modelo = next((key for key in dataset_paths if key in caminho), 'UFPR05')
    model_save_path = f"Modelos_keras/{nome_Modelo}_{'Congelado' if num == 0 else 'Descongelado'}_mobilenetv3.keras"
    weights_save_path = f"weights_finais/{nome_Modelo}_{'Congelado' if num == 0 else 'Descongelado'}_mobilenetv3.weights.h5"

    model.save(model_save_path)
    model.save_weights(weights_save_path)

    filename = f"Resultados/teste_{nome_Modelo}_{'Congelado' if num == 0 else 'Descongelado'}"
    with open(filename, 'w') as f:
        _, dataset_1 = preprocessamento_dataframe(dataset_paths.get(nome_Modelo))
        _, dataset_2 = preprocessamento_dataframe('Datasets_csv/df_UFPR05.csv' if nome_Modelo != 'UFPR05' else 'Datasets_csv/df_UFPR04.csv')

        modelo = keras.models.load_model(model_save_path)  # Carregar o modelo original

        modelo.load_weights(weights_save_path)

        test_results = {
            f"{nome_Modelo} X {nome_Modelo}": modelo.evaluate(teste)[1],
            f"{nome_Modelo} X {'UFPR04' if nome_Modelo == 'PUC' else 'PUC'}": modelo.evaluate(dataset_1)[1],
            f"{nome_Modelo} X {'UFPR05' if nome_Modelo != 'UFPR05' else 'UFPR04'}": modelo.evaluate(dataset_2)[1]
        }

        for modelos, accuracy in test_results.items():
            print(f"{modelos} - precisão de {accuracy:.3f} ", file=f)
            
#Exemplo de criação dos modelos:
print("Modelos sem o Fine Tuning:\n")
teste_modelos('Datasets_csv/df_PUC.csv', 0)
teste_modelos('Datasets_csv/df_UFPR04.csv', 0)
teste_modelos('Datasets_csv/df_UFPR05.csv', 0)

print("\nModelos com o Fine Tuning:\n")
teste_modelos('Datasets_csv/df_PUC.csv', 2)
teste_modelos('Datasets_csv/df_UFPR04.csv', 2)
teste_modelos('Datasets_csv/df_UFPR05.csv', 2)

def predict_e_matriz_de_confusao():
    PUC_df, PUC= preprocessamento_dataframe('Datasets_csv/df_PUC.csv')
    UFPR04_df, UFPR04,  = preprocessamento_dataframe('Datasets_csv/df_UFPR04.csv')
    UFPR05_df, UFPR05 = preprocessamento_dataframe('Datasets_csv/df_UFPR05.csv')

    modelos = {
        'PUC_Congelado': r'Modelos_keras/PUC_Congelado_mobilenetv3.keras',
        'PUC_Descongelado': r'Modelos_keras/PUC_Descongelado_mobilenetv3.keras', 
        'UFPR04_Congelado': r'Modelos_keras/UFPR04_Congelado_mobilenetv3.keras',
        'UFPR04_Descongelado': r'Modelos_keras/UFPR04_Descongelado_mobilenetv3.keras',
        'UFPR05_Congelado': r'Modelos_keras/UFPR05_Congelado_mobilenetv3.keras',
        'UFPR05_Descongelado': r'Modelos_keras/UFPR05_Descongelado_mobilenetv3.keras',
    }

    pesos = {
        'PUC_Congelado': r'weights_finais/PUC_Congelado_mobilenetv3.weights.h5',
        'PUC_Descongelado': r'weights_finais/PUC_Descongelado_mobilenetv3.weights.h5', 
        'UFPR04_Congelado': r'weights_finais/UFPR04_Congelado_mobilenetv3.weights.h5',
        'UFPR04_Descongelado': r'weights_finais/UFPR04_Descongelado_mobilenetv3.weights.h5',
        'UFPR05_Congelado': r'weights_finais/UFPR05_Congelado_mobilenetv3.weights.h5',
        'UFPR05_Descongelado': r'weights_finais/UFPR05_Descongelado_mobilenetv3.weights.h5',
    }

    modelos_carregados = {}

    for modelo_nome, modelo_caminho in modelos.items():
        pesos_caminho = pesos[modelo_nome]

        modelo = keras.models.load_model(modelo_caminho)
        modelo.load_weights(pesos_caminho)

        modelos_carregados[modelo_nome] = modelo 

    for modelo_nome, modelo in modelos_carregados.items():
        for dataset_nome, dataset, dataset_df in [('PUC', PUC, PUC_df), ('UFPR04', UFPR04, UFPR04_df), ('UFPR05', UFPR05, UFPR05_df)]:
            y_verdadeiro = dataset_df['classe'].values
            y_binario = mapear_rotulos_binarios(y_verdadeiro) 

            caminhos_imagens = dataset_df['caminho_imagem'].tolist() 
            imagens = carregar_e_preprocessar_imagens(caminhos_imagens)
            y_predicao = modelo.predict(imagens).argmax(axis=1)

            labels = ['Empty', 'Occupied']

            save_path_matriz = os.path.join('Resultados', 'Matriz_de_confusao', modelo_nome, f'{modelo_nome}_vs_{dataset_nome}_matriz_de_confusao.png')
            titulo_matriz = f'Matriz de Confusão - {modelo_nome} vs {dataset_nome}'
            plot_confusion_matrix(y_binario, y_predicao, labels, save_path_matriz, titulo_matriz)
            plot_imagens_incorretas(y_binario, y_predicao, caminhos_imagens, modelo_nome, dataset_nome, 3)

# Exemplo: 
predict_e_matriz_de_confusao(5)




