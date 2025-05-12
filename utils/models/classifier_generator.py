import os
import shutil
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score


class GeradorClassificador:
    def __init__(self, encoder=None, pesos=None, nome_modelo:str=None, nome_autoencoder:str='Sem-Peso'):
        self.encoder = encoder
        self.nome_modelo = nome_modelo

        self.model = self.modelo(self.encoder)
        self.carrega_pesos(pesos)

        if self.encoder != None:
            self.compila()

        self.treino = None
        self.validacao = None
        self.teste = None
        self.nome_autoencoder = nome_autoencoder
        
        if nome_modelo is None:
            raise ValueError("O nome do modelo não foi definido.")
        else:
            self.verifica_dirs()

    def modelo(self, encoder):
        if encoder != None:
            #eu congelo as camadas do encoder, não faço fine-tunning
            for layer in self.encoder.layers:
                layer.trainable = False
            encoder.trainable = False

            #crio o classificador com o enconder
            classificador = keras.models.Sequential([
                    self.encoder,  
                    keras.layers.Dropout(0.2),  
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(128,activation='relu'),  
                    keras.layers.Dense(2, activation='softmax')  
                ], name=f'classificador{self.nome_modelo}')
            
        else:
            classificador = keras.models.Sequential([ 
                    keras.layers.Dropout(0.2),  
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(128,activation='relu'),  
                    keras.layers.Dense(2, activation='softmax')  
                ], name=f'classificador{self.nome_modelo}')
            
        return classificador
    
    def verifica_dirs(self):
        save_dir = os.path.join(path, "Modelos", self.nome_modelo)
        dir_raiz = os.path.join(save_dir, f"Classificador-{self.nome_autoencoder}")
        dir_modelo = os.path.join(dir_raiz, "Estrutura")
        dir_pesos = os.path.join(dir_raiz, "Pesos")

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if not os.path.isdir(dir_raiz):
            os.mkdir(dir_raiz)
        if not os.path.isdir(dir_modelo):
            os.mkdir(dir_modelo)
        if not os.path.isdir(dir_pesos):
            os.mkdir(dir_pesos)

    def salva_modelo(self, save_dir=''):
        if save_dir == '':
            path_save_modelo = os.path.join(path, f'Modelos/{self.nome_modelo}/Classificador-{self.nome_autoencoder}/Estrutura/Classificador_{self.nome_modelo}.keras')
        else:
            path_save_modelo = os.path.join(path, save_dir, f'Modelos/{self.nome_modelo}/Classificador-{self.nome_autoencoder}/Estrutura/Classificador_{self.nome_modelo}.keras' )
        self.model.save(path_save_modelo)
    
    def setNome(self, nome):
        self.nome_modelo = nome
    
    def compila(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def carrega_pesos(self, peso):
        if peso == None:
            print("Criação do modelo de classificação sem pesos")
        elif peso == False:
            print("Criação do modelo sem carregar os pesos")
        else:
            try:
                self.model.load_weights(peso, skip_mismatch=True)
                print("Pesos carregados com sucesso")
            except Exception as e:
                print(f"Erro ao carregar os pesos: {e}")
        limpa_memoria()

    def treinamento(self, salvar=False, epocas=10, batch_size=64, n_batchs=None, nome_base=None, pesos=True):

        #Indico se no meu treina em batch, estou carregando ou não os pesos do encoder
        #para poder salvar de forma correta, ou seja, muda o diretório de onde irá salvar
        if pesos:
            path_slv = path
        else:
            path_slv = os.path.join(path)

        checkpoint_path = os.path.join(path_slv, 'Pesos/Pesos_parciais/weights-improvement-{epoch:02d}-{val_loss:.2f}.weights.h5')

        cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                        save_weights_only=True, 
                                        monitor='accuracy', 
                                        mode='max', 
                                        save_best_only=True, 
                                        verbose=1)

        agora = datetime.datetime.now().strftime("%d%m%y-%H%M")
        log_dir = f"logs/fit/{self.nome_modelo}-{agora}"
        log_dir = os.path.join(path_slv, f'Modelos/{self.nome_modelo}/Classificador-{self.nome_autoencoder}', log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        #Se quiser acompanhar
        history = self.model.fit(self.treino, epochs=epocas, callbacks=[cp_callback, tensorboard_callback], batch_size=batch_size ,validation_data=self.validacao)
        

        if os.path.isdir(os.path.join(path_slv,"Pesos/Pesos_parciais")):
            shutil.rmtree(os.path.join(path_slv,"Pesos/Pesos_parciais"), ignore_errors=True)
            os.makedirs(os.path.join(path_slv,"Pesos/Pesos_parciais"), exist_ok=True)

        if salvar == True:
            save_dir = os.path.join(path_slv, "Modelos", self.nome_modelo)
            dir_raiz = os.path.join(save_dir, f"Classificador-{self.nome_autoencoder}")
            dir_modelo = os.path.join(dir_raiz, "Estrutura")
            dir_pesos = os.path.join(dir_raiz, "Pesos")
            dir_pesos_base = os.path.join(dir_pesos, f'Treinado_em_{nome_base}')
            dir_imagens = os.path.join(save_dir, 'Plots')

            # criando o ../classificador 
            if not os.path.isdir(dir_raiz):
                os.makedirs(dir_raiz)

            # criando o ../Estrutura e ../Pesos
            if not os.path.isdir(dir_modelo) and not os.path.isdir(dir_pesos):
                os.makedirs(dir_modelo)
                os.makedirs(dir_pesos)

            # salvo o modelo
            self.model.save(os.path.join(dir_modelo, f'Classificador_{self.nome_modelo}.keras'))

            if n_batchs != None:
                #crio o ../Pesos/Treinado_em_PUC/
                if not os.path.isdir(dir_pesos_base):
                    os.makedirs(dir_pesos_base)

                # salvo em ../Pesos/Treinado_em_PUC/Classificador-NomeModelo-batchs-64
                #print("aqui")
                self.model.save_weights(f"{dir_pesos_base}/Classificador_{self.nome_modelo}_batchs-{n_batchs}.weights.h5")
            else:
                self.model.save_weights(f"{dir_pesos_base}/Classificador_{self.nome_modelo}.weights.h5")  

            if not os.path.isdir(dir_imagens):
                os.makedirs(dir_imagens)   

            save_history = os.path.join(dir_imagens, 'History')
            if not os.path.isdir(save_history):
                os.makedirs(save_history)
            plot_history_batch(history, save_history, self.nome_modelo, nome_base, self.nome_autoencoder ,n_batchs) 
            del history                  

    def Dataset(self, treino, validacao, teste):
        self.treino = treino
        self.validacao = validacao
        self.teste = teste

    def setTreino(self, treino):
        self.treino = treino

    def setTeste(self, teste):
        self.teste = teste

    def predicao(self, teste_csv):
        predicoes_np = self.model.predict(self.teste)
        predicoes = np.argmax(predicoes_np, axis=1)

        print(predicoes)

        y_verdadeiro = mapear(teste_csv['classe'])

        #plot_confusion_matrix(y_verdadeiro, predicoes, ['Empty', 'Occupied'], title=f'{self.nome_modelo}')

        accuracia = accuracy_score(y_verdadeiro, predicoes)

        return predicoes_np, accuracia
    
    def carrega_modelo(self, modelo:str, pesos:str):
        modelo_carregado = tf.keras.models.load_model(modelo)
        self.model = modelo_carregado
        self.carrega_pesos(pesos)

        self.model.summary(show_trainable=True)

        return self.model


#Exemplo de uso:
#classificador = GeradorClassificador(encoder=encoder, pesos="pesos.weights.h5") -> crio o classificador encima do encoder e seus pesos
#classificador.Dataset(treino, validacao, teste)
#classificador.treinamento() 
#classificador.predicao(teste_df) -> cria a matriz de confusão

"""------------------Funções para usar diversos classificadores----------------------"""
def cria_classificadores(n_modelos=10, nome_modelo=None, base_autoencoder='Sem-Peso', treino=None, validacao=None, teste=None, 
                         input_shape=(64,64,3)):
    gerador = Gerador(input_shape)
    for i in range(n_modelos):  
        limpa_memoria()

        gerador.carrega_modelo(os.path.join(path,f'Modelos/{nome_modelo}-{i}/Modelo-Base/Estrutura/{nome_modelo}-{i}.keras'))
        encoder = gerador.encoder

        pesos= os.path.join(path,f'Modelos/{nome_modelo}-{i}/Modelo-Base/Pesos/{nome_modelo}-{i}_Base-{base_autoencoder}.weights.h5')
        classificador = GeradorClassificador(encoder=encoder, pesos=pesos, nome_autoencoder=base_autoencoder, nome_modelo=f'{nome_modelo}-{i}')
        classificador.Dataset(treino, validacao, teste)
        #classificador.compila()
        classificador.salva_modelo()
        #classificador.treinamento(epocas=10)
        #classificador.predicao(teste_csv)

        limpa_memoria()

def treinamento_em_batch(nome_modelo, base_usada, base_autoencoder, treino_csv, validacao, teste, teste_csv ,salvar=True, n_epocas=10, pesos=True, input_shape=(64,64,3)):
    path_slv = path

    #nome da base de treino do classificador
    nome, _ = retorna_nome_base(treino_csv)
    nome_base_teste = retorna_nome_df(teste_csv)

    batch_dir = f"CSV/{nome}/batches"
    batchs = sorted(os.listdir(batch_dir), key=lambda x: int(x.split("batch-")[1].split(".")[0]))
    print(batchs)
    precisoes = []
    n_batchs = [64,128,256,512,1024] 

    if not pesos:
        base_autoencoder = 'Sem-Peso'


    #modelo = classificador.model

    # crio o Modelo/Classificador/Resultados
    if not os.path.isdir(os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{base_autoencoder}/Resultados')):
        os.makedirs(os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{base_autoencoder}/Resultados'))

    # crio o Modelo/Classificador/Resultados/Treinados_em_PUC
    dir_resultados_base = os.path.join(path_slv, f'Modelos/{nome_modelo}/Classificador-{base_autoencoder}/Resultados/Treinados_em_{base_usada}')
    if not os.path.isdir(dir_resultados_base):
        os.makedirs(dir_resultados_base)
    
    #para cada um dos meus batches sizes, eu vou pegar seu correspondendo em csv 
    for batch, batch_size in zip(batchs, n_batchs):
        limpa_memoria()

        gerador = Gerador(input_shape)
        gerador.carrega_modelo(os.path.join(path,f'Modelos/{nome_modelo}/Modelo-Base/Estrutura/{nome_modelo}.keras'), pesos=False)
        encoder = gerador.encoder

        #caso eu queira carregar o pesos do encoder
        if pesos:
            classificador = GeradorClassificador(encoder=encoder, 
                                                pesos=os.path.join(path,f'Modelos/{nome_modelo}/Modelo-Base/Pesos/{nome_modelo}_Base-{base_autoencoder}.weights.h5'), 
                                                nome_modelo=nome_modelo, 
                                                nome_autoencoder=base_autoencoder)
        else:
            #se eu não quiser carregar os pesos do encoder, o meu gerador irá mudar o nome para nome_autoencoder='Sem-Peso'
            classificador = GeradorClassificador(encoder=encoder, pesos=False, nome_modelo=nome_modelo, nome_autoencoder=base_autoencoder)

        classificador.compila()
        classificador.Dataset(treino=None, validacao=validacao, teste=teste)

        treino, _ = preprocessamento_dataframe(os.path.join(batch_dir, batch), autoencoder=False)
        classificador.setTreino(treino)
        classificador.treinamento(epocas=n_epocas, salvar=salvar ,n_batchs=batch_size, nome_base=base_usada, pesos=pesos)
        predicoes_np, acuracia = classificador.predicao(teste_csv)
        precisoes.append(acuracia)

        #Modelo-Kyoto-1/Classificador/Resultados/Treinados_em_PUC/PUC
        resultados_dir = os.path.join(dir_resultados_base, nome_base_teste)
        if not os.path.isdir(resultados_dir):
            os.makedirs(resultados_dir)

        #Salvo o resultado npy
        arquivo = os.path.join(path_slv, resultados_dir, f"batches-{batch_size}.npy")
        np.save(arquivo, predicoes_np)

        #Salvar a precisão 
        dir_prec = os.path.join(path_slv, f"Modelos/{nome_modelo}/Classificador-{base_autoencoder}/Precisao")
        if not os.path.isdir(dir_prec):
            os.makedirs(dir_prec)

        #cria ../precisao/Treinado_em_UFPR04
        dir_prec_base = os.path.join(dir_prec, f'Treinado_em_{base_usada}')
        recria_diretorio(dir_prec_base)#Apaga e cria a pasta nova se já tiver

        #Salva a precisão 
        print(nome_base_teste)
        caminho_arquivo = os.path.join(dir_prec_base, f'precisao-{nome_base_teste}.txt')
        with open(caminho_arquivo, 'w') as f:
            for prec in precisoes:
                f.write(f"{prec}\n")
        
        limpa_memoria() 

    dir_graf = os.path.join(path_slv,f'Modelos/{nome_modelo}/Plots/Graficos')
    if not os.path.isdir(dir_graf):
        os.makedirs(dir_graf)

    dir_graf_facul = os.path.join(dir_graf, f'Treinado_em_{base_usada}')
    if not os.path.isdir(dir_graf_facul):
        os.makedirs(dir_graf_facul)

    grafico_batchs(n_batchs=n_batchs, precisoes=precisoes, nome_modelo=nome_modelo, 
                   nome_base_treino=base_usada, 
                   base_usada_teste=base_usada, 
                   nome_autoencoder=base_autoencoder, 
                   caminho_para_salvar=dir_graf_facul)

    #plot_model(encoder, show_shapes=True,show_layer_names=True,to_file=os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{base_autoencoder}/encoder-{nome_modelo}.png'))
    #plot_model(modelo, show_shapes=True,show_layer_names=True,to_file=os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{base_autoencoder}/classificador-{nome_modelo}.png'))
    
    print(precisoes)
    #return (n_batchs, precisoes, nome_modelo)

def treina_modelos_em_batch(nome_modelo, base_usada, base_autoencoder, treino_csv, validacao, teste, teste_csv, salvar=True, n_epocas=10, input_shape=(64,64,3)):
    path_modelos = os.path.join(path, "Modelos")
    modelos = os.listdir(path_modelos)
    modelos_para_treinar = []
    for modelo in modelos:
        if os.path.exists(os.path.join(path_modelos, modelo)):
            if nome_modelo in modelo and 'Fusoes' not in modelo:
                modelo_base = os.path.join(path_modelos, modelo, 'Modelo-Base')
                peso, estrutura = os.listdir(modelo_base)
                m = os.listdir(os.path.join(modelo_base , estrutura))
                dir_modelo = os.path.join(modelo_base, estrutura, m[0])
                modelos_para_treinar.append(dir_modelo)
            else:
                pass
        else:
            print(f"O diretório {modelo} não existe.")

    for i, m in enumerate(sorted(modelos_para_treinar)):
        nome = nome_modelo + f"-{i}"
        #treino cada um dos modelos em batches 
        treinamento_em_batch(nome, base_usada, base_autoencoder, treino_csv, validacao, teste, teste_csv, salvar, n_epocas, True, input_shape)

    #faço o plot comparando os modelos
    comparacao(caminho_para_salvar=os.path.join(path_modelos, "Plots"), 
               nome_modelo=nome_modelo, 
               base_usada=base_usada, 
               base_autoencoder=base_autoencoder,
               base_de_teste=base_usada)

def testa_modelos_em_batch(nome_modelo, teste, teste_df, base_do_classificador, pesos=True, nome_autoencder='Kyoto'):
    path_slv = path

    if not pesos:
        nome_autoencder='Sem-Peso'

    classificador = GeradorClassificador(nome_modelo=nome_modelo, nome_autoencoder=nome_autoencder)

    nome_base = retorna_nome_df(teste_df)

    classificador.setTeste(teste)

    if not os.path.isdir(os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Resultados')):
        recria_diretorio(os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Resultados'))
        
    acuracias = []
    batchs = [64,128,256,512,1024]

    estrutura = os.path.join(path_slv,f'Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Estrutura/Classificador_{nome_modelo}.keras')
    for batch_size in [64,128,256,512,1024]:
        dir_peso = f'Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Pesos/Treinado_em_{base_do_classificador}/Classificador_{nome_modelo}_batchs-{batch_size}.weights.h5'
        peso = os.path.join(path_slv,dir_peso)
        classificador.carrega_modelo(estrutura, peso)
        predicoes_np, acuracia = classificador.predicao(teste_df)

        #Modelo-Kyoto-1/Classificador-CNR/Resultados/Treinados_em_PUC/UFPR04/batchs-64-npy
        dir_base = os.path.join(path_slv, f"Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Resultados/Treinados_em_{base_do_classificador}/{nome_base}")
        
        if not os.path.isdir(dir_base):
            os.makedirs(dir_base)
            print(f"Diretório criado: {dir_base}")
        else:
            print(f"Diretório {dir_base} já existia!")

        arquivo = os.path.join(dir_base, f'batchs-{batch_size}.npy')
        np.save(arquivo, predicoes_np)
        limpa_memoria()
        acuracias.append(acuracia)
        del peso, predicoes_np, acuracia


    print(acuracias)

    #Salvar as precisões no arquivo
    #print(nome_base_teste)
    print(nome_base)
    caminho_arquivo = os.path.join(path_slv, f'Modelos/{nome_modelo}/Classificador-{nome_autoencder}/Precisao/Treinado_em_{base_do_classificador}', f'precisao-{nome_base}.txt')
    with open(caminho_arquivo, 'w') as f:
        for prec in acuracias:
            f.write(f"{prec}\n")

    dir_graf_facul = os.path.join(path_slv,f'Modelos/{nome_modelo}/Plots/Graficos/Treinado_em_{base_do_classificador}')
    grafico_batchs(n_batchs=batchs, precisoes=acuracias, nome_modelo=nome_modelo, 
                   nome_base_treino=base_do_classificador, 
                   base_usada_teste=nome_base, 
                   nome_autoencoder=nome_autoencder ,
                   caminho_para_salvar=dir_graf_facul) 

def testa_modelos(nome_modelo, teste, teste_df, base_do_classificador, nome_autoencoder):
    #classificador = GeradorClassificador()
    caminho_modelos = os.path.join(path, "Modelos")
    modelos = os.listdir(caminho_modelos) #todos as pastas no dir Modelos
    modelos_usados = []
    print("Os modelos em pastas são:", modelos)
    for modelo in modelos:
        if os.path.exists(os.path.join(caminho_modelos, modelo)):
            if nome_modelo in modelo and 'Fusoes' not in modelo: #Se nome modelo tiver em modelo e fusão não
                modelo_base = os.path.join(caminho_modelos, modelo, f'Classificador-{nome_autoencoder}') #Modelo_Kyoto-0/Classificador_Modelo
                print("A estrutura disponível é:" , os.listdir(os.path.join(modelo_base, 'Estrutura')))
                estrutura = os.listdir(os.path.join(modelo_base, 'Estrutura'))[0]
                modelos_usados.append(estrutura)
            else:
                pass
        else:
            print(f"O diretório {modelo} não existe.")
    
    print(modelos_usados)
    for modelo in modelos_usados:
        nome = extrair_nome_modelo1(modelo)
        print(nome)
        testa_modelos_em_batch(
            nome_modelo=nome, teste=teste, teste_df=teste_df, 
            base_do_classificador=base_do_classificador, 
            pesos=True, 
            nome_autoencder=nome_autoencoder)
        limpa_memoria()

    base_testada = retorna_nome_df(teste_df)

    comparacao(caminho_para_salvar=os.path.join(path, 'Modelos/Plots'), 
               nome_modelo=nome_modelo, 
               base_usada=base_do_classificador, 
               base_de_teste=base_testada, 
               base_autoencoder=nome_autoencoder)


