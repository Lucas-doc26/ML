[TENSORFLOW_BADGE]:https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white

[KERAS_BADGE]:https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white

[NUMPY_BADGE]:https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white

[PANDAS_BADGE]:https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white

[SCIKIT-LEARN_BADGE]:https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white

[PYTHON_BADGE]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54

<h1 align="center" style="font-weight: bold;">Iniciação Científica💻</h1>

![Python][PYTHON_BADGE]
![Keras][KERAS_BADGE]
![TensorFlow][TENSORFLOW_BADGE]
![Pandas][PANDAS_BADGE]
![Scikit-Learn][SCIKIT-LEARN_BADGE]
![NumPy][NUMPY_BADGE]

<p align="center">
  <a href="#Projeto">Projeto</a> •
  <a href="#Objetivos">Objetivos</a> • 
  <a href="#Datasets">Bases</a> •
  <a href="#Autor">Autor</a> 
</p>

<p align="center">
  <b></b>
</p>

<h2 id="Projeto">📫Projeto</h2>

Programa Institucional de Bolsas de Iniciação Científica (PIBIC), desenvolvendo uma pesquisa intitulada: **"Adaptação de Domínio em Modelos Profundos de Classificação Aplicados à Detecção de Vagas de Estacionamento"**, com apoio da CNPq (Conselho Nacional de Desenvolvimento Científico e Tecnológico).


![Estado do projeto!](https://img.shields.io/badge/Estado:-Em%20produção-FFFF00.svg)

<h2 id="Objetivos">🚀Objetivos do Projeto</h2>

Implementar técnicas de aprendizado não supervisionado e adaptação de domínio para a classificação de vagas de estacionamento, visando reduzir o esforço de anotação, otimizar o desempenho do modelo em cenários variados e manter um baixo custo computacional, buscando um equilíbrio entre tempo de processamento e precisão. <br>
Para atingir o objetivo geral, foram definidos os seguintes objetivos específicos:

<ul>
    <li>Criação de autoencoders</li>
    <li>Aplicar a técnica de Fine-Tunning</li>
    <li>Testar técnicas de fusão</li>
</ul>

<h2 id="Datasets">📍 Bases usadas</h2>

<h3>PKLot</h3>
A base de dados contém imagens capturadas de três estacionamentos diferentes (PUC, UFPR04, UFPR05) em dias ensolarados, nublados e chuvosos, onde cada estacionamento tem sua camêra posicionada em ângulos diferentes. 
<br> 
<br>
<img src ="https://ars.els-cdn.com/content/image/1-s2.0-S0957417422002032-gr1.jpg">
<h3>CNR-EXT</h3>
Composto por imagens coletadas de novembro de 2015 a fevereiro de 2016 sob diversas condições climáticas por 9 câmeras com diferentes perspectivas e ângulos de visão. O CNR-EXT captura diferentes situações de condições de luz e inclui padrões de oclusão parcial devido a obstáculos (árvores, postes de iluminação, outros carros) e carros com sombra parcial ou global.
<br>
<br>
<img src="https://lh3.googleusercontent.com/proxy/p-9dUexhRfdxfbh58L61VNsaFatf8KH-Bh7mVzWT8d35bqPE4GXaKuT_BNE5z-RLwJR6">

<h2>Estrutura de pastas:</h2>

```
Modelos/
├── Fusao-Modelo-Nome                                          # Plots das fusões entre os diferentes modelos  
├── Modelo_Kyoto-1/
│   ├── classificador/                                         # Classificador sem carregar os pesos do encoder
│   │   ├── estrutura/                                         # arquitetura do modelo .keras
│   │   ├── pesos/                                             # .weights.h5
│   │   │    ├── Modelo_Kyoto-1_Base-autoencoder1.weights.h5
│   │   │    ├── Modelo_Kyoto-1_Base-autoencoder2.weights.h5
│   │   │ 
│   │   ├── resultados/                                        # Resultados numpy.
│   │   │    ├── Treinado_em_Base1/
│   │   │    ├── Treinado_em_Base2/
|   |   |         ├── Base1/                                   # Testado contra a base 1
|   |   |         ├── Base2/                                   # Testado contra a base 2
|   |   |             ├── batches-64.npy 
|   |   |             ├── batches-128.npy 
|   |   |             ├── batches-256.npy 
|   |   |             ├── batches-512.npy 
|   |   |             ├── batches-1024.npy 
│   │   ├── logs/                                              # Arquivos de log do treinamento (TensorBoard, .txt etc.)
│   │   ├── precisao/                                          # Precisões contra as bases de teste 
|   |   |    ├── Treinado_em_Base1/  
|   |   |    ├── Treinado_em_Base2/
|   |   |         ├── precisao-base1.txt                       # Precisões em txt 
|   |   |         ├── precisao-base2.txt                                                    
│   │   |
│   ├── classificador_x/                                       # Classificador carregando os pesos do encoder com a base x 
│   ├── classificador_y/                                       
│   ├── classificador_z/                                       
│   ├── modelo_base/                                           # Estrutura e peso do Autoencoder treinado nas diferentes bases
|   └── Plots/
|       ├── Gráficos/
│       │    ├── Treinado_em_Base1/
│       │    ├── Treinado_em_Base2/
|       |         ├── Grafico-Modelo-Nome-1-BaseClassificador-BaseTeste.png                            
|       ├── History/
|       ├── Autoencoder-Base1.png
|       ├── History-Autoencoder-Base1.png
|       ├── Autoencoder-Base2.png
|       ├── History-Autoencoder-Base2.png
│
├── Modelo_Kyoto-2/
│   ├── classificador/
│   ├── ...
│
└── README.md
```

<h2 id="usar">Como usar:</h2>

![Estado do projeto!](https://img.shields.io/badge/Estado:-Em%20produção-FFFF00.svg)

```sh
$ git clone https://github.com/Lucas-doc26/ML 
$ git pull origin master
pip install -r requirements.txt 
```

<h2 id="Autor">🤝Autor</h2>
<table>
  <tr>
    <td align="center">
      <a href="https://www.linkedin.com/in/lucasdoc/">
        <img src="https://avatars.githubusercontent.com/u/89359426?v=4" width="100px;" alt="Lucas Cunha"/><br>
        <sub>
          <b>Lucas Cunha</b>
        </sub>
      </a>
    </td>
  </tr>
</table>
