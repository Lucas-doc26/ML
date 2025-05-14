[TENSORFLOW_BADGE]:https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white

[KERAS_BADGE]:https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white

[NUMPY_BADGE]:https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white

[PANDAS_BADGE]:https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white

[SCIKIT-LEARN_BADGE]:https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white

[PYTHON_BADGE]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54

<h1 align="center" style="font-weight: bold;">Machine Learning & Pesquisa Científica 💻</h1>

<p align="center">
  ![Python][PYTHON_BADGE]
  ![Keras][KERAS_BADGE]
  ![TensorFlow][TENSORFLOW_BADGE]
  ![Pandas][PANDAS_BADGE]
  ![Scikit-Learn][SCIKIT-LEARN_BADGE]
  ![NumPy][NUMPY_BADGE]
</p>

<p align="center">
  <a href="#projeto">Projeto</a> •
  <a href="#objetivos">Objetivos</a> • 
  <a href="#datasets">Bases de Dados</a> •
  <a href="#rodar">Como rodar</a> •
  <a href="#autor">Autor</a> 
</p>

<p align="center">
  <i>Este repositório documenta um projeto de iniciação científica focado na adaptação de domínio em modelos profundos para detecção de vagas de estacionamento.</i>
</p>

<hr>

<h2 id="projeto">📫 Projeto Atual</h2>

O projeto é desenvolvido no âmbito do Programa Institucional de Bolsas de Iniciação Científica (PIBIC), com o título: <em>"Adaptação de Domínio em Modelos Profundos de Classificação Aplicados à Detecção de Vagas de Estacionamento"</em>. A pesquisa conta com o apoio do CNPq (Conselho Nacional de Desenvolvimento Científico e Tecnológico).

<p align="center">
  <img src="https://img.shields.io/badge/Estado:-Em%20Desenvolvimento-yellow?style=for-the-badge" alt="Estado do projeto: Em Desenvolvimento"/>
</p>

<hr>

<h2 id="objetivos">🚀 Objetivos do Projeto</h2>

A pesquisa busca analisar soluções que melhorem a escalabilidade dos modelos aplicados ao problema de classificação de vagas de estacionamento. Os macro-objetivos deste projeto são:

<ul>
    <li>Revisão dos métodos no estado da arte no contexto de Adaptação de Domínio.</li>
    <li>Construção de um benchmark das técnicas mais promissoras.</li>
    <li>Avaliação das técnicas nas bases de dados PKLot e CNR-EXT</li>
    <li>Proposição de um framework para detecção de vagas de estacionamento que reduza a necessidade de anotação de dados, utilizando técnicas de Adaptação de Domínio.</li>
    <li>Análise de erros e elaboração de um relatório crítico sobre os resultados alcançados.</li>
    <li>Divulgação da pesquisa (publicação de artigo) para a comunidade científica.</li>
</ul>

<hr>

<h2 id="datasets">📍 Bases de Dados (Datasets)</h2>

<h3>PKLot</h3>
<p>A base de dados PKLot contém imagens capturadas de três estacionamentos diferentes (PUC, UFPR04, UFPR05) sob diversas condições climáticas (ensolarado, nublado, chuvoso). Cada estacionamento possui câmeras posicionadas em ângulos distintos.</p>
<p align="center">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0957417422002032-gr1.jpg" alt="Exemplo de imagens da base PKLot" width="600px">
</p>

<br>

<h3>CNR-EXT</h3>
<p>A base CNR-EXT é composta por imagens coletadas entre novembro de 2015 e fevereiro de 2016, abrangendo várias condições climáticas e utilizando 9 câmeras com diferentes perspectivas e ângulos de visão. Esta base de dados captura diversas situações de iluminação e inclui padrões de oclusão parcial (devido a obstáculos como árvores, postes de iluminação, outros carros) e carros com sombreamento parcial ou total.</p>
<p align="center">
  <img src="https://www.researchgate.net/profile/Razib-Iqbal/publication/357722449/figure/fig1/AS:1147004549894144@1650478603121/mage-samples-from-the-CNRPark-EXT-and-PKLot-datasets.ppm" alt="Exemplo de imagens da base CNR-EXT" width="600px">
</p>

<hr>

<h2 id="rodar">🚀 Como Rodar o Projeto</h2>

Para executar este projeto em sua máquina local, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Lucas-doc26/ML.git
    cd ML
    ```

2.  **Crie um ambiente Conda e ative-o:**
    ```bash
    conda create --name venv python=3.10
    conda activate venv
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute o projeto:**
    ```bash
    ./main.sh
    ```

<hr>

<h2 id="autor">🤝 Autor</h2>
<table align="left">
  <tr>
    <td align="left">
      <a href="https://www.linkedin.com/in/lucasdoc/">
        <img src="https://avatars.githubusercontent.com/u/89359426?v=4" width="100px;" alt="Foto de Lucas Cunha"/><br>
        <sub>
          <b>Lucas Cunha</b>
        </sub>
      </a>
    </td>
  </tr>
</table>



<hr>
<p align="center">
  <a href="#top">Voltar ao topo</a>
</p>