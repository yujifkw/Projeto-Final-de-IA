# 🤟 Tradutor de Libras em Tempo Real (MediaPipe + CNN)

Este projeto é um sistema avançado de Visão Computacional e Inteligência Artificial criado para reconhecer e traduzir as letras do alfabeto de Libras (Língua Brasileira de Sinais) em tempo real, utilizando a câmara do computador.

O sistema combina o rastreio de mãos do **Google MediaPipe** com uma rede neural profunda de **Transfer Learning (MobileNet via TensorFlow/Keras)**. O grande diferencial deste projeto é a sua arquitetura de extração de imagem, que utiliza máscaras dinâmicas e operações morfológicas para garantir que a Inteligência Artificial se foque apenas na geometria da mão, ignorando o ambiente em redor.

---

## 🧠 Como Funciona (A Ciência por Trás do Projeto)

Para garantir a máxima precisão (evitando que o modelo adivinhe as letras com base na cor da parede ou na roupa do utilizador), o projeto foi dividido em dois grandes pilares:

### 1. O Cérebro: Inteligência Artificial (TensorFlow & Keras)
Para reconhecer as letras, não criámos uma rede do zero. Utilizámos uma técnica chamada **Transfer Learning**. Pegámos no modelo **MobileNet** (uma rede neural previamente treinada no dataset ImageNet para reconhecer milhares de objetos), congelámos a sua base de conhecimento de formas e texturas, e substituímos as últimas camadas para que ele aprendesse a classificar especificamente o alfabeto de Libras.

**A Luta contra o "Viés de Fundo" (Background Bias):**
Durante o desenvolvimento, mapas de calor (Grad-CAM) revelaram que a rede estava a memorizar o fundo da imagem em vez do formato da mão. Para resolver isso, o pipeline de treino foi reforçado com:
* **Data Augmentation:** As imagens de treino sofrem rotações de 20 graus, zoom e deslocamentos aleatórios a cada época. Isto destrói o padrão do fundo e força a IA a focar-se na forma central.
* **Dropout Agressivo:** Inserimos camadas de Dropout (que desligam 50% e 30% dos neurónios aleatoriamente durante o treino). Isto impede a rede de decorar píxeis específicos do ambiente e obriga-a a generalizar o formato dos dedos.
* **Leitura Dinâmica de Classes:** O modelo lê automaticamente a quantidade de categorias a partir dos dados de treino (`y_train.shape[1]`), adaptando a camada de saída sem gerar erros de índice.

### 2. Os Olhos: Visão Computacional (MediaPipe & OpenCV)
Se enviássemos a imagem inteira da câmara, a IA iria confundir-se com o rosto do utilizador ou objetos de fundo. Precisávamos de extrair **apenas a mão num fundo perfeitamente branco**.

Para isso, desenvolvemos uma **Máscara Morfológica Inteligente**:
1. O **MediaPipe** processa a frame da câmara e deteta as coordenadas exatas de 21 pontos articulares (*landmarks*) da mão.
2. Com o **OpenCV**, desenhamos círculos nestas articulações e linhas a conectá-las.
3. **Pontos de Âncora Extras:** Como posições complexas (como a letra "C") deixavam falhas no centro da mão, o código calcula matematicamente novos pontos (no centro da palma e entre o polegar e o indicador) para dar "volume" à máscara.
4. **Dilatação Matemática:** Aplicamos a função `cv2.dilate` (Dilatação), que expande os limites da máscara de forma uniforme, garantindo que as pontas dos dedos e a palma não sejam cortadas.
5. **O Resultado:** A silhueta perfeita da mão é isolada e colada num fundo 100% branco. É esta imagem cirurgicamente limpa (redimensionada para 64x64 píxeis) que é entregue à IA.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.12**
* **TensorFlow (2.16.1):** Construção, treino e inferência da Rede Neural (MobileNet).
* **MediaPipe (0.10.14):** Deteção e rastreio espacial das mãos em milissegundos.
* **OpenCV (4.10.0.84):** Captura de vídeo, processamento da máscara, extração de ROI e operações morfológicas.
* **Protobuf (4.25.3):** Serialização de dados e gestão interna dos modelos do TensorFlow/MediaPipe.
* **Pandas & Matplotlib:** Para a extração e visualização do histórico de treino e gráficos de precisão.

---

## ⚙️ Instalação e Configuração (Ambiente Blindado)

Devido a conflitos estruturais conhecidos entre as versões mais recentes do TensorFlow, as novas APIs do MediaPipe e as atualizações do Numpy 2.0, criámos um ambiente "blindado" para garantir que o projeto funciona perfeitamente em **Python 3.12** sem erros de compatibilidade.

Abra o terminal (Prompt de Comando ou PowerShell) e instale as dependências exatas executando:

```bash
py -3.12 -m pip install tensorflow==2.16.1 protobuf==4.25.3 opencv-python==4.10.0.84 mediapipe==0.10.14 pandas matplotlib scikit-learn
