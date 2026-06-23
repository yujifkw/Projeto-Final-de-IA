# Tradutor de Libras em Tempo Real

Este projeto é um sistema de Inteligência Artificial criado para reconhecer e traduzir as letras do alfabeto de Libras (Língua Brasileira de Sinais) em tempo real, utilizando a câmara do computador.

Diferente de abordagens tradicionais baseadas em processamento de imagem (CNNs), este projeto utiliza técnicas avançadas de **Extração de Características (Feature Extraction)** com o MediaPipe, transformando o problema de Visão Computacional num problema de **Dados Tabulares**. Isso garante um treinamento incrivelmente rápido e elimina completamente o "viés de fundo" (Background Bias).

---

## A Ciência por Trás do Projeto

Para garantir precisão máxima e performance em tempo real, o projeto foi estruturado em três grandes pilares:

### 1. Extração de Coordenadas (O "Esqueleto" da Mão)
Em vez de analisar milhares de píxeis de uma imagem, utilizamos o **Google MediaPipe** para detetar a mão e extrair as coordenadas matemáticas (X, Y, Z) de 21 articulações (*landmarks*). Cada imagem do dataset é convertida numa simples linha de 63 números.
* **Normalização pelo Pulso:** Para evitar que a IA memorize "onde" a mão está na tela, aplicamos uma normalização matemática. Subtraímos as coordenadas do Pulso (Ponto 0) de todos os outros dedos. Assim, a IA foca-se puramente na **geometria e formato** do sinal, independentemente da distância ou posição da mão na câmara.

### 2. A Batalha das IAs (Comparação de Modelos)
Para fins de análise comparativa de Machine Learning, treinamos dois modelos clássicos e robustos usando a biblioteca `scikit-learn`:
* **Rede Neural Multicamadas (MLP):** Capaz de encontrar padrões complexos não-lineares nos ângulos dos dedos.
* **Random Forest (Floresta Aleatória):** Um conjunto de árvores de decisão perfeito para dados tabulares, que nos permite até mesmo extrair um "Mapa de Atenção" para saber quais dedos foram mais importantes para a IA.

### 3. O Tradutor em Tempo Real (Câmara)
A aplicação ao vivo aplica a mesma extração e padronização (`StandardScaler`) feita no treino. Adicionámos um limiar de confiança (*Confidence Threshold*): o sistema calcula a probabilidade da letra e só valida o sinal e o exibe a verde se tiver mais de **60% de certeza**, evitando flutuações e erros de leitura durante o movimento da mão.

---

## Tecnologias Utilizadas

* **Python 3.12**
* **MediaPipe (0.10.14):** Deteção e rastreio espacial das mãos.
* **Scikit-Learn:** Padronização de dados, treino da Rede Neural (MLP), Random Forest e métricas de avaliação.
* **OpenCV:** Captura de vídeo e processamento de frames em tempo real.
* **Joblib:** Serialização e salvamento do modelo e dos padronizadores.
* **Pandas, Matplotlib & Seaborn:** Manipulação de dados e geração de gráficos (Curva de Loss e Matrizes de Confusão).

---

## Instalação e Configuração

Certifique-se de usar o **Python 3.12** para garantir compatibilidade total com as bibliotecas. No seu terminal, execute o comando abaixo para instalar as dependências exatas:

```bash
py -3.12 -m pip install tensorflow==2.16.1 protobuf==4.25.3 opencv-python==4.10.0.84 mediapipe==0.10.14 pandas matplotlib scikit-learn seaborn==0.13.2 kagglehub==1.0.2
```

## Estrutura e Como Executar

O projeto está dividido de forma modular, separando a extração de dados, o treinamento e a aplicação final.

### Fase 1: Preparação e Treinamento (Jupyter Notebook)
Abra o arquivo de treinamento e execute as células sequencialmente:
1. **Extração de Dados:** Baixa o dataset, passa o MediaPipe em todas as imagens, normaliza as coordenadas pelo pulso e salva as variáveis de ambiente (`dados_ambiente.pkl`).
2. **Treinamento MLP:** Treina a Rede Neural Artificial e mede o tempo de execução.
3. **Treinamento Random Forest:** Treina a Floresta Aleatória.
4. **Avaliação Final:** Compara a Acurácia dos dois modelos, desenha a curva de Loss, gera as Matrizes de Confusão, salva o modelo vencedor como `modelo_libras_pontos_final.pkl` e extrai o Mapa de Atenção (Feature Importance).

### Fase 2: O Tradutor em Tempo Real
Com os arquivos `.pkl` gerados e salvos na pasta raiz, abra o terminal e execute a aplicação da câmara:

```bash
py -3.12 app_camera.py
```

**O que vai acontecer:**
A sua webcam será ativada. Ao fazer um sinal de Libras, o MediaPipe desenhará o esqueleto da sua mão. No canto superior esquerdo, você verá a previsão da letra juntamente com a **percentagem de confiança** da Inteligência Artificial. Pressione a tecla **'q'** para fechar a aplicação em segurança.