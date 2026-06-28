# Tradutor de Libras em Tempo Real: Abordagem Geométrica vs. Visão Computacional

Este repositório documenta um projeto de pesquisa acadêmica desenvolvido na **Universidade Federal de São Paulo (UNIFESP)**, focado no reconhecimento da Língua Brasileira de Sinais (Libras). O trabalho investiga a transição entre modelos de *Deep Learning* baseados em imagem pura (CNNs) e uma abordagem geométrica baseada em extração de *landmarks* (MediaPipe) combinada com algoritmos clássicos de Machine Learning.

## O Problema
Modelos tradicionais baseados em Redes Neurais Convolucionais (CNNs) sofrem frequentemente de *Background Bias* (viés de fundo), onde a IA aprende a identificar o ambiente da foto em vez da anatomia da mão. Além disso, o custo computacional de treinar modelos de imagem profundos em tempo real é proibitivo para dispositivos de *Edge Computing*.

## A Solução
Propomos um *pipeline* que transforma o problema de Visão Computacional em um problema de classificação de dados tabulares. Utilizamos o **Google MediaPipe** para extrair 21 marcos topológicos da mão, aplicamos uma **normalização geométrica** (translação pelo pulso e escala) e validamos a eficácia de algoritmos como **Random Forest**, **SVM** e **MLP**.

## Resultados Principais
Após testes rigorosos com 30 execuções independentes, o **Random Forest** consolidou-se como o modelo vencedor pela sua robustez e eficiência.

| Arquitetura | Acurácia (Média) | Tempo de Treino |
| :--- | :--- | :--- |
| MLP (2 Camadas) | 99.97% | ~18.25s |
| **Random Forest** | **99.96%** | **~3.48s** |
| SVM | 99.91% | ~2.74s |

*Nota: Em inferência em tempo real no "Dashboard", o Random Forest atingiu 97.62% de acurácia global, com 100% de estabilidade ao processar a mão esquerda, provando a eficácia da normalização geométrica.*

## Stack Tecnológica
Para garantir a reprodutibilidade dos resultados, o ambiente foi rigorosamente versionado:

* **Linguagem:** Python 3.12
* **Visão Computacional:** `mediapipe==0.10.14`, `opencv-python==4.10.0.84`
* **Deep Learning:** `tensorflow==2.16.1`, `protobuf==4.25.3`
* **Machine Learning:** `scikit-learn`, `pandas`, `seaborn==0.13.2`

## Como Executar
1. Clone este repositório:
   ```bash
   git clone https://github.com/yujifkw/Projeto-Final-de-IA
   cd mediapipe
   ```

2. Instale as dependências:
    ```bash
    pip install tensorflow==2.16.1 protobuf==4.25.3 opencv-python==4.10.0.84 mediapipe==0.10.14 pandas matplotlib scikit-learn seaborn==0.13.2
    ```

3. Execute o Dashboard de inferência em tempo real:
    ```bash
    py -3.12 app_camera.py
    ```

## Declaração de Uso de IA
Durante o desenvolvimento desta pesquisa e a redação do relatório final, utilizamos ferramentas de IA Generativa como assistentes de suporte para:
* Revisão técnica e formatação (LaTeX/IEEE).
* Brainstorming sobre a normalização geométrica (cálculo de translação e escala).
* Depuração de código e otimização do fluxo de trabalho.

*Ressalta-se que a execução dos treinamentos, a extração dos dados, a análise estatística e a verificação empírica dos resultados são de total autoria dos pesquisadores.*

**Projeto desenvolvido pelos alunos Lucas Yuji Sapia Furukawa e Celine Kazumi Miyajima (BCT - UNIFESP).**