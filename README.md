# Projeto Final IA

# 🤟 Tradutor de Libras em Tempo Real (MediaPipe + CNN)

Este projeto é um sistema de Visão Computacional criado para reconhecer e traduzir as letras do alfabeto de Libras (Língua Brasileira de Sinais) em tempo real, utilizando a webcam.

O sistema combina o rastreamento de mãos avançado do **Google MediaPipe** com uma rede neural profunda de **Transfer Learning (MobileNet via TensorFlow/Keras)**, focada em reconhecer a geometria da mão extraída por meio de máscaras dinâmicas.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.12**
* **TensorFlow (2.16.1)**: Treinamento e inferência da Rede Neural (MobileNet).
* **MediaPipe (0.10.14)**: Detecção dos *landmarks* (pontos articulares) da mão.
* **OpenCV (4.10.0.84)**: Captura de vídeo, manipulação de imagens e criação da máscara isoladora.
* **Protobuf (4.25.3)**: Gerenciamento dos modelos do MediaPipe/TensorFlow.

---

## ⚙️ Instalação e Configuração (Ambiente Blindado)

Devido a conflitos conhecidos de versão entre o TensorFlow mais recente, as novas APIs do MediaPipe e as atualizações do Numpy 2.0, criamos um ambiente "blindado" para garantir que tudo funcione perfeitamente em **Python 3.12**.

Abra o terminal (Prompt de Comando) e instale as dependências exatas utilizando o comando abaixo:

```bash
py -3.12 -m pip install tensorflow==2.16.1 protobuf==4.25.3 opencv-python==4.10.0.84 mediapipe==0.10.14