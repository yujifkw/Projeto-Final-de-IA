import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input

print("A carregar a Inteligência Artificial (224x224)...")
modelo = tf.keras.models.load_model('modelo_transfer_libras.keras') 
letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'Y']

print("A carregar o Detector de Mãos (MediaPipe)...")
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
detector_maos = mp_maos.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

captura = cv2.VideoCapture(0)

while True:
    sucesso, frame = captura.read()
    if not sucesso: break
    
    frame = cv2.flip(frame, 1)
    altura, largura, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = detector_maos.process(frame_rgb)
    
    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)
            
            # Captura as coordenadas extremas da mão
            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]
            
            # Define o Bounding Box com uma margem de segurança
            x_min, x_max = int(min(x_coords)*largura)-40, int(max(x_coords)*largura)+40
            y_min, y_max = int(min(y_coords)*altura)-40, int(max(y_coords)*altura)+40
            
            # Faz o recorte NATURAL da imagem (sem fundo branco artificial)
            # Isto alinha o que a câmara vê com as imagens do seu dataset!
            recorte = frame[max(0, y_min):min(altura, y_max), max(0, x_min):min(largura, x_max)]
            
            if recorte.size > 0:
                # REDIMENSIONAMENTO PARA A NOVA RESOLUÇÃO PROFISSIONAL: 224x224
                recorte_ia = cv2.resize(recorte, (224, 224))
                
                # Exibe o que a IA está a ver
                cv2.imshow('Visao da IA (Natural - 224x224)', recorte_ia)
                cv2.moveWindow('Visao da IA (Natural - 224x224)', 700, 50)
                
                # Pré-processamento do MobileNet
                img_array = np.expand_dims(recorte_ia, axis=0)
                img_preprocessada = preprocess_input(img_array.astype(np.float32))
                
                # Previsão
                previsoes = modelo.predict(img_preprocessada, verbose=0)
                indice = np.argmax(previsoes[0])
                
                if 0 <= indice < len(letras) and previsoes[0][indice] > 0.7:
                    cv2.putText(frame, f'Letra: {letras[indice]}', (max(0, x_min), max(0, y_min) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Tradutor de Libras', frame)
    cv2.moveWindow('Tradutor de Libras', 50, 50)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

captura.release()
cv2.destroyAllWindows()