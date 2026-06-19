import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

print("Carregando a Inteligência Artificial...")
modelo = tf.keras.models.load_model('modelo_transfer_libras.keras') 
letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'Y']

print("Carregando o Detector de Mãos (MediaPipe)...")
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

detector_maos = mp_maos.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

captura = cv2.VideoCapture(0)

while True:
    sucesso, frame = captura.read()
    if not sucesso: break
    
    frame = cv2.flip(frame, 1)
    altura_frame, largura_frame, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    resultados = detector_maos.process(frame_rgb)
    
    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)
            
            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]
            
            x_min = int(min(x_coords) * largura_frame)
            x_max = int(max(x_coords) * largura_frame)
            y_min = int(min(y_coords) * altura_frame)
            y_max = int(max(y_coords) * altura_frame)
            
            margem = 40
            x_min = max(0, x_min - margem)
            y_min = max(0, y_min - margem)
            x_max = min(largura_frame, x_max + margem)
            y_max = min(altura_frame, y_max + margem)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            recorte_mao = frame[y_min:y_max, x_min:x_max]
            
            if recorte_mao.shape[0] > 0 and recorte_mao.shape[1] > 0:
                recorte_redimensionado = cv2.resize(recorte_mao, (64, 64))
                
                # --- JANELA DE VISÃO DA IA ---
                # Exibe a imagem de 64x64 que a IA está analisando
                cv2.imshow('Visao da IA (64x64)', recorte_redimensionado)
                
                recorte_normalizado = recorte_redimensionado / 255.0
                roi_pronta = np.expand_dims(recorte_normalizado, axis=0)
                
                previsoes = modelo.predict(roi_pronta, verbose=0)
                probabilidades = previsoes[0]
                
                indice_vencedor = np.argmax(probabilidades)
                confianca = probabilidades[indice_vencedor]
                
                if 0 <= indice_vencedor < len(letras):
                    if confianca > 0.50:
                        letra_predita = letras[indice_vencedor]
                        texto = f'Letra: {letra_predita} ({confianca*100:.1f}%)'
                        cv2.putText(frame, texto, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Tradutor de Libras - MediaPipe + CNN', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()