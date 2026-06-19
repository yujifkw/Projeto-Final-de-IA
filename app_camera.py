import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input

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
    altura, largura, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = detector_maos.process(frame_rgb)
    
    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)
            
            mask = np.zeros((altura, largura), dtype=np.uint8)
            
            # 1. Desenha as conexões padrão
            for connection in mp_maos.HAND_CONNECTIONS:
                p1 = landmarks.landmark[connection[0]]
                p2 = landmarks.landmark[connection[1]]
                cv2.line(mask, (int(p1.x*largura), int(p1.y*altura)), (int(p2.x*largura), int(p2.y*altura)), 255, 20)
            
            # 2. Adiciona Pontos de Tracking Extras (Âncoras)
            # Palma: Média entre pulso (0) e base do dedo médio (9)
            palma_x = int(((landmarks.landmark[0].x + landmarks.landmark[9].x) / 2) * largura)
            palma_y = int(((landmarks.landmark[0].y + landmarks.landmark[9].y) / 2) * altura)
            cv2.circle(mask, (palma_x, palma_y), 30, 255, -1)
            
            # Entre Dedão e Indicador: Média entre base do polegar (2) e base do indicador (5)
            entre_x = int(((landmarks.landmark[2].x + landmarks.landmark[5].x) / 2) * largura)
            entre_y = int(((landmarks.landmark[2].y + landmarks.landmark[5].y) / 2) * altura)
            cv2.circle(mask, (entre_x, entre_y), 25, 255, -1)
            
            # 3. Dilatação para suavizar tudo
            kernel = np.ones((20, 20), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Isola a mão
            mao_isolada = cv2.bitwise_and(frame, frame, mask=mask)
            mao_no_branco = np.where(mask[:, :, np.newaxis] == 255, mao_isolada, 255)
            
            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]
            x_min, x_max = int(min(x_coords)*largura)-50, int(max(x_coords)*largura)+50
            y_min, y_max = int(min(y_coords)*altura)-50, int(max(y_coords)*altura)+50
            
            recorte = mao_no_branco[max(0, y_min):min(altura, y_max), max(0, x_min):min(largura, x_max)]
            
            if recorte.size > 0:
                recorte_ia = cv2.resize(recorte, (64, 64))
                cv2.imshow('Visao da IA (Mascarada)', cv2.resize(recorte_ia, (300, 300)))
                cv2.moveWindow('Visao da IA (Mascarada)', 700, 50)
                
                img_array = np.expand_dims(recorte_ia, axis=0)
                img_preprocessada = preprocess_input(img_array.astype(np.float32))
                
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