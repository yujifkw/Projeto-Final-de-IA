import cv2
import mediapipe as mp
import joblib
import numpy as np

print("A carregar a Inteligência Artificial e o Padronizador...")
# Carrega o modelo vencedor e o Scaler
# modelo = joblib.load('modelo_libras_pontos_final.pkl')
modelo = joblib.load('modelo_mlp.pkl')
scaler = joblib.load('scaler_libras.pkl')

print("A iniciar a câmara e o MediaPipe...")
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

detector_maos = mp_maos.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

captura = cv2.VideoCapture(0)

while True:
    sucesso, frame = captura.read()
    if not sucesso:
        print("Erro ao aceder à câmara!")
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = detector_maos.process(frame_rgb)
    
    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)
            
            linha_pontos = []
            
            # Captura a posição do pulso (Ponto 0)
            pulso_x = landmarks.landmark[0].x
            pulso_y = landmarks.landmark[0].y
            pulso_z = landmarks.landmark[0].z
            
            for ponto in landmarks.landmark:
                # Subtrai o pulso da coordenada atual antes de enviar para a IA
                linha_pontos.extend([
                    ponto.x - pulso_x, 
                    ponto.y - pulso_y, 
                    ponto.z - pulso_z
                ])
            
            pontos_array = np.array(linha_pontos).reshape(1, -1)
            pontos_escalonados = scaler.transform(pontos_array)
            
            previsao = modelo.predict(pontos_escalonados)[0]
            probabilidades = modelo.predict_proba(pontos_escalonados)[0]
            confianca = np.max(probabilidades) * 100
            
            if confianca > 60:
                texto = f'Letra: {previsao} ({confianca:.1f}%)'
                cor = (0, 255, 0)
            else:
                texto = f'A analisar...'
                cor = (0, 165, 255)
            
            cv2.putText(frame, texto, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)

    cv2.imshow('Tradutor de Libras (MediaPipe + Machine Learning)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()