import cv2
import mediapipe as mp
import joblib
import numpy as np

print("A carregar os 3 Modelos de Inteligência Artificial e o Padronizador...")
modelo_mlp = joblib.load('modelo_mlp.pkl')
modelo_rf = joblib.load('modelo_rf.pkl')
modelo_svm = joblib.load('modelo_svm.pkl')
scaler = joblib.load('scaler_libras.pkl')

print("A iniciar a câmara e o MediaPipe...")
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

detector_maos = mp_maos.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
captura = cv2.VideoCapture(0)

def obter_cor(confianca):
    """Devolve verde se a confiança for alta, laranja se for baixa."""
    return (0, 255, 0) if confianca > 60 else (0, 165, 255)

while True:
    sucesso, frame = captura.read()
    if not sucesso:
        print("Erro ao aceder à câmara!")
        break
    
    # Espelha a imagem
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Cria um painel lateral escuro de 350 píxeis de largura
    painel = np.zeros((h, 350, 3), dtype=np.uint8)
    
    # Título do Dashboard
    cv2.putText(painel, "DASHBOARD DA IA", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.line(painel, (30, 55), (320, 55), (255, 255, 255), 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = detector_maos.process(frame_rgb)
    
    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)
            
            # 1. Normalização de Posição (Pulso)
            pulso_x, pulso_y, pulso_z = landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z
            
            linha_temp = []
            for ponto in landmarks.landmark:
                linha_temp.extend([ponto.x - pulso_x, ponto.y - pulso_y, ponto.z - pulso_z])
                
            # 2. Normalização de Escala (Proporção)
            valor_maximo = max(map(abs, linha_temp))
            if valor_maximo == 0: valor_maximo = 1.0
            linha_normalizada = [valor / valor_maximo for valor in linha_temp]
            
            # Prepara os dados
            pontos_array = np.array(linha_normalizada).reshape(1, -1)
            pontos_escalonados = scaler.transform(pontos_array)
            
            # ==========================================
            # PREVISÕES DOS 3 MODELOS
            # ==========================================
            def prever_sinal(modelo):
                previsao = modelo.predict(pontos_escalonados)[0]
                probabilidade = np.max(modelo.predict_proba(pontos_escalonados)[0]) * 100
                return previsao, probabilidade

            prev_mlp, conf_mlp = prever_sinal(modelo_mlp)
            prev_rf, conf_rf = prever_sinal(modelo_rf)
            prev_svm, conf_svm = prever_sinal(modelo_svm)
            
            # DESENHAR RESULTADOS NO PAINEL LATERAL
            # Texto MLP
            cv2.putText(painel, "Rede Neural (MLP):", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(painel, f"{prev_mlp} ({conf_mlp:.1f}%)", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.1, obter_cor(conf_mlp), 2)
            
            # Texto RF
            cv2.putText(painel, "Random Forest:", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(painel, f"{prev_rf} ({conf_rf:.1f}%)", (30, 245), cv2.FONT_HERSHEY_SIMPLEX, 1.1, obter_cor(conf_rf), 2)
            
            # Texto SVM
            cv2.putText(painel, "Support Vector Machine:", (30, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(painel, f"{prev_svm} ({conf_svm:.1f}%)", (30, 345), cv2.FONT_HERSHEY_SIMPLEX, 1.1, obter_cor(conf_svm), 2)

    # Juntar a câmara com o painel lateral (Dashboard)
    tela_final = np.hstack((frame, painel))
    
    cv2.imshow('Tradutor de Libras - Comparativo de Modelos', tela_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()