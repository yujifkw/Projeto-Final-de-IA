import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

# Esconde os avisos do TensorFlow no terminal para uma execução mais limpa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

print("A carregar os Modelos Geométricos (Sklearn)...")
modelo_mlp = joblib.load('modelo_mlp.pkl')
modelo_rf = joblib.load('modelo_rf.pkl')
modelo_svm = joblib.load('modelo_svm.pkl')
scaler = joblib.load('scaler_libras.pkl')

print("A carregar os Modelos de Imagem Profunda (TensorFlow/Keras)...")
# Nota: O carregamento dos modelos .keras pode demorar alguns segundos
modelo_cnn = load_model('modelo_cnn_basico.keras')
modelo_transfer = load_model('modelo_transfer_libras.keras')

# Puxa a lista de classes (A, B, C...) de um dos modelos para traduzir o output das CNNs
lista_letras = modelo_rf.classes_

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
    
    # Espelha a imagem para o efeito "espelho" natural
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Cria o painel lateral do Dashboard com 400 píxeis de largura
    painel = np.zeros((h, 400, 3), dtype=np.uint8)
    cv2.putText(painel, "ULTIMATE DASHBOARD", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.line(painel, (30, 55), (370, 55), (255, 255, 255), 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = detector_maos.process(frame_rgb)
    
    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)
            
            # PROCESSAMENTO GEOMÉTRICO (SKLEARN)
            pulso_x, pulso_y, pulso_z = landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z
            
            linha_temp = []
            x_coords, y_coords = [], [] # Para o recorte de imagem
            
            for ponto in landmarks.landmark:
                linha_temp.extend([ponto.x - pulso_x, ponto.y - pulso_y, ponto.z - pulso_z])
                x_coords.append(ponto.x)
                y_coords.append(ponto.y)
                
            # Normalização de Escala
            valor_maximo = max(map(abs, linha_temp))
            if valor_maximo == 0: valor_maximo = 1.0
            linha_normalizada = [valor / valor_maximo for valor in linha_temp]
            
            # Padroniza e prevê com os modelos geométricos
            pontos_escalonados = scaler.transform(np.array(linha_normalizada).reshape(1, -1))
            
            prev_mlp = modelo_mlp.predict(pontos_escalonados)[0]
            conf_mlp = np.max(modelo_mlp.predict_proba(pontos_escalonados)[0]) * 100
            
            prev_rf = modelo_rf.predict(pontos_escalonados)[0]
            conf_rf = np.max(modelo_rf.predict_proba(pontos_escalonados)[0]) * 100
            
            prev_svm = modelo_svm.predict(pontos_escalonados)[0]
            conf_svm = np.max(modelo_svm.predict_proba(pontos_escalonados)[0]) * 100
            
            # PROCESSAMENTO DE IMAGEM (KERAS)
            margem = 40
            x_min = max(0, int(min(x_coords) * w) - margem)
            y_min = max(0, int(min(y_coords) * h) - margem)
            x_max = min(w, int(max(x_coords) * w) + margem)
            y_max = min(h, int(max(y_coords) * h) + margem)
            
            # Desenha um retângulo azul à volta da mão
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            recorte_mao = frame[y_min:y_max, x_min:x_max]
            
            prev_cnn, conf_cnn = "...", 0.0
            prev_transf, conf_transf = "...", 0.0
            
            if recorte_mao.size > 0:
                # CNN Básica
                # Lê o tamanho dinamicamente
                _, altura_cnn, largura_cnn, _ = modelo_cnn.input_shape
                recorte_para_cnn = cv2.resize(recorte_mao, (largura_cnn, altura_cnn))
                img_array_cnn = np.expand_dims(recorte_para_cnn, axis=0)
                
                prob_cnn = modelo_cnn.predict(img_array_cnn, verbose=0)[0]
                idx_cnn = np.argmax(prob_cnn)
                prev_cnn = lista_letras[idx_cnn]
                conf_cnn = prob_cnn[idx_cnn] * 100
                
                # Transfer Learning
                # Lê o tamanho dinamicamente
                _, altura_transf, largura_transf, _ = modelo_transfer.input_shape
                recorte_para_transf = cv2.resize(recorte_mao, (largura_transf, altura_transf))
                img_array_transf = np.expand_dims(recorte_para_transf, axis=0)
                
                img_preprocessada = preprocess_input(img_array_transf.astype(np.float32))
                prob_transf = modelo_transfer.predict(img_preprocessada, verbose=0)[0]
                idx_transf = np.argmax(prob_transf)
                prev_transf = lista_letras[idx_transf]
                conf_transf = prob_transf[idx_transf] * 100
            
            # ATUALIZAR O DASHBOARD
            modelos_ui = [
                ("Random Forest (Geometria)", prev_rf, conf_rf),
                ("SVM (Geometria)", prev_svm, conf_svm),
                ("Rede Neural MLP (Geometria)", prev_mlp, conf_mlp),
                ("Transfer Learning (Imagem)", prev_transf, conf_transf),
                ("CNN Basica (Imagem)", prev_cnn, conf_cnn)
            ]
            
            y_offset = 80
            for nome, prev, conf in modelos_ui:
                cv2.putText(painel, nome, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(painel, f"{prev} ({conf:.1f}%)", (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, obter_cor(conf), 2)
                y_offset += 65 # Espaçamento dinâmico para caber tudo na tela

    # Une a janela da câmara ao painel lateral
    tela_final = np.hstack((frame, painel))
    cv2.imshow('Tradutor de Libras', tela_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()