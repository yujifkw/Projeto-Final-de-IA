import cv2
import mediapipe as mp
import joblib
import numpy as np

print("A carregar a Inteligência Artificial e o Padronizador...")
# Carrega o modelo vencedor (Random Forest ou MLP) e o Scaler
modelo = joblib.load('modelo_libras_pontos_final.pkl')
scaler = joblib.load('scaler_libras.pkl')

print("A iniciar a câmara e o MediaPipe...")
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

# min_tracking_confidence ajuda a manter a estabilidade do esqueleto (menos tremidos)
detector_maos = mp_maos.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

captura = cv2.VideoCapture(0)

while True:
    sucesso, frame = captura.read()
    if not sucesso:
        print("Erro ao aceder à câmara!")
        break
    
    # Espelha a imagem para ficar natural (efeito espelho)
    frame = cv2.flip(frame, 1)
    
<<<<<<< Updated upstream
    # O MediaPipe exige que a imagem esteja no formato RGB
=======
    # Cria o painel lateral do Dashboard com 400 píxeis de largura
    painel = np.zeros((h, 400, 3), dtype=np.uint8)
    cv2.putText(painel, "DASHBOARD", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.line(painel, (30, 55), (370, 55), (255, 255, 255), 1)

>>>>>>> Stashed changes
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = detector_maos.process(frame_rgb)
    
    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            # 1. Desenha o esqueleto da mão na tela (Linhas e Pontos)
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)
            
            # 2. Extrai as 63 coordenadas matemáticas (X, Y, Z dos 21 pontos)
            linha_pontos = []
            for ponto in landmarks.landmark:
                linha_pontos.extend([ponto.x, ponto.y, ponto.z])
            
            # 3. Prepara a lista para o formato que a IA exige (1 linha, 63 colunas)
            pontos_array = np.array(linha_pontos).reshape(1, -1)
            
            # 4. APLICA O PADRONIZADOR (Crucial! O mesmo que usamos no treino)
            pontos_escalonados = scaler.transform(pontos_array)
            
            # 5. Pede à IA para prever a letra e calcular a percentagem de certeza
            previsao = modelo.predict(pontos_escalonados)[0]
            probabilidades = modelo.predict_proba(pontos_escalonados)[0]
            confianca = np.max(probabilidades) * 100
            
            # 6. Exibe o resultado na tela consoante a confiança da IA
            if confianca > 60:
                texto = f'Letra: {previsao} ({confianca:.1f}%)'
                cor = (0, 255, 0) # Verde = Sinal Reconhecido e Validado
            else:
                texto = f'A analisar...'
                cor = (0, 165, 255) # Laranja = IA está confusa ou a mão está a mover-se
            
            # Coloca o texto no canto superior esquerdo
            cv2.putText(frame, texto, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)

<<<<<<< Updated upstream
    # Mostra a janela da câmara
    cv2.imshow('Tradutor de Libras (MediaPipe + Machine Learning)', frame)
=======
    # Une a janela da câmara ao painel lateral
    tela_final = np.hstack((frame, painel))
    cv2.imshow('Tradutor de Libras', tela_final)
>>>>>>> Stashed changes
    
    # Pressione 'q' no teclado para fechar a janela em segurança
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()