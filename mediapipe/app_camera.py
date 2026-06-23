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
    
    # O MediaPipe exige que a imagem esteja no formato RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = detector_maos.process(frame_rgb)
    
    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            # Desenha o esqueleto da mão na tela (Linhas e Pontos)
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)
            
            # 1. NORMALIZAÇÃO DE POSIÇÃO (Subtrair o Pulso)
            pulso_x = landmarks.landmark[0].x
            pulso_y = landmarks.landmark[0].y
            pulso_z = landmarks.landmark[0].z
            
            linha_temp = []
            for ponto in landmarks.landmark:
                linha_temp.extend([
                    ponto.x - pulso_x, 
                    ponto.y - pulso_y, 
                    ponto.z - pulso_z
                ])
                
            # 2. NORMALIZAÇÃO DE ESCALA (Proporção)
            # Encontra a maior distância absoluta desta mão
            valor_maximo = max(map(abs, linha_temp))
            if valor_maximo == 0: 
                valor_maximo = 1.0 # Evita divisão por zero
            
            # Divide todos os valores pelo valor máximo
            linha_normalizada = [valor / valor_maximo for valor in linha_temp]
            
            # Prepara a lista para o formato que a IA exige (1 linha, 63 colunas)
            pontos_array = np.array(linha_normalizada).reshape(1, -1)
            
            # Aplica o Padronizador (Crucial! O mesmo que usamos no treino)
            pontos_escalonados = scaler.transform(pontos_array)
            
            # Pede à IA para prever a letra e calcular a percentagem de certeza
            previsao = modelo.predict(pontos_escalonados)[0]
            probabilidades = modelo.predict_proba(pontos_escalonados)[0]
            confianca = np.max(probabilidades) * 100
            
            # Exibe o resultado na tela consoante a confiança da IA
            if confianca > 60:
                texto = f'Letra: {previsao} ({confianca:.1f}%)'
                cor = (0, 255, 0) # Verde = Sinal Reconhecido e Validado
            else:
                texto = f'A analisar...'
                cor = (0, 165, 255) # Laranja = IA está confusa ou a mão a mover-se
            
            # Coloca o texto no canto superior esquerdo
            cv2.putText(frame, texto, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)

    # Mostra a janela da câmara
    cv2.imshow('Tradutor de Libras (MediaPipe + Machine Learning)', frame)
    
    # Pressione 'q' no teclado para fechar a janela em segurança
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()