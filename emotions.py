import cv2
from deepface import DeepFace


# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Ler o quadro da câmera
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar o quadro")
        break

    # Tentar analisar as emoções com DeepFace
    try:
        # Usar o DeepFace para analisar emoções no quadro
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Extrair a emoção mais provável
        dominant_emotion = result['dominant_emotion']

        # Exibir a emoção no quadro
        cv2.putText(frame, f"Emocao: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"Erro ao detectar emoções: {e}")

    # Exibir o quadro
    cv2.imshow("Detecção de Emoções", frame)

    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar as janelas
cap.release()
cv2.destroyAllWindows()
