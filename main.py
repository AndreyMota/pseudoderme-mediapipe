import cv2
import mediapipe as mp

# Inicializando o módulo de detecção facial do Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Converter o quadro para RGB (Mediapipe usa RGB, não BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar o quadro com o Mediapipe
    results = face_mesh.process(rgb_frame)

    # Se landmarks foram detectados
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i in range(0, 468):  # Mediapipe detecta 468 pontos faciais
                pt1 = face_landmarks.landmark[i]
                x = int(pt1.x * frame.shape[1])
                y = int(pt1.y * frame.shape[0])

                # Desenhar os pontos na imagem
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Exibir o resultado com os pontos
    cv2.imshow("Mediapipe - Landmarks Faciais", frame)

    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
