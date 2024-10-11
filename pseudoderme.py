import cv2
import mediapipe as mp
import numpy as np
import scipy.spatial

# Inicializar Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Função para obter a cor média da pele
def get_skin_color(frame, landmarks):
    skin_colors = []

    # Calcular a média das cores de todos os pontos da malha facial
    for i, landmark in enumerate(landmarks):
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])

        # Adicionar a cor do pixel à lista
        skin_colors.append(frame[y, x])

    # Calcular a cor média
    return np.mean(skin_colors, axis=0).astype(int)

# Captura de vídeo
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(static_image_mode=False,
                           max_num_faces=1,
                           refine_landmarks=True) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converter a imagem para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Obter a cor média da pele
                skin_color = get_skin_color(frame, face_landmarks.landmark)

                # Criar máscara branca do tamanho da face apenas
                mask = np.zeros_like(frame, dtype=np.uint8)

                # Obter os pontos da malha facial
                points = np.array([(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                    int(face_landmarks.landmark[i].y * frame.shape[0])) 
                                   for i in range(len(face_landmarks.landmark))], np.int32)

                # Criar uma triangulação Delaunay
                tri = scipy.spatial.Delaunay(points)

                # Preencher a máscara com triângulos usando a cor média da pele
                for simplex in tri.simplices:
                    triangle = points[simplex]
                    cv2.fillPoly(mask, [triangle], (int(skin_color[0]), int(skin_color[1]), int(skin_color[2])))  # Máscara com cor da pele

                # Criar uma máscara binária para o rosto
                gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

                # Inverter a máscara para aplicar o fundo original
                inverted_mask = cv2.bitwise_not(binary_mask)

                # Manter o fundo original fora da máscara
                face_area = cv2.bitwise_and(frame, frame, mask=inverted_mask)

                # Combinar o fundo original com o rosto filtrado
                final_frame = cv2.add(face_area, mask)  # Combinando a máscara colorida e o fundo original

                # Exibir o resultado final
                cv2.imshow('Filtro Pseudoderme', final_frame)

        else:
            # Exibir uma tela de fundo preto ou branco quando não houver rosto detectado
            standby_screen = np.zeros_like(frame)  # Fundo preto
            cv2.imshow('Filtro Pseudoderme', standby_screen)  # Para branco use np.ones_like(frame) * 255

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
