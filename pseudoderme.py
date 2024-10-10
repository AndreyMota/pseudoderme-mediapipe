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

                # Criar máscara branca
                mask = np.zeros_like(frame, dtype=np.uint8)

                # Obter os pontos da malha facial
                points = np.array([(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                    int(face_landmarks.landmark[i].y * frame.shape[0])) 
                                   for i in range(len(face_landmarks.landmark))], np.int32)

                # Criar uma triangulação Delaunay
                tri = scipy.spatial.Delaunay(points)

                # Preencher a máscara com triângulos brancos
                for simplex in tri.simplices:
                    triangle = points[simplex]
                    cv2.fillPoly(mask, [triangle], (255, 255, 255))  # Máscara branca

                # Criar uma imagem da cor da pele
                skin_color_image = np.zeros_like(frame, dtype=np.uint8)
                skin_color_image[:] = (int(skin_color[0]), int(skin_color[1]), int(skin_color[2]))  # Preencher com a cor média da pele

                # Aplicar a máscara da cor da pele
                final_skin_color = cv2.bitwise_and(skin_color_image, mask)

                # Combinar a máscara branca com a cor média da pele
                mask_opacity = 1.0  # Opacidade total para a máscara
                skin_color_opacity = 0.5  # Opacidade da cor média da pele

                # Adicionar as duas camadas
                final_frame = cv2.addWeighted(mask, mask_opacity, final_skin_color, skin_color_opacity, 0)

                # Exibir o resultado final
                cv2.imshow('Filtro Pseudoderme', final_frame)

        else:
            # Exibir o quadro original se não houver rostos detectados
            cv2.imshow('Filtro Pseudoderme', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
