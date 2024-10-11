import cv2
import mediapipe as mp
import numpy as np
import scipy.spatial

# Inicializar Mediapipe
mp_face_mesh = mp.solutions.face_mesh

# Lista de cores em escala de cinza
gray_colors = [
    (220, 220, 220),  # Claro
    (200, 200, 200),
    (180, 180, 180),
    (160, 160, 160),
    (140, 140, 140),
    (120, 120, 120),
    (100, 100, 100),
    (80, 80, 80),
    (60, 60, 60),
    (40, 40, 40),
    (20, 20, 20),
    (0, 0, 0)        # Preto
]

# Função para atribuir cores em escala de cinza aos triângulos
def assign_colors_to_triangles(num_triangles):
    colors = []
    for i in range(num_triangles):
        # Atribui uma cor da lista a cada triângulo
        color = gray_colors[i % len(gray_colors)]  # Loop pelas cores se necessário
        colors.append(color)
    return colors

# Função para calcular as cores dos triângulos
def calculate_triangle_colors(points):
    tri = scipy.spatial.Delaunay(points)
    triangle_colors = assign_colors_to_triangles(len(tri.simplices))
    return triangle_colors

# Captura de vídeo
cap = cv2.VideoCapture(0)

# Variáveis para armazenar cores dos triângulos
triangles_colors = None

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
                # Obter os pontos da malha facial
                points = np.array([(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                    int(face_landmarks.landmark[i].y * frame.shape[0])) 
                                   for i in range(len(face_landmarks.landmark))], np.int32)

                # Calcular as cores dos triângulos apenas na primeira detecção
                if triangles_colors is None:  # Se ainda não calculou as cores
                    triangles_colors = calculate_triangle_colors(points)

                # Preencher a máscara com as cores dos triângulos
                mask = np.zeros_like(frame, dtype=np.uint8)
                tri = scipy.spatial.Delaunay(points)  # Recalcular triângulos para preencher corretamente
                for index, simplex in enumerate(tri.simplices):
                    if index < len(triangles_colors):
                        triangle = points[simplex]
                        color = triangles_colors[index]
                        cv2.fillPoly(mask, [triangle], color)

                # Criar uma máscara binária para o rosto
                gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

                # Inverter a máscara para aplicar o fundo original
                inverted_mask = cv2.bitwise_not(binary_mask)

                # Manter o fundo original fora da máscara
                face_area = cv2.bitwise_and(frame, frame, mask=inverted_mask)

                # Combinar o fundo original com o rosto filtrado
                final_frame = cv2.add(face_area, mask)

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
