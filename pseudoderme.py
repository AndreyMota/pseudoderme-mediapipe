import cv2
import mediapipe as mp
import numpy as np
import scipy.spatial

# Inicializar o Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# Lista de cores em escala de cinza
gray_colors = [
    (220, 220, 220),  # Cinza claro
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

# Função para atribuir cores fixas aos triângulos
def assign_colors_to_triangles(num_triangles):
    colors = []
    for i in range(num_triangles):
        color = gray_colors[i % len(gray_colors)]  # Loop pelas cores se necessário
        colors.append(color)
    return colors

# Função para criar uma máscara colorida com triângulos
def create_static_mask(triangles, points, frame_shape):
    mask = np.zeros(frame_shape, dtype=np.uint8)
    triangle_colors = assign_colors_to_triangles(len(triangles))
    
    for index, triangle in enumerate(triangles):
        pts = points[triangle].astype(np.int32)
        color = triangle_colors[index]
        cv2.fillPoly(mask, [pts], color)
        
    return mask

# Captura de vídeo
cap = cv2.VideoCapture(0)

# Variáveis de controle
static_mask = None
triangles = None

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

        # Se houver um rosto detectado
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Obtenha os pontos faciais
                points = np.array([(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                    int(face_landmarks.landmark[i].y * frame.shape[0])) 
                                   for i in range(len(face_landmarks.landmark))], np.int32)

                # Calcular triângulos de Delaunay com base nos pontos faciais
                if triangles is None:
                    delaunay_triangles = scipy.spatial.Delaunay(points)
                    triangles = delaunay_triangles.simplices

                    # Crie a máscara estática apenas uma vez
                    static_mask = create_static_mask(triangles, points, frame.shape)

                # Mapeie a máscara estática para o rosto detectado, mas sem recalcular as cores
                mask_copy = np.zeros_like(frame)
                for index, triangle in enumerate(triangles):
                    pts = points[triangle].astype(np.int32)
                    color = gray_colors[index % len(gray_colors)]
                    cv2.fillPoly(mask_copy, [pts], color)
                
                # Criar uma máscara binária para sobrepor ao rosto
                gray_mask = cv2.cvtColor(mask_copy, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

                # Inverter a máscara binária
                inverted_mask = cv2.bitwise_not(binary_mask)

                # Aplicar a área do rosto original fora da máscara
                face_area = cv2.bitwise_and(frame, frame, mask=inverted_mask)

                # Combinar o rosto original com a máscara estática
                final_frame = cv2.add(face_area, mask_copy)

                # Exibir o resultado final
                cv2.imshow('Filtro Pseudoderme', final_frame)

        else:
            # Se não houver rosto detectado, mostre a tela de standby
            standby_screen = np.zeros_like(frame)
            cv2.imshow('Filtro Pseudoderme', standby_screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
