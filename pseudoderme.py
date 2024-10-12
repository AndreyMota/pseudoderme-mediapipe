import cv2
import mediapipe as mp
import numpy as np
import scipy.spatial
import time

# Inicializar o Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# Função para calcular a cor média de uma região triangular
def calculate_average_color(frame, triangle):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [triangle], 255)

    # Extrair os pixels dentro do triângulo
    triangle_pixels = cv2.bitwise_and(frame, frame, mask=mask)

    # Encontrar os pixels não pretos
    triangle_pixels = triangle_pixels[np.where(mask == 255)]

    # Verificar se há pixels não pretos
    non_black_pixels = triangle_pixels[np.all(triangle_pixels != [0, 0, 0], axis=1)]

    if len(non_black_pixels) == 0:
        return (127, 127, 127)  # Retorna cinza padrão se não houver pixels válidos

    # Calcular a cor média
    mean_color = np.mean(non_black_pixels, axis=0)

    return tuple(int(v) for v in mean_color)


# Função para calcular a média das cores ao longo de várias capturas
def calculate_average_color_over_time(cap, triangles, points, duration=3, fps=5):
    total_frames = int(duration * fps)
    triangle_colors_sum = np.zeros((len(triangles), 3))  # Somar as cores de todos os triângulos

    frames_captured = 0
    start_time = time.time()
    
    while frames_captured < total_frames and (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Converter a imagem para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calcular as cores de cada triângulo
        for idx, triangle in enumerate(triangles):
            pts = points[triangle].astype(np.int32)
            color = calculate_average_color(frame, pts)  # Cor média do triângulo
            triangle_colors_sum[idx] += color  # Somar as cores para calcular a média ao final

        frames_captured += 1

        # Mostrar a imagem da câmera enquanto coleta frames
        cv2.imshow('Filtro Pseudoderme', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calcular a média das cores sobre todas as capturas
    triangle_colors_avg = triangle_colors_sum / max(frames_captured, 1)
    return [tuple(map(int, color)) for color in triangle_colors_avg]


# Função para criar uma máscara colorida com triângulos
def create_static_mask_with_average_colors(frame, triangles, points, triangle_colors):
    mask = np.zeros(frame.shape, dtype=np.uint8)

    for idx, triangle in enumerate(triangles):
        pts = points[triangle].astype(np.int32)
        color = triangle_colors[idx]  # Usar a cor média calculada
        cv2.fillPoly(mask, [pts], color)

    return mask

# Captura de vídeo
cap = cv2.VideoCapture(0)

# Variáveis de controle
static_mask = None
triangles = None
triangle_colors = None
calibration_done = False

# Adicionar um delay para a câmera ajustar a iluminação
pre_capture_delay = 3  # Segundos

with mp_face_mesh.FaceMesh(static_image_mode=False,
                           max_num_faces=1,
                           refine_landmarks=True) as face_mesh:

    print(f"Preparando... Aguarde {pre_capture_delay} segundos.")

    # Exibir a imagem da câmera enquanto aguarda o delay
    start_time = time.time()
    while time.time() - start_time < pre_capture_delay:
        ret, frame = cap.read()
        if not ret:
            break

        # Mostrar a imagem da câmera enquanto o programa aguarda
        cv2.imshow('Filtro Pseudoderme', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Iniciando a captura de cores...")

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

                if not calibration_done:
                    # Exibir mensagem de calibração
                    cv2.putText(frame, "Por favor, mantenha o rosto de frente por 3 segundos", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow('Filtro Pseudoderme', frame)
                    cv2.waitKey(1)  # Atualizar a janela

                    # Calcular as cores médias ao longo de 3 segundos com uma taxa de fps reduzida
                    triangle_colors = calculate_average_color_over_time(cap, triangles, points, duration=3, fps=5)

                    # Criar a máscara estática com as cores médias calculadas
                    static_mask = create_static_mask_with_average_colors(frame, triangles, points, triangle_colors)
                    calibration_done = True

                # Atualizar a posição da máscara, mas usar as mesmas cores calculadas
                mask_copy = np.zeros_like(frame)
                for index, triangle in enumerate(triangles):
                    pts = points[triangle].astype(np.int32)
                    color = triangle_colors[index]  # Usar a cor média calculada
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
