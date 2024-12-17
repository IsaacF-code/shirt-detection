import cv2
import numpy as np
from ultralytics import YOLO

def detect_yellow_shirt(video_path, output_file="yellow_shirt_times.txt"):
    # Carregar modelo YOLO pré-treinado para detectar pessoas
    model = YOLO("yolov8n.pt")  # Baixe o YOLOv8 modelo antes de usar
    cap = cv2.VideoCapture(video_path)

    # Variáveis de controle
    fps = cap.get(cv2.CAP_PROP_FPS)  # Obter frames por segundo
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Contar total de frames
    duration = frame_count / fps  # Duração do vídeo em segundos

    # Para salvar timestamps
    yellow_shirt_times = []

    # Processar frames
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Processar o frame a cada 10 frames para desempenho
        if frame_index % 10 == 0:
            results = model(frame, conf=0.5, classes=[0])  # Detectar apenas pessoas
            for result in results:
                boxes = result.boxes  # Coletar bounding boxes detectadas
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box
                    person_crop = frame[y1:y2, x1:x2]  # Recortar a pessoa detectada
                    
                    # Converter para HSV e aplicar máscara para detectar amarelo
                    hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
                    lower_yellow = np.array([20, 100, 100])  # Tons de amarelo
                    upper_yellow = np.array([30, 255, 255])
                    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

                    # Verificar se há quantidade significativa de amarelo
                    if cv2.countNonZero(mask) > 500:  # Ajuste o threshold conforme necessário
                        time_in_seconds = frame_index / fps
                        yellow_shirt_times.append(time_in_seconds)

        frame_index += 1

    cap.release()

    # Salvar resultados
    with open(output_file, "w") as f:
        for t in yellow_shirt_times:
            minutes = int(t // 60)
            seconds = int(t % 60)
            f.write(f"{minutes:02}:{seconds:02}\n")
    
    print(f"Análise concluída! Timestamps salvos em {output_file}")

# Executar a função
video_path = "C:\\Users\\DTI\\Documents\\Pasta\\Python\\shirt-detection\\shirt-detection\\assets\\teste.avi"  # Substitua pelo caminho do seu vídeo
detect_yellow_shirt(video_path)
