import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

def detect_yellow_shirt(video_path):
    # Carregar modelo YOLO pré-treinado para detectar pessoas
    model = YOLO("yolov8n.pt")  # Baixe o modelo YOLOv8 antes de usar
    cap = cv2.VideoCapture(video_path)

    # Variáveis de controle
    fps = cap.get(cv2.CAP_PROP_FPS)  # Obter frames por segundo
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Contar total de frames

    # Para salvar timestamps
    yellow_shirt_times = []

    # Barra de progresso
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Processar frames
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Atualizar barra de progresso
        progress = frame_index / frame_count
        progress_bar.progress(min(progress, 1.0)) # Garantir que não ultrapasse 1.0
        progress_text.text(f"Processando... {int(progress * 100)}% concluído")

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
    return yellow_shirt_times

# Streamlit app
st.title("Detecção de Camisas Amarelas em Vídeos")
st.write("Carregue um vídeo e veja os momentos em que aparecem pessoas com camisas amarelas!")

uploaded_file = st.file_uploader("Carregar vídeo", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Salvar vídeo temporariamente
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    st.video(uploaded_file)

    # Botão para iniciar a análise
    if st.button("Iniciar análise"):
        with st.spinner("Processando... isso pode levar alguns minutos dependendo do vídeo."):
            yellow_shirt_times = detect_yellow_shirt(video_path)

        # Exibir resultados
        if yellow_shirt_times:
            st.success("Análise concluída! Momentos detectados:")
            for time in yellow_shirt_times:
                minutes = int(time // 60)
                seconds = int(time % 60)
                st.write(f"📌 {minutes:02}:{seconds:02}")
        else:
            st.warning("Nenhuma pessoa com camisa amarela foi detectada.")

    # Remover o vídeo temporário
    os.remove(video_path)
