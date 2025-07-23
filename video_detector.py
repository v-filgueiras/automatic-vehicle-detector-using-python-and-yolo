from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train/weights/best.pt')

video_path = "road.mp4"

print(f"Processando vídeo: {video_path}")

def processar_video_automatico():
    """Usa o método automático do YOLO"""
    results = model.predict(
        source=video_path,
        save=True,
        show=False,
        conf=0.3,
        stream=True,
        verbose=True
    )
    
    print("Processando frames...")
    for result in results:
        pass
    
    print("Vídeo processado e salvo em: runs/detect/predict/")

def processar_video_manual():
    """Processa o vídeo com controle manual"""
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Vídeo: {width}x{height}, {fps} FPS, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('video_processado.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"Processando frame {frame_count}/{total_frames}", end='\r')

        results = model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence >= 0.3:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        label = f'Carro: {confidence:.2f}'
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nVídeo processado salvo como: video_processado.mp4")

if __name__ == "__main__":
    print("Escolha o método:")
    print("1 - Automático)")
    print("2 - Manual")
    
    escolha = input("Digite 1 ou 2: ").strip()
    
    if escolha == "1":
        processar_video_automatico()
    elif escolha == "2":
        processar_video_manual()
    else:
        print("Opção inválida! Usando método automático...")
        processar_video_automatico()