from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('runs/detect/train/weights/best.pt')

def detectar_carros(caminho_imagem, salvar_resultado=True):
    """

    Detecta carros em uma imagem e desenha bounding boxes

    """

    results = model(caminho_imagem)
   
    img = cv2.imread(caminho_imagem)

    for result in results:
        boxes = result.boxes
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                confidence = box.conf[0].cpu().numpy()
                
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                print(f"Carro detectado: {label} - Coordenadas: ({x1}, {y1}, {x2}, {y2})")

    if salvar_resultado:
        cv2.imwrite('resultado_deteccao.jpg', img)
        print("Imagem salva como 'resultado_deteccao.jpg'")

    cv2.imshow('Detecção de Carros', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return results

if __name__ == "__main__":
    caminho_da_imagem = "images/train/carro1.jpg"

    resultados = detectar_carros(caminho_da_imagem)