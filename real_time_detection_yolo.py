from ultralytics import YOLO
import cv2

model = YOLO("best.pt")  # Load a pre-trained best model given to us(YOLOv11)

cap = cv2.VideoCapture('data_videos/15sec_input_720p.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Perform inference on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with detection results
    cv2.imshow('YOLOv11 Detection', annotated_frame)  # Display the annotated frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


