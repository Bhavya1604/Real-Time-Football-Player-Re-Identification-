from ultralytics import YOLO

model = YOLO("best.pt")  # Load a pre-trained best model given to us(YOLOv11)

results1 = model.predict('data_videos/15sec_input_720p.mp4',save=True)  
print(results1[0])



for box in results1[0].boxes:
    print(box)


 

