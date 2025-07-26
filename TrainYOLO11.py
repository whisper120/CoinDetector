from ultralytics import YOLO

# we need a path to the model that we want to train on , if we dont have one we just type the previous model name and items
# will download automatically and run 
#model = YOLO('runs/detect/train/weights/best.pt') # if we want to be using latest model that we trained previously
model = YOLO('yolo11l.pt')

#Actual training run
results = model.train(data="datasets/data.yaml", epochs=150, imgsz=640, workers=0, patience = 20)
