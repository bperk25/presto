from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
#model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

# Train the model
# results = model.train(data='/Users/bryantperkins/Documents/CS/CS131/FinalProject/Code/presto/datasets/notes_dataset', epochs=100, imgsz=64)

model = YOLO('./runs/classify/train3/weights/best.pt')  # load an official model

# Predict with the model
results = model('./cropped_note_imgs/note_6.png')  # predict on an image
names_dict = results[0].names
probs = results[0].probs
print(names_dict)
print(probs)