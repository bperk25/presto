from ultralytics import YOLO

# train model (if needed), then load it
def initialize_model(trained=True):
    if not trained:
        # Load a base model
        model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights
    
        # Train the model
        results = model.train(data='/Users/bryantperkins/Documents/CS/CS131/FinalProject/Code/presto/datasets/notes_dataset', epochs=100, imgsz=64)

    model = YOLO('./runs/classify/train3/weights/best.pt')  # load a trained model
    
    return model

# predict time of note image
def predict(model, predict_dir):
    # Predict with the model
    results = model(predict_dir)  # predict on an image
    names_dict = results[0].names
    probs = results[0].probs
    
    print(names_dict)
    print(probs)
    
    return names_dict, probs