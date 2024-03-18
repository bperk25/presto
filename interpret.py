from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from musics import Note
from collections import defaultdict 

# train YOLO classification model
def train_model(data='/Users/bryantperkins/Documents/CS/CS131/FinalProject/Code/presto/datasets/notes_dataset'):
    # Load a base model
    model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=data, epochs=100, imgsz=64)


# load chosen trained model
def initialize_model():
    model = YOLO('./runs/classify/train3/weights/best.pt')  # load a trained model
    return model


# predict time of note using yolo model & update each note obj's time attribute
# NOTE: notes_objs must be in same order as notes_imgs
def predict_note_times(model, notes_imgs, notes_objs, predict_dir='cropped_note_imgs/', predict_from_dir=False):
    # if predicting on all imgs saved as files in directory predict_dir
    if predict_from_dir:
        # for all cropped note imgs in predict dir
        note_files = [predict_dir + file for file in os.listdir(predict_dir)]
        # predict time of all note imgs
        results = model(note_files)
    # if using passed in numpy/cv2 imgs
    else:
        # predict time of all note imgs
        results = model(notes_imgs)

    # add predicted time to corresponding note attribute
    for i, result in enumerate(results):
        # assign time as most likely time
        notes_objs[i].time = result.names[0]
        

## ---------  Configure Notes  ------------- ##
C_MAJOR = ["C", "D", "E", "F", "G", "A", "B"]

'''
    Create staff dictionary; key: staff id, val: list of notes
    input:
        base_lines -> list of staff base lines sorted (lo->hi)
        note_set -> list of Note object instances to be configured

    output:
        staff_dict-> key: staff id, val: list of notes
'''
def create_staff_dict(base_lines: list[float], note_set: list[Note],
                      gap_size: float) -> dict[tuple[int, float], list[Note]]:
    note_set.sort(key=lambda n: n.x)
    note_set.sort(key=lambda n: n.y)
    staff_dict = defaultdict(list)
    staff_id = 0
    for line in base_lines:
        i = 0
        lowest = line + (gap_size * 1.5)
        while(note_set[i].y < lowest and i < len(note_set) - 1):
            i+= 1
        staff_dict[staff_id] = note_set[:i]
        note_set = note_set[i:]
        staff_id += 1

    return staff_dict


'''
    Create note_range; key: note, val: range
    input:
        line -> staff base line y coordinates
        gap_size -> space between staff lines
        key_sig -> list of notes for each gap space; top to bottom

    output:
        ys -> y coordinates of lines found
'''
def create_nrange_dict(line, gap_size, key_sig):
    start = int(line + gap_size)
    step = int(-1 * gap_size // 2)
    end = int(start + (step * 13))
    note_ranges = defaultdict(list)
    for i, y in enumerate(range(start, end, step)):
        cur_note = key_sig[i % len(key_sig)]
        lower_bound = y + step // 2
        upper_bound = y - step // 2
        note_ranges[cur_note].append((lower_bound, upper_bound))

    return note_ranges


'''
    Add Notes and ID's to Note instances
    input:
        base_lines -> list of staff base lines sorted (lo->hi)
        note_set -> Note object instances to be configured

    output:
        ys -> y coordinates of lines found
'''
def config_notes(base_lines: list[float], note_set: list[Note], gap_size: float, key_sig: list[str]=C_MAJOR):
    staff_dict = create_staff_dict(base_lines, note_set, gap_size)
    # print([(k, [n.y for n in v]) for k, v in staff_dict.items()])
    note_count = 0
    for staff_id in staff_dict.keys():
        note_ranges = create_nrange_dict(base_lines[staff_id], gap_size, key_sig)
        # print(f"{staff_id}: {note_ranges}\n")
        # Assign note properties
        for note in staff_dict[staff_id]: # for every note obj
            # print(f"y: {note.y}, staff: {staff_id}, ranges: {note_ranges}\n")
            for key in note_ranges.keys(): # each possible note name
                for r in note_ranges[key]: # each range for note name
                    if int(note.y) in range(r[0], r[1] + 1): # if note obj in note name range
                        #print(f"y: {note.y}, staff: {staff_id}, ranges: {r}\n")
                        note.staff_id = staff_id
                        note.id = note_count
                        note.key = key
                        note_count += 1


