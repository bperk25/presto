from edge import canny
import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Note:
    x: int
    y: int
    staff: int
    letter: str
    time: str
    note_img: np.ndarray

notes = dict()  # dictionary of {int noteID num : Note class instance}


# reads input file as a numpy image
# returns grayscale version of image, and guassian blurred image
def filter_img(input_file):
    # open img & convert frame to grayscale
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    # smooth image using gaussian smoothing and parameters found before
    smoothed = cv2.GaussianBlur(img, (7, 7), 1.4)  # kernel size 7x7, sigma 1.4

    return img, smoothed


# display all images in imgs
# imgs: list of numpy array imgs to display
# titles: optional, title for image corresponding to index. len must match len of imgs
def display_imgs(imgs, titles=[]):
    num_imgs_per_row = 2
    num_rows = math.ceil(len(imgs) / num_imgs_per_row)  # num rows needed

    for index, img in enumerate(imgs):
        plt.subplot(num_rows, num_imgs_per_row, index + 1)
        plt.imshow(img)
        plt.axis('off')
        if titles:
            plt.title(titles[index])


# Set parameters for blob detector
def set_blob_params(img_shape):
    H, W = img_shape

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    # 16 pixel note blob height on img of height 1552 pixels --> use 12 pixels for min area, 18 for max
    # use ratio of note height:img height, then scale to height of current image
    # divide by 2 to get radius instead of diameter
    min_pixel_radius = int(((12 / 1552) * H) / 2)
    max_pixel_radius = int(((18 / 1552) * H) / 2)
    # compute min area from estimated pixel radius of note blob
    params.minArea = int(min_pixel_radius ** 2 * np.pi)
    params.maxArea = int(max_pixel_radius ** 2 * np.pi)

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.maxCircularity = 0.85

    return params


# Finds blobs of notes in image
# inputs: smoothed/filtered image, whether to display image or not
# output: found keypoints for note blobs
def find_blobs(img, display=False):
    params = set_blob_params(img.shape)

    # Set up the detector with default parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(img)

    # Draw detected blobs as blue circles
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    img_w_keypts = cv2.drawKeypoints(img.copy(), keypoints, np.array([]), (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # show process images
    if display:
        plt.imshow(img_w_keypts)
        plt.axis('off')
        plt.show()

        display_imgs([img], ["original"])

    # clean up once processes is done
    cv2.destroyAllWindows()

    return keypoints


# crop full-sized image to just a window containing the note corresponding
# to the note blob
def crop_to_note(blob, full_img, i):
    x, y = blob.pt
    lr_space = blob.size + 5  # pixels left and right of center of blob to include
    ud_space = blob.size + 30  # pixels below and above center of blob to include

    cropped_img = full_img[int(y - ud_space): int(y + ud_space),
                  int(x - lr_space): int(x + lr_space)]

    cv2.imwrite('cropped_note_imgs/note_' + str(i) + '.png', cropped_img)

    return cropped_img


# crop all blobs to images of individual notes
# return list of cropped images
def get_cropped_notes(blobs, full_img):
    cropped_notes = []

    for i, blob in enumerate(blobs):
        cropped = crop_to_note(blob, full_img, i)
        cropped_notes.append(cropped)

    return cropped_notes


