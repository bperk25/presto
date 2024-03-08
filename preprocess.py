# from edge import canny
import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from musics import Note
# from dataclasses import dataclass

# @dataclass
# class Note:
#     x: int
#     y: int
#     staff: int
#     letter: str
#     time: str
#     note_img: np.ndarray

# notes = dict()  # dictionary of {int noteID num : Note class instance}


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

    plt.show()


# Set parameters for blob detector
def set_blob_params(img_shape):
    H, W = img_shape

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    '''
    # Set Area filtering parameters
    params.filterByArea = True
    # 16 pixel note blob height on img of height 1552 pixels --> use 12 pixels for min area, 18 for max
    # use ratio of note height:img height, then scale to height of current image
    # divide by 2 to get radius instead of diameter
    min_pixel_radius = int(((10 / 1552) * H) / 2)
    max_pixel_radius = int(((20 / 1552) * H) / 2)
    # compute min area from estimated pixel radius of note blob
    params.minArea = int(min_pixel_radius ** 2 * np.pi)
    params.maxArea = int(max_pixel_radius ** 2 * np.pi)

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.maxCircularity = 0.8
    '''

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
def crop_to_note(blob, full_img):
    x, y = blob.pt
    lr_space = blob.size + 5  # pixels left and right of center of blob to include
    ud_space = blob.size + 30  # pixels below and above center of blob to include

    cropped_img = full_img[int(y - ud_space): int(y + ud_space),
                  int(x - lr_space): int(x + lr_space)]

    return cropped_img


# crop all blobs to images of individual notes
# return list of cropped images
def get_cropped_notes(blobs, full_img, save=False):
    cropped_notes = []

    for i, blob in enumerate(blobs):
        cropped = crop_to_note(blob, full_img)
        cropped_notes.append(cropped)

        if save:
            cv2.imwrite('cropped_note_imgs/note_' + str(i) + '.png', cropped)

    return cropped_notes

##  ---------  Line Removal  -------------


##  ---------  Line Removal  -------------
'''
    Remove horizontal lines from input img
    input:
        img -> cv2 original sheet music image
        len -> length of kernel line
        kern_size -> size of blur kernel
        sig -> sigma for Gaussian blur

    output:
        no_lines -> image reconstructed around removed lines
'''
def remove_horizontal(img, len=13, kern_size=5, sig=0):
    hor_lines = horizontal_canny(img.copy(), len)
    hor_lines = cv2.GaussianBlur(hor_lines, (kern_size, kern_size), sig)
    no_lines = img*(hor_lines == 0) + 255*(hor_lines > 0)
    return np.uint8(no_lines)


## ---------  Line Detection  -------------

# canny on crack
def horizontal_canny(img, len=13):
    img = cv2.Canny(img, 200, 200)
    horizontal_kernel = np.ones((1, len), np.uint8)
    horizontal_lines = cv2.erode(img, horizontal_kernel, iterations=2)
    return horizontal_lines

'''
    Finds y coordinates of staff lines using HoughLines
    input:
        img -> cv2 original sheet music image
        edges -> super_canny version of sheet music
        min_gap -> minimum distance between line y coordinates
        show_img -> flag to show image with lines drawn

    output:
        ys -> y coordinates of lines found
'''
def get_line_coords(img, edges, min_gap = 3, show_img=False):
    # Get lines using cv2's HoughLinesP: 
    # https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 2, minLineLength=edges.shape[0]//50)
    lines = sorted(lines.tolist(), key=lambda x: x[0][1])   # sort by y coord
    ys = [-100] # start with -100 for first y1 check; never in range with reasonable thresh val 

    # Filter lines based on proximity to previous line
    for line in lines:
        x1, y1, x2, y2 = line[0]
        low_thresh = ys[-1] - min_gap
        hi_thresh = ys[-1] + min_gap + 1
        if y1 not in range(low_thresh, hi_thresh):
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.line(img, (0, y1), (edges.shape[0]//2, y2), (255, 0, 0), 1)
            ys.append(y1)

    if show_img: 
        cv2.imshow("Lines Found", img)
        cv2.waitKey(0)

    ys = ys[1:] # remove dummy value of -100
    ys, avg = refine_line_coords(ys)
    return ys, avg

'''
    Removes outlier lines from y coordinates
    input:
        ys -> y coordinates found
    outpus: 
        refined_ys -> ys with outliers removed
'''
def refine_line_coords(ys: list[float]) -> list[float]:
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    avg_gap = np.mean(gaps) * .8
    max_gap = np.mean([y for y in gaps if y < avg_gap]) * 1.2
    avg_gap = np.mean([y for y in gaps if y < avg_gap])

    refined_ys = []
    for i in range(len(ys)):
        cur_y = ys[i]
        dif_above = ys[i] - ys[i - 1] if i > 0 else max_gap + 1
        dif_below = ys[i + 1] - ys[i] if i < len(ys) - 1 else max_gap + 1
        if dif_above <= max_gap or dif_below <= max_gap:
            # print((max_gap, dif_above, dif_below))
            refined_ys.append(ys[i])

    return refined_ys, avg_gap

def get_base_lines(img, edges, min_gap=3, show_img=False):
    ys, avg = get_line_coords(img, edges, min_gap, show_img)
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    base_lines = [ys[i + 1] for i in range(len(gaps) - 1) if gaps[i + 1] - gaps[i] > avg]
    base_lines.append(ys[-1])

    # print(base_lines)
    return base_lines, avg


