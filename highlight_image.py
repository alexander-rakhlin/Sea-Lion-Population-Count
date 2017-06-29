import pickle
from os.path import join
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2


CELL_SZ = 200
DATA_PATH = "data"
IMAGE_PATH = join(DATA_PATH, "train")
COORDS_FILE = join(DATA_PATH, "coords.csv")


def highlight_cells(img, cells, cell_sz, highlight=80):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for cell in cells:
        top, left = [c * cell_sz for c in cell]
        crop = img[top: top + cell_sz, left: left + cell_sz, :]
        crop[..., 2] = cv2.add(crop[..., 2], 80)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    plt.figure(figsize=(25, 10))
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    image_id = 41
    cells = (
        (10, 15),
        (10, 16),
        (11, 16),
        (13, 16),
    )
    img = ndimage.imread(join(IMAGE_PATH, "{}.jpg".format(image_id)))
    highlight_cells(img, cells, CELL_SZ, highlight=180)