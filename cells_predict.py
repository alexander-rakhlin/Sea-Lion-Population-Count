"""
Filter out empty tiles
"""

from keras import backend as K
from time import time
import numpy as np
from predict import frame_image
from os.path import join, splitext
from os import listdir
import scipy.ndimage as ndimage
from itertools import product
from image_cache import scale
from def_vgg16 import vgg16fcn
from collections import defaultdict
import pickle
import pandas as pd
from threaded_generator import threaded_generator


IMAGE_DIR = "data/test"
all_ids = sorted([int(splitext(f)[0]) for f in listdir(IMAGE_DIR) if splitext(f)[1] == ".jpg"])

CELL_SZ = 200
PAD_SZ = 0
CONTRAST = False
NORMALIZE = True
IMG_BATCH = 3

SAVED_MODEL = "models/cell_model_vgg16fcn.epoch170-0.96.h5"
CELL_DICT_FILE = "data/cell_dict_test.pkl"

model = vgg16fcn(CELL_SZ, saved_model=SAVED_MODEL)

cell_dict = defaultdict(dict)


def gen_images(ids):
    for i, image_id in enumerate(ids):
        img = ndimage.imread(join(IMAGE_DIR, "{}.jpg".format(image_id)))
        # img = imread("{}.jpg".format(image_id), image_dir=IMAGE_DIR, dotted_dir="data/TrainDotted", verbose=0)

        h, w, ch = img.shape
        cells = list(product(range(int(np.ceil(h / CELL_SZ))), range(int(np.ceil(w / CELL_SZ)))))
        data = frame_image(img, CELL_SZ, PAD_SZ, cells=cells, do_padding=True)

        print("Image {}: {} cells".format(image_id, len(cells)))
        yield [image_id] * len(cells), cells, data


def gen_batches(images):
    data_batch = []
    ids_batch = []
    cells_batch = []
    for image_id, cells, data in images:
        data_batch.append(data)
        ids_batch.extend(image_id)
        cells_batch.extend(cells)
        if len(data_batch) >= IMG_BATCH:
            data_batch = np.vstack(data_batch)
            if K.image_data_format() == "channels_first":
                data_batch = np.rollaxis(data_batch, 3, 1)
            if NORMALIZE:
                data_batch = scale(data_batch)
            yield ids_batch, cells_batch, data_batch
            data_batch = []
            ids_batch = []
            cells_batch = []
    if len(data_batch) > 0:
        data_batch = np.vstack(data_batch)
        if K.image_data_format() == "channels_first":
            data_batch = np.rollaxis(data_batch, 3, 1)
        if NORMALIZE:
            data_batch = scale(data_batch)
        yield ids_batch, cells_batch, data_batch


def gen_features(batches):
    for ids_batch, cells_batch, data_batch in batches:
        p_ids = sorted(set(ids_batch))
        print("....Starting {}".format(p_ids))
        preds_batch = model.predict(data_batch)
        print("....Done with {}".format(p_ids))
        yield ids_batch, cells_batch, preds_batch


def test_dicts():
    dict_files = ("dumps/cell_dict7.pkl", "dumps/cell_dict8.pkl")
    d = []
    for d_fl in dict_files:
        with open(d_fl, mode="rb") as f_:
            d.append(pickle.load(f_))
    assert d[0].keys() == d[1].keys()
    for k in d[0].keys():
        assert d[0][k].keys() == d[1][k].keys()
        for j in d[0][k].keys():
            print(j, d[0][k][j], d[1][k][j])
            np.testing.assert_almost_equal(d[0][k][j][0], d[1][k][j][0], decimal=5)


def stats_and_thresholds():
    global IMAGE_DIR
    from linear_layer_utils import get_cells_ground_truth
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt
    test_ids = list(range(1, 51))
    dict_file = "dumps/cell_dict_train.pkl"
    with open(dict_file, mode="rb") as f_:
        d = pickle.load(f_)
    COORDS = pd.read_csv("data/coords.csv")
    COORDS["cls"] += 1
    IMAGE_DIR = "data/train"
    y_true = []
    y_pred = []
    for image_id in test_ids:
        img = ndimage.imread(join(IMAGE_DIR, "{}.jpg".format(image_id)))
        h, w, _ = img.shape
        cells, ground_truth = get_cells_ground_truth(image_id, h, w)
        ground_truth = ground_truth.sum(1).clip(0, 1)

        filled_cells = [c for c, g in zip(cells, ground_truth) if g > 0]
        all_cells = product(range(int(np.ceil(h / CELL_SZ))), range(int(np.ceil(w / CELL_SZ))))
        empty_cells = [cell for cell in all_cells if cell not in filled_cells]
        cells = filled_cells + empty_cells
        ground_truth = [1] * len(filled_cells) + [0] * len(empty_cells)

        y_true.extend(ground_truth)
        y_pred.extend([d[image_id][cell][0] for cell in cells])
    y_pred = np.array(y_pred)
    for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
        print("threshold", threshold)
        acc = accuracy_score(y_true, y_pred > threshold)
        print("Accuracy", acc)

        cmx = confusion_matrix(y_true, y_pred > threshold)
        print("Confusion matrix", cmx)
    # threshold
    # 0.3
    # Accuracy
    # 0.969385151799
    # Confusion
    # matrix[[22073   697]
    # [23 725]]
    # threshold
    # 0.35
    # Accuracy
    # 0.973977379029
    # Confusion
    # matrix[[22183   587]
    # [25 723]]
    # threshold
    # 0.4
    # Accuracy
    # 0.977336508206
    # Confusion
    # matrix[[22265   505]
    # [28 720]]
    # threshold
    # 0.45
    # Accuracy
    # 0.979632621822
    # Confusion
    # matrix[[22324   446]
    # [33 715]]
    # threshold
    # 0.5
    # Accuracy
    # 0.981205884854
    # Confusion
    # matrix[[22366   404]
    # [38 710]]
    # threshold
    # 0.55
    # Accuracy
    # 0.982694106642
    # Confusion
    # matrix[[22404   366]
    # [41 707]]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()


def highlight(image_id, image_dir, preds_dict, threshold=0.5):
    from highlight_image import highlight_cells
    with open(preds_dict, mode="rb") as f_:
        d = pickle.load(f_)
    img = ndimage.imread(join(image_dir, "{}.jpg".format(image_id)))
    cells = [cell for cell in d[image_id] if d[image_id][cell][0] > threshold]
    highlight_cells(img, cells, CELL_SZ, highlight=80)


if __name__ == "__main__":

    images = gen_images(all_ids)
    batches = gen_batches(images)
    batches_ = threaded_generator(batches, num_cached=2)
    features = gen_features(batches_)

    total_ids_done = 0
    total_cells_done = 0
    t0 = time()
    for i, (ids_batch, cells_batch, preds_batch) in enumerate(features):
        for image_id, cell, pred in zip(ids_batch, cells_batch, preds_batch):
            cell_dict[image_id][cell] = pred
        if (i % 100) == 0:
            print("Dumping to disk")
            with open(CELL_DICT_FILE, mode="wb") as f_:
                pickle.dump(cell_dict, f_, protocol=pickle.HIGHEST_PROTOCOL)

        elapsed = time() - t0
        processed_ids = sorted(set(ids_batch))
        total_ids_done += len(processed_ids)
        total_cells_done += len(cells_batch)
        seconds_per_image = elapsed / total_ids_done
        seconds_per_cell = elapsed / total_cells_done
        print("Processed images {}. Elapsed {} minutes. {:0.2f} seconds/image. {:0.2f} seconds/cell. {} minutes to go".
              format(processed_ids, int(elapsed / 60), seconds_per_image, seconds_per_cell,
                     int(seconds_per_image * (len(all_ids) - total_ids_done) / 60)))

    with open(CELL_DICT_FILE, mode="wb") as f_:
        pickle.dump(cell_dict, f_, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time() - t0
    print("Elapsed {} minutes. {:0.2f} seconds/image".format(elapsed // 60, elapsed / len(all_ids)))

    with open(CELL_DICT_FILE, mode="rb") as f_:
        cell_dict = pickle.load(f_)

    high_values = [cell[0] for cells in cell_dict.values() for cell in cells.values() if cell[0] >= 0.50 and cell[0] < np.inf]
    print(len(high_values), high_values[:10], len(high_values) * 0.17 / 3600)
    print(len(high_values) // len(cell_dict))

    highlight(10, IMAGE_DIR, CELL_DICT_FILE, 0.5)
