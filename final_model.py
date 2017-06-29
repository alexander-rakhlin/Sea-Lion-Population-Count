from def_linear_layer_models import resnet, super_resnet
from def_drn18 import drn18_linear_regression
from sklearn.externals import joblib
from time import time
import pandas as pd
from os.path import join, splitext
from os import listdir
import scipy.ndimage as ndimage
import numpy as np
from image_cache import lcn, scale
from predict import frame_image, CLASSES
import pickle
from sklearn.metrics import mean_squared_error
from threaded_generator import threaded_generator

from keras.layers import Input
from keras.models import Model
from keras import backend as K

LOW_THRESHOLD, HIGH_THRESHOLD = 0.8, np.inf
SLICES = (
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, np.inf),
)

CELL_SZ = 200
PAD_SZ = 38
CONTRAST = False
NORMALIZE = True

FEATURE_MODEL = "models/model_drn18.epoch14-0.90_nocontrast.h5"
LINEAR_CLASSIFIERS = (
    "models/linear_classifiers-1.pkl",
    "models/linear_classifiers-2_abbr2.pkl"
)
EXIT_MODELS = (
    "models/model_resnet.epoch98-0.73.h5",
    "models/model_resnet.epoch116-0.73.h5",
    "models/model_resnet.epoch160-0.66.h5",
)
BATCH_SIZE = 16
IMG_BATCH = 5

IMAGE_DIR = "data/test"
ids = sorted([int(splitext(f)[0]) for f in listdir(IMAGE_DIR) if splitext(f)[1] == ".jpg"])

CELL_DICT_FILE = "data/cell_dict_test.pkl"
with open(CELL_DICT_FILE, mode="rb") as f_:
    CELL_DICT = pickle.load(f_)

PRED_DICT_FILE = "submission/preds_dict_test.{:0.2f}-{:0.2f}.pkl"

MODE = None  # None|"assertion"

if MODE == "assertion":
    IMAGE_DIR = "data/train"
    COORDS = pd.read_csv("data/coords.csv")
    COORDS["cls"] += 1

    ids = list(range(1000))
    ids = [i for i in ids if i in COORDS["tid"].values]

    CELL_DICT = {}
    for image_id in ids:
        img_coords = COORDS[COORDS["tid"] == image_id]
        dots = img_coords[["row", "col"]].as_matrix()
        CELL_DICT[image_id] = list(set([(dot[0] // CELL_SZ, dot[1] // CELL_SZ) for dot in dots]))

    PRED_DICT_FILE = "dumps/preds_dict_train.pkl"
    SLICES = (
        (0, np.inf),
    )

def format_list(lst, fun=None):
    return ", ".join(map("{:0.2f}".format, [l if fun is None else fun(l) for l in lst]))


def build_model(frame_sz):
    linear_classifiers_set = [joblib.load(cls) for cls in LINEAR_CLASSIFIERS]
    feature_model = drn18_linear_regression(frame_sz, FEATURE_MODEL, linear_classifiers_set)
    exit_model = super_resnet(EXIT_MODELS)

    if K.image_data_format() == "channels_first":
        input_shape = (3, frame_sz, frame_sz)
    else:
        input_shape = (frame_sz, frame_sz, 3)

    img_input = Input(shape=input_shape)
    x = feature_model(img_input)
    x = exit_model(x)
    model = Model(inputs=img_input, outputs=x)
    return model


def final_model_assert():
    with open(PRED_DICT_FILE, mode="rb") as f_:
        m_pred, m_direct = pickle.load(f_)

    gt = {}
    for image_id in ids:
        img_coords = COORDS[COORDS["tid"] == image_id]
        gt[image_id] = [len(img_coords[img_coords["cls"] == cl]) for cl in CLASSES]
    gt = np.array([gt[i] for i in ids])

    m_pred = np.array([m_pred[i] for i in ids]).round().astype(int)
    m_pred = np.maximum(0, m_pred)
    m_err = [np.sqrt(mean_squared_error(x, y)) for x, y in zip(gt.T, m_pred.T)]
    print("err={:0.2f} | ({})".format(np.mean(m_err), ", ".join((map("{:0.2f}".format, m_err)))))

    m_direct = np.array([m_direct[i] for i in ids]).round().astype(int)
    m_direct = np.maximum(0, m_direct)
    m_err = [np.sqrt(mean_squared_error(x, y)) for x, y in zip(gt.T, m_direct.T)]
    print("direct err={:0.2f} | ({})".format(np.mean(m_err), ", ".join((map("{:0.2f}".format, m_err)))))


def assert_slices():
    SLICES = (
        (0.80, 0.85),
        (0.85, np.inf),
    )

    keys, d = None, None
    for slc in SLICES:
        with open(PRED_DICT_FILE.format(*slc), mode="rb") as f_:
            m_p, m_d = pickle.load(f_)
        if keys is None:
            keys = sorted(list(m_p))
            d = np.array([m_p[k] for k in keys])
        else:
            d += np.array([m_p[k] for k in keys])
    with open("submission/preds_dict_test.0.80-inf.pkl", mode="rb") as f_:
        m_p, m_d = pickle.load(f_)
    d_conrol = np.array([m_p[k] for k in keys])
    np.testing.assert_almost_equal(d_conrol, d, decimal=5)
    print("Slices asserted")


def prob2slice(v):
    for slc in SLICES:
        l, h = slc
        if (v >= l) and (v < h):
            return slc
    return None


def gen_images(ids):
    for image_id in ids:
        img = ndimage.imread(join(IMAGE_DIR, "{}.jpg".format(image_id)))
        # img = imread("{}.jpg".format(image_id), image_dir=IMAGE_DIR, dotted_dir="data/TrainDotted", verbose=0)

        if MODE == "assertion":
            cells = CELL_DICT[image_id]
            cell_slice = [SLICES[0]] * len(cells)
            print("Image {}: {} cells.".format(image_id, len(cells)))
        else:
            d = CELL_DICT[image_id]
            cells = []
            cell_slice = []
            for cell in d:
                v = d[cell][0]
                if (v >= LOW_THRESHOLD) and (v < HIGH_THRESHOLD):
                    cells.append(cell)
                    slc = prob2slice(v)
                    assert slc is not None
                    cell_slice.append(slc)
            print("Image {}: {} cells. {}".format(image_id, len(cells), format_list(cells, fun=lambda x: d[x][0])))

        if len(cells) > 0:
            data = frame_image(img, CELL_SZ, PAD_SZ, cells=cells, do_padding=True)
            yield [image_id] * len(cells), cells, data, cell_slice


def gen_batches(images):
    data_batch = []
    ids_batch = []
    cells_batch = []
    slice_batch = []
    for image_id, cells, data, cell_slice in images:
        data_batch.append(data)
        ids_batch.extend(image_id)
        cells_batch.extend(cells)
        slice_batch.extend(cell_slice)
        if len(data_batch) >= IMG_BATCH:
            data_batch = np.vstack(data_batch)
            if K.image_data_format() == "channels_first":
                data_batch = np.rollaxis(data_batch, 3, 1)
            if CONTRAST:
                data_batch = np.stack([lcn(x, rgb=True) for x in data_batch])
            if NORMALIZE:
                data_batch = scale(data_batch)
            yield ids_batch, cells_batch, data_batch, slice_batch

            data_batch = []
            ids_batch = []
            cells_batch = []
            slice_batch = []
    if len(data_batch) > 0:
        data_batch = np.vstack(data_batch)
        if K.image_data_format() == "channels_first":
            data_batch = np.rollaxis(data_batch, 3, 1)
        if CONTRAST:
            data_batch = np.stack([lcn(x, rgb=True) for x in data_batch])
        if NORMALIZE:
            data_batch = scale(data_batch)
        yield ids_batch, cells_batch, data_batch, slice_batch


def gen_features(batches):
    for ids_batch, cells_batch, data_batch, slice_batch in batches:
        p_ids = sorted(set(ids_batch))
        print("....Starting {}".format(p_ids))
        model_pred_batch, model_direct_batch = model.predict(data_batch, batch_size=BATCH_SIZE)
        print("....Done with {}".format(p_ids))
        yield ids_batch, cells_batch, model_pred_batch, model_direct_batch, slice_batch


if __name__ == "__main__":
    model = build_model(CELL_SZ + 2 * PAD_SZ)
    images = gen_images(ids)
    batches = gen_batches(images)
    batches_ = threaded_generator(batches, num_cached=2)
    features = gen_features(batches_)

    m_pred, m_direct = {}, {}
    for slc in SLICES:
        m_pred[slc] = dict(zip(ids, np.zeros((len(ids), len(CLASSES)), dtype=np.float32)))
        m_direct[slc] = dict(zip(ids, np.zeros((len(ids), len(CLASSES)), dtype=np.float32)))
    total_ids_done = total_cells_done = 0
    t0 = time()

    for i, (ids_batch, cells_batch, model_pred_batch, model_direct_batch, slice_batch) in enumerate(features):
        for image_id, cell, pred, direct, slc in zip(ids_batch, cells_batch, model_pred_batch, model_direct_batch, slice_batch):
            m_pred[slc][image_id] += pred
            m_direct[slc][image_id] += direct

        if (i % 100) == 0:
            print("Dumping to disk")
            for slc in SLICES:
                with open(PRED_DICT_FILE.format(*slc), mode="wb") as f_:
                    pickle.dump((m_pred[slc], m_direct[slc]), f_, protocol=pickle.HIGHEST_PROTOCOL)

        # timing
        elapsed = time() - t0
        processed_ids = sorted(set(ids_batch))
        total_ids_done += len(processed_ids)
        total_cells_done += len(cells_batch)
        seconds_per_image = elapsed / total_ids_done
        seconds_per_cell = elapsed / total_cells_done
        print("Processed images {}. Elapsed {} minutes. {:0.2f} seconds/image. {:0.2f} seconds/cell. {} minutes to go".
              format(processed_ids, int(elapsed / 60), seconds_per_image, seconds_per_cell,
                     int(seconds_per_image * (ids[-1] - processed_ids[-1]) / 60)))

    for slc in SLICES:
        with open(PRED_DICT_FILE.format(*slc), mode="wb") as f_:
            pickle.dump((m_pred[slc], m_direct[slc]), f_, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time() - t0
    print("Elapsed {} minutes. {:0.2f} seconds/image".format(elapsed // 60, elapsed / len(ids)))

    # just to see
    for slc in SLICES:
        with open(PRED_DICT_FILE.format(*slc), mode="rb") as f_:
            m_p, m_d = pickle.load(f_)
        print("\nSlice", slc)
        for im in list(m_p)[:5]:
            print(im, format_list(m_p[im]), "|", format_list(m_d[im]))

    if MODE == "assertion":
        final_model_assert()
