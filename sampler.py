from os.path import join, splitext
from os import listdir
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from PIL import ImageDraw, Image, ImageFont
from image_cache import Cache
from keras import backend as K
import threading


SIGMA = 3.16227766016838
HIGH_DENSITY = 1e-2
MID_DENSITY = 1e-3
PATCH_SZ = 75
PAD_SZ = 100
CELL_SZ = 200
DOT_DICT_FILE = "cache/dot_dict_sig{:0.2}_hid{:1.0e}_mid{:1.0e}.pkl".format(SIGMA, HIGH_DENSITY, MID_DENSITY)
CLASSES = (1, 2, 3, 4, 5)

data_path = "data"
image_path = join(data_path, "train")
coords_file = join(data_path, "coords.csv")


def gaussian(x):
    return gaussian_filter(x, SIGMA)


def sample_coords(verbose=0):
    coords = pd.read_csv(coords_file)
    coords["cls"] += 1
    ids = [int(splitext(f)[0]) for f in listdir(image_path) if (splitext(f)[1] == ".jpg") and
                     (int(splitext(f)[0]) in coords["tid"].values)]
    ids = sorted(ids)

    dot_dict = defaultdict(dict)

    for image_id in ids:
        if verbose >= 0:
            print("id", image_id)
        im_file = join(image_path, "{}.jpg".format(image_id))
        img = ndimage.imread(im_file)
        neg_a = np.zeros(img.shape[:2])
        pos_area = 0
        dots = coords[coords["tid"] == image_id][["row", "col"]].as_matrix()
        cells = list(set(tuple(dot) for dot in (dots // CELL_SZ)))
        for cls in CLASSES:
            dots = coords[(coords["tid"] == image_id) & (coords["cls"] == cls)][["row", "col"]].as_matrix()

            a = np.zeros(img.shape[:2])
            a[dots.T[0], dots.T[1]] = 1
            a = gaussian(a)
            neg_a = np.max(np.stack((neg_a, a)), axis=0)

            i, j = np.where(a >= HIGH_DENSITY)
            pos_dots = np.stack((i, j)).T.astype(np.uint16)
            idx = np.array([tuple(dot) in cells for dot in (pos_dots // CELL_SZ)], dtype=bool)
            pos_dots = pos_dots[idx]
            np.random.shuffle(pos_dots)
            dot_dict[image_id][cls] = pos_dots
            pos_area += len(pos_dots)

        ij = [np.where(neg_a[cell[0] * CELL_SZ: (cell[0] + 1) * CELL_SZ,
                       cell[1] * CELL_SZ: (cell[1] + 1) * CELL_SZ] < MID_DENSITY) for cell in cells]
        ij = [np.stack(x).T + np.array(cell) * CELL_SZ for x, cell in zip(ij, cells)]
        neg_dots = np.vstack(ij).astype(np.uint16)
        np.random.shuffle(neg_dots)
        assert np.alltrue(neg_a[list(neg_dots.T)] < MID_DENSITY)

        if verbose >= 2:
            fig = plt.figure(figsize=(16, 16))
            fig.add_subplot(2, 1, 1)
            plt.imshow(img)

            fig.add_subplot(2, 1, 2)
            b = np.ones(img.shape[:2], dtype=np.uint8) * 255
            b[list(zip(*neg_dots))] = 0
            plt.imshow(b)
            plt.show()

        dot_dict[image_id][0] = neg_dots[0: 10 * pos_area]

        if verbose >= 1:
            cells_area = sum(
                [(img.shape[0] % CELL_SZ if (cell[0] + 1) * CELL_SZ > img.shape[0] else CELL_SZ) *
                 (img.shape[1] % CELL_SZ if (cell[1] + 1) * CELL_SZ > img.shape[1] else CELL_SZ)
                 for cell in cells])
            print("cells area", cells_area, "pos + neg area", pos_area + len(neg_dots), "pos area", pos_area,
                  "neg area", len(neg_dots))

    with open(DOT_DICT_FILE, mode="wb") as f_:
        pickle.dump(dot_dict, f_, protocol=pickle.HIGHEST_PROTOCOL)


def plot_patches(store="memory"):
    with open(DOT_DICT_FILE, mode="rb") as f:
        dot_dict = pickle.load(f)
    z = [(dot_dict[image_id][cl], (cl,) * len(dot_dict[image_id][cl]),
          (image_id,) * len(dot_dict[image_id][cl])) for image_id in dot_dict for cl in dot_dict[image_id]]
    X, y, im = zip(*z)
    X = np.vstack(X)
    y = np.concatenate(y).astype(int)
    im = np.concatenate(im).astype(int)

    idx = list(range(len(X)))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    im = im[idx]

    neg = np.where(y == 0)[0]
    pos = np.where(y > 0)[0]
    X_pos = X[pos[:50]]
    y_pos = y[pos[:50]]
    im_pos = im[pos[:50]]
    X_neg = X[neg[:50]]
    y_neg = y[neg[:50]]
    im_neg = im[neg[:50]]

    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg)).astype(int)
    im = np.concatenate((im_pos, im_neg)).astype(int)
    idx = list(range(len(X)))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    im = im[idx]

    plot_mosaic(im, X, y, store)


class PatchIterator(object):

    def __init__(self, dot_dict, cache, patch_sz, image_ids=None,
                 batch_size=32, shuffle=True, seed=None, normalize=True,
                 output_ids=False, neg_to_pos_ratio=1.0, num_class=6, svm_labels=False,
                 local_contrast_normalize=False, channel_shift=0, gen_id=""):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.cache = cache
        self.dot_dict = dot_dict
        self.normalize = normalize
        self.patch_sz = patch_sz
        self.output_ids = output_ids
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.num_class = num_class
        self.svm_labels = svm_labels
        self.local_contrast_normalize = local_contrast_normalize
        self.channel_shift = channel_shift
        self.normalize = normalize
        self.gen_id = gen_id
        if K.image_data_format() == "channels_last":
            self.image_shape = (self.patch_sz, self.patch_sz, 3)
        else:
            self.image_shape = (3, self.patch_sz, self.patch_sz)
        filtered_ids = list(dot_dict)
        if image_ids:
            filtered_ids = [im_id for im_id in filtered_ids if im_id in image_ids]

        z = [(dot_dict[image_id][cl], (cl,) * len(dot_dict[image_id][cl]),
              (image_id,) * len(dot_dict[image_id][cl])) for image_id in filtered_ids for cl in dot_dict[image_id]]
        dots, classes, image_ids = zip(*z)
        self.dots = np.vstack(dots)
        self.classes = np.concatenate(classes).astype(int)
        self.image_ids = np.concatenate(image_ids).astype(int)

        neg_idx = np.where(self.classes == 0)[0]
        pos_idx = np.where(self.classes > 0)[0]

        self.index_generator = self._flow_index(pos_idx, neg_idx, batch_size=batch_size, shuffle=shuffle, seed=seed)

    def next(self):

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        # build batch of image data
        for i, j in enumerate(index_array):
            x = self.cache.get_patch(self.image_ids[j], self.dots[j], self.patch_sz,
                                     channel_shift=self.channel_shift, normalize=self.normalize)
            batch_x[i] = x

        # build batch of labels
        batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
        for i, label in enumerate(self.classes[index_array]):
            batch_y[i, label] = 1.
        if self.svm_labels:
            batch_y = batch_y * 2. - 1.
        if self.output_ids:
            return batch_x, batch_y, self.image_ids[index_array]
        else:
            return batch_x, batch_y

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, pos_idx, neg_idx, batch_size=32, shuffle=True, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            random_seed = None if seed is None else seed + self.total_batches_seen
            if self.batch_index == 0:
                print("\n************** New epoch. Generator", self.gen_id, "*******************\n")
                if shuffle:
                    np.random.RandomState(random_seed).shuffle(neg_idx)
                cut_off = int(len(pos_idx) * self.neg_to_pos_ratio)
                index_array = np.concatenate((pos_idx, neg_idx[:cut_off]))
                if shuffle:
                    np.random.RandomState(random_seed).shuffle(index_array)
            n = len(index_array)
            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


def plot_mosaic(image_ids, x, classes, store):
    if x.ndim == 2:
        cache = Cache(CELL_SZ, PAD_SZ, store=store, local_contrast_normalize=True)
        patches = np.array([cache.get_patch(image_id, dot, PATCH_SZ) for image_id, dot in zip(image_ids, x)])
    else:
        patches = x
    if K.image_data_format() == "channels_first":
        patches = np.rollaxis(patches, 1, 4)

    font = ImageFont.truetype("times", 96)
    tag_sz = (30, 20)
    for image_id, patch, cls in zip(image_ids, patches, classes):
        tag = Image.new("RGB", (180, 120), color=(255, 255, 255))
        draw = ImageDraw.Draw(tag)
        draw.text((0, 0), "{} {}".format(cls, image_id), font=font, fill=0)
        tag = tag.resize(tag_sz, resample=Image.BILINEAR)
        patch[:tag_sz[1], :tag_sz[0], :] = np.array(tag)

    mosaic = np.ones(((PATCH_SZ + 1) * 10 + 1, (PATCH_SZ + 1) * 10 + 1, 3), dtype=np.uint8) * 255
    for k, patch in enumerate(patches[:100]):
        i = k // 10
        j = k % 10
        mosaic[i * (PATCH_SZ + 1) + 1: i * (PATCH_SZ + 1) + 1 + PATCH_SZ,
        j * (PATCH_SZ + 1) + 1: j * (PATCH_SZ + 1) + 1 + PATCH_SZ, :] = patch
    plt.figure(figsize=(16, 16))
    plt.imshow(mosaic)
    plt.show()


if __name__ == "__main__":
    from itertools import islice

    # sample_coords(verbose=0)
    # plot_patches(store="hd5")

    # Test PatchIterator
    with open(DOT_DICT_FILE, mode="rb") as f:
        dot_dict = pickle.load(f)
    cache = Cache(CELL_SZ, PAD_SZ, store="hd5", local_contrast_normalize=True)
    patch_iterator = PatchIterator(dot_dict, cache, PATCH_SZ, batch_size=10, normalize=False,
                                   output_ids=True, neg_to_pos_ratio=1.0, channel_shift=5, gen_id="Test")

    z = [(X_, y_, image_ids_) for X_, y_, image_ids_ in islice(patch_iterator, 10)]
    X, y, image_ids = zip(*z)
    X = np.vstack(X)
    y = np.vstack(y)
    image_ids = np.concatenate(image_ids)
    y = y.argmax(axis=1)
    print(["class {}: {}".format(i, sum(y==i)) for i in range(6)])
    plot_mosaic(image_ids, X, y, store="hd5")

