# -*- coding: utf-8 -*-
"""
Implements "Interactive Object Counting" by Arteta, Lempitsky, Noble, Zisserman
# Reference

- [Interactive Object Counting](https://www.robots.ox.ac.uk/~vgg/publications/2014/Arteta14/arteta14.pdf)
- data sets used in the original paper can be obtained [here](http://www.robots.ox.ac.uk/~vgg/software/interactive_counting)
"""

from os.path import join
from os import listdir
import numpy as np
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as ndimage
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, Input, Activation, Conv2D, Flatten
from keras.regularizers import l2
from keras import backend as K
from keras.layers.pooling import _GlobalPooling2D
from keras.callbacks import ReduceLROnPlateau
from theano.tensor.shared_randomstreams import RandomStreams

TRAIN_DIR = "./Counting_MATLAB_package/syntheticCells/train"
TEST_DIR = "./Counting_MATLAB_package/syntheticCells/test"
SIGMA = 3.16227766016838
PATCH_SZ = 10
IMAGE_SZ = 256
MID_DENSITY = 1e-3
HIGH_DENSITY = 1e-2
N_FEATURES = 128
SAMPLES_PER_FRAME = 2000

srng = RandomStreams(seed=234)


def gaussian(x):
    return gaussian_filter(x, SIGMA)


class GlobalSumPooling2D(_GlobalPooling2D):
    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_squared_error(y_true, y_pred):
    return K.mean(K.mean(K.square(y_pred - y_true), axis=[1,2,3]), axis=-1)

chk = None
def mean_squared_error_region(y_true, y_pred):
    global chk
    pos = K.T.nonzero(y_true >= MID_DENSITY)
    if chk is None:
        ind_p = srng.choice(size=(8*SAMPLES_PER_FRAME,), a=pos[0].shape[0], replace=False)
    pos = tuple(j[ind_p] for j in pos)

    neg = K.T.nonzero(y_true < MID_DENSITY)
    if chk is None:
        ind_n = srng.choice(size=(pos[0].shape[0],), a=neg[0].shape[0], replace=False)
        chk = 1
    neg = tuple(j[ind_n] for j in neg)

    region = tuple(K.T.concatenate([p_, n_]) for p_, n_ in zip(pos, neg))

    return K.mean(K.square(y_pred[region] - y_true[region]))

def mean_squared_error_region2(y_true, y_pred):
    threshold = 1e-4
    region = K.T.nonzero(y_true >= threshold)
    return K.mean(K.square(y_pred[region] - y_true[region]))


def read_mat(f):
    return loadmat(f, struct_as_record=False, verify_compressed_data_integrity=False, squeeze_me=True)["gt"].T[::-1] - 1


def get_patches(directory):
    images = [join(directory, f) for f in listdir(directory) if f.endswith(".png")]
    images = [(f, read_mat(f.replace(".png", ".mat"))) for f in images]

    X = []
    y = []
    for image in images:
        im_file = image[0]
        dots = image[1]

        img = ndimage.imread(im_file).astype(float)
        assert all([shape == IMAGE_SZ for shape in img.shape])

        a = np.zeros(img.shape)
        a[dots[0], dots[1]] = 1
        a = gaussian(a)

        padded = np.ones((IMAGE_SZ + PATCH_SZ - 1, IMAGE_SZ + PATCH_SZ - 1)) * img.flatten().min()
        padded[PATCH_SZ // 2: PATCH_SZ // 2 + IMAGE_SZ, PATCH_SZ // 2: PATCH_SZ // 2 + IMAGE_SZ] = img

        # x_pos = [padded[dot[0]: dot[0] + PATCH_SZ, dot[1]: dot[1] + PATCH_SZ] for dot in dots.T]

        i, j = np.where(a >= HIGH_DENSITY)
        pos_dots = list(zip(i, j))
        np.random.shuffle(pos_dots)
        pos_dots = pos_dots[:SAMPLES_PER_FRAME]
        x_pos = [padded[dot[0]: dot[0] + PATCH_SZ, dot[1]: dot[1] + PATCH_SZ] for dot in pos_dots]

        i, j = np.where(a < MID_DENSITY)
        neg_dots = list(zip(i, j))
        np.random.shuffle(neg_dots)
        neg_dots = neg_dots[:len(x_pos)]
        x_neg = [padded[dot[0]: dot[0] + PATCH_SZ, dot[1]: dot[1] + PATCH_SZ] for dot in neg_dots]

        X.extend(x_pos)
        X.extend(x_neg)
        y.extend([1] * len(x_pos))
        y.extend([0] * len(x_neg))

    X = np.stack(X)
    if K.image_dim_ordering() == 'th':
        X = X[:, None, ...]
    else:
        X = X[..., None]
    X /= 255.
    return X, y


def encode_images(directory, model):
    X = []
    images = [join(directory, f) for f in listdir(directory) if f.endswith(".png")]
    for im_file in images:
        img = ndimage.imread(im_file).astype(float)
        padded = np.ones((IMAGE_SZ + PATCH_SZ - 1, IMAGE_SZ + PATCH_SZ - 1)) * img.flatten().min()
        padded[PATCH_SZ // 2:PATCH_SZ // 2 + IMAGE_SZ, PATCH_SZ // 2:PATCH_SZ // 2 + IMAGE_SZ] = img
        X.append(padded)

    X = np.stack(X)
    if K.image_dim_ordering() == 'th':
        X = X[:, None, ...]
    else:
        X = X[..., None]
    X /= 255.
    return model.predict(X, batch_size=10), images


def sample_encoded(encoded, images):
    dot_set = [read_mat(f.replace(".png", ".mat")) for f in images]

    X = []
    Y = []
    for enc, dots in zip(encoded, dot_set):
        if K.image_dim_ordering() == 'th':
            enc = np.stack([gaussian(ch) for ch in enc])
        else:
            enc = np.stack([gaussian(ch) for ch in enc.transpose((2, 0, 1))]).transpose((1, 2, 0))

        a = np.zeros((IMAGE_SZ, IMAGE_SZ))
        a[dots[0], dots[1]] = 1
        a = gaussian(a)

        i, j = np.where(a >= MID_DENSITY)
        pos_dots = list(zip(i, j))
        np.random.shuffle(pos_dots)
        pos_dots = pos_dots[:SAMPLES_PER_FRAME]
        x_pos = [enc[:, dot[0], dot[1]] for dot in pos_dots]
        y_pos = [a[dot[0], dot[1]] for dot in pos_dots]

        i, j = np.where(a < MID_DENSITY)
        neg_dots = list(zip(i, j))
        np.random.shuffle(neg_dots)
        neg_dots = neg_dots[:len(x_pos)]
        x_neg = [enc[:, dot[0], dot[1]] for dot in neg_dots]
        y_neg = [a[dot[0], dot[1]] for dot in neg_dots]

        X.extend(x_pos + x_neg)
        Y.extend(y_pos + y_neg)

    X = np.stack(X)
    return X, Y


def predict(encoded, images, classifier):
    dot_set = [read_mat(f.replace(".png", ".mat")) for f in images]

    X = []
    Y = []
    for enc, dots in zip(encoded, dot_set):
        if K.image_dim_ordering() == 'th':
            enc = enc.transpose((1, 2, 0))
        enc = enc.reshape(-1, N_FEATURES)
        x = classifier.predict(enc).reshape(-1, IMAGE_SZ, IMAGE_SZ)

        a = np.zeros((IMAGE_SZ, IMAGE_SZ))
        a[dots[0], dots[1]] = 1
        y = gaussian(a)

        X.extend(x)
        Y.extend(y)

    X = np.stack(X)
    return X, Y


def get_model(x, y, x_test, y_test, n_features, batch_size=512, num_epochs=20):
    def m():
        side = IMAGE_SZ + PATCH_SZ - 1
        shape = (1, side, side) if K.image_dim_ordering() == 'th' else (side, side, 1)
        input = Input(shape=shape)
        z = Conv2D(n_features, kernel_size=(PATCH_SZ, PATCH_SZ), kernel_initializer="he_normal")(input)
        z = Activation("elu")(z)
        z = Dropout(0.3)(z)
        return Model(input, z)

    features = m()
    shape = (1, PATCH_SZ, PATCH_SZ) if K.image_dim_ordering() == 'th' else (PATCH_SZ, PATCH_SZ, 1)
    patch_input = Input(shape=shape)
    z = features(patch_input)
    flat = Flatten()(z)
    model_output = Dense(1, activation="sigmoid")(flat)

    m = Model(patch_input, model_output)
    m.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    m.fit(x, y, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=2, shuffle=True)

    return features


def get_data(directory, model):
    X = []
    Y = []
    images = [join(directory, f) for f in listdir(directory) if f.endswith(".png")]
    dot_set = [read_mat(f.replace(".png", ".mat")) for f in images]

    for im_file, dots in zip(images, dot_set):
        img = ndimage.imread(im_file).astype(float)
        padded = np.ones((IMAGE_SZ + PATCH_SZ - 1, IMAGE_SZ + PATCH_SZ - 1)) * img.flatten().min()
        padded[PATCH_SZ // 2:PATCH_SZ // 2 + IMAGE_SZ, PATCH_SZ // 2:PATCH_SZ // 2 + IMAGE_SZ] = img

        X.append(padded)

        a = np.zeros((IMAGE_SZ, IMAGE_SZ))
        a[dots[0], dots[1]] = 1
        Y.append(gaussian(a))

    X = np.stack(X)
    Y = np.stack(Y)
    if K.image_dim_ordering() == 'th':
        X = X[:, None, ...]
        Y = Y[:, None, ...]
    else:
        X = X[..., None]
        Y = Y[..., None]
    X /= 255.

    pred = model.predict(X, batch_size=10)
    X = []
    for x in pred:
        if K.image_dim_ordering() == 'th':
            x = np.stack([gaussian(ch) for ch in x])
        else:
            x = np.stack([gaussian(ch) for ch in x.transpose((2, 0, 1))]).transpose((1, 2, 0))
        X.append(x)
    X = np.stack(X)
    return X, Y, dot_set


def get_unet(x, y, x_test, y_test, model, batch_size=8, num_epochs=20):
    for layer in model.layers: layer.trainable = False
    side = IMAGE_SZ
    shape = (N_FEATURES, side, side) if K.image_dim_ordering() == 'th' else (side, side, N_FEATURES)
    input = Input(shape=shape)
    area = Conv2D(1, kernel_size=(1, 1), padding="same", activation="linear", kernel_initializer="he_normal",
                  kernel_regularizer=l2(0.01), name="area")(input)
    m = Model(input, outputs=area)

    opt = Adam(lr=0.05)
    m.compile(loss=mean_squared_error_region, optimizer=opt)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, epsilon=0,
                                  patience=3, min_lr=0.001)
    m.fit(x, y, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
          callbacks = [reduce_lr], verbose=2, shuffle=True)
    return m


X, y = get_patches(TRAIN_DIR)
X_test, y_test = get_patches(TEST_DIR)
model = get_model(X, y, X_test, y_test, N_FEATURES, num_epochs=20)

encoded, images = encode_images(TRAIN_DIR, model)
X, Y = sample_encoded(encoded, images)

clf = Ridge(alpha=0.01)
clf.fit(X, Y)
del X, Y

encoded, images = encode_images(TEST_DIR, model)
pred, _ = predict(encoded, images, clf)
res = np.stack([gaussian(p) for p in pred]).sum(axis=(1,2))
ground = [read_mat(f.replace(".png", ".mat")).shape[1] for f in images]
print("Ground:", ground[:20])
print("Pred:", list(res[:20].astype(int)))
print("MAPE", mean_absolute_percentage_error(ground, res))

# X, Y, dot_train = get_data(TRAIN_DIR, model)
# ground_train = np.array([dots.shape[1] for dots in dot_train])
# X_test, Y_test, dot_test = get_data(TEST_DIR, model)
# ground_test = np.array([dots.shape[1] for dots in dot_test])
# unet = get_unet(X, Y, X_test, Y_test, model, batch_size=8, num_epochs=120)
#
# pred = unet.predict(X_test)
# res = pred.sum(axis=(1,2,3))
# print("Ground:", ground_test[:20])
# print("Pred:", list(res[:20].astype(int)))
# print("MAPE", mean_absolute_percentage_error(ground_test, res))
#
# print("Train")
# pred = unet.predict(X)
# res = pred.sum(axis=(1,2,3))
# print("Ground:", ground_test[:20])
# print("Pred:", list(res[:20].astype(int)))
# print("MAPE", mean_absolute_percentage_error(ground_test, res))
# pass
