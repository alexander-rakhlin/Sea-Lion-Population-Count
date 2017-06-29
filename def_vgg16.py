from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling2D
import keras.backend as K
from keras.optimizers import Adam, SGD
from load_model import load_model
from time import time
import numpy as np


num_classes = 6

def vgg_block(num_filters, sz=3, repeats=1, pooling=True):
    def f(x):
        bn_axis = 1 if K.image_data_format() == "channels_first" else 3
        for _ in range(repeats):
            x = Conv2D(num_filters, (sz, sz), padding="same", activation="elu", kernel_initializer="he_normal")(x)
            x = BatchNormalization(axis=bn_axis, scale=False)(x)
        if pooling:
            x = MaxPooling2D((3, 3), strides=2)(x)
        return x
    return f


def vgg16(patch_sz, lr=1e-4, saved_model=None):
    if saved_model is not None:
        print("Load model from", saved_model)
        model = load_model(saved_model, lr=lr)
        return model

    if K.image_data_format() == 'channels_first':
        input_shape = (3, patch_sz, patch_sz)
    else:
        input_shape = (patch_sz, patch_sz, 3)

    inputs = Input(shape=input_shape)

    z = vgg_block(64, sz=2, repeats=1, pooling=False)(inputs)
    z = vgg_block(64, repeats=1)(z)
    z = vgg_block(128, repeats=2)(z)
    z = vgg_block(256, repeats=3)(z)
    z = vgg_block(512, repeats=3)(z)

    z = Flatten()(z)
    z = Dropout(0.3)(z)
    z = Dense(256, activation='elu', kernel_initializer="he_normal")(z)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(z)

    model = Model(inputs=inputs, outputs=outputs, name="vgg16")
    opt = Adam(lr=lr)
    # opt = SGD(lr=lr, momentum=0.9, decay=0.0005, nesterov=True)
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


def vgg16fcn(cell_sz, patch_model=None, saved_model=None, lr=1e-4):
    if saved_model is not None:
        print("Load model from", saved_model)
        model = load_model(saved_model, lr=lr)
        return model
    """
    specified for 75x75 input conversion: 512x3x3 last conv layer
    """
    assert patch_model is not None
    if K.image_data_format() == 'channels_first':
        input_shape = (3, cell_sz, cell_sz)
    else:
        input_shape = (cell_sz, cell_sz, 3)

    model = vgg16(patch_sz=cell_sz)
    model.load_weights(patch_model)

    x = model.get_layer("max_pooling2d_4").output   # -5 layer
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3, 3), activation="elu", name="fcn")(x)  # (3, 3) comes from 75x75 patch
    x = GlobalMaxPooling2D()(x)
    outputs = Dense(1, activation="sigmoid", kernel_initializer="he_normal")(x)

    model_fcn = Model(inputs=model.get_input_at(0), outputs=outputs, name="vgg16fcn")
    opt = SGD(lr=lr, momentum=0.9, decay=0.0005, nesterov=True)
    model_fcn.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    donor_layer = model.get_layer("dense_1")
    donor_weights = donor_layer.get_weights()
    recipient_layer = model_fcn.get_layer("fcn")

    # Important
    donor_weights[0] = np.transpose(donor_weights[0].T.reshape((256, 512, 3, 3)), [2, 3, 1, 0])[::-1, ::-1, ...]
    recipient_layer.set_weights(donor_weights)

    return model_fcn


def timeit():
    saved_model = "models/model_vgg16.epoch115-0.91.h5"
    CELL_SZ = 200
    N = 10000
    X = np.random.randn(N, 3, CELL_SZ, CELL_SZ)
    model = vgg16fcn(CELL_SZ, saved_model)
    t0 = time()
    pred = model.predict(X)
    print(pred.shape)
    elapsed = time() - t0
    print("Elapsed {} seconds. {:0.2f} seconds/cell".format(int(elapsed), elapsed / len(X)))


if __name__ == "__main__":
    timeit()