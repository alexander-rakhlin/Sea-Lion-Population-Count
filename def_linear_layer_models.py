from keras.layers import Input
from keras import layers
from keras.layers import Activation, Lambda, Concatenate
from keras.layers import Conv2D, Flatten, Dense, Cropping2D, Average, Reshape
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D
from keras.layers import BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD
from load_model import load_model
from keras.layers.pooling import _GlobalPooling2D


num_classes = 5


def rmse(y_true, y_pred):
    return K.mean(K.sqrt(K.mean(K.square(y_pred - y_true), axis=0)))


def mape_custom(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), 1, None))
    return 100. * K.mean(diff, axis=-1)


def msle_custom(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, 0, None) + 1.)
    second_log = K.log(y_true + 1.)
    return K.mean(K.mean(K.square(first_log - second_log), axis=0))


class GlobalVariancePooling2D(_GlobalPooling2D):
    """Global variance pooling operation for spatial data.

    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.var(inputs, axis=[1, 2])
        else:
            return K.var(inputs, axis=[2, 3])


class GlobalMinPooling2D(_GlobalPooling2D):
    """Global min pooling operation for spatial data.

    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.min(inputs, axis=[1, 2])
        else:
            return K.min(inputs, axis=[2, 3])


class GlobalSumPooling2D(_GlobalPooling2D):
    """Global sum pooling operation for spatial data.

    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])


def identity_block(input_tensor, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: "a","b"..., current block label, used for generating layer names

    # Returnsimport numpy as np
        Output tensor for the block.
    """
    filters1, filters2 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
    act_name_base = "act" + str(stage) + block + "_branch"

    x = Conv2D(filters1, (3, 3), padding="same", kernel_initializer="he_normal",
               name=conv_name_base + "2a")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("elu")(x)

    x = Conv2D(filters2, (3, 3), padding="same", kernel_initializer="he_normal", name=conv_name_base + "2b")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)

    x = layers.add([x, input_tensor])
    x = Activation("elu", name=act_name_base)(x)
    return x


def conv_block(input_tensor, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: "a","b"..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(filters1, (3, 3), padding="same", strides=strides, kernel_initializer="he_normal",
               name=conv_name_base + "2a")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("elu")(x)

    x = Conv2D(filters2, (3, 3), padding="same", kernel_initializer="he_normal",
               name=conv_name_base + "2b")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)

    shortcut = Conv2D(filters2, (3, 3), padding="same", strides=strides, kernel_initializer="he_normal",
                      name=conv_name_base + "1")(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(shortcut)

    x = layers.add([x, shortcut])
    x = Activation("elu")(x)
    return x


def resnet(input_shape, cropping=19, lr=1e-2, saved_model=None):
    if saved_model is not None:
        print("Load model from", saved_model)
        model = load_model(saved_model, lr=lr, custom_objects={
            "GlobalVariancePooling2D": GlobalVariancePooling2D,
            "GlobalMinPooling2D": GlobalMinPooling2D,
            "GlobalSumPooling2D": GlobalSumPooling2D,
            "rmse": rmse, "mape_custom": mape_custom, "msle_custom": msle_custom})
        return model

    img_input = Input(shape=input_shape)

    z = Cropping2D(cropping=cropping)(img_input)
    z1 = GlobalAveragePooling2D(name="input_average_pool")(z)
    z2 = GlobalMaxPooling2D(name="input_max_pool")(z)
    z3 = GlobalVariancePooling2D(name="input_var_pool")(z)

    x = Activation("elu")(img_input)
    x = conv_block(x, [32, 32], stage=2, block="a")
    x = identity_block(x, [32, 32], stage=2, block="b")

    x = conv_block(x, [32, 32], stage=3, block="a")
    x = identity_block(x, [32, 32], stage=3, block="b")

    x = conv_block(x, [32, 32], stage=4, block="a")
    x = identity_block(x, [32, 32], stage=4, block="b")

    x = conv_block(x, [64, 64], stage=5, block="a")
    x = identity_block(x, [64, 64], stage=5, block="b")

    x = GlobalAveragePooling2D(name="avg_pool")(x)

    x = layers.concatenate([x, z1, z2, z3])

    x = Dense(num_classes, activation="linear", kernel_initializer="he_normal")(x)

    model = Model(inputs=img_input, outputs=x, name="resnet")

    opt = SGD(lr=lr, momentum=0.9, decay=1e-4, nesterov=True)
    model.compile(optimizer=opt, loss="mse", metrics=["mse", rmse])

    return model


def super_resnet(model_files):
    outputs = []
    img_input, crop_branch, aux_model = None, None, None
    for i, m_file in enumerate(model_files):
        print("Load model from", m_file)
        model = load_model(m_file, custom_objects={"GlobalVariancePooling2D": GlobalVariancePooling2D, "rmse": rmse})

        if img_input is None:
            input_shape = model.get_input_shape_at(0)[1:]
            img_input = Input(shape=input_shape)
            crop_branch = [model.get_layer(name).output for name in
                           ("input_average_pool", "input_max_pool", "input_var_pool")]

            aux_output = model.layers[66].output    # crop layer
            aux_output = GlobalSumPooling2D()(aux_output)
            output_shape = model.output_shape[-1]
            aux_output = Reshape((-1, output_shape))(aux_output)
            aux_output = GlobalAveragePooling1D()(aux_output)
            aux_model = Model(inputs=model.inputs[0], outputs=aux_output, name="direct_sum")
        else:
            model.layers[71].inbound_nodes = []
            model.layers[71]([model.layers[67].output] + crop_branch)
            [model.layers.pop(i) for i in (70, 69, 68, 66)]

        for layer in model.layers:
            layer.name += "_{}".format(i)
        model.name += "_{}".format(i)

        outputs.append(model(img_input))
    output = Average()(outputs)
    aux_output = aux_model(img_input)
    model = Model(inputs=img_input, outputs=[output, aux_output])
    return model


"""
M46
"""


def vgg_block(num_filters, kernel_sz=3, strides=(1, 1)):
    def f(x):
        bn_axis = 1 if K.image_data_format() == "channels_first" else 3
        x = Conv2D(num_filters, (kernel_sz, kernel_sz), padding="same", activation="elu",
                   strides = strides, kernel_initializer="he_normal")(x)
        x = BatchNormalization(axis=bn_axis, scale=False)(x)
        x = Conv2D(num_filters, (1, 1), padding="same", activation="elu", kernel_initializer="he_normal")(x)
        x = BatchNormalization(axis=bn_axis, scale=False)(x)
        x = MaxPooling2D((3, 3), strides=2)(x)
        return x
    return f


def m46(input_shape, cropping=19, lr=1e-2, saved_model=None):
    if saved_model is not None:
        print("Load model from", saved_model)
        model = load_model(saved_model, lr=lr, loss=rmse)
        return model

    inputs = Input(shape=input_shape)
    z = vgg_block(16, strides=(2, 2))(inputs)
    z = vgg_block(32, strides=(2, 2))(z)
    z = vgg_block(64)(z)
    # z = vgg_block(128)(z)

    z = Flatten()(z)
    z = Dropout(0.0)(z)

    x = Cropping2D(cropping=cropping)(inputs)
    x1 = GlobalAveragePooling2D(name="input_average_pool")(x)
    x2 = GlobalMaxPooling2D(name="input_max_pool")(x)
    z = layers.concatenate([z, x1, x2])

    z = Dense(256, activation='elu', kernel_initializer="he_normal")(z)
    outputs = Dense(num_classes, activation="linear", kernel_initializer="he_normal")(z)

    model = Model(inputs=inputs, outputs=outputs, name="m46")
    opt = SGD(lr=lr, momentum=0.9, decay=5e-4, nesterov=True)
    model.compile(optimizer=opt, loss=rmse)

    return model


"""
Maxout
"""

def maxout_block(k):
    """
    `k` Number of linear feature extractors
    """
    feature_axis = 1 if K.image_data_format() == "channels_first" else 3

    def output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 4  # only valid for 4D tensors
        shape[feature_axis] = 1
        return tuple(shape)

    def f(x):
        x = Conv2D(k, (1, 1), strides=1, kernel_initializer="he_normal")(x)
        x = Lambda(lambda z: K.max(z, axis=feature_axis, keepdims=True), output_shape=output_shape)(x)
        return x

    return f

def maxout_layer(k, m):
    """
    `m` Number of units in each linear feature extractor (complexity)
    """
    feature_axis = 1 if K.image_data_format() == "channels_first" else 3

    def f(x):
        x = Concatenate(axis=feature_axis)([maxout_block(k)(x) for _ in range(m)])
        return x

    return f


def maxout(input_shape, k=10, m=6, cropping=19, lr=1e-2, saved_model=None):
    if saved_model is not None:
        print("Load model from", saved_model)
        model = load_model(saved_model, lr=lr, custom_objects={"rmse": rmse})
        return model

    img_input = Input(shape=input_shape)

    x = conv_block(img_input, [32, 32], strides=(1, 1), stage=0, block="0")

    x = maxout_layer(k, m)(x)
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(num_classes, activation="linear", kernel_initializer="he_normal")(x)

    model = Model(inputs=img_input, outputs=x, name="maxout")

    opt = SGD(lr=lr, momentum=0.9, decay=1e-4, nesterov=True)
    model.compile(optimizer=opt, loss="mse", metrics=["mse", rmse])

    return model
