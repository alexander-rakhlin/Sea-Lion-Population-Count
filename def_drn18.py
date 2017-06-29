from keras.layers import Input
from keras import layers
from keras.layers import Activation
from keras.layers import Conv2D, Flatten, Dense
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD
from load_model import load_model
from keras.activations import softmax
import numpy as np


num_classes = 6


def identity_block(input_tensor, filters, stage, block, dilations=(1, 1)):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returnsimport numpy as np
        Output tensor for the block.
    """
    filters1, filters2 = filters
    dilation1, dilation2 = dilations
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    act_name_base = 'act' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (3, 3), padding='same', name=conv_name_base + '2a', dilation_rate=dilation1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (3, 3), dilation_rate=dilation2,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu', name=act_name_base)(x)
    return x


def conv_block(input_tensor, filters, stage, block, strides=(2, 2), dilations=(1, 1)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2 = filters
    dilation1, dilation2 = dilations
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (3, 3), padding='same', strides=strides, dilation_rate=dilation1,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (3, 3), padding='same', dilation_rate=dilation2,
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    shortcut = Conv2D(filters2, (3, 3), padding='same', strides=strides, dilation_rate=dilation1,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def drn18(patch_sz, lr=1e-1, saved_model=None):
    if saved_model is None:
        if K.image_data_format() == 'channels_first':
            input_shape = (3, patch_sz, patch_sz)
        else:
            input_shape = (patch_sz, patch_sz, 3)

        img_input = Input(shape=input_shape)

        x = conv_block(img_input, [64, 64], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, [64, 64], stage=2, block='b')

        x = conv_block(x, [128, 128], stage=3, block='a')
        x = identity_block(x, [128, 128], stage=3, block='b')

        x = conv_block(x, [256, 256], stage=4, block='a', strides=(1, 1), dilations=(1, 2))
        x = identity_block(x, [256, 256], stage=4, block='b', dilations=(2, 2))

        x = conv_block(x, [512, 512], stage=5, block='a', strides=(1, 1), dilations=(2, 4))
        x = identity_block(x, [512, 512], stage=5, block='b', dilations=(4, 4))

        pooling_shape = Model(img_input, x).output_layers[0].output_shape[-2:]
        x = AveragePooling2D(pooling_shape, name='avg_pool')(x)

        x = Flatten()(x)
        x = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(x)

        model = Model(inputs=img_input, outputs=x, name="drn18")

        opt = SGD(lr=lr, momentum=0.9, decay=1e-4, nesterov=True)
        model.compile(optimizer=opt,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
    else:
        print("Load model from", saved_model)
        model = load_model(saved_model, lr=lr)

    return model


def drn18_feat(frame_sz, saved_model):
    model = drn18(frame_sz)
    model.load_weights(saved_model)
    model_feat = Model(inputs=model.get_input_at(0), outputs=model.get_layer("act5b_branch").output)
    return model_feat


def drn18_linear_regression(frame_sz, saved_model, linear_classifiers):
    if type(linear_classifiers) in (list, tuple):
        linear_classifiers = [cls for c in linear_classifiers for _, cls in sorted(c.items())]
        linear_classifiers = {key: value for key, value in enumerate(linear_classifiers)}

    model = drn18(frame_sz)
    model.load_weights(saved_model)

    x = model.get_layer("act5b_branch").output
    output = Conv2D(len(linear_classifiers), (1, 1), activation="linear", name="linear_regression")(x)
    model_feat = Model(inputs=model.get_input_at(0), outputs=output)

    w = np.stack([cls.coef_ for _, cls in sorted(linear_classifiers.items())]).T[None, None, ...]
    b = np.concatenate([cls.intercept_ for _, cls in sorted(linear_classifiers.items())])
    layer = model_feat.get_layer("linear_regression")
    layer.set_weights([w, b])

    return model_feat


def drn18_field_view(frame_sz, saved_model):

    def conv_softmax():
        if K.image_data_format() == "channels_last":
            softmax_axis = 3
        else:
            softmax_axis = 1
        def f(x):
            return softmax(x, axis=softmax_axis)
        return f

    model = drn18(frame_sz)
    model.load_weights(saved_model)

    x = model.get_layer("act5b_branch").output
    output = Conv2D(num_classes, (1, 1), activation=conv_softmax(), name="field_view")(x)

    model_field_view = Model(inputs=model.get_input_at(0), outputs=output)

    layer_donor = model.get_layer("dense_1")
    weights = layer_donor.get_weights()
    weights[0] = weights[0][None, None, ...]
    layer_recipient = model_field_view.get_layer("field_view")
    layer_recipient.set_weights(weights)

    return model_field_view
