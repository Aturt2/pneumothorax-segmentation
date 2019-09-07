import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Concatenate, Dropout, Add, Activation
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf

def main_block(X, f, layer_n, name, l2_reg=0, double_conv=False):
    layer_n = str(layer_n)
    main = BatchNormalization(name="batchnorm_" + name + "_" + layer_n + "a")(X)
    main = Activation("relu", name="relu_" + name + "_" + layer_n + "a")(main)
    main = Conv2D(f, 3, kernel_initializer="he_normal", name="conv2d_" + name + "_" + layer_n + "a",
                  kernel_regularizer=l2(l2_reg), padding="same")(main)

    if double_conv:
        main = BatchNormalization(name="batchnorm_" + name + "_" + layer_n + "b")(main)
        main = Activation("relu", name="relu_" + name + "_" + layer_n + "b")(main)
        main = Conv2D(f, 3, kernel_initializer="he_normal", name="conv2d_" + name + "_" + layer_n + "b",
                      kernel_regularizer=l2(l2_reg), padding="same")(main)

    short = Conv2D(f, 1, kernel_initializer="he_normal", name="conv2d_" + name + "_short_" + layer_n)(X)
    main = Add(name="add_" + name + "_" + layer_n)([short, main])

    return main

def resunet(input_shape, n_layers, f_layer1, l2_reg=0, double_conv=False):
    inputs = Input(input_shape, name="input")

    X_down_list = []

    ## ENCODER
    # First down block w/o BN
    main = Conv2D(f_layer1, 3, kernel_initializer="he_normal", name="conv2d_down_1a",
                  kernel_regularizer=l2(l2_reg), padding="same")(inputs)
    main = BatchNormalization(name="batchnorm_down_1a")(main)
    main = Activation("relu", name="relu_down_1a")(main)
    if double_conv:
        main = Conv2D(f_layer1, 3, kernel_initializer="he_normal", name="conv2d_down_1b",
                      kernel_regularizer=l2(l2_reg), padding="same")(main)
    short = Conv2D(f_layer1, 1, kernel_initializer="he_normal", name="conv2d_down_res")(inputs)
    main = Add(name="add_down_1")([short, main])
    X_down_list.append(main)
    main = MaxPooling2D((2,2), name="pool_1")(main)

    for layer_n in range(2, n_layers):
        main = main_block(main, f_layer1 * 2**(layer_n - 1), layer_n, "down", l2_reg, double_conv)
        X_down_list.append(main)
        main = MaxPooling2D((2,2), name="pool_" + str(layer_n))(main)

    ## BRIDGE
    main = main_block(main, f_layer1 * 2**(n_layers - 1), n_layers, "bridge", l2_reg, double_conv)

    ## DECODER
    for layer_n in range(1, n_layers):
        layer_n = n_layers - layer_n
        main = UpSampling2D((2,2), name="up_" + str(layer_n))(main)
        main = Concatenate(axis=3, name="merge_" + str(layer_n))([main, X_down_list[layer_n - 1]])
        main = main_block(main, f_layer1 * 2**(layer_n - 1), layer_n, "up", l2_reg, double_conv)

    ## OUTPUT
    outputs = Conv2D(1, 1, activation="sigmoid", name="output")(main)

    model = Model(inputs=inputs, outputs=outputs)

    return model
