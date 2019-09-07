import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Concatenate, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf

def down_block(X, f, layer_n, l2_reg=0, double_conv=False):
    conv = Conv2D(f, 3, activation="relu", padding="same", kernel_initializer="he_normal",
                  name="conv2d_down_"+str(layer_n)+"a", kernel_regularizer=l2(l2_reg))(X)
    conv = BatchNormalization(name="batchnorm_down_"+str(layer_n)+"a")(conv)

    if double_conv:
        conv = Conv2D(f, 3, activation="relu", padding="same", kernel_initializer="he_normal",
                      name="conv2d_down_"+str(layer_n)+"b", kernel_regularizer=l2(l2_reg))(conv)
        conv = BatchNormalization(name="batchnorm_down_"+str(layer_n)+"b")(conv)

    X = MaxPooling2D((2,2), name="pool_"+str(layer_n))(conv)

    return conv, X

def up_block(X, f, X_down, layer_n, l2_reg=0, double_conv=False):
    X = UpSampling2D((2,2), name="up_"+str(layer_n))(X)
    X = Concatenate(axis=3, name="merge_"+str(layer_n))([X_down, X])

    X = Conv2D(f, 3, activation="relu", padding="same", kernel_initializer="he_normal",
               name="conv2d_up_"+str(layer_n)+"a", kernel_regularizer=l2(l2_reg))(X)
    X = BatchNormalization(name="batchnorm_up_"+str(layer_n)+"a")(X)

    if double_conv:
        X = Conv2D(f, 3, activation="relu", padding="same", kernel_initializer="he_normal",
                   name="conv2d_up_"+str(layer_n)+"b")(X)
        X = BatchNormalization(name="batchnorm_up_"+str(layer_n)+"b")(X)

    return X

def unet(input_shape, n_layers, f_layer1, l2_reg=0, double_conv=False):
    inputs = Input(input_shape, name="input")

    X_down_list = []
    X = inputs
    for layer_n in range(1, n_layers):
        conv, X = down_block(X, f_layer1 * 2**(layer_n - 1), layer_n, l2_reg, double_conv)
        X_down_list.append(conv)

    X = Conv2D(f_layer1 * 2**(n_layers - 1), 3, activation="relu", padding="same", kernel_initializer="he_normal",
               name="conv2d_"+str(n_layers)+"a")(X)
    X = BatchNormalization(name="batchnorm_"+str(n_layers)+"a")(X)
    if double_conv:
        X = Conv2D(f_layer1 * 2**(n_layers - 1), 3, activation="relu", padding="same", kernel_initializer="he_normal",
                   name="conv2d_"+str(n_layers)+"b", kernel_regularizer=l2(l2_reg))(X)
        X = BatchNormalization(name="batchnorm_"+str(n_layers)+"b")(X)
    X = Dropout(0.5)(X)

    for layer_n in range(1, n_layers):
        layer_n = n_layers - layer_n
        X = up_block(X, f_layer1 * 2**(layer_n - 1), X_down_list[layer_n - 1], layer_n, l2_reg, double_conv)

    outputs = Conv2D(1, 1, activation="sigmoid", name="output")(X)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def dice_coef(y_true, y_pred):
    '''
    Params: y_true -- the labeled mask corresponding to an rgb image
            y_pred -- the predicted mask of an rgb image
    Returns: dice_coeff -- A metric that accounts for precision and recall
                           on the scale from 0 - 1. The closer to 1, the
                           better.
    Citation (MIT License): https://github.com/jocicmarko/
                            ultrasound-nerve-segmentation/blob/
                            master/train.py
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1.0
    return (2.0*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)


def dice_coef_loss(y_true, y_pred):
    '''
    Params: y_true -- the labeled mask corresponding to an rgb image
            y_pred -- the predicted mask of an rgb image
    Returns: 1 - dice_coeff -- a negation of the dice coefficient on
                               the scale from 0 - 1. The closer to 0, the
                               better.
    Citation (MIT License): https://github.com/jocicmarko/
                            ultrasound-nerve-segmentation/blob/
                            master/train.py
    '''
    return 1-dice_coef(y_true, y_pred)
