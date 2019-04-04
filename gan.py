import os, math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from keras.layers import (
    Input, 
    Dense, 
    Reshape, 
    Dropout, 
    BatchNormalization, 
    LeakyReLU, 
    Deconv2D, 
    Conv2D, 
    Flatten
)
from keras.models import Model
from keras.utils import multi_gpu_model


def load_data(input_shape):
    filelist = [os.path.join(root, filename) for root, dirs, files in os.walk('../data') 
        for filename in files if filename.endswith('.jpg')]
    idxs = np.random.choice(len(filelist), size=len(filelist), replace=False)

    for i in idxs:
        img = Image.open(filelist[i]).convert('RGB')
        max_axis = img.size.index(max(img.size))
        scale = input_shape[max_axis] / img.size[max_axis]
        img = img.resize((math.ceil(scale * img.width), math.ceil(scale * img.height)))

        x = np.asarray(img)
        #normalize and pad
        c = 255 / 2
        x = (x - c) / c
        min_axis = 1 if len(x) == input_shape[0] else 0
        pad_width = [(0,0)] * 3
        pad_width[min_axis] = (0, input_shape[min_axis] - x.shape[min_axis])
        x = np.pad(x, pad_width=pad_width, mode='constant')

        yield x


#https://arxiv.org/pdf/1511.06434.pdf
def generator(input_shape, 
              img_shape, 
              nlayers, 
              D_filters, 
              drop_rate, 
              kernel_size, 
              ngpus=0):
    filters = D_filters * (2 ** (nlayers - 2))

    input_ = Input(input_shape)
    new_shape = np.zeros(3)
    new_shape[:-1] = img_shape[0] / (2 ** nlayers)
    new_shape[-1] = 2 * filters
    dense = Dense(new_shape.prod())(input_)
    layer = Reshape(new_shape)(dense)

    for i in range(nlayers - 1):
        if drop_rate > 0:
            layer = Dropout(drop_rate)(layer, training=True)
        batch_norm = BatchNormalization()(layer)
        relu = LeakyReLU(alpha=.2)(batch_norm)
        layer = Deconv2D(
            filters=int(filters / (2 ** i)), 
            kernel_size=kernel_size, 
            strides=(2,2), 
            padding='same'
        )(relu)

    relu = LeakyReLU(alpha=.2)(layer)
    deconv = Deconv2D(
        filters=3, 
        kernel_size=kernel_size, 
        strides=(2,2), 
        padding='same', 
        activation='tanh'
    )(relu)

    model = Model(input_, deconv)
    if ngpus > 0:
        model = multi_gpu_model(model, gpus=ngpus)

    return model


#https://arxiv.org/pdf/1511.06434.pdf
def discriminator(input_shape, 
                  filters, 
                  kernel_size, 
                  nlayers, 
                  ngpus=0):
    input_ = Input(input_shape)
    layer = Conv2D(
        filters=filters, 
        kernel_size=kernel_size, 
        strides=(2,2), 
        padding='same'
    )(input_)

    for i in range(1, nlayers):
        relu = LeakyReLU(alpha=.2)(layer)
        conv = Conv2D(
            filters=filters * (2 ** i), 
            kernel_size=kernel_size, 
            strides=(2,2), 
            padding='same'
        )(relu)
        layer = BatchNormalization()(conv)

    flat = Flatten()(layer)
    relu = LeakyReLU(alpha=.2)(flat)
    dense = Dense(1, activation='sigmoid')(relu)

    model = Model(input_, dense)
    if ngpus > 0:
        model = multi_gpu_model(model, gpus=ngpus)

    return model


if __name__ == '__main__':
    img_shape = (128,128,3)

    # while True:
    #     img = next(load_data(img_shape))
    #     img = (img / 2 + .5) * 255
    #     img = img.astype(np.uint8)

    #     plt.imshow(img)
    #     plt.pause(.001)
    #     input('')