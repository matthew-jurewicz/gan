import os, math, sys
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
import keras.backend as K
from keras.optimizers import Adam


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
    new_shape = new_shape.astype(np.int64)
    dense = Dense(new_shape.prod())(input_)
    layer = Reshape(new_shape)(dense)

    for i in range(nlayers - 1):
        if drop_rate > 0:
            layer = Dropout(drop_rate)(layer, training=True)
        batch_norm = BatchNormalization()(layer)
        relu = LeakyReLU(alpha=.2)(batch_norm)
        layer = Deconv2D(
            filters=filters // (2 ** i), 
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
        strides=(2,2)
    )(input_)

    for i in range(1, nlayers):
        relu = LeakyReLU(alpha=.2)(layer)
        conv = Conv2D(
            filters=filters * (2 ** i), 
            kernel_size=kernel_size, 
            strides=(2,2)
        )(relu)
        layer = BatchNormalization()(conv)

    flat = Flatten()(layer)
    relu = LeakyReLU(alpha=.2)(flat)
    dense = Dense(1, activation='sigmoid')(relu)

    model = Model(input_, dense)
    if ngpus > 0:
        model = multi_gpu_model(model, gpus=ngpus)

    return model


def loss(label, pred):
    return -K.mean(K.log(pred))


if __name__ == '__main__':
    img_shape = (128,128,3)
    z_shape = (100,)
    nlayers = 4
    filters = 128
    drop_rate = .5
    kernel_size = 5
    ngpus = 0

    # while True:
    #     img = next(load_data(img_shape))
    #     img = (img / 2 + .5) * 255
    #     img = img.astype(np.uint8)

    #     plt.imshow(img)
    #     plt.pause(.001)
    #     input('')

    G = generator(
        z_shape, 
        img_shape, 
        nlayers, 
        filters, 
        drop_rate, 
        kernel_size, 
        ngpus
    )
    D = discriminator(
        img_shape, 
        filters, 
        kernel_size, 
        nlayers, 
        ngpus
    )
    D.compile(
        loss='mse', 
        optimizer='sgd'
    )
    D.trainable = False

    z = Input(z_shape)
    img = G(z)
    valid = D(img)
    gan = Model(z, valid)
    adam = Adam(
        lr=.0002, 
        beta_1=.5
    )
    gan.compile(
        loss=loss, 
        optimizer=adam
    )

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    batch_size = 32
    nepochs = 100
    for i in range(nepochs):
        load_data_inst = load_data(img_shape)
        has_next = True
        D_loss = 0
        gan_loss = 0
        count = 0

        while has_next:
            x = []
            for j in range(batch_size // 2):
                img = next(load_data_inst, None)
                if img is None:
                    has_next = False
                    break
                x.append(img)

            if len(x) == 0:
                break
            x = np.array(x)

            z = np.random.normal(size=(len(x), z_shape[0]))
            imgs = G.predict_on_batch(z)

            #add image noise w/o changing padding
            noise = .1
            x *= np.random.uniform(low=1 - noise, high=1, size=x.shape)
            imgs *= np.random.uniform(low=1 - noise, high=1, size=imgs.shape)

            #one-sided soft labels
            c = .2
            y1 = np.random.uniform(low=1 - c / 2, high=1 + c / 2, size=len(x))
            y2 = np.zeros(len(imgs))

            #swap labels
            idxs = np.random.choice(len(y1), size=int(.1 * len(y1)), replace=False)
            tmp = y1[idxs]
            y1[idxs] = y2[idxs]
            y2[idxs] = tmp

            D_loss += .5 * D.train_on_batch(x, y1)
            D_loss += .5 * D.train_on_batch(imgs, y2)

            z = np.random.normal(size=(batch_size, z_shape[0]))
            gan_loss += gan.train_on_batch(z, np.ones(len(z)))

            count += 1

        D_loss /= count
        gan_loss /= count
        print('epoch {0}: D_loss -> {1:.4f}\t\tgan_loss -> {2:.4f}'.format(i + 1, D_loss, gan_loss))
        sys.stdout.flush()
        filepath = os.path.join('checkpoints', 'G_{:.4f}.h5'.format(gan_loss))
        G.save(filepath)