import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


def gen_imgs(model):
    z_shape = (1,100)
    while True:
        z = np.random.normal(size=z_shape)
        img = model.predict_on_batch(z)[0]
        img = (img / 2 + .5) * 255
        img = img.astype(np.uint8)

        plt.imshow(img)
        plt.pause(.001)
        input('Press any key to continue...')


if __name__ == '__main__':
    filepath = 'checkpoints/G_0..h5'
    model = load_model(filepath)
    gen_imgs(model)