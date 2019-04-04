import os, math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    img_shape = (128,128,3)

    # while True:
    #     img = next(load_data(img_shape))
    #     img = (img / 2 + .5) * 255
    #     img = img.astype(np.uint8)

    #     plt.imshow(img)
    #     plt.pause(.001)
    #     input('')