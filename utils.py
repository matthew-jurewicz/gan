import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
import wget, os, time, re, io, progressbar
from google.cloud import vision
from PIL import Image


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


def dload_data(xls_filepath, n):
        df = pd.read_excel(xls_filepath).sample(n, replace=False)
        count = 0

        if not os.path.exists('data'):
                os.mkdir('data')

        for i, row in df.iterrows():
                count += 1
                url = row['URL'].replace('html', 'art', 1).replace('.html', '.jpg')
                print('')
                print('{}/{}: {}'.format(count, n, url))

                dst = 'data/' + re.sub(r'(?!\.jpg)\W+', '_', url)
                if not os.path.exists(dst):
                        wget.download(url, dst)
                        time.sleep(.25)


def detect_faces(data_dir):
        files = os.listdir(data_dir)
        faces_dir = os.path.join(data_dir, 'faces')
        if not os.path.exists(faces_dir):
                os.mkdir(faces_dir)
                
        client = vision.ImageAnnotatorClient()
        with progressbar.ProgressBar(max_value=len(files)) as bar:
                for i in range(len(files)):
                        filepath = os.path.join(data_dir, files[i])
                        with open(filepath, 'rb') as f:
                                img = Image.open(f)
                                f.seek(0)
                                
                                response = client.annotate_image({
                                        'image':{'content':f.read()}, 
                                        'features':[{'type':vision.enums.Feature.Type.FACE_DETECTION}]
                                })
                                for j in range(len(response.face_annotations)):
                                        ul, ur, lr, ll = response.face_annotations[j].bounding_poly.vertices
                                        face = img.crop((ul.x, ul.y, lr.x, lr.y))
                                        filepath = os.path.join(faces_dir, 
                                                files[i].replace('.jpg', '_face{}.jpg'.format(j)))
                                        face.save(filepath)

                        bar.update(i + 1)


if __name__ == '__main__':
#     filepath = 'checkpoints/G_0..h5'
#     model = load_model(filepath)
#     gen_imgs(model)

#     dload_data('catalog.xls', n=4618)

        detect_faces(r'C:\Users\matth\Documents\gan\data')