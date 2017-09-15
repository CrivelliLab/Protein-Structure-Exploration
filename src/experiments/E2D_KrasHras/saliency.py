'''
saliency.py
Updated: 7/12/17
[NOT PASSING]

README:

'''
import os, sys
import numpy as np
from tqdm import tqdm
from vis.visualization import visualize_cam, visualize_saliency
from models_0003 import *
from keras_extra import make_parallel_gpu
from keras.metrics import categorical_accuracy
from scipy import misc

model_def = CIFAR_512_3CHAN_98ACC2
gpus = 1

if __name__ == '__main__':

    # Load Model
    model, loss, optimizer, metrics = model_def()
    if gpus > 1 : model = make_parallel_gpu(model, gpu)
    model.load_weights('weights/'+model_def.__name__+'.hdf5')
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Load Data
    x = []
    pdb_imgs = []
    for line in sorted(os.listdir('../../../data/raw/NEW_KRAS_HRAS3/HRASBOUNDED0%64-T45-MS-HH512/')):
        if line.endswith(".png"): pdb_imgs.append(line)
    pdb_imgs = np.array(pdb_imgs)

    for i in tqdm(range(len(pdb_imgs))):
        img = misc.imread('../../../data/raw/NEW_KRAS_HRAS3/HRASBOUNDED0%64-T45-MS-HH512/'+ pdb_imgs[i])
        img = img.astype('float')
        img = np.expand_dims(img, 0)
        p = model.predict(img, batch_size=1, verbose=1)
        atten_map = visualize_saliency(model, 28, [np.argmax(p[0])], img[0])
        atten_map = atten_map/255.0
        atten_map = np.dot(atten_map[...,:3], [0.299, 0.587, 0.114])
        misc.imsave('../../../data/raw/sal_maps/'+pdb_imgs[i].split('.')[0]+'.png', atten_map)
