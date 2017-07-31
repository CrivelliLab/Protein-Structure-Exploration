'''
GenSaliency.py
Updated: 7/12/17
[NOT PASSING]

README:

'''
import os, sys
import numpy as np
from tqdm import tqdm
from vis.visualization import visualize_cam, visualize_saliency
from HKRAS_CIFAR_512 import CIFAR_512
from scipy import misc

encoded_folder = "NEWKRAS-T45-MS-HH512"
if __name__ == '__main__':

    # Load Model
    net = CIFAR_512(nb_channels=3, nb_class=2, nb_gpu=1)
    net.model.load_weights("kras_hras_full-data_50-epochs_98A_weights.hdf5")

    # Load Data
    x = []
    pdb_imgs = []
    for line in sorted(os.listdir('../../data/processed/tars/'+encoded_folder + '/')):
        if line.endswith('.png'): pdb_imgs.append(line)
    pdb_imgs = np.array(pdb_imgs)

    good = 0
    for i in tqdm(range(len(pdb_imgs))):
        img = misc.imread('../../data/processed/tars/'+encoded_folder + '/' + pdb_imgs[i])
        img = img.astype('float')
        img = img/255.0
        img = np.expand_dims(img, 0)
        p = net.model.predict(img)
        if np.argmax(p[0]) == 1:
            good += 1
        print p
        #atten_map = visualize_saliency(net.model, 28, [np.argmax(p[0])], img[0])
        #atten_map = np.dot(atten_map[...,:3], [0.299, 0.587, 0.114])
        #misc.imsave('HRAS_SALIENCY/'+pdb_imgs[i].split('.')[0]+'.png', atten_map)
    print good / len(pdb_imgs)
