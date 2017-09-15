'''
inference.py
Updated: 08/30/17

README:

Calculate Auc And Map

'''
import os, sys; sys.path.insert(0, '../../misc')
import numpy as np
from scipy import misc
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from models import *
from keras_extra import make_parallel_gpu

batch_size = 1
model_def = CIFAR_512_3CHAN_98ACC

image_size = (512, 512)
seed = 125
gpus = 1

################################################################################

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_folder = "../../../data/raw/NEW_KRAS_HRAS/"

    # Load Model
    model, loss, optimizer, metrics = model_def()
    if gpus > 1 : model = make_parallel_gpu(model, gpu)
    model.load_weights('weights/'+model_def.__name__+'.hdf5')
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Infer Data
    results = {}
    y_true = []
    y_score = []
    classes = sorted(os.listdir(data_folder))
    for i in range(len(classes)):
        for fn in os.listdir(data_folder + classes[i]):
            path = data_folder + classes[i] + '/' + fn
            protein = fn.split('-')[0]
            img = misc.imread(path)
            img = img.astype('float')
            img = np.expand_dims(img, 0)
            p = model.predict(img, batch_size=1, verbose=1)
            arg_max = np.argmax(p[0])
            if not results[protein]: results[protein] = [arg_max, 0, 0]
            results[protein][1] += p[arg_max]
            results[protein][2] += 1
            y = [0 for x in range(len(classes))]; y[i] = 1
            y_true.append(y)
            y.score(p[0])
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Evaluate And Write To File
    with open('logs/inference_'+model_def.__name__+'.csv', 'w') as fw:
        for key in sorted(results.keys()):
            hits = results[key][0]
            miss = results[key][1]
            acc = hits / float(hits + miss)
            line = str(key) + ',' + "{0:.4f}".format(acc) +'\n'
            print(line[:-2])
            fw.write(line)
