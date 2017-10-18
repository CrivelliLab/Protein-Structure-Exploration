''''
inference.py
Updated: 09/25/17

README:

This script is used to run inference on KRAS/HRAS data using a trained
classifier.

'''
import os, sys; sys.path.insert(0, '../')
import numpy as np
from models import *
from scipy import misc
from sklearn.metrics import roc_auc_score
from keras_extra import make_parallel_gpu

model_def = CIFAR_NET
weight_file = 'weights/'+model_def.__name__+'.hdf5'
gpus = 1

################################################################################

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_folder = "../../../data/raw/NEW_KRAS_HRAS/"

    # Load Model
    model, loss, optimizer, metrics = model_def(3, 2)
    if gpus > 1 : model = make_parallel_gpu(model, gpu)
    model.load_weights(weight_file)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Infer Data
    results = {}
    classes = sorted(os.listdir(data_folder))
    for i in range(len(classes)):
        for fn in os.listdir(data_folder + classes[i]):
            path = data_folder + classes[i] + '/' + fn
            image_name = fn.split('-')[0]
            img = misc.imread(path)
            img = img.astype('float')
            img = np.expand_dims(img, 0)
            p = model.predict(img, batch_size=1, verbose=0)
            arg_max = np.argmax(p[0])
            if image_name not in results.keys(): results[image_name] = [i, 0, 0]
            results[image_name][1] += p[0][i]
            results[image_name][2] += 1

    # Evaluate And Write To File
    inference_file = 'logs/inf_'+weight_file.split('.')[0].split('/')[1]+'.csv'
    with open(inference_file, 'w') as fw:
        y_true = []
        y_score = []
        for key in sorted(results.keys()):
            acc = results[key][1] / results[key][2]
            line = str(key) + ',' + "{0:.4f}".format(acc)+','+str(results[key][0]) +'\n'
            print(line[:-1])
            fw.write(line)
            i = results[key][0]
            y = [0 for x in range(len(classes))]; y[i] = 1
            p = [0,0]
            if i == 0:
                p[0] = acc
                p[1] = 1 - acc
            else:
                p[1] = acc
                p[0] = 1 - acc
            y_true.append(y)
            y_score.append(p)
        print(y_true, y_score)
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        auc = roc_auc_score(y_true, y_score)
        print(auc)
