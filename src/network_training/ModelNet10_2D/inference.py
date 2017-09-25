'''
inference.py
Updated: 09/25/17

README:

This script is used to run inference on ModelNet10 data using a trained
classifier.

'''
import os, sys; sys.path.insert(0, '../../misc')
import numpy as np
from models import *
from scipy import misc
from sklearn.metrics import roc_auc_score
from keras_extra import make_parallel_gpu

batch_size = 1
model_def = CIFAR_NET
weight_file = 'weights/'+model_def.__name__+'22092017.hdf5'

seed = 125
gpus = 1

################################################################################

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_folder = "../../../data/split/AUG_ModelNet10_E2D/test/"

    # Load Model
    model, loss, optimizer, metrics = model_def(1, 10)
    if gpus > 1 : model = make_parallel_gpu(model, gpu)
    model.load_weights(weight_file)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    datagen = ImageDataGenerator()

    # Infer Data
    results = {}
    classes = sorted(os.listdir(data_folder))
    for i in range(len(classes)):
        if not os.path.isdir(data_folder+classes[i]): continue
        for fn in os.listdir(data_folder + classes[i]):
            path = data_folder + classes[i] + '/' + fn
            image_name = fn.split('.')[0].split('_')
            image_name = image_name[0] +'_' +image_name[1]
            img = misc.imread(path)
            img = img.astype('float')
            img = np.expand_dims(img, -1)
            img = np.expand_dims(img, 0)
            p = model.predict(img, batch_size=1, verbose=0)
            p = p[0]
            if image_name not in results.keys(): results[image_name] = [i, p, 0]
            else:results[image_name][1] = p + results[image_name][1]
            results[image_name][2] += 1

    # Evaluate And Write To File
    inference_file = 'logs/inf_'+weight_file.split('.')[0].split('/')[1]+'.csv'
    with open(inference_file, 'w') as fw:
        #y_true = []
        #y_score = []
        avg_acc = 0
        for key in sorted(results.keys()):
            acc = results[key][1] / results[key][2]
            line = str(key) + ',' + "{0:.4f}".format(acc)+','+str(results[key][0]) +'\n'
            fw.write(line)
            if np.argmax(acc) == results[key][0]: avg_acc +=1
            else: print(str(key), np.argmax(acc), results[key][0])
        avg_acc = avg_acc / len(results.keys())
        print(avg_acc)
        #y_true = np.array(y_true)
        #y_score = np.array(y_score)
        #auc = roc_auc_score(y_true, y_score)
