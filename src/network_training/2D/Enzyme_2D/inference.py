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

model_def = SMPLENET_MODIFIED4
weight_file = 'weights/'+model_def.__name__+'.hdf5'
gpus = 1

################################################################################

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_folder = "../../../data/split/ENZYME_split102517/validation/"

    # Load Model
    model, loss, optimizer, metrics = model_def(5, 6)
    if gpus > 1 : model = make_parallel_gpu(model, gpus)
    model.load_weights(weight_file)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Infer Data
    results = {}
    classes = sorted(os.listdir(data_folder))
    for i in range(len(classes)):
        if not os.path.isdir(data_folder+classes[i]): continue
        for fn in os.listdir(data_folder + classes[i]):
            path = data_folder + classes[i] + '/' + fn
            image_name = fn.split('-')[0]

            array = imread(path)
            array = array[:,:,0] + (array[:,:,1] * 2**8) + (array[:,:,2] * 2**16)
            array = np.expand_dims(array.astype('>i8'), axis=-1)
            nb_chans = 5
            array = np.unpackbits(array.view('uint8'),axis=-1)[:,:,-nb_chans:]
            array = np.flip(array, axis=-1)
            array = array * 255

            resized_array = []
            for i in range(nb_chans):
                temp = imresize(array[:,:,i], target_size, interp='bicubic')
                resized_array.append(temp)
            img = np.transpose(np.array(resized_array), (1,2,0))

            img = np.expand_dims(img, 0)
            p = model.predict(img, batch_size=1, verbose=0)
            p = p[0]
            if image_name not in results.keys(): results[image_name] = [i, p, 0]
            else:results[image_name][1] = p + results[image_name][1]
            results[image_name][2] += 1

    # Evaluate
    avg_acc = 0
    for key in sorted(results.keys()):
        acc = results[key][1] / results[key][2]
        if np.argmax(acc) == results[key][0]: avg_acc +=1
        else: print(str(key), np.argmax(acc), results[key][0])
    avg_acc = avg_acc / len(results.keys())
    print(avg_acc)
