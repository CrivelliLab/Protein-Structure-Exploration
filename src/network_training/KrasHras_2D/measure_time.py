''''
measure_time.py
Updated: 09/25/17

README:

'''
import os, sys; sys.path.insert(0, '../')
from time import perf_counter
import numpy as np
from models import *
from scipy import misc
from binvox_io import read_binvox
from sklearn.metrics import roc_auc_score
from keras_extra import make_parallel_gpu

model_def = VOXNET_64_OG
gpus = 1
batch_size = 12
samples = 100

################################################################################

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    path = "../../../data/raw/NEW_KRAS_HRAS_BINVOX/KRAS/5o2tA-r0.binvox"

    # Load Model
    model, loss, optimizer, metrics = model_def(3, 2)
    if gpus > 1 : model = make_parallel_gpu(model, gpus)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Infer Data
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        y = [0 for j in range(2)]
        #x = misc.imread(path).astype('float')
        x = read_binvox(path).astype('int')
        x = np.transpose(x, (1,2,3,0))
        batch_x.append(x)
        batch_y.append(y)
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    # Measure Time
    model.train_on_batch(batch_x, batch_y)
    times = []
    for i in range(samples):
        time_ = perf_counter()
        model.train_on_batch(batch_x, batch_y)
        time_ = perf_counter() - time_
        times.append(time_)
        print(i, time_)

    # Average Time
    avg = np.average(times)
    print(avg)
