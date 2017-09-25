'''
gen_tsne.py
Updated: 09/25/17

README:

This script is used to generate T-SNE embedding of trained Kras/Hras classifier.

'''
import os, sys; sys.path.insert(0, '../')
import numpy as np
from models import *
from tqdm import tqdm
from scipy import misc
from sklearn.manifold import TSNE
from keras_extra import make_parallel_gpu

model_def = CIFAR_NET
weight_file = 'weights/'+model_def.__name__+'.hdf5'
gpus = 1
nb_rot = 512

################################################################################

if __name__ == '__main__':

    # File paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_folders = ["../../../data/split/KRAS_HRAS_seed45/test/",
                    "../../../data/raw/NEW_KRAS_HRAS/"]

    # Get file paths
    file_paths = []
    for data_folder in data_folders:
        classes = sorted(os.listdir(data_folder))
        for i in range(len(classes)):
            for fn in sorted(os.listdir(data_folder + classes[i])):
                path = data_folder + classes[i] + '/' + fn
                file_paths.append(path)

    # Load model
    model, loss, optimizer, metrics, x, l = model_def(3, 2)
    if gpus > 1 : model = make_parallel_gpu(model, gpu)
    model.load_weights(weight_file)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    inter_model = Model(inputs=x, outputs=l)

    # Infer data
    img_names = []
    predictions = []
    for path in tqdm(file_paths):
        img_name = path.split('-')[0]
        img = misc.imread(path)
        img = img.astype('float')
        img = np.expand_dims(img, 0)
        p = inter_model.predict(img, batch_size=1, verbose=0)
        predictions.append(p[0])
        image_names.append(img_name)
    predictions = np.array(predictions)
    X_embedded = TSNE(n_components=2, perplexity=2400, learning_rate=1500, verbose=1).fit_transform(predictions)
    x = X_embedded[:,0]
    y = X_embedded[:,1]

    # Calculate average embedded activation
    x_avgs = []
    y_avgs = []
    for i in range(len(list(set(filenames)))):
        x_avgs.append(np.sum(x[(nb_rot*i):(nb_rot*i)+nb_rot])/nb_rot)
        y_avgs.append(np.sum(y[(nb_rot*i):(nb_rot*i)+nb_rot])/nb_rot)

    # Write results to file
    with open('logs/tsne_'+model_def.__name__+'.csv', 'w') as fw:
        for i in range(len(list(set(image_names)))):
            line = image_names[(i*nb_rot)] + ',' + "{0:.4f}".format(x_avgs[i]) +','+"{0:.4f}".format(y_avgs[i])+'\n'
            print(line[:-1]); fw.write(line)
