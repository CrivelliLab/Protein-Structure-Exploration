'''
tsne.py
Updated: 08/30/17

README:

'''
import os
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.manifold import TSNE
from models_0003 import *
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
    data_folder = "../../../data/split/KRAS_HRAS_30_AUGUST/test"

    # Intiate Keras Flow From Directory
    datagen = ImageDataGenerator()
    inference_flow = datagen.flow_from_directory(data_folder, color_mode="rgb",
                target_size=image_size, batch_size=batch_size, class_mode='categorical',
                shuffle=False)

    # Load Model
    model, loss, optimizer, metrics = model_def()
    if gpus > 1 : model = make_parallel_gpu(model, gpu)
    model.load_weights('weights/'+model_def.__name__+'.hdf5')
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    inter_model = Model(inputs=[model.get_layer(index=0)], outputs=[model.get_layer(index=-1)])

    # Predict Model
    X = inter_model.predict_generator(inference_flow, steps=22900//batch_size)
    X_embedded = TSNE(n_components=2).fit_transform(X)
    np.savetxt('logs/tsne.csv', X_embedded, delimiter=',')
