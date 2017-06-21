'''
BinaryClassifier.py
Updated: 06/20/17

'''
import os, sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.measure import compare_ssim as ssim

# Neural Network
from keras.models import Model
from keras.layers import *
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.metrics import categorical_accuracy

# Global Variables

# Verbose Settings
debug = True

class ProteinNet:

    def __init__(self, shape=(64, 64, 3)):
        # Network Parameters
        self.shape = shape
        self.loss_fun = 'categorical_crossentropy'
        self.optimizer = 'sgd'

        # Input Layer
        x = Input(shape=self.shape)

        l = Conv2D(32, (5, 5), activation='relu')(x)
        l = MaxPooling2D((2, 2))(l)
        l = Conv2D(32, (5, 5), activation='relu')(l)
        l = MaxPooling2D((2, 2))(l)
        l = Conv2D(32, (5, 5), activation='relu')(l)
        l = MaxPooling2D((2, 2))(l)
        l = Flatten()(l)
        l = Dense(512, activation='relu')(l)
        l = Dropout(0.5)(l)

        # Output Layer
        y = Dense(2, activation='softmax')(l)

        # Compile Model
        self.model = Model(inputs=x, outputs=y)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun, metrics=[categorical_accuracy])
        self.model.summary()

if __name__ == '__main__':

    if debug: print "Generating Dataset..."

    x_train = []
    y_train = []

    # Read Ras PDB IMG File Names
    pdb_imgs = []
    for line in sorted(os.listdir('../data/ERS64/')):
        if line != '.gitignore': pdb_imgs.append(line)

    if debug: print "Loading RAS Encoded Images..."

    # Load Ras PDB Images
    for i in tqdm(range(len(pdb_imgs))):
        if pdb_imgs[i][5:7] == '0.':
            img = misc.imread('../data/ERS64/' + pdb_imgs[i])
            img = img.astype('float')
            img[:,:,0] = img[:,:,0]/255.0
            img[:,:,1] = img[:,:,1]/255.0
            img[:,:,2] = img[:,:,2]/255.0
            x_train.append(img)
            y_train.append([0, 1])

    '''
    for i in range(len(pdb_imgs)):
        for j in range(len(pdb_imgs)):
            print pdb_imgs[i], pdb_imgs[j], ssim(x_train[i], x_train[j], multichannel=True)
    '''

    # Read Ras PDB IMG File Names
    pdb_imgs = []
    for line in sorted(os.listdir('../data/EWS64/')):
        if line != '.gitignore': pdb_imgs.append(line)

    if debug: print "Loading WD40 Encoded Images..."

    # Load Ras PDB Images
    for i in tqdm(range(len(pdb_imgs))):
        if pdb_imgs[i][5:7] == '0.':
            img = misc.imread('../data/EWS64/' + pdb_imgs[i])
            img = img.astype('float')
            img[:,:,0] = img[:,:,0]/255.0
            img[:,:,1] = img[:,:,1]/255.0
            img[:,:,2] = img[:,:,2]/255.0
            x_train.append(img)
            y_train.append([1, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_data, x_test, y_data, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=45)

    if debug: print "Training Network..."

    # Fit Training Data to Model with 0.7/0.3 split between data set
    net = ProteinNet(shape=x_train.shape[1:])
    for i in range(100):
        net.model.fit(x_data[:1000], y_data[:1000], epochs=1, batch_size=25)
        print(net.model.evaluate(x_test[:1000], y_test[:1000], batch_size=25))
        print(net.model.predict(x_test[:25], batch_size=25))
        print(y_test[:25])
