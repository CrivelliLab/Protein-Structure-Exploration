'''
PDBClassifier.py
Updated: 06/20/17

README:

The following script is used to test preliminary training of 2D PDB encodings.

Global variables used during training are defined under #- Global Variables.
data_folders defines the list of folder containing encoded PDBs. Folders must
be under data/final/.

Network utilizes convolutional layers to do multi-class classification betweeen
the different encoded images defined in data_folders.

There is a 0.7/0.3 split of the data to generate training and testing data.

'''
import os, sys
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from tqdm import tqdm

# Neural Network
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.metrics import categorical_accuracy
from vis.visualization import visualize_cam, visualize_saliency
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator

#- Global Variables
data_folder = 'RAS-WD40-MD512-HH'
resize = (512, 512)
seed = 125

# Verbose Settings
debug = True

################################################################################

class ProteinNet:

    def __init__(self, shape=(64, 64, 1), nb_class=2):
        '''
        '''
        # Network Parameters
        self.shape = shape
        self.loss_fun = 'categorical_crossentropy'
        self.optimizer = SGD(lr=0.0001, momentum=0.9, decay=0.0001/100, nesterov=False)

        # Input Layer
        x = Input(shape=self.shape)

        l = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(x)
        l = Dropout(0.2)(l)
        l = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.2)(l)
        l = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.2)(l)
        l = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.2)(l)
        l = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.2)(l)
        l = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = Flatten()(l)
        l = Dense(512, activation='relu', kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.5)(l)
        '''
        l = Conv2D(64, (5, 5))(x)
        l = MaxPooling2D((2, 2))(l)
        l = LeakyReLU(0.2)(l)
        l = Conv2D(64, (5, 5))(l)
        l = MaxPooling2D((2, 2))(l)
        l = LeakyReLU(0.2)(l)
        l = Conv2D(64, (5, 5))(l)
        l = MaxPooling2D((2, 2))(l)
        l = LeakyReLU(0.2)(l)
        l = Flatten()(l)
        l = Dense(512, activation='relu')(l)
        l = Dropout(0.5)(l)
        '''
        # Output Layer
        y = Dense(nb_class, activation='softmax')(l)

        # Compile Model
        self.model = Model(inputs=x, outputs=y)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun,
                            metrics=[categorical_accuracy])
        self.model.summary()

if __name__ == '__main__':

    if debug: print "Training Network..."

    datagen = ImageDataGenerator()
    train_flow = datagen.flow_from_directory("../../data/final/"+ data_folder +'/train',
                    batch_size=8, class_mode='categorical',
                    seed=seed)
    test_flow = datagen.flow_from_directory("../../data/final/"+ data_folder +'/test',
                    batch_size=8, class_mode='categorical',
                    seed=seed)

    # Fit Training Data
    net = ProteinNet(shape=[resize[0], resize[1], 3])
    net.model.fit_generator(train_flow, epochs=100, steps_per_epoch=27597, validation_data=test_flow,
        validation_steps=7393)
    #print(net.model.evaluate(x_test, y_test, batch_size=5))
