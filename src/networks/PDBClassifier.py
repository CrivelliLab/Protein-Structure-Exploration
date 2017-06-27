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
from keras.models import Model
from keras.layers import *
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split
from vis.visualization import visualize_cam, visualize_saliency

#- Global Variables
data_folders = ['RAS-SD512-HH', 'WD40-SD512-HH']
sample = 10000
seed = 1234
resize = (64, 64, 3)

# Verbose Settings
debug = True

################################################################################

class ProteinNet:

    def __init__(self, shape=(64, 64, 3), nb_class=2):
        '''
        '''
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
        y = Dense(nb_class, activation='softmax')(l)

        # Compile Model
        self.model = Model(inputs=x, outputs=y)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fun,
                            metrics=[categorical_accuracy])
        self.model.summary()


def load_pdb_train(encoded_folders, i, sample=None, resize=None, names=False):
    '''
    Method loads in training images from defined folder. Images values are normalized
    betweeen 0.0 and 1.0.

    '''

    x_train = []
    y_train = []

    # Read PDB IMG File Names
    pdb_imgs = []
    for line in sorted(os.listdir('../../data/final/' + encoded_folders[i] + '/')):
        if line.endswith('.png'): pdb_imgs.append(line)
    pdb_imgs = np.array(pdb_imgs)

    # Take Random Sample
    if sample:
        np.random.seed(seed)
        np.random.shuffle(pdb_imgs)
        pdb_imgs = pdb_imgs[:sample]

    if debug: print "Loading Encoded Images From", encoded_folders[i], '...'

    # Load PDB Images
    for j in tqdm(range(len(pdb_imgs))):
        img = misc.imread('../../data/final/' + encoded_folders[i] + '/' + pdb_imgs[j])
        img = img.astype('float')
        img[:,:,0] = img[:,:,0]/255.0
        img[:,:,1] = img[:,:,1]/255.0
        img[:,:,2] = img[:,:,2]/255.0
        if resize:
            img = misc.imresize(img, resize, interp='bicubic')
            #if j < 10:
                #plt.imshow(img)
                #plt.show()
        y_ = [0 for z in range(len(encoded_folders))]
        y_[i] = 1
        x_train.append(img)
        y_train.append(y_)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if names: return x_train, y_train, pdb_imgs
    else: return x_train, y_train

if __name__ == '__main__':

    if debug: print "Generating Dataset..."

    # Load Training Data
    x_train = None
    y_train = None
    for i in range(len(data_folders)):
        x_, y_ = load_pdb_train(data_folders, i, sample, resize)
        if x_train is None: x_train = x_
        else: x_train = np.concatenate([x_train, x_], axis=0)
        if y_train is None: y_train = y_
        else:y_train = np.concatenate([y_train, y_], axis=0)

    if debug: print "Splitting Training and Test Data..."
    # 0.7/0.3 Train/Test Data Split
    x_data, x_test, y_data, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=45)

    if debug: print "Training Network..."

    # Fit Training Data
    net = ProteinNet(shape=x_train.shape[1:])
    for i in range(10):
        print "Epoch", i
        net.model.fit(x_data, y_data, epochs=1, batch_size=25)
        print(net.model.evaluate(x_test, y_test, batch_size=25))

    # Generate Test Attention Maps
    x_atten, y_atten, pdb_files = load_pdb_train(data_folders, 0, 100, resize, True)
    for i in range(len(pdb_files)):
        p = net.model.predict(x_atten[i:i+1])
        atten_map = visualize_saliency(net.model, 10, [np.argmax(p[0])], x_atten[i], alpha=0.0)
        atten_map = np.dot(atten_map[...,:3], [0.299, 0.587, 0.114])
        atten_map = misc.imresize(atten_map, (512, 512), interp='nearest')
        misc.imsave('../../data/valid/attenmaps/'+pdb_files[i].split('.')[0]+'.png', atten_map)
