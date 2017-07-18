'''
CIFAR_512.py
Updated: 7/12/17
[PASSING]

README:

This script contains a keras neural network definition inspired by CIFAR 10 image
recognition network. Network utilizes convolutional layers to do multi-class
classification betweeen the different images.

'''
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Neural Network
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm
from ParallelModels import make_parallel

################################################################################

class CIFAR_512:

    def __init__(self, nb_channels, nb_class=2, nb_gpu=1):
        '''
        '''
        # Network Parameters
        self.optimizer = SGD(lr=0.00001, momentum=0.9, decay=0.00001/100, nesterov=False)

        # Input Layer
        x = Input(shape=(512, 512, nb_channels))

        # Hidden Layers

        ## Convolution Layers
        l = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(x)
        l = Dropout(0.2)(l)
        l = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.2)(l)
        l = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
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

        ## Fully Connected Layers
        l = Dense(1024, activation='relu', kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.5)(l)

        # Output Layer
        y = Dense(nb_class, activation='softmax')(l)

        # Compile Model
        self.model = Model(inputs=x, outputs=y)
        self.model = make_parallel(self.model, nb_gpu)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=[categorical_accuracy])
        self.model.summary()

        # Save Model Diagram
        model_path = '../../models/CIFAR_512/'
        if not os.path.exists(model_path): os.makedirs(model_path)
        #plot_model(self.model, to_file=model_path+'model.png')

        # Save Model JSON
        with open(model_path+'model.json', 'w') as f:
            f.write(self.model.to_json())
