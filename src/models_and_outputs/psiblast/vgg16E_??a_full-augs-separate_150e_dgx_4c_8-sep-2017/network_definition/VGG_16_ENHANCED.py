'''
VGG_16_ENHANCED.py
Updated: 8 September 2017
[PASSING]

README:

    This network is inspired by the VGG team's ILSVRC14-winning
    submission. See: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

    The network has been updated in a number of ways to take advantage of
    recent advances in deep learning research (e.g some versions have SELU
    activations), as well as the powerful hardware that we are training on (the 
    Nvidia DGX-1 with Tesla P100s is the target platform for this architecture). 
    However, it is largely the same. 

'''
# *****************************************************************************
# IMPORTS
# *****************************************************************************
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense 
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm
import os
from ParallelModels import make_parallel

# *****************************************************************************
# NETWORK DEFINITION
# *****************************************************************************
class VGG_16_ENHANCED:

    def __init__(self, nb_channels, nb_class=2, nb_gpu=1):
        
        # Parameters
        self.optimizer = SGD(lr=0.00001, momentum=0.9, decay=0.00001/100, nesterov=False)

        # Input Layer
        x = Input(shape=(512, 512, nb_channels))

        # Convolution Layers
        l = Conv2D(64, (3, 3), padding='same', activation='relu', 
                kernel_constraint=maxnorm(3))(x)
        l = Dropout(0.2)(l)
        l = Conv2D(64, (3, 3), padding='same', activation='relu',
                kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        
        l = Conv2D(128, (3, 3), padding='same', activation='relu', 
                kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.2)(l)
        l = Conv2D(128, (3, 3), padding='same', activation='relu',
                kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)

        l = Conv2D(256, (3, 3), padding='same', activation='relu', 
                kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.2)(l)
        l = Conv2D(256, (3, 3), padding='same', activation='relu',
                kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)

        l = Conv2D(512, (3, 3), padding='same', activation='relu', 
                kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.2)(l)
        l = Conv2D(512, (3, 3), padding='same', activation='relu',
                kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)

        l = Conv2D(512, (3, 3), padding='same', activation='relu', 
                kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.2)(l)
        l = Conv2D(512, (3, 3), padding='same', activation='relu',
                kernel_constraint=maxnorm(3))(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)

        l = Flatten()(l)

        ## Fully Connected Layers
        l = Dense(2048, activation='relu', kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.5)(l)

        l = Dense(2048, activation='relu', kernel_constraint=maxnorm(3))(l)
        l = Dropout(0.5)(l)
        
        # Output Layer
        y = Dense(nb_class, activation='softmax')(l)

        self.model = Model(inputs=x, outputs=y)

        # Save Model Diagram
        model_path = '../outputs/'
        if not os.path.exists(model_path): os.makedirs(model_path)
        plot_model(self.model, to_file=model_path+'model.png', 
                show_shapes=True, show_layer_names=False)

        # Save Model JSON
        with open(model_path+'model.json', 'w') as f:
            f.write(self.model.to_json())

        # Compile Model
        if nb_gpu > 1: self.model = make_parallel(self.model, nb_gpu)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', 
                metrics=[categorical_accuracy])
        self.model.summary()
