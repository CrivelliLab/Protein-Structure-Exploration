'''
SIMPLENET_MODIFIED.py
Updated: 14 September 2017
[PASSING]

README:
    
    Trainable Parameters: 5.4 million (as reported by the cited paper).

    This network is inspired by the design principles underlying the SimpleNet 
    architecture, which is a 13-layer design intended to attain state-of-the-art 
    performance on the CIFAR10 dataset while holding 2 to 25 times fewer 
    parameters compared to previous deep architectures. The results on 
    major competitions have been promising. 

    However, this is not a 1-1 reproduction of the SimpleNet architecture. In
    particular, it follows the recent trend toward applying batch normalization
    after the activation function instead of before (as appeared to be the case
    in the SimpleNet design). There are also changes to the output layers (e.g.
    dense layer addition, number of neurons, etc.).

    Additionally, this network also doesn't use the 1x1 kernels suggested for
    layers 11 and 12 in the original paper, nor does it make use of dropout on
    the hidden layers. 

    See this paper for a detailed explanation of the design of this network:
        https://arxiv.org/pdf/1608.06037.pdf
'''
# *****************************************************************************
# IMPORTS
# *****************************************************************************
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense 
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm
import os
from ParallelModels import make_parallel

# *****************************************************************************
# NETWORK DEFINITION
# *****************************************************************************
class SIMPLENET_MODIFIED:

    def __init__(self, nb_channels, nb_class=2, nb_gpu=1):
        
        # Parameters
        self.optimizer = SGD(lr=0.00001, momentum=0.9, decay=.00001/100, nesterov=False)

        # Input Layer
        x = Input(shape=(512, 512, nb_channels))
        
        # Layer 1
        l = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        l = BatchNormalization()(l)

        # Layer 2
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = BatchNormalization()(l)
        
        # Layer 3
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = BatchNormalization()(l)
        
        # Layer 4
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = BatchNormalization()(l)
        
        # Layer 5
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = BatchNormalization()(l)
        
        # Layer 6
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = BatchNormalization()(l)
        
        # Layer 7
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = BatchNormalization()(l)

        # Layer 8
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = BatchNormalization()(l)

        # Layer 9
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = BatchNormalization()(l)

        # Layer 10
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = BatchNormalization()(l)

        # Layer 11 
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = BatchNormalization()(l)

        # Layer 12 
        l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = BatchNormalization()(l)

        l = Flatten()(l)                                                                                                   

        # Layer 13
        l = Dense(1024, activation='relu')(l)
        l = Dropout(.05)(l)
        
        # Output Layer
        y = Dense(nb_class, activation='softmax')(l)

        # Creating Model
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
