'''
models.py
Updated: 09/27/17

README:

This script contains an array of keras neural network definitions inspired by
varous architectures in the current literature.

    CIFAR 10 image recognition network.

The networks utilize convolutional layers to perform multi-class classification
betweeen the different images.

'''

# For Neural Network
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense 
from keras.layers import Conv3D, AveragePooling2D, Activation, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adamax, Adadelta, RMSprop
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm

################################################################################

def CIFAR_NET(nb_chans, nb_class):
    '''
    Trainable Parameters: 4,321,002 trainable (as reported by Keras)

    This network is inspired by the CIFAR10 network.

    '''
    x = Input(shape=(512, 512, nb_chans))
    l = Conv2D(32, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(x)
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
    l = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = Flatten()(l)
    l = Dense(2048, activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.5)(l)

    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def VOXNET_64(nb_chans, nb_class):
    '''
    A slight modification of the network proposed in the VoxNet paper (see:
    http://www.dimatura.net/publications/voxnet_maturana_scherer_iros15.pdf).
    The network has been extended to accomodate the 64x64x64 space that we are
    operating in vs. the 32x32x32 space of the original VoxNet. 
    '''
    x = Input(shape=(64, 64, 64, nb_chans))
    l = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), activation='relu', padding='valid')(x)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), activation='relu', padding='valid')(x)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='valid')(x)
    l = Dense(128, activation='relu')
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def SIMPLENET_MODIFIED3(nb_chans, nb_class):
    '''
    This is the most successful of the SimpleNet-inspired architectures that were
    developed for the ModelNet10, Kras / Hras, and PSIBLAST datasets. Detailed information
    regarding the development process for this network, as well as information
    regarding its performance on the datasets mentioned, is available in the
    network development and experiemnt log.

    This network differs from the original SIMPLENET_MODIFIED definition in
    that it makes use of the Adam optimizer (with tuned learning rate and decay parameters)
    and that it uses a 5x5 kernel in the first layer of the network. Other
    verisons of this network (e.g., #2, #4, and #5) made use of more elaborate
    kernel sizing strategies to little benefit. 
    '''
    # Input Layer
    x = Input(shape=(512, 512, nb_chans))

    # Layer 1
    l = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
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
    l = Dense(2048, activation='relu')(l)
    l = Dropout(.05)(l)
    
    # Output Layer
    y = Dense(nb_class, activation='softmax')(l)
    
    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def SIMPLENET_MODIFIED(nb_chans, nb_class):
    '''
    Trainable Parameters: 5.4 million (as reported by the cited paper).
        3,392,514 trainable and 2,944 non-trainable (as reported by keras).

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
    x = Input(shape=(512, 512, nb_chans))

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
    l = Dense(2048, activation='relu')(l)
    l = Dropout(.05)(l)

    # Output layer.
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = optimizer = Adadelta()
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def VGG_16_REFERENCE_3KERN(nb_chans, nb_class):
    '''
    Trainable Parameters: 287,352,642 (as reported by Keras on DGX-1).
                          138 Million (as reported by literature).

    Reference VGG 16 net translated to updated API w/ recommended hyperparameters.
    Only major difference from reference is # of dense layer neurons (2048 here
    vs. 4096 in original paper due to memory constraints).

    OG source available from:
        https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3#file-vgg-16_keras-py-L32
    '''
    x = Input(shape=(512, 512, nb_chans))

    # Convolution Layers
    l = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    l = Conv2D(64, (3, 3), padding='same', activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(256, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(256, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(256, (3, 3), padding='same', activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(512, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(512, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(512, (3, 3), padding='same', activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(512, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(512, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(512, (3, 3), padding='same', activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Flatten()(l)

    # Fully-connected layers. NOTE: This differs from OG VGG16 in that the
    # dense layers have only 2048 neurons vs. 4096 in reference. This was done
    # to meet memory constraints.
    l = Dense(2048, activation='relu')(l)
    l = Dropout(0.5)(l)

    l = Dense(2048, activation='relu')(l)
    l = Dropout(0.5)(l)

    # Output layer.
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = optimizer = Adadelta()
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def VGG_16_REFERENCE_5KERN(nb_chans, nb_class):
    '''
    Trainable Parameters: 287,352,642 (as reported by Keras on DGX-1).
                          138 Million (as reported by literature).

    Reference VGG 16 net translated to updated API w/ recommended hyperparameters.
    Only major difference from reference is # of dense layer neurons (2048 here
    vs. 4096 in original paper due to memory constraints), as well as the use
    of a 5x5 kernel vs. the original 3x3 (for comparison against relevant 3D
    convolutional network kernels).

    OG source available from:
        https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3#file-vgg-16_keras-py-L32
    '''
    x = Input(shape=(512, 512, nb_chans))

    # Convolution Layers
    l = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    l = Conv2D(64, (5, 5), padding='same', activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(128, (5, 5), padding='same', activation='relu')(l)
    l = Conv2D(128, (5, 5), padding='same', activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(256, (5, 5), padding='same', activation='relu')(l)
    l = Conv2D(256, (5, 5), padding='same', activation='relu')(l)
    l = Conv2D(256, (5, 5), padding='same', activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(512, (5, 5), padding='same', activation='relu')(l)
    l = Conv2D(512, (5, 5), padding='same', activation='relu')(l)
    l = Conv2D(512, (5, 5), padding='same', activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(512, (5, 5), padding='same', activation='relu')(l)
    l = Conv2D(512, (5, 5), padding='same', activation='relu')(l)
    l = Conv2D(512, (5, 5), padding='same', activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Flatten()(l)

    # Fully-connected layers. NOTE: This differs from OG VGG16 in that the
    # dense layers have only 2048 neurons vs. 4096 in reference. This was done
    # to meet memory constraints.
    l = Dense(2048, activation='relu')(l)
    l = Dropout(0.5)(l)

    l = Dense(2048, activation='relu')(l)
    l = Dropout(0.5)(l)

    # Output layer.
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = optimizer = SGD(lr=0.1, momentum=0.9, decay=0.000001, nesterov=True)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def VGG_16_MODIFIED_3KERNEL(nb_chans, nb_class):
    '''
    Trainable Parameters: 282,042,946 (as reported by Keras on DGX-1).

    This version of VGG16 is modified from the reference. This is the
    pared-down version that was used to obtain a non-scaled classification
    accuracy of 95.5% on the original, augments-separated PSIBLAST dataset.
    This differs from the reference VGG16 in that it contains dropouts in the
    convolutional blocks and it contains only 13 layers vs. VGG16's 16 layers.
    Furthermore, it uses the same hyperparameters (i.e., learning rate, decay,
    etc.) as the other network definitions in this file (models.py), whereas
    the VGG_16_REFERENCE() implementation above makes use of the original VGG16
    hyperparameters.
    '''
    x = Input(shape=(512, 512, 1))

    # Convolution Layers
    l = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(x)
    l = Dropout(0.2)(l)
    l = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Flatten()(l)

    # Fully-connected layers. NOTE: This differs from OG VGG16 in that the
    # dense layers have only 2048 neurons vs. 4096 in reference. This was done
    # to meet memory constraints.
    l = Dense(2048, activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.5)(l)

    l = Dense(2048, activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.5)(l)

    # Output layer.
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = optimizer = SGD(lr=0.00001, momentum=0.9, decay=0.00001/100, nesterov=False)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def VGG_16_MODIFIED_5KERNEL(nb_chans, nb_class):
    '''
    Trainable Parameters: 282,042,946 (as reported by Keras on DGX-1).

    This version of VGG16 is modified from the reference. This is a 5-kernel
    version of the net that was used to obtain a non-scaled classification
    accuracy of 95.5% on the original, augments-separated PSIBLAST dataset.
    This differs from the reference VGG16 in that it contains dropouts in the
    convolutional blocks and it contains only 13 layers vs. VGG16's 16 layers.
    Furthermore, it uses the same hyperparameters (i.e., learning rate, decay,
    etc.) as the other network definitions in this file (models.py), whereas
    the VGG_16_REFERENCE() implementations above makes use of the original VGG16
    hyperparameters. As noted, this also used a 5x5 kernel for comparison
    against relevant 3D convolution kernels.
    '''
    x = Input(shape=(512, 512, nb_chans))

    # Convolution Layers
    l = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(x)
    l = Dropout(0.2)(l)
    l = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(128, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(128, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(256, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(256, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(512, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(512, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(512, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(512, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Flatten()(l)

    # Fully-connected layers. NOTE: This differs from OG VGG16 in that the
    # dense layers have only 2048 neurons vs. 4096 in reference. This was done
    # to meet memory constraints.
    l = Dense(2048, activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.5)(l)

    l = Dense(2048, activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.5)(l)

    # Output layer.
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = optimizer = SGD(lr=0.00001, momentum=0.9, decay=0.00001/100, nesterov=False)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

if __name__ == '__main__':
    model, loss, optimizer, metrics = CIFAR_NET(3, 10)
    model.summary()
