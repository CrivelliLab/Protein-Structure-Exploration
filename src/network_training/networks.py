'''
networks.py
Updated: 12/29/17

README:

'''

# For Neural Network
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Add, Dense, Input, Activation
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense, Concatenate
from keras.layers import Conv3D, AveragePooling2D, Activation, MaxPooling3D
from keras.optimizers import SGD, Adam, Adamax, Adadelta, RMSprop
from keras.constraints import maxnorm

################################################################################

def D1NET(nb_chans, nb_class):
    '''
    '''
    # Input Layer
    x = Input(shape=(None, nb_chans))

    # Layers
    l = Conv1D(filters=32, kernel_size=169, strides=81, padding='valid', activation='relu')(x)
    l = Conv1D(filters=32, kernel_size=169, strides=81, padding='valid', activation='relu')(l)
    l = GlobalMaxPooling1D()(l)

    l = Dense(128, activation='relu')(l)

    # Output Layer
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D2NET(nb_chans, nb_class):
    '''
    '''
    x = Input(shape=(512,  512, nb_chans))
    l = Conv2D(32, (15, 15), strides = (3,3), padding='valid', activation='relu')(x)
    l = MaxPooling2D((3,3))(l)
    l = Conv2D(32, (15, 15), strides = (3,3), padding='valid', activation='relu')(l)
    l = MaxPooling2D((3,3))(l)

    # Fully Connected Layer
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)

    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D3NET(nb_chans, nb_class):
    '''
    A slight modification of the network proposed in the VoxNet paper (see:
    http://www.dimatura.net/publications/voxnet_maturana_scherer_iros15.pdf).
    The network has been extended to accomodate the 64x64x64 space that we are
    operating in vs. the 32x32x32 space of the original VoxNet.
    '''
    x = Input(shape=(64, 64, 64, nb_chans))
    l = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), activation='relu', padding='valid')(x)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), activation='relu', padding='valid')(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics
