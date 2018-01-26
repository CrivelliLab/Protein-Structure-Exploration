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

def D1NET_v1(nb_chans, nb_class):
    '''
    Parameters: 242,114

    '''
    # Input Layer
    x = Input(shape=(262144, nb_chans))

    # Layers
    l = Conv1D(filters=32, kernel_size=64, strides=9, padding='valid', activation='relu')(x)
    l = MaxPooling1D(9)(l)
    l = Conv1D(filters=32, kernel_size=64, strides=9, padding='valid', activation='relu')(l)
    l = MaxPooling1D(9)(l)
    l = Flatten()(l)

    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    # Output Layer
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics


def D1NET_v2(nb_chans, nb_class):
    '''
    Parameters: 310,978

    '''
    # Input Layer
    x = Input(shape=(262144, nb_chans))

    # Layers
    l = Conv1D(filters=32, kernel_size=121, strides=9, padding='valid', activation='relu')(x)
    l = MaxPooling1D(9)(l)
    l = Conv1D(filters=32, kernel_size=121, strides=9, padding='valid', activation='relu')(l)
    l = MaxPooling1D(9)(l)
    l = Flatten()(l)

    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    # Output Layer
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D1NET_v3(nb_chans, nb_class):
    '''
    Parameters: 440,002

    '''
    # Input Layer
    x = Input(shape=(262144, nb_chans))

    # Layers
    l = Conv1D(filters=32, kernel_size=225, strides=9, padding='valid', activation='relu')(x)
    l = MaxPooling1D(9)(l)
    l = Conv1D(filters=32, kernel_size=225, strides=9, padding='valid', activation='relu')(l)
    l = MaxPooling1D(9)(l)
    l = Flatten()(l)

    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    # Output Layer
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D2NET_v1(nb_chans, nb_class):
    '''
    Parameters: 184,770

    '''
    x = Input(shape=(512,  512, nb_chans))
    l = Conv2D(32, (8, 8), strides = (3,3), padding='valid', activation='relu')(x)
    l = MaxPooling2D((3,3))(l)
    l = Conv2D(32, (8, 8), strides = (3,3), padding='valid', activation='relu')(l)
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

def D2NET_v2(nb_chans, nb_class):
    '''
    Parameters: 257,730

    '''
    x = Input(shape=(512,  512, nb_chans))
    l = Conv2D(32, (11, 11), strides = (3,3), padding='valid', activation='relu')(x)
    l = MaxPooling2D((3,3))(l)
    l = Conv2D(32, (11, 11), strides = (3,3), padding='valid', activation='relu')(l)
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

def D2NET_v3(nb_chans, nb_class):
    '''
    Parameters: 353,986

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

def D3NET_v1(nb_chans, nb_class):
    '''
    Parameters: 192,962

    '''
    x = Input(shape=(64, 64, 64, nb_chans))
    l = Conv3D(32, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='valid')(x)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(32, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='valid')(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D3NET_v2(nb_chans, nb_class):
    '''
    Parameters: 271,042

    '''
    x = Input(shape=(64, 64, 64, nb_chans))
    l = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), activation='relu', padding='valid')(x)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), activation='relu', padding='valid')(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def D3NET_v3(nb_chans, nb_class):
    '''
    Parameters: 309,698

    '''
    x = Input(shape=(64, 64, 64, nb_chans))
    l = Conv3D(32, (6, 6, 6), strides=(2, 2, 2), activation='relu', padding='valid')(x)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(32, (6, 6, 6), strides=(2, 2, 2), activation='relu', padding='valid')(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Flatten()(l)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    y = Dense(nb_class, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics
