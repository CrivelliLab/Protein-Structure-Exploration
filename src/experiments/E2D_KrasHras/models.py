'''
models.py
Updated: 8/29/17
[PASSING]

README:

This script contains a keras neural network definition inspired by CIFAR 10 image
recognition network. Network utilizes convolutional layers to do multi-class
classification betweeen the different images.

'''

# For Neural Network
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm

################################################################################

def CIFAR_512_3CHAN():
    '''
    Trainable Parameters: 4,922,114

    '''
    x = Input(shape=(512, 512, 3))
    l = Conv2D(64, (4, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3))(x)
    l = Dropout(0.2)(l)
    l = Conv2D(64, (4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = Conv2D(64, (4, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(64, (4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = Conv2D(64, (4, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(64, (4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = Conv2D(64, (4, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(64, (4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = Conv2D(64, (4, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(64, (4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = Conv2D(64, (4, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(64, (4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = Flatten()(l)
    l = Dense(1024, activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.5)(l)
    y = Dense(2, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = SGD(lr=0.00001, momentum=0.9, decay=0.00001/100, nesterov=False)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def CIFAR_512_3CHAN_98ACC():
    '''
    Trainable Parameters: 2,202,850

    '''
    x = Input(shape=(512, 512, 3))
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
    l = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = Flatten()(l)
    l = Dense(1024, activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.5)(l)
    y = Dense(2, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = SGD(lr=0.00001, momentum=0.9, decay=0.00001/100, nesterov=False)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

if __name__ == '__main__':
    model, loss, optimizer, metrics = CIFAR_512_3CHAN()
    model.summary()
