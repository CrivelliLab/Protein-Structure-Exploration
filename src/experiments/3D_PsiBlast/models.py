'''
models.py
Updated: 8/29/17
[NOT PASSING]

README:

'''

# For Neural Network
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm

################################################################################

def CONV3D_64_3CHAN():
    '''
    Trainable Parameters: 6,045,186

    '''
    x = Input(shape=(64, 64, 64, 3))
    l = Conv3D(64, (4, 4, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3))(x)
    l = Dropout(0.2)(l)
    l = Conv3D(64, (4, 4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(64, (4, 4, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv3D(64, (4, 4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(64, (4, 4, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv3D(64, (4, 4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Conv3D(64, (4, 4, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.2)(l)
    l = Conv3D(64, (4, 4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3))(l)
    l = MaxPooling3D(pool_size=(2, 2, 2))(l)
    l = Flatten()(l)
    l = Dense(1024, activation='relu', kernel_constraint=maxnorm(3))(l)
    l = Dropout(0.5)(l)
    y = Dense(2, activation='softmax')(l)

    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = optimizer = SGD(lr=0.00001, momentum=0.9, decay=0.00001/100, nesterov=False)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

if __name__ == '__main__':
    model, loss, optimizer, metrics = CONV3D_64_3CHAN()
    model.summary()
