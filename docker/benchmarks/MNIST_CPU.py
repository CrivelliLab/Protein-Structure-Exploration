'''
MNIST_CPU.py
Updated: 09/25/17

README:

This script is used to benchmark CPU system on the MNIST dataset using the Keras
nueral network library.

The network defined in this benchmark gets to 99.25% test accuracy after 12
epochs (204 seconds per epoch). The network utilizes convolutional
layers to preform multi-class classification between the different handwritten
character images.

'''
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

epochs = 10
batch_size = 128

################################################################################

def define_model():
    '''
    Method defines Keras model. Network is made up of 2 convolutional layers,
    one with 32 feature maps and kernel size of 3, followed by another with 64
    feature maps and kernel size of 3. Both layers use the RELU activation
    function. A max pooling of 2 is applied to the convolutions with a dropout
    of 0.25 followed by a fully-connected layer of 128 nuerons with 0.5 dropout.
    The output layer consist of a fully-connected layer containing 10 output
    neurons (for each MNIST character class) with softmax activation.

    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

def load_mnist():
    '''
    Method loads MNIST dataset. The images are 28x28 pixels with one channel.
    The pixel values are normailized between 0.0 and 1.0.

    '''
    # Input image dimensions
    img_rows, img_cols = 28, 28

    # The data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Data reshaping
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # Normalize data
    x_train = x_train.astype('float32')/ 255.0
    x_test = x_test.astype('float32')/ 255.0

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    # Load training and test images
    x_train, x_test, y_train, y_test = load_mnist()

    # Define and compile Keras model
    model = define_model()
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(),
                  metrics=['accuracy'])

    # Train Keras model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_test, y_test))
