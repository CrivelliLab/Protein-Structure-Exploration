'''
MNIST_GPU.py
Updated: 09/25/17

README:

This script is used to benchmark GPU system on the MNIST dataset using the Keras
nueral network library.

The network defined in this benchmark gets to 99.25% test accuracy after 12
epochs (3 seconds per epoch on Tesla P100). The network utilizes convolutional
layers to preform multi-class classification between the different handwritten
character images.

NOTE: To benchmark multi-GPUs, available GPUs must be defined in
CUDA_VISIBLE_DEVICES environment variable.

'''
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.layers.core import Lambda
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.losses import categorical_crossentropy
from keras.layers import merge, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

nb_gpus = 4
epochs = 12
batch_size = 128

################################################################################

def make_parallel(model, gpu_count):
    '''
    Method distributes equal-length training batches to n-defined GPUs.

    '''
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)): outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:
                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape,
                            arguments={'idx':i, 'parts':gpu_count})(x)
                    inputs.append(slice_n)
                outputs = model(inputs)

                if not isinstance(outputs, list): outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)): outputs_all[l].append(outputs[l])

    # Merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)

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
    if nb_gpus > 1: make_parallel(model, nb_gpus)
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(),
                  metrics=['accuracy'])

    # Train Keras model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_test, y_test))
