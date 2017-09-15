'''
MNIST_GPU.py
Updated: 08/28/17

README:

Script is used to benchmark GPU system on the MNIST dataset.

The network defined in this benchmark gets to 99.25% test accuracy after 12
epochs (there is still a lot of margin for parameter tuning). 16 seconds per
epoch on a GRID K520 GPU.

'''
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import merge, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Lambda

import numpy as np
import tensorflow as tf

nb_gpus = 4
epochs = 10
batch_size = 128

################################################################################

def make_parallel(model, gpu_count):
    '''
    Method distributes training batches to n-defined GPUs.

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
    Method defines Keras model.

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
    Method loads MNIST dataset.

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
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    x_train, x_test, y_train, y_test = load_mnist()

    model = define_model()

    if nb_gpus > 1: make_parallel(model, nb_gpus)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs, verbose=1,
                        validation_data=(x_test, y_test))
