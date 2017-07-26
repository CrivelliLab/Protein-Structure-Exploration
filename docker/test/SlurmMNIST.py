from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pickle
import os
import tensorflow as tf
import numpy as np
import sys
import time
from tensorflow_on_slurm import tf_config_from_slurm

cluster, my_job_name, my_task_index = tf_config_from_slurm(ps_number=1)
cluster_spec = tf.train.ClusterSpec(cluster)
server = tf.train.Server(server_or_cluster_def=cluster_spec,
                         job_name=my_job_name,
                         task_index=my_task_index)

if my_job_name == 'ps':
    server.join()
    sys.exit(0)

data, labels = [], []
is_chief = my_task_index == 0

# Dataset
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

with tf.device('/job:worker/task:{}'.format(my_task_index)):
    y = tf.placeholder(tf.uint8, shape=[None, num_classes], name='y')
    K.set_learning_phase(1)

    x = Input(shape=input_shape))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
    loss = tf.reduce_mean(cross_entropy)
    opt = tf.train.AdamOptimizer(1e-3)
    opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=len(cluster['worker']),
                                total_num_replicas=len(cluster['worker']))
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_step = opt.minimize(loss, global_step=global_step)
    sync_replicas_hook = opt.make_session_run_hook(is_chief)

sess = tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief,
                                         hooks=[sync_replicas_hook])

def batch_generator(data, labels, batch_size=32):
    x_batch, y_batch = [], []
    for d, l in zip(data, labels):
        x_batch.append(d)
        y_batch.append(l)
        if len(x_batch) == batch_size:
            yield np.vstack(x_batch),np.vstack(y_batch)
            x_batch = []
            y_batch = []

step = 0

for i in range(epochs):
    bg = batch_generator(x_train, y_train, batch_size)
    for j, (data_batch, label_batch) in enumerate(bg):
        if (j+i) % len(cluster['worker']) != my_task_index:
            continue
        _, loss_, acc = sess.run([train_step, loss, accuracy],
                                feed_dict={x: data_batch,
                                          y: label_batch.reshape(-1,1),
                                          keep_prob: 0.5})
        step += 1
        print(step, my_task_index, loss_, acc)
        sys.stdout.flush()
