'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
import keras
from keras.objectives import categorical_crossentropy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow_on_slurm import setup_slurm_cluster
import pickle, os, sys, time
import tensorflow as tf
import numpy as np

# Dataset
batch_size = 128
num_classes = 10
epochs = 12

################################################################################

cluster, server, task_index, num_tasks, job_name = setup_slurm_cluster(num_ps=1)

if job_name == 'ps':
    server.join()
    sys.exit(0)

elif job_name == "worker":
    is_chief=(task_index == 0)

    # Load Data
    data, labels = [], []
    img_rows, img_cols = 28, 28
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

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Replica Device Setter Assigns ops to the local worker by default and stores all variables in the parameter server (ps).
    device = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,cluster=cluster)
    with tf.device(device):

        K.set_learning_phase(1)
        img = tf.placeholder(tf.float32, shape=([None,] + list(input_shape)))
        y = tf.placeholder(tf.float32, shape=[None,num_classes])

        # Model
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(img)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        preds = Dense(num_classes, activation='softmax')(x)

        loss = tf.reduce_mean(categorical_crossentropy(y, preds))
        opt = tf.train.AdamOptimizer(1e-3)
        global_step = tf.contrib.framework.get_or_create_global_step()
        opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_tasks, total_num_replicas=num_tasks)
        train_op = opt.minimize(loss, global_step=global_step)

    #a hook that will stop training at
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]
    hooks.append(opt.make_session_run_hook(is_chief=is_chief))

    with tf.train.MonitoredTrainingSession(is_chief=is_chief, master=server.target, hooks=hooks) as sess:
        step = 0
        for i in range(epochs):
            x_train_split = np.split(x_train[:59904], 59904//batch_size)
            y_train_split = np.split(y_train[:59904], 59904//batch_size)
            for j in range(len(x_train_split)):
                if (j+i) % num_tasks != task_index: continue
                data_batch = x_train_split[j]
                label_batch = y_train_split[j]
                _, loss_ = sess.run([train_op, loss], feed_dict={img: data_batch, y: label_batch})
                step += 1
                print(step, task_index, loss_)
                sys.stdout.flush()
