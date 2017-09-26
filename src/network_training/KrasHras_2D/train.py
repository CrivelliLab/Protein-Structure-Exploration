'''
train.py
Updated: 9/25/17

README:

This script is used to train neural network models on the 2D encoded KRAS/HRAS
dataset.

'''
import os, time, sys; sys.path.insert(0, '../')
import numpy as np
from models import *
from keras_extra import make_parallel_gpu
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

epochs = 100
batch_size = 10
model_def = CIFAR_NET
gpus = 1
seed = 125

################################################################################

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_folder = "../../../data/split/KRAS_HRAS_seed45"
    train_count = 0
    for root, dirs, files in os.walk(data_folder+'/train'):
        for file_ in files:
            if file_.endswith(".png"): train_count += 1
    val_count = 0
    for root, dirs, files in os.walk(data_folder+'/test'):
        for file_ in files:
            if file_.endswith(".png"): val_count += 1
    if not os.path.exists('logs'): os.makedirs('logs')
    if not os.path.exists('weights'): os.makedirs('weights')

    # Intiate Keras Flow From Directory
    datagen = ImageDataGenerator(preprocessing_function=standard)
    train_flow = datagen.flow_from_directory(data_folder +'/train', color_mode="rgb",
                target_size=(512, 512), batch_size=batch_size, class_mode='categorical',
                seed=seed)
    validation_flow = datagen.flow_from_directory(data_folder +'/validation', color_mode="rgb",
                target_size=(512, 512), batch_size=batch_size, class_mode='categorical',
                seed=seed)

    # Load Model
    model, loss, optimizer, metrics = model_def(3, 2)
    if gpus > 1 : model = make_parallel_gpu(model, gpu)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Callbacks
    date = time.strftime("%d%m%Y")
    earlystopper = EarlyStopping(patience=5)
    csv_logger = CSVLogger('logs/'+model_def.__name__+date+".csv", separator=',')
    checkpointer = ModelCheckpoint(filepath='weights/'+model_def.__name__+date+'.hdf5',
                                   verbose=0, save_best_only=True)

    # Train Model
    model.fit_generator(train_flow, epochs=epochs, steps_per_epoch=train_count//batch_size,
                        validation_data=validation_flow, callbacks=[csv_logger,checkpointer,earlystopper],
                        validation_steps=val_count//batch_size)
