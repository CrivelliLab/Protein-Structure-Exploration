'''
train.py
Updated: 9/12/17

README:

'''
import os, time, sys; sys.path.insert(0, '../../misc')
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint
from BinvoxDataGenerator import ImageDataGenerator
from keras_extra import make_parallel_gpu
from models import *

epochs = 100
batch_size = 10
seed = 125
gpus = 1

model_def = CONV3D_64_3CHAN

################################################################################

image_size = (64,64,64)

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_folder = "../../../data/split/PSIBLAST_BINVOX"
    if not os.path.exists('logs'): os.makedirs('logs')
    if not os.path.exists('weights'): os.makedirs('weights')

    # Intiate Keras Flow From Directory
    datagen = ImageDataGenerator()
    train_flow = datagen.flow_from_directory(data_folder +'/train', color_mode="rgb",
                target_size=image_size, batch_size=batch_size, class_mode='categorical',
                seed=seed)
    validation_flow = datagen.flow_from_directory(data_folder +'/validation', color_mode="rgb",
                target_size=image_size, batch_size=batch_size, class_mode='categorical',
                seed=seed)

    # Load Model
    model, loss, optimizer, metrics = model_def(3, 2)
    if gpus > 1: model = make_parallel_gpu(model, gpu)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Callbacks
    date = time.strftime("%d%m%Y")
    csv_logger = CSVLogger('logs/'+model_def.__name__+date+".csv", separator=',')
    checkpointer = ModelCheckpoint(filepath='weights/'+model_def.__name__+date+'.hdf5', verbose=0, save_best_only=True)

    # Train Model
    model.fit_generator(train_flow, epochs=epochs, steps_per_epoch=92000//batch_size,
                        validation_data=validation_flow, callbacks=[csv_logger, checkpointer],
                        validation_steps=22900//batch_size)
