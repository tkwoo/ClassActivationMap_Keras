from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras
import cv2
import numpy as np
import os
from glob import glob
import argparse
import random
import math

import models

def centering(np_image):
        return 2*(np_image - 128)

class Trainer:
    def __init__(self, flag):
        self.flag = flag

    def lr_step_decay(self, epoch):
        init_lr = self.flag.initial_learning_rate
        lr_decay = self.flag.learning_rate_decay_factor
        epoch_per_decay = self.flag.epoch_per_decay
        lrate = init_lr * math.pow(lr_decay, math.floor((1+epoch)/epoch_per_decay))
        # print lrate
        return lrate

    def train(self):
        img_size = self.flag.image_size
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        train_datagen = ImageDataGenerator(
            preprocessing_function=centering,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1)

        test_datagen = ImageDataGenerator(
            preprocessing_function=centering,
            rescale=1./255
            )

        train_generator = train_datagen.flow_from_directory(
                os.path.join(self.flag.data_path, 'train'),
                target_size=(img_size, img_size),
                batch_size=batch_size,
                # color_mode='grayscale',
                class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
                os.path.join(self.flag.data_path, 'validation'),
                target_size=(img_size, img_size),
                batch_size=batch_size,
                # color_mode='grayscale',
                class_mode='categorical')

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        model = models.vgg_like(self.flag)

        if self.flag.pretrained_weight_path != None:
            model.load_weights(self.flag.pretrained_weight_path)
            print "[*] loaded pretrained model: %s"%self.flag.pretrained_weight_path
        if not os.path.exists(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name)):
            os.mkdir(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name))
        model_json = model.to_json()
        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'w') as json_file:
            json_file.write(model_json)
        
        model_checkpoint = ModelCheckpoint(
                    os.path.join(
                                self.flag.ckpt_dir, 
                                self.flag.ckpt_name,
                                'weights.{epoch:02d}.h5'), 
                    save_best_only=False,
                    verbose=1,
                    monitor='val_loss',
                    period= 2, #self.flag.total_epoch // 10 + 1, 
                    save_weights_only=True)
        learning_rate = LearningRateScheduler(self.lr_step_decay)

        model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.n // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=train_generator.n // batch_size,
            callbacks=[model_checkpoint, learning_rate]
        )