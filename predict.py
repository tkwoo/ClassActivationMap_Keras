from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json
import keras.backend as K
import tensorflow as tf
import keras
import cv2
import numpy as np
import os
from glob import glob
import argparse
import pandas as pd

import models
from utils import centering, un_centering
from utils import get_classmap_keras, get_classmap_numpy
from utils import image_read

class predictor:
    def __init__(self, flag):
        self.flag = flag
    
    def cam(self):
        img_size = self.flag.image_size
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        
        #model = model_from_json(loaded_model_json)
        model = models.vgg_like(self.flag)

        weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'weight*')))
        model.load_weights(weight_list[-1])
        print '[*] model load : %s'%weight_list[-1]
        
        label_list = [os.path.basename(path) for path 
                    in sorted(glob(os.path.join(self.flag.data_path, '*')))]
        dict_name_list = dict()
        for name in label_list:
            dict_name_list[name] = [path for path 
                    in sorted(glob(os.path.join(self.flag.data_path, name, '*')))]
            dict_name_list[name] = dict_name_list[name][:50]
        
        data_list = []
        [data_list.append(image_read(name, color_mode=1, target_size=128)) 
                for name in dict_name_list['dog']]
        # print len(data_list)
        np_Inference_data_list = np.array(data_list)[:,0,:,:,:]
        np_original_data_list = np.array(data_list)[:,1,:,:,:].astype(np.uint8)
        # print np_Inference_data_list.shape

        result = model.predict(np_Inference_data_list, self.flag.batch_size)
        # print result.shape
        prediction_labels = np.argmax(result, axis=1)

        classmap = get_classmap_keras(self.flag, model, np_Inference_data_list, prediction_labels)

        assert classmap.shape[0] == np_original_data_list.shape[0] == prediction_labels.shape[0]
        
        cv2.namedWindow("show", 0)
        cv2.resizeWindow("show", 500, 500)
        
        for idx in range(classmap.shape[0]):
            predicted_label = prediction_labels[idx]
            print "[*] %d's label : %s"%(idx, label_list[predicted_label])
            img_original = np_original_data_list[idx,:,:,:]
            img_classmap = classmap[idx,:,:,predicted_label]
            color_classmap = cv2.applyColorMap(img_classmap, cv2.COLORMAP_JET)
            img_show = cv2.addWeighted(img_original, 0.8, color_classmap, 0.5, 0.)
            # cv2.imshow("show", img_show)
            cv2.imwrite("./result/dog_%d.png"%idx, img_show)
            # if cv2.waitKey(1) == 27:
            #     break
        print "[*] done"
   

    def evaluate(self):
        img_size = self.flag.image_size
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        t_start = cv2.getTickCount()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        # model = model_from_json(loaded_model_json)
        model = models.vgg_like(self.flag)        
        weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, "weight*")))
        model.load_weights(weight_list[-1])
        print "[*] model load : %s"%weight_list[-1]
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print "[*] model loading Time: %.3f ms"%t_total

        test_datagen = ImageDataGenerator(
            preprocessing_function=centering,
            rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
                os.path.join(self.flag.data_path, 'validation'),
                target_size=(img_size, img_size),
                batch_size=batch_size,
                shuffle=False,
                #color_mode='grayscale',
                class_mode='categorical')
        cv2.namedWindow('classmap', 0)
        cv2.resizeWindow('classmap', 500, 500)
        for x_batch, y_batch in test_generator:
            classmap = get_classmap2(self.flag, model, x_batch, int(np.argmax(y_batch[0])))
            
            classmap_vis = (classmap-classmap.min())/(np.max(classmap)-np.min(classmap))
            classmap_vis = (classmap_vis*255).astype(np.uint8)
            classmap_vis = np.expand_dims(classmap_vis, axis=2)
            classmap_color = cv2.applyColorMap(classmap_vis, cv2.COLORMAP_JET)
            
            b,g,r = cv2.split(x_batch[0])
            img_show = un_centering(cv2.merge((r,g,b))*255).astype(np.uint8)
            
            classmap_color = cv2.addWeighted(img_show, 0.8, classmap_color, 0.8, 0)
            
            cv2.imshow("classmap", classmap_color)
            cv2.imshow("show", img_show)
            if cv2.waitKey() == 27:
                break

        t_start = cv2.getTickCount()
        loss, acc = model.evaluate_generator(test_generator, test_generator.n // self.flag.batch_size)
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print '[*] test loss : %.4f'%loss
        print '[*] test acc  : %.4f'%acc
        print "[*] evaluation Time: %.3f ms"%t_total
    
    def inference(self):
        img_size = self.flag.image_size
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        t_start = cv2.getTickCount()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        # model = models.vgg_like(self.flag)        
        weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, "weight*")))
        model.load_weights(weight_list[-1])
        print "[*] model load : %s"%weight_list[-1]
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print "[*] model loading Time: %.3f ms"%t_total

        image_name_list = glob(os.path.join(self.flag.data_path, '*.jpg'))
        print len(image_name_list)
        # print image_name_list

        data_test = pd.read_csv('./submission.csv', index_col=0).fillna(0)
        # np_data_test = data_test.values
        
        for img_name in image_name_list:
            img = cv2.imread(img_name, 1)
            img = cv2.resize(img, (128,128))
            show = img
            # img = np.expand_dims(img, 2)
            img = np.expand_dims(img, 0).astype(np.float32)
            img -= 128
            img *= 2
            img /= 255.
            # print img.shape

            result = model.predict(img, 1)
            
            print img_name, 
            # print result
            
            predict_label = np.argmax(result)
            print predict_label
            # print predict_label.dtype
            # print predict_label.astype(np.str)
            # exit()

            data_test.loc[os.path.basename(img_name)] = predict_label.astype(np.str)

            # cv2.imshow("show", show)
            # if cv2.waitKey(0) == 27:
            #     break
        data_test.to_csv('./submission.csv')