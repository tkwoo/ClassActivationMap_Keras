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

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def centering(np_image):
    return 2*(np_image - 128)
def un_centering(np_image):
    return (np_image/2) + 128
class predictor:
    def __init__(self, flag):
        self.flag = flag
    
    def get_classmap(self, model, img, label):
        inc = model.layers[0].output
        conv6 = model.layers[-3].output
        channel = int(conv6.shape[3])
        # print conv6 #last relu ?, 16,16,1024
        conv6_resized = tf.image.resize_bilinear(conv6, [int(inc.shape[1]), int(inc.shape[2])])
        # print conv6_resized # ?,128,128,1024

        weight = model.layers[-1].kernel
        # print weight
        # print weight # 1024, 2
        weight = weight[:,label]
        weight = K.reshape(weight, [-1, channel, 1])
        # print weight # 1, 1024, 2
        conv6_resized = K.reshape(conv6_resized, [-1, int(inc.shape[1]) * int(inc.shape[2]),  channel])
        # print conv6_resized # ?, 16384, 1024
        classmap = K.dot(conv6_resized, weight)
        classmap = K.reshape(classmap, [-1, int(inc.shape[1]), int(inc.shape[2])])
        get_cmap = K.function([inc, keras.backend.learning_phase()], [classmap])

        return get_cmap([img, 0 if self.flag.mode!='train' else 1])
    
    def get_classmap2(self, model, img, label):
        _, width, height, _ = img.shape
        # # Reshape to the network input shape (3, w, h)
        # img = np.array([np.transpose(np.float32(img), (2,0,1))])
        # Get the 1024 input weights to softmax
        class_weights = model.layers[-1].get_weights()[0]
        # print class_weights.shape
        # exit()
        final_conv_layer = model.layers[-3] #get_output_layer(model, 'conv6')
        get_output = K.function([model.layers[0].input, keras.backend.learning_phase()], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, predictions] = get_output([img, 0 if self.flag.mode!='train' else 1])
        # print conv_outputs.shape
        conv_outputs = conv_outputs[0,:,:,:]

        #create the class activation map
        cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
        # print cam.shape
        for i,w in enumerate(class_weights[:,label]):
            # print conv_outputs[:,:,i].shape
            cam += w*conv_outputs[:,:,i]
        # print "predictions", predictions
        
        cam = cv2.resize(cam, (height, width))
        debug = cam/np.max(cam)
        # cv2.namedWindow('debug', 0)
        # cv2.imshow('debug', debug)
        return cam


    def predict(self, image_path=None):
        t_start = cv2.getTickCount()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, "weight*")))
        model.load_weights(weight_list[-1])
        print "[*] model load : %s"%weight_list[-1]
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print "[*] model loading Time: %.3f ms"%t_total

        # image path ex) './dataset/sagital/odd/ori/AD_154_103.png'
        if image_path == None:
            imgInput = cv2.imread(self.flag.test_image_path, 0)
        else:
            imgInput = cv2.imread(image_path, 0)
        input_data = imgInput.reshape((1,self.flag.image_size,self.flag.image_size,1))
        input_data /= 1./255

        t_start = cv2.getTickCount()
        result = model.predict(input_data, 1)
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
        print "Predict Time: %.3f ms"%t_total

        print result

        window_name = "show"
        cv2.namedWindow(window_name, 0)
        cv2.resizeWindow(window_name, 800, 800)
        cv2.imshow(window_name, imgInput)
    
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
            # print x_batch.shape # 48, 128, 128, 3
            # print y_batch.shape # 48, 2
            # [classmap] = self.get_classmap(model, x_batch, int(np.argmax(y_batch[0])))
            # classmap = classmap[0]

            # img = centering(x_batch[0].astype(np.float32))/255
            classmap = self.get_classmap2(model, x_batch, int(np.argmax(y_batch[0])))
            # classmap = classmap[0]
            # print classmap.shape

            # cv2.normalize(classmap[0], classmap[0], 0, 1, cv2.NORM_MINMAX)
            # classmap_vis = np.array(map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap[0]))
            # classmap_vis = np.array(map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap))
            classmap_vis = (classmap-classmap.min())/(np.max(classmap)-np.min(classmap))
            # classmap_vis[np.where(classmap_vis<0.)]=0
            classmap_vis = (classmap_vis*255).astype(np.uint8)
            classmap_vis = np.expand_dims(classmap_vis, axis=2)
            # print classmap_vis.shape
            # classmap_vis = classmap_vis.astype(np.uint8)
            classmap_color = cv2.applyColorMap(classmap_vis, cv2.COLORMAP_JET)
            # classmap_color[np.where(classmap_vis<int(255*0.2))]=0
            
            # print classmap_color.shape

            b,g,r = cv2.split(x_batch[0])
            # print np.min(x_batch[0])
            img_show = un_centering(cv2.merge((r,g,b))*255).astype(np.uint8)
            # print img_show.dtype
            # print classmap_color.dtype

            classmap_color = cv2.addWeighted(img_show, 0.8, classmap_color, 0.8, 0)
            # print classmap[0].shape
            cv2.imshow("classmap", classmap_color)
            cv2.imshow("show", img_show)
            if cv2.waitKey() == 27:
                break
            # break
        exit()

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