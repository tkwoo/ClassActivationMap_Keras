import tensorflow as tf
import keras.backend as K
import keras
import cv2
import numpy as np
import os, errno


def image_read(path, color_mode=1, target_size=128):
    img = cv2.imread(path, color_mode)
    img = cv2.resize(img, (target_size, target_size))
    show = img
    img = img.astype(np.float32)
    img = centering(img)/255
    return [img, show]

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: #Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else : raise

def Rot90cw(img):
    height, width = img.shape[:2]
    matRot = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
    imgRotate = cv2.warpAffine(img, matRot, (width,height))
    return imgRotate

def Rot90ccw(img):
    height, width = img.shape[:2]
    matRot = cv2.getRotationMatrix2D((width/2, height/2), -90, 1)
    imgRotate = cv2.warpAffine(img, matRot, (width,height))
    return imgRotate

def centering(np_image):
    return 2*(np_image - 128)

def un_centering(np_image):
    return (np_image/2) + 128

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def get_classmap_keras(flag, model, img, labels):
    inc = model.layers[0].output
    conv6 = model.get_layer('last_logit').output # last relu ?,16,16,1024
    channel = int(conv6.shape[3])
    conv6_resized = tf.image.resize_bilinear(conv6, [int(inc.shape[1]), int(inc.shape[2])]) # ?,128,128,1024
    weight = model.layers[-1].kernel # 1024, 2
    weight = K.reshape(weight, [-1, channel, 2]) # 1, 1024, 2
    conv6_resized = K.reshape(conv6_resized, 
                    [-1, int(inc.shape[1])*int(inc.shape[2]),  channel])
                    # ?, 16384, 1024
    classmap = K.dot(conv6_resized, weight) # ?, 16384, 2
    classmap = K.reshape(classmap, [-1, int(inc.shape[1]), int(inc.shape[2]), 2])
    get_cmap = K.function([inc, keras.backend.learning_phase()], [classmap])
    [np_classmap] = get_cmap([img, 0 if flag.mode!='train' else 1])
    
    np_classmap[np.where(np_classmap < 0.0)] = 0
    np_classmap = np.array(map(lambda x: 
                ((x-x.min())/(x.max()-x.min())), 
                np_classmap))
    # np_classmap /= np.max(np_classmap)
    return (np_classmap*255).astype(np.uint8)
    
def get_classmap_numpy(flag, model, img, label):
    _, width, height, _ = img.shape
    # # Reshape to the network input shape (3, w, h)
    # img = np.array([np.transpose(np.float32(img), (2,0,1))])
    # Get the 1024 input weights to softmax
    class_weights = model.layers[-1].get_weights()[0]
    # print class_weights.shape
    # exit()
    #final_conv_layer = get_output_layer(model, 'last_logit') #model.layers[-3]
    final_conv_layer = model.get_layer('last_logit')
    get_output = K.function([model.layers[0].input, keras.backend.learning_phase()], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img, 0 if flag.mode!='train' else 1])
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
    return cam