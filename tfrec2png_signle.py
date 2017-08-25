import numpy as np
import cv2
import os
import sys
import readers
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2', '3'}

# def get_reader():
#   reader = readers.QuickDrawFeatureReader()
#   return reader

def decode_and_resize(image_str_tensor, size):
      """Decodes png string, resizes it and returns a uint8 tensor."""
  
      # Output a grayscale (channels=1) image
      image = tf.image.decode_jpeg(image_str_tensor, channels=3)
  
      # Note resize expects a batch_size, but tf_map supresses that index,
      # thus we have to expand then squeeze.  Resize returns float32 in the
      # range [0, uint8_max]
      image = tf.expand_dims(image, 0)
    #   image = tf.image.resize_bilinear(
    #       image, [size, size], align_corners=False)
      image = tf.squeeze(image, squeeze_dims=[0])
      image = tf.cast(image, dtype=tf.uint8)
      return image

data_name = 'validation-00002-of-00002'

def main():
    with tf.Session() as sess:
        tfrec_file = "./data/{}".format(data_name)
        total_num = sum(1 for _ in tf.python_io.tf_record_iterator(tfrec_file))
        # filename_queue = tf.train.string_input_producer([tfrec_file])
        filename_queue = tf.train.string_input_producer([tfrec_file], num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        
        feature_map = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/class/label': tf.FixedLenFeature([], tf.int64)
            # 'image_id': tf.FixedLenFeature((), tf.string, default_value='')
                }
        features = tf.parse_single_example(serialized_example, features=feature_map)
        label = features['image/class/label']
        # label = features['image_id']
        image_raw = features['image/encoded']
        image_name = features['image/filename']
        image = decode_and_resize(image_raw, 128)
        
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        cv2.namedWindow("show", 0)
        cv2.resizeWindow("show", 500, 500)
        for idx in range(int(total_num)):
            img, np_label, np_name = sess.run([image, label, image_name])
            # np_label = sess.run(label)
            # np_name = sess.run(image_name)
            
            # print np_label.shape
            # print img.shape
            # cv2.imshow("show", img)
            # if cv2.waitKey(1) == 27:
            #     break
            
            r, g, b = cv2.split(img)
            img = cv2.merge((b,g,r))

            cv2.imwrite("./image_data/val/%d/%s.png"%(np_label, np_name), img)
            print '(%d / %d), label: %s'%(idx, total_num, np_label)
    
    # coord.join(thread)
    exit()
                                                
        # total_num = sum(1 for _ in tf.python_io.tf_record_iterator(tfrec_file))
        # print total_num
        # for serialized_example in tf.python_io.tf_record_iterator(tfrec_file):
        #     example = tf.train.Example()
        #     example.ParseFromString(serialized_example)

        #     # feature_map = {
        #     #     'image': tf.FixedLenFeature((), tf.string, default_value=''),
        #     #     'label': tf.FixedLenFeature([], tf.int64)
        #     #         }
        #     # features = tf.parse_example(serialized_example, features=feature_map)
        
        #     image = example.features.feature["image"].bytes_list.value
        #     label = example.features.feature["label"].int64_list.value
        #     # image = features["image"]
        #     # label = features["label"]

        #     print image.Message
        #     exit()
            # print("Feature: {}, label: {}".format(feature, label))
 
        # reader = get_reader()
        # # images, labels = reader.prepare_reader(filename_queue, 32)
        # # print images
        # # print labels

        # training_data = reader.prepare_reader(filename_queue)

        # # image, label, height, width, depth = read_and_decode(filename_queue)
        # # image = tf.reshape(image, tf.stack([height, width, 3]))
        # # # image.set_shape([32,32,3])
        # init_op = tf.global_variables_initializer()
        # sess.run(init_op)
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        # data = sess.run(training_data)
        # print len(data)
        # images = np.array(data[0])
        # print data[0].shape
        # for idx in range(data[0].shape[0]):
        #     image = images[idx]
        #     cv2.namedWindow("image", 0)
        #     cv2.resizeWindow("image", 500, 500)
        #     cv2.imshow("image", image)
        #     if cv2.waitKey() == 27:
        #         break
    
        # # for i in range(1000):
        #     # example, l = sess.run([image, label])
        #     # print (example,l)
        # coord.request_stop()
        # coord.join(threads)
        
if __name__=='__main__':
    main()