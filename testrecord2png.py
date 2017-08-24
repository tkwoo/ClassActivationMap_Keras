import numpy as np
import cv2
import os
import sys
import readers
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2', '3'}

def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.
  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.
  Returns:
    A tuple containing the image id tensor, image tensor and labels tensor.
  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))

    filename_queue = tf.train.string_input_producer(
        files, num_epochs=None, shuffle=False)
    examples_and_ids = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    image_id_batch, image_batch = (
        tf.train.batch_join(examples_and_ids,
                            batch_size=batch_size,
                            allow_smaller_final_batch=True,
                            enqueue_many=True))
    return image_id_batch, image_batch

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    
    reader = readers.QuickDrawTestFeatureReader()
    tfrec_file = "./data/quickdraw/test/test.tfrecords"
    
    image_id_batch, image_batch = get_input_data_tensors(reader, tfrec_file, 32)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            image_id_batch_val, image_batch_val  = sess.run([image_id_batch, image_batch])
            # print image_batch_val.shape
            # print image_id_batch_val
            # print image_batch_val
            
            for idx in range(image_batch_val.shape[0]):
                cv2.imwrite("./image_data/test/%s.png"%(image_id_batch_val[idx]), image_batch_val[idx])
                print '%s, done'%image_id_batch_val[idx]
                # print '(%d / %d), label: %s'%(idx, total_num, np_label)
            # exit()

    # except tf.errors.OutOfRangeError:
    #     # logging.info('Done with inference. The output file was written to ' + out_file_location)
    #     print ('hihi')
    finally:
        coord.request_stop()

# coord.join(threads)
sess.close()
