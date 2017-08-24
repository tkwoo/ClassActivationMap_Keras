import cv2
import numpy as np
import os
from glob import glob
import argparse
import sys

import train
import predict

parser = argparse.ArgumentParser()
### training
parser.add_argument("--data_path", help="training data path", default="./data")
parser.add_argument("--total_epoch", help="number of epochs", default=10, type=int)
parser.add_argument("--initial_learning_rate", help="init lr", default=0.007, type=float)
parser.add_argument("--learning_rate_decay_factor", help="learning rate decay", default=0.5, type=float)
parser.add_argument("--epoch_per_decay", help="lr decay period", default=3, type=int)
parser.add_argument("--pretrained_weight_path", help="weight.h5 path", default=None)
### testing
parser.add_argument("--output_dir", help="output directory", default="./result")
parser.add_argument("--test_image_path", 
                    help="[mode:predict] ex) .../Image.png, [mode:predict_imgDir] ex).../dirname",
                    default=None)
### common
parser.add_argument("--num_classes", help="number of classes", default=2, type=int)
parser.add_argument("--image_size", help="image size", default=128, type=int)
parser.add_argument("--batch_size", help="batch size", default=48, type=int)
parser.add_argument("--ckpt_dir", help="checkpoint root directory", default='./checkpoint')
parser.add_argument("--ckpt_name", help="[.../ckpt_dir/ckpt_name/weights.h5]", default='vgg')
parser.add_argument("--debug", help="['True' or 'false']", default=False, type=bool)
parser.add_argument("--mode",
                    help="[train], [predict], [eval]",
                    default='train')
parser.add_argument("--tf_log_level", help="0, 1, 2, 3", default='2', type=str)
parser.add_argument("--visible_gpu", help="visible gpu", default='0', type=str)

flag = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = flag.tf_log_level # or any {'0', '1', '2', '3'}
os.environ["CUDA_VISIBLE_DEVICES"] = flag.visible_gpu

def main():
    if not os.path.isdir(flag.output_dir):
        os.mkdir(flag.output_dir)
    if flag.mode == 'train':
        train_op = train.Trainer(flag)
        train_op.train()
    elif flag.mode == 'predict':
        predict_op = predict.predictor(flag)
        predict_op.inference()
    elif flag.mode == 'eval':
        eval_op = predict.predictor(flag)
        eval_op.evaluate()
    else:
        print 'not supported'

if __name__ == '__main__':
    main()