"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import scipy.io as sio
import cv2

from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label, decode_labels_old

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/home/VOCdevkit'
DATA_LIST_PATH = './dataset/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = 1449 # Number of images in the validation set.
RESTORE_FROM = './deeplab_resnet.ckpt'
PRINT_IMG = True
PRINT_PROB = True
PRINT_IND = True

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY, help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH, help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL, help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("--print-prob", action="store_true", help="Whether to print propabilities.")    
    parser.add_argument("--print-img", action="store_true", help="Whether to print propabilities.")        
    parser.add_argument("--print-ind", action="store_true", help="Whether to print propabilities.")    
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    # print(args)
    print(str(args).replace(',','\n'))

    # Output dirs
    OUTPUT_ROOT = os.path.dirname(args.restore_from)
    PRAEFIX = args.restore_from.split('/')[-1]
    PRAEFIX = PRAEFIX.split('-')[1]
    SAVE_DIR_IND = OUTPUT_ROOT + '/' 'res_' + PRAEFIX + '_ind/'
    SAVE_DIR_PROB = OUTPUT_ROOT + '/' 'res_' + PRAEFIX + '_prob/'
    SAVE_DIR_RGB = OUTPUT_ROOT + '/' 'res_' + PRAEFIX + '_rgb/'
    print("OUTPUT_ROOT = " + OUTPUT_ROOT)
    print("SAVE_DIR_IND = " + SAVE_DIR_IND)
    print("SAVE_DIR_PROB = " + SAVE_DIR_PROB)
    print("SAVE_DIR_RGB = " + SAVE_DIR_RGB)

    # Create save_dir
    if not os.path.exists(SAVE_DIR_IND):
        os.makedirs(SAVE_DIR_IND)
    if not os.path.exists(SAVE_DIR_PROB):
        os.makedirs(SAVE_DIR_PROB)
    if not os.path.exists(SAVE_DIR_RGB):
        os.makedirs(SAVE_DIR_RGB)

    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label

    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    image_batch075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
    image_batch05 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.5)), tf.to_int32(tf.multiply(w_orig, 0.5))]))
    
    # Create network.
    with tf.variable_scope('', reuse=False):
        net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)
    with tf.variable_scope('', reuse=True):
        net075 = DeepLabResNetModel({'data': image_batch075}, is_training=False, num_classes=args.num_classes)
    with tf.variable_scope('', reuse=True):
        net05 = DeepLabResNetModel({'data': image_batch05}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output100 = net.layers['fc1_voc12']
    raw_output075 = tf.image.resize_images(net075.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    raw_output05 = tf.image.resize_images(net05.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    
    raw_output = tf.reduce_max(tf.stack([raw_output100, raw_output075, raw_output05]), axis=0)
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    pred_square = tf.argmax(raw_output, dimension=3)
    pred_square = tf.expand_dims(pred_square, dim=3) # Create 4-d tensor.
    
    # mIoU
    pred = tf.reshape(pred_square, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    weights = tf.cast(tf.less_equal(gt, args.num_classes - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes, weights=weights)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Filelist
    data_info = pd.read_csv(args.data_list, sep = " ", header=None)
    num_test_files = data_info.shape[0] # For computing Iter
    imgList = np.asarray(data_info.iloc[:, 0])

    # Iterate over training steps.
    for step in range(num_test_files):
        preds, preds_square, raw_outputs, _ = sess.run([pred, pred_square, raw_output, update_op])
        if step % 100 == 0:
            print('step {:d}'.format(step))

        imName = imgList[step].split("/")[1][:-4]
        print("imName = " + imName)
        print("raw_outputs.shape = " + str(np.array(preds_square).shape))
        preds_square_np = np.array(preds_square)[0, :, :, 0]
        print(preds_square_np.shape)
        if args.print_img:
            preds_square_rgb = decode_labels_old(preds_square_np, args.num_classes)
            im = Image.fromarray(preds_square_rgb)
            im.save(SAVE_DIR_RGB + imName + '.png')

        if args.print_ind:
            cv2.imwrite(SAVE_DIR_IND + imName + '.png', preds_square_np)
            
        # Store probabilities
        if args.print_prob:
            sio.savemat(SAVE_DIR_PROB + imName,{'data':np.array(raw_outputs)[0, :, :, :]})

    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
