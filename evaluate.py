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

from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label

#n_classes = 21

#DATA_DIRECTORY = '/home/VOCdevkit'
#DATA_LIST_PATH = './dataset/val.txt'
#NUM_STEPS = 1449 # Number of images in the validation set.
#RESTORE_FROM = './deeplab_resnet.ckpt'


### Cityscapes (19 classes + BG)
n_classes=20
ignore_label=19
DATA_DIRECTORY='/home/garbade/datasets/cityscapes/'

# Validation fully visible
DATA_LIST_PATH='./dataset/city/small_50/val_nc20.txt'
DATA_LIST_PATH_ID='./dataset/city/small_50/val_id.txt'
MASK_FILE = './dataset/city/masks/images_50_mask_320x640.png'

# Validation on Training set
#DATA_LIST_PATH='./dataset/city/small_50/train_aug_nc20.txt'
#DATA_LIST_PATH_ID='./dataset/city/small_50/train_id.txt'

#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/18_predNet_nc20/'
EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/21_nc20_ic19_msc/'
RESTORE_FROM = EXP_ROOT + '/snapshots_finetune/model.ckpt-20000'
SAVE_DIR = EXP_ROOT + '/images_val_pred_masked_input/'
SAVE_DIR_IND = EXP_ROOT + '/images_val_pred_ind_masked_input/'




imgList = []
with open(DATA_LIST_PATH_ID, "rb") as fp:
    for i in fp.readlines():
        tmp = i[:-1]
        try:
            imgList.append(tmp)
        except:pass

if imgList == []:
    print('Error: Filelist is empty')
else:
    print('Filelist loaded successfully')
NUM_STEPS = len(imgList)
print(NUM_STEPS)





def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--n_classes", type=int, default=n_classes,
                        help="How many classes to predict (default = n_classes).")
    parser.add_argument("--ignore_label", type=int, default=ignore_label,
			help="All labels >= ignore_label are beeing ignored")    
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
            coord)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, args.n_classes , is_training=False)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    # mIoU
    pred = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    weights = tf.cast(tf.less_equal(gt, n_classes - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=n_classes, weights=weights)
    
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
    
    # Iterate over training steps.
    for step in range(args.num_steps):
        preds, _ = sess.run([pred, update_op])
        if step % 100 == 0:
            print('step {:d}'.format(step))
    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
