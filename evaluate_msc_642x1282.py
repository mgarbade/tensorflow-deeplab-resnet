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

from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label, decode_labels_old

#n_classes = 21

### Voc12
#n_classes = 21
#ignore_label = 255
#DATA_DIRECTORY = '/home/garbade/datasets/VOC2012/'
#DATA_LIST_PATH = './dataset/voc12/val_Bndry255.txt'
#DATA_LIST_PATH_ID = './dataset/voc12/val_id.txt'
#
#EXP_ROOT = '/home/garbade/models_tf/01_voc12/22_nc21_fullResolution_Bndry255/'
#RESTORE_FROM = EXP_ROOT + '/snapshots_finetune/model.ckpt-20000'
#SAVE_DIR = EXP_ROOT + '/images_val_pred_masked_input/'
#SAVE_DIR_IND = EXP_ROOT + '/images_val_pred_ind_masked_input/'

### CamVid
#n_classes = 11
#ignore_label = 255
#DATA_DIRECTORY = '/home/garbade/datasets/CamVid/'
## DATA_LIST_PATH = './dataset/camvid/test_70.txt'
#DATA_LIST_PATH = './dataset/camvid/test.txt'
#DATA_LIST_PATH_ID = './dataset/camvid/test_id.txt'
#SAVE_DIR = '/home/garbade/models_tf/03_CamVid/14_fixedRandomCropping/images_val_full/'
## RESTORE_FROM = '/home/garbade/models_tf/03_CamVid/12_higherLR/snapshots_finetune/model.ckpt-6600'
#RESTORE_FROM = '/home/garbade/models_tf/03_CamVid/14_fixedRandomCropping/snapshots_finetune/model.ckpt-20000'

OUTPUT_IMGS = True

### Cityscapes (19 classes + BG)
n_classes = 19
ignore_label = 19
ignore_labels_below_equal = 18
DATA_DIRECTORY='/home/garbade/datasets/cityscapes/'

# Validation fully visible
#DATA_LIST_PATH='./dataset/city/val_nc20_new.txt'
DATA_LIST_PATH_ID='./dataset/city/val_id.txt'

# sz50
#DATA_LIST_PATH='./dataset/city/small_50/val_nc20.txt'
#MASK_FILE = './dataset/city/masks_50/01_center_visible/mask_sz50_320x640.png'

# sz100

## Phase1
DATA_LIST_PATH='./dataset/city/val_642x1282.txt'
MASK_FILE = './dataset/city/masks/no_mask_642x1282.png'

## Validation on Training set
#DATA_LIST_PATH='./dataset/city/small_50/train_aug_nc20.txt'
#DATA_LIST_PATH_ID='./dataset/city/small_50/train_id.txt'

EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/22_msc_fullSizeInput_nc19/'
RESTORE_FROM = EXP_ROOT + '/snapshots_finetune/model.ckpt-20000'
SAVE_DIR = EXP_ROOT + '/val_semseg_642x1282/'
SAVE_DIR_IND = EXP_ROOT + '/val_semseg_642x1282_ind/'




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
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted masks.")
    parser.add_argument("--save_dir_ind", type=str, default=SAVE_DIR_IND,
                        help="Where to save predicted masks index.")   
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
            coord,
            args.ignore_label)
        image, label = reader.image, reader.label

    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    image_batch075 = tf.image.resize_images(image_batch, tf.pack([tf.to_int32(tf.mul(h_orig, 0.75)), tf.to_int32(tf.mul(w_orig, 0.75))]))
    image_batch05 = tf.image.resize_images(image_batch, tf.pack([tf.to_int32(tf.mul(h_orig, 0.5)), tf.to_int32(tf.mul(w_orig, 0.5))]))
    
    # Create network.
    with tf.variable_scope('', reuse=False):
        net = DeepLabResNetModel({'data': image_batch}, args.n_classes, is_training=False)
    with tf.variable_scope('', reuse=True):
        net075 = DeepLabResNetModel({'data': image_batch075}, args.n_classes, is_training=False)
    with tf.variable_scope('', reuse=True):
        net05 = DeepLabResNetModel({'data': image_batch05}, args.n_classes, is_training=False)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output100 = net.layers['fc1_voc12']
    raw_output075 = tf.image.resize_images(net075.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    raw_output05 = tf.image.resize_images(net05.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    
    raw_output = tf.reduce_max(tf.stack([raw_output100, raw_output075, raw_output05]), axis=0)
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    # mIoU
    pred_lin = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    weights = tf.cast(tf.less_equal(gt, ignore_labels_below_equal), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_lin, gt, num_classes=n_classes, weights=weights)
    
#    MASK_FILE = './dataset/city/masks/01_center_visible/mask_sz100_1024x2048.png'
#    if MASK_FILE is not None:
#        mask = tf.image.decode_png(tf.read_file(MASK_FILE),channels=1)
#        mask = tf.cast(mask, dtype=tf.float32) 
#        # Downsample to input image size -> needs same size for evaluation of IoU
#        mask_int = tf.cast(mask, dtype=tf.int32)     
#    mask_inside = tf.reshape(mask_int, [-1])
#    mask_outside = (mask_inside - 1)  * -1
#    mIoU_inside, update_op_inside = tf.contrib.metrics.streaming_mean_iou(pred_lin, gt, num_classes = n_classes, weights = mask_inside)
#    mIoU_outside, update_op_outside = tf.contrib.metrics.streaming_mean_iou(pred_lin, gt, num_classes = n_classes, weights = mask_outside)    
    
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

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)    # Iterate over training steps.
    if not os.path.exists(SAVE_DIR_IND):
        os.makedirs(SAVE_DIR_IND)    # Iterate over training steps.   
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
    if not os.path.exists(args.save_dir_ind):
        os.makedirs(args.save_dir_ind)        
    # Iterate over training steps.
    for step in range(args.num_steps):
#        preds, preds_lin, _, _, _ = sess.run([pred, pred_lin, update_op, update_op_inside, update_op_outside])
        preds, preds_lin, _, = sess.run([pred, pred_lin, update_op])
#        preds, _ = sess.run([pred, update_op])
        if step % 100 == 0:
            print('step {:d}'.format(step))
        if OUTPUT_IMGS:
            # print(np.array(preds).shape)
            msk = decode_labels_old(np.array(preds)[0, :, :, 0], args.n_classes)
            im = Image.fromarray(msk)
            im.save(args.save_dir + imgList[step] + '.png')

            mask_ind = np.array(preds)[0, :, :, 0]
            cv2.imwrite(args.save_dir_ind + imgList[step] + '.png', mask_ind)
            
    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
#    print('Mean IoU_inside: {:.3f}'.format(mIoU_inside.eval(session=sess)))
#    print('Mean IoU_outside: {:.3f}'.format(mIoU_outside.eval(session=sess)))    
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
