import argparse
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import scipy.io as sio
import cv2
from deeplab_resnet import ImageReaderEval, \
                           DeepLabResNetModel, \
                           prepare_label, \
                           decode_labels_old

EXP_FOLDER = '01_firstTry_nyu11' # MG 2018-01-29
EXP_ROOT = '/home/garbade/models_tf/08_nyu_depth_v2/' # MG 2018-01-29
DATA_DIRECTORY = '/home/garbade/datasets/nyu_depth_v2/'
DATA_LIST_PATH = '/home/garbade/datasets/nyu_depth_v2/filelists/test_id.txt'
RESTORE_FROM = '/home/garbade/models_tf/08_nyu_depth_v2/01_firstTry_nyu11/snapshots_finetune/model.ckpt-20000'
OUTPUT_IMGS = True
PRINT_PROPABILITIES = True


BATCH_SIZE = 1
n_classes = 11  # TODO: When "0" is the background class -> subtract 1 from gt vector before 1-hot encoding!!!!!!!!!!!!!!!
ignore_label = 0

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

# Print out params
print 'BATCH_SIZE: ', BATCH_SIZE

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--n_classes", type=int, default=n_classes,help="Number of classes.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,help="Where restore model parameters from.")    
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Number of images sent to the network in one step.")    
    parser.add_argument("--exp-folder", type=str, default=EXP_FOLDER, help="Specify expFolder")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY, help="Directory containing dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH, help="File listing the images.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    args = get_arguments()

    # Output dirs
    OUTPUT_ROOT = EXP_ROOT + args.exp_folder
    PRAEFIX = RESTORE_FROM.split('/')[-1]
    PRAEFIX = PRAEFIX.split('-')[1]
    SAVE_DIR_IND = OUTPUT_ROOT + '/' 'res_' + PRAEFIX + '_ind/'
    SAVE_DIR_PROB = OUTPUT_ROOT + '/' 'res_' + PRAEFIX + '_prob/'
    SAVE_DIR_RGB = OUTPUT_ROOT + '/' 'res_' + PRAEFIX + '_rgb/'

    # Create save_dir
    if not os.path.exists(SAVE_DIR_IND):
        os.makedirs(SAVE_DIR_IND)
    if not os.path.exists(SAVE_DIR_PROB):
        os.makedirs(SAVE_DIR_PROB)
    if not os.path.exists(SAVE_DIR_RGB):
        os.makedirs(SAVE_DIR_RGB)

    print('OUTPUT_ROOT: ' + OUTPUT_ROOT)


    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReaderEval(args.data_dir + '/images/', 
                                 args.data_list, 
                                 coord, 
                                 mask = None)         
        image = reader.image
    image_batch = tf.expand_dims(image, dim=0) # Add one batch dimension.

    # Start session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    sess = tf.InteractiveSession(config=config)
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    net = DeepLabResNetModel({'data': image_batch}, args.n_classes, is_training = True)

    # Predictions.
    raw_output_small = net.layers['fc1_voc12']
    raw_output_big = tf.image.resize_bilinear(raw_output_small, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output_big, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.

    # Init all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Load weights
    restore_var = tf.global_variables()
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, RESTORE_FROM)
    



    # Iterate over test files
    data_info = pd.read_csv(args.data_list, header=None)
    num_test_files = data_info.shape[0] # For computing Iter
    imgList = np.asarray(data_info.iloc[:, 0])


    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    print("Iterate over " + str(num_test_files) + " test files")
    for step in range(num_test_files):
        start_time = time.time()
        feed_dict = { step_ph : step}

        if PRINT_PROPABILITIES:
            preds, probs_small, probs = sess.run([pred, raw_output_small, raw_output_big])
        else:
            preds = sess.run([pred])
        if step % 1 == 0:
            print('step {:d}'.format(step))
        if OUTPUT_IMGS:
            # print(np.array(preds).shape)
            preds_rgb = decode_labels_old(np.array(preds)[0, :, :, 0], args.n_classes)
            im = Image.fromarray(preds_rgb)
            im.save(SAVE_DIR_RGB + imgList[step] + '.png')

            mask_ind = np.array(preds)[0, :, :, 0]
            cv2.imwrite(SAVE_DIR_IND + imgList[step] + '.png', mask_ind)
            
        # Store probabilities
        if PRINT_PROPABILITIES:
            sio.savemat(SAVE_DIR_PROB + imgList[step],{'data':np.array(probs_small)[0, :, :, :]})

        duration = time.time() - start_time
        print('step {:d} \t, ({:.3f} sec/step) '.format(step, duration))
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()


