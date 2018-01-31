import argparse
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from deeplab_resnet import ImageReader, DeepLabResNetModel, prepare_label, decode_labels_old

#EXP_FOLDER = '01_firstTry'
EXP_FOLDER = '01_firstTry_nyu11' # MG 2018-01-29
#EXP_ROOT = '/home/garbade/datasets/sscnet/models_tf/'
EXP_ROOT = '/home/garbade/models_tf/08_nyu_depth_v2/' # MG 2018-01-29
#DATA_DIRECTORY = '/home/garbade/datasets/sscnet/data/depthbin/NYUCADtrain'
DATA_DIRECTORY = '/home/garbade/datasets/nyu_depth_v2/'
DATA_LIST_PATH = '/home/garbade/datasets/nyu_depth_v2/filelists/train.txt'
phase = 'restore_all_but_last'
# RESTORE_FROM = '/home/garbade/models_tf/05_Cityscapes/20_nc20_ic19/snapshots_finetune/model.ckpt-20000'
RESTORE_FROM = '/home/garbade/models_tf/dr_sleep_models/tf_v0.12/deeplab_resnet_init.ckpt' # MG 2018-01-29

## OPTIMISATION PARAMS
BATCH_SIZE = 7
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
BASE_LR = LEARNING_RATE
POWER = 0.9
MOMENTUM = 0.9

# n_classes = 40
# n_classes = 11 # MG 2018-01-29  TODO: BUG:: For the function 'tf.nn.sparse_softmax_cross_entropy_with_logits' \
                              # TODO:       the gt labels have to go from [0:n_classes -1]
                              # TODO:  else -> internal conversion to 1-hot tensor fails
# n_classes = 12  # TODO: This might be bad practice since now 'background' is one of the predicted classes
n_classes = 11  # TODO: When "0" is the background class -> subtract 1 from gt vector before 1-hot encoding!!!!!!!!!!!!!!!
ignore_label = 0
weight_l2_loss = 1
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

NUM_STEPS = 20001
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 100

# Print out params
print 'BATCH_SIZE: ', BATCH_SIZE

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--n_classes", type=int, default=n_classes,help="Number of classes.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,help="Where restore model parameters from.")    
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,help="How many images to save.")    
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,help="Save summaries and checkpoint every often.")    
    parser.add_argument("--phase", type=str, default=phase,help="Phases: restore_all, restore_all_but_last.")     
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Number of images sent to the network in one step.")    
    parser.add_argument("--exp-folder", type=str, default=EXP_FOLDER, help="Specify expFolder")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY, help="Directory containing dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH, help="File listing the images.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,help="Number of training steps.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    args = get_arguments()

    # Output dirs
    OUTPUT_ROOT = EXP_ROOT + args.exp_folder
    SAVE_DIR = OUTPUT_ROOT + '/images_finetune/'
    SNAPSHOT_DIR = OUTPUT_ROOT + '/snapshots_finetune/'
    LOG_DIR = OUTPUT_ROOT + '/logs/'
    print('OUTPUT_ROOT: ' + OUTPUT_ROOT)


    coord = tf.train.Coordinator()


    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            (args.data_dir + 'images/',
             args.data_dir + 'labels_11/',  # args.data_dir + 'labels_36/'
             args.data_dir + 'depths/'),
            args.data_list,
            coord)
        image_batch, label_depth_batch = reader.dequeue(args.batch_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    sess = tf.InteractiveSession(config=config)
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    net = DeepLabResNetModel({'data': image_batch}, args.n_classes, is_training = True)

    raw_output = net.layers['fc1_voc12']
    raw_prediction = tf.reshape(raw_output, [-1, args.n_classes])

    # Get variables by name
    restore_var = tf.global_variables()
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
    vars_restore_gist = [v for v in tf.global_variables() if not 'fc' in v.name]

    # Labels and masks are still concatenated until here
    label_batch, mask_batch = tf.split(3,2,label_depth_batch)
    label_proc = prepare_label(label_batch, tf.pack(raw_output.get_shape()[1:3]), args.n_classes, one_hot=False) # [10,321,321,20] --> [batch_size,h,w]
    mask_proc = prepare_label(mask_batch, tf.pack(raw_output.get_shape()[1:3]), args.n_classes, one_hot=False)
    raw_gt = tf.reshape(label_proc, [-1,])
    
    # Loss 2D Semantic segmentation
    indices = tf.squeeze(tf.where(tf.not_equal(raw_gt, ignore_label)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    # Cross Entropy Loss --> Be careful about 'gt' input format --> [0:n_classes -1]
    if ignore_label == 0:
        loss_pixel = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels = gt - 1) # TODO: The gt labels have to go from [0:n_classes -1]
    else:
        loss_pixel = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)

    loss = tf.reduce_mean(loss_pixel)
    l2_losses = [WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = loss + weight_l2_loss * tf.add_n(l2_losses)

    # Processed predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
    # Optimizer
    base_lr = tf.constant(BASE_LR)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / 20000), POWER))
    tf.summary.scalar('learning_rate', learning_rate)
    opt_conv = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, MOMENTUM)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, MOMENTUM)
    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]
    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))
    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

    # Log variables
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    tf.summary.scalar("reduced_loss", reduced_loss)
    for v in conv_trainable + fc_w_trainable + fc_b_trainable:
        tf.summary.histogram(v.name.replace(":", "_"), v)
    merged_summary_op = tf.summary.merge_all()

    # Init all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=1)

    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        if args.phase ==  'restore_all_but_last':
            print('Restore everything but last layer')
            loader = tf.train.Saver(var_list=vars_restore_gist)
        elif args.phase ==  'restore_all':
            print('Restore all layers')            
            loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)
    
    # Create save_dir
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT) 

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step}

        if step % args.save_pred_every == 0:
            # total summary
            loss_value, images, labels, preds, summary, _ = sess.run([reduced_loss, 
                                                                  image_batch, 
                                                                  label_batch, 
                                                                  pred, 
                                                                  merged_summary_op,
                                                                  train_op], feed_dict=feed_dict) 
            summary_writer.add_summary(summary, step)
            ### Print intermediary images
            fig, axes = plt.subplots(args.save_num_images, 3, figsize = (16, 12))
            for i in xrange(args.save_num_images):
                if ignore_label == 0:
                    labels = labels - 1  # --> see explanation above
                axes.flat[i * 3].set_title('data')
                axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))
                axes.flat[i * 3 + 1].set_title('mask')
                axes.flat[i * 3 + 1].imshow(decode_labels_old(labels[i, :, :, 0], args.n_classes))
                axes.flat[i * 3 + 2].set_title('pred')
                axes.flat[i * 3 + 2].imshow(decode_labels_old(preds[i, :, :, 0], args.n_classes))
                plt.savefig(SAVE_DIR + str(start_time) + ".png")
                plt.close(fig)
            if args.save_pred_every is not 2:
              save(saver, sess, SNAPSHOT_DIR, step)
        else:
            loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step) '.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()


