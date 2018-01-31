import scipy.misc
import numpy as np
import tensorflow as tf

n_classes = 19
ignore_labels_below_equal = 18
DATA_LIST_PATH_ID='./dataset/city/val_id.txt'
PRED_PATH = '/home/garbade/models_tf/05_Cityscapes/22_msc_fullSizeInput_nc19/images_val_ind/'
GT_PATH = '/home/garbade/datasets/cityscapes/labels_ic19/'


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


pred = tf.placeholder(tf.float32, [1024, 2048], name='pred')
gt = tf.placeholder(tf.float32, [1024, 2048], name='gt')

# mIoU
pred_lin = tf.reshape(pred, [-1,])
gt_lin = tf.reshape(gt, [-1,])
weights = tf.cast(tf.less_equal(gt_lin, ignore_labels_below_equal), tf.int32) 

MASK_FILE = './dataset/city/masks/01_center_visible/mask_sz100_1024x2048.png'
if MASK_FILE is not None:
    mask = tf.image.decode_png(tf.read_file(MASK_FILE),channels=1)
    mask = tf.cast(mask, dtype=tf.float32) 
    # Downsample to input image size -> needs same size for evaluation of IoU
    mask_int = tf.cast(mask, dtype=tf.int32)     
mask_inside = tf.reshape(mask_int, [-1])
mask_outside = (mask_inside - 1)  * -1
mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_lin, gt_lin, num_classes = n_classes, weights=weights)
mIoU_inside, update_op_inside = tf.contrib.metrics.streaming_mean_iou(pred_lin, gt_lin, num_classes = n_classes, weights = mask_inside)
mIoU_outside, update_op_outside = tf.contrib.metrics.streaming_mean_iou(pred_lin, gt_lin, num_classes = n_classes, weights = mask_outside)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
sess.run(tf.local_variables_initializer())

for i in xrange(NUM_STEPS):
  print(i)
  myPred = scipy.misc.imread(PRED_PATH + imgList[i] + '.png', flatten = True).astype(np.float)
  myLabel = scipy.misc.imread(GT_PATH + imgList[i] + '.png', flatten = True).astype(np.float)

  fd = { 
    pred: myPred,
    gt: myLabel
  }  
#  _ = sess.run([update_op], feed_dict=fd)
  _ = sess.run([update_op_inside], feed_dict=fd)  
#  _, _, _ = sess.run([update_op, update_op_inside, update_op_outside], feed_dict=fd)
  
  if i % 100 == 0:
    print('step {:d}'.format(i))
#    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    print('Mean IoU_inside: {:.3f}'.format(mIoU_inside.eval(session=sess)))
#    print('Mean IoU_outside: {:.3f}'.format(mIoU_outside.eval(session=sess))) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    