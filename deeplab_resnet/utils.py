from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

# n_classes = 21
# colour map voc
#label_colours = [(0,0,0)
#                # 0=background
#                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
#                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
#                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
#                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
#                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
#                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
#                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

# colour map cityscapes
#label_colours = [(128,64,128)
#                # 0='road'
#                ,(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153)
#                # 1='sidewalk', 2='building', 3='wall', 4='fence', 5='pole'
#                ,(250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180)
#                # 6='traffic light', 7='traffic sign', 8='vegetation', 9='terrain', 10='sky'
#                ,(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100)
#                # 11='person', 12='rider', 13='car', 14='truck', 15='bus'
#                ,(0,80,100),(0,0,230),(119,11,32),(0,0,0)]
#                # 16='train', 17='motorcycle', 18='bicycle', 19='background'                

## NYU_v2 labels_40
#label_colours = [
#                    (  0,  0,153),
#                    (  0,  0,179),
#                    (  0,  0,204),
#                    (  0,  0,230),
#                    (  0,  0,255),
#                    (  0, 26,255),
#                    (  0, 51,255),
#                    (  0, 77,255),
#                    (  0,102,255),
#                    (  0,128,255),
#                    (  0,153,255),
#                    (  0,179,255),
#                    (  0,204,255),
#                    (  0,230,255),
#                    (  0,255,255),
#                    ( 26,255,230),
#                    ( 51,255,204),
#                    ( 77,255,179),
#                    (102,255,153),
#                    (128,255,128),
#                    (153,255,102),
#                    (179,255, 77),
#                    (204,255, 51),
#                    (230,255, 26),
#                    (255,255,  0),
#                    (255,230,  0),
#                    (255,204,  0),
#                    (255,179,  0),
#                    (255,153,  0),
#                    (255,128,  0),
#                    (255,102,  0),
#                    (255, 77,  0),
#                    (255, 51,  0),
#                    (255, 26,  0),
#                    (255,  0,  0),
#                    (230,  0,  0),
#                    (204,  0,  0),
#                    (179,  0,  0),
#                    (153,  0,  0),
#                    (128,  0,  0)]

## NYU_v2 labels_11
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0)]
                # 11=potted plant
 
# image mean
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
def decode_labels(mask, n_classes, num_images=1):
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < n_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def decode_labels_old(mask, n_classes):
    mask = mask.astype(np.int32)
    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < n_classes:
                pixels[k_,j_] = label_colours[k]
    return np.array(img)



def prepare_label(input_batch, new_size, n_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
          input_batch = tf.one_hot(input_batch, depth=n_classes)
    return input_batch

def inv_preprocess(imgs, num_images):
    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
      outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)
    return outputs


def get_delocalized_loss(images_one_hot,
			labels_one_hot,
			mask_proc_1c, # [batch_size,41,41,1]
			indices_ignore_class, # [batch_size,41,41,1]
			n_classes,
			batch_size,
			loss_type='square_error',
			kernel_size=3,
			stride=1):
	"""Computes custom loss.

	Args:
	  images_one_hot: input tensor of shape [batch_size H W 1].
	  labels: input tensor of shape [batch_size H W 1].
	  mask: shape [batch_size H W 1].
  
	Returns:
	  Outputs a tensor of shape [1]
	"""
	images_deloc = tf.nn.max_pool(images_one_hot,[1, kernel_size, kernel_size, 1],[1,stride,stride,1],'SAME')
#	images_deloc_lin = tf.reshape(images_deloc,[-1,])
	images_deloc_lin = tf.reshape(images_deloc, [-1, n_classes]) ###

	labels_deloc = tf.nn.max_pool(labels_one_hot,[1, kernel_size, kernel_size, 1],[1,stride,stride,1],'SAME')
#	labels_deloc_lin = tf.reshape(labels_deloc,[-1,])
	labels_deloc_lin = tf.reshape(labels_deloc, [-1, n_classes]) ###
 
	# Mask needs to be downsampled according to size of pred and gt after max_pooling
	h_new = tf.shape(labels_deloc)[1]
	w_new = tf.shape(labels_deloc)[2]
	new_shape = tf.pack([h_new, w_new])
	mask_small = tf.image.resize_nearest_neighbor(mask_proc_1c, new_shape) # mask_proc_1c: 1 inside, 0 outside
	indices_ignore_class_small = tf.image.resize_nearest_neighbor(tf.cast(indices_ignore_class,tf.int32), new_shape)

	### This is the cross_entropy loss
	indices_outside = tf.less(mask_small, 0.5)
	mask_combined = tf.logical_and(tf.cast(indices_outside,tf.bool),tf.cast(indices_ignore_class_small,tf.bool))
	mask_combined_lin = tf.reshape(mask_combined,[-1,])
	indices = tf.squeeze(tf.where(mask_combined_lin),1)# 10x41x41x20
 
#	mask_lin = tf.reshape(mask_small, [-1,]) # mask_small has been downscaled by pooling
	
	images_deloc_lin_sel = tf.gather(images_deloc_lin, indices)
	labels_deloc_lin_sel = tf.gather(labels_deloc_lin, indices)

	if loss_type == 'square_error':
		print('square_error')
		images_deloc_lin_sel = tf.nn.sigmoid(images_deloc_lin_sel)
		loss_square_pixel = tf.squared_difference(images_deloc_lin_sel,labels_deloc_lin_sel)
		loss = tf.reduce_mean(loss_square_pixel)
	elif loss_type == 'sigmoid_cross_entropy':
		print('Sigmoid cross entropy loss')     
		loss_sel_pixel = tf.nn.sigmoid_cross_entropy_with_logits(logits=images_deloc_lin_sel, targets=labels_deloc_lin_sel)
		loss = tf.reduce_mean(loss_sel_pixel)
	elif loss_type == 'softmax_cross_entropy':
		print('Softmax cross entropy loss')          
		loss_sel_pixel = tf.nn.softmax_cross_entropy_with_logits(logits=images_deloc_lin_sel, labels=labels_deloc_lin_sel)
		loss = tf.reduce_mean(loss_sel_pixel)     
	else:
		print('Error: Loss type must be either cross_entropy or square_error' )
	return loss
 
def get_random_mask(h, w, max_h = 0.4, max_w = 0.4):
	"""Compute a random mask in the same size as the image

	Args:
	  h: image height
	  w: image width
  
	Returns:
	  Outputs a mask of shape [h,w]
	"""     
	scale_h = tf.random_uniform([],-max_h*0.1,max_h)
 	scale_w = tf.random_uniform([],-max_w*0.1,max_w) 
	
	# TODO: This avoids that Deloc loss gets an empty mask as input --> better use a function like gather that can handle empty masks as well
	randHeight = tf.cast(tf.maximum(scale_h * h,1),tf.int32)
	randWidth = tf.cast(tf.maximum(scale_w * w,1),tf.int32)
 
#	randHeight = tf.cast(tf.random_uniform([],0,max_h*h),tf.int32)
#	randWidth = tf.cast(tf.random_uniform([],0,max_w*w),tf.int32)
	mask_ht = tf.ones([tf.constant(w,tf.int32), randHeight])
	mask_wl = tf.ones([randWidth,tf.constant(h,tf.int32)])
	mask_ht = tf.squeeze(tf.image.pad_to_bounding_box(tf.expand_dims(mask_ht, 2), 0, 0, h, w),squeeze_dims=[2])
	mask_wl = tf.squeeze(tf.image.pad_to_bounding_box(tf.expand_dims(mask_wl, 2), 0, 0, h, w),squeeze_dims=[2])        

	mask_hb = tf.reverse(mask_ht,[False,True])
	mask_wr = tf.reverse(mask_wl,[True,False])

	a=tf.logical_or(tf.cast(mask_ht,tf.bool),tf.cast(mask_wl,tf.bool))
	b=tf.logical_or(tf.cast(mask_hb,tf.bool),tf.cast(mask_wr,tf.bool))
	c=tf.logical_or(a,b)
	# Use either a, b or c as a mask or   mask_ht, mask_wl, mask_hb, mask_wr   
	# TODO: Probably the worst implementation of switch case possible :-/ 
	rand_num = tf.random_uniform([], minval=0, maxval=2.0, dtype=tf.float32, seed=None)
 	# First try
	def if_true():
         return b
    
	def if_false():
         return a
 
	mask_sel = tf.cond(tf.less(rand_num , tf.constant(1.0)),if_true,if_false)
 
 
	# Second try 
#	mask_sel = tf.cond(tf.less(rand_num , 2.0),c,a)
#	mask_sel = tf.cond(tf.less(rand_num , 3.0),mask_ht,a)
#	mask_sel = tf.cond(tf.less(rand_num , 4.0),mask_wl,a)
#	mask_sel = tf.cond(tf.less(rand_num , 5.0),mask_hb,a)
#	mask_sel = tf.cond(tf.less(rand_num , 6.0),mask_wr,a)
	mask = tf.expand_dims(tf.cast(mask_sel,tf.float32),2)
	return mask
 
 
