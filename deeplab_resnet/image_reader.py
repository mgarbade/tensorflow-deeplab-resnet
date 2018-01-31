import os

import numpy as np
import tensorflow as tf

from utils import get_random_mask

# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
# IMG_MEAN = np.array((104.0,116.0,122.0), dtype=np.float32)

def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label


def read_labeled_image_list(data_dirs, data_list):
    f = open(data_list, 'r')
    images = []
    labels2D = []
    depth = []
    for line in f:
        name = line.strip("\n")
        images.append(data_dirs[0] + name + '.jpg')
        labels2D.append(data_dirs[1] + name + '.png')
        depth.append(data_dirs[2] + name + '.png')
    firstImg = images[0]  # Get image type
    if firstImg.endswith('.jpg'):
        img_type = 1
        print('Image type is jpg')
    elif firstImg.endswith('.png'):
        img_type = 2
        print('Image type is png')
    else:
        print('Error: Wrong image type. Must be either png or jpg')

    return images, labels2D, depth, img_type


def read_labeled_image_list_eval(data_dirs, data_list):
    f = open(data_list, 'r')
    images = []
    for line in f:
        name = line.strip("\n")
        im_name = data_dirs + name + '.jpg'
        print(im_name)
        images.append(im_name)
    firstImg = images[0] # Get image type
    if firstImg.endswith('.jpg'):
        img_type = 1
        print('Image type is jpg')
    elif firstImg.endswith('.png'):
        img_type = 2
        print('Image type is png')
    else:
        print('Error: Wrong image type. Must be either png or jpg')
    
    return images, img_type


def random_crop_and_pad_image_and_labels(image, labels, mask, crop_h, crop_w):
    """Randomly crops `image` together with `labels`.
        To ensure labels are padded with "ignore_label" this has to be subtracted from the label image, then
        after padding with 0, that value is added again. Make sure dtype allows negative values.
    Args:
        image: A Tensor with shape [D_1, ..., D_K, N]
        labels: A Tensor with shape [D_1, ..., D_K, M]
        size: A Tensor with shape [K] indicating the crop size.
    Returns:
        A tuple of (cropped_image, cropped_label).
    """
    combined = tf.concat(2, [image, labels, mask]) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(
        combined, 0, 0,
        tf.maximum(crop_h, image_shape[0]),
        tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(labels)[-1]
    combined_crop = tf.random_crop(combined_pad,[crop_h,crop_w,5]) # TODO: Make cropping size a variable

    return (combined_crop[:, :, :last_image_dim],
            combined_crop[:, :, last_image_dim:last_image_dim + 1],
            combined_crop[:, :, last_image_dim + 1:])

    
def preprocess_input_train( img, label, 
                            ignore_label = 255, 
                            input_size = (321,321),
                            scale = False,
                            mirror = False,
                            crop = True):
    """Read one image and its corresponding mask with optional pre-processing.    
    Args:
      input_queue: tf queue with paths to the image and its mask.

      Returns:
      Two tensors: the decoded image and its mask.
    """    
    label, mask = tf.split(split_dim=2, num_split=2, value=label)
#    if scale:
#        scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
#        h_new = tf.to_int32(tf.mul(tf.to_float(tf.shape(img)[0]), scale))
#        w_new = tf.to_int32(tf.mul(tf.to_float(tf.shape(img)[1]), scale))
#        new_shape = tf.squeeze(tf.pack([h_new, w_new]), squeeze_dims=[1])
#        img = tf.image.resize_images(img, new_shape)
#        label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
#        mask = tf.image.resize_nearest_neighbor(tf.expand_dims(mask, 0), new_shape)
#        label = tf.squeeze(label, squeeze_dims=[0])
#        mask = tf.squeeze(mask, squeeze_dims=[0])
    if mirror:
        random_number = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
        img = image_mirroring(img, random_number)
        label = image_mirroring(label, random_number)
        mask = image_mirroring(mask, random_number)
    # Crop and pad image
    label = tf.cast(label, dtype=tf.float32) # Needs to be subtract and later added due to 0 padding
    mask = tf.cast(mask, dtype=tf.float32)
    if crop:
        label = label - ignore_label
        crop_h, crop_w = input_size
        img, label, mask = random_crop_and_pad_image_and_labels(img, label, mask, crop_h, crop_w)
        label = label + ignore_label
        # Set static shape so that tensorflow knows shape at compile time 
        img.set_shape((crop_h, crop_w, 3))
        label.set_shape((crop_h, crop_w, 1)) 
        mask.set_shape((crop_h, crop_w, 1)) 
    label_with_mask = tf.concat(2,[label,mask])
    return img, label_with_mask

def read_bin_file():
    print('read_bin_file has not been implemented yet')

def read_images_from_disk(input_queue, img_type, phase):
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    depth_contents = tf.read_file(input_queue[2])
    if img_type == 1:
        img = tf.image.decode_jpeg(img_contents, channels=3) # VOC12
    else:
        img = tf.image.decode_png(img_contents, channels=3) # CamVid
    label = tf.image.decode_png(label_contents, channels=1)
    depth = tf.image.decode_png(depth_contents, channels=1)
    combined = tf.concat(2, [label, depth]) 

    # Change RGB to BGR
    img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img)
    img = tf.cast(tf.concat(2, [img_b, img_g, img_r]), dtype=tf.float32)    
    # Mean subtraction 
    IMG_MEAN = tf.constant([104.00698793,116.66876762,122.67891434],shape=[1,1,3], dtype=tf.float32) # BGR
    IMG_MEAN = tf.reshape(IMG_MEAN,[1,1,3]) 
    img = img - IMG_MEAN
    

    # Optional preprocessing for training phase    
    if phase == 'train':
        img, label = preprocess_input_train(img, combined)
    elif phase == 'valid':
        # TODO: Perform only a central crop -> size should be the same as during training
        pass
    elif phase == 'test':
        pass

    return img, label

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, 
                 data_list,
                 coord,
                 phase = 'train'):
        self.data_dir = data_dir
        self.data_list = data_list
        self.coord = coord

        self.image_list, self.label_list, self.depth_list, self.img_type = read_labeled_image_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.depths = tf.convert_to_tensor(self.depth_list, dtype=tf.string)
        self.isTrain = phase == 'train'
        self.queue = tf.train.slice_input_producer([self.images, self.labels, self.depths],shuffle=self.isTrain)
        self.image, self.label = read_images_from_disk(self.queue,
                                                       self.img_type, 
                                                       phase)
    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                  num_elements)
#        mask = tf.expand_dims(self.mask, 0)
#        mask = tf.tile(mask,(num_elements,1,1,self.n_classes))                                          
        return image_batch, label_batch       
        
        
class ImageReaderEval(object):        
    def __init__(self, data_dir, 
                 data_list, 
                 coord, 
                 mask = None):
        self.data_dir = data_dir
        self.data_list = data_list
        self.coord = coord
        self.image_list, self.img_type = read_labeled_image_list_eval(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images],shuffle=False)
        self.img_contents = tf.read_file(self.queue[0])
        if self.img_type == 1:
            img = tf.image.decode_jpeg(self.img_contents, channels=3) # VOC12
        else:
            img = tf.image.decode_png(self.img_contents, channels=3) # CamVid
        # Change RGB to BGR
        img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img)
        img = tf.cast(tf.concat(2, [img_b, img_g, img_r]), dtype=tf.float32)    
        # Mean subtraction 
        IMG_MEAN = tf.constant([104.00698793,116.66876762,122.67891434],shape=[1,1,3], dtype=tf.float32) # BGR
        IMG_MEAN = tf.reshape(IMG_MEAN,[1,1,3]) 
        img = img - IMG_MEAN            
        if mask is not None:
            self.image = tf.multiply(img, mask)
        else:
            self.image = img


