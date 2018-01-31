import math
import pprint
import scipy.misc
import numpy as np

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, c_dim, is_crop=True):
    im = imread(image_path,c_dim)
    im_trans = transform(im, image_size, is_crop)
    return im_trans

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path,c_dim):
    if c_dim == 3:
        im  = scipy.misc.imread(path, mode='RGB').astype(np.float)
    elif c_dim == 1:
        im  = scipy.misc.imread(path, mode='L').astype(np.float)
        im = np.expand_dims(im, 3)
    else:
        print('Error: c_dim must be either 3 or 1')
            
    return im
    

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=(64,128), is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx[0], npx[1])
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


