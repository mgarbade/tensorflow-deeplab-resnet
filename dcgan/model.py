#from __future__ import division
from __future__ import print_function
import os
import time
from glob import glob
import tensorflow as tf
#from six.moves import xrange
from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, d_net_input, image_size=(64,128),
                 df_dim=64, 
                 c_dim=1):
        """
        Args:
            batch_size: The size of batch. Should be specified before training.
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.df_dim = df_dim
        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')


        self.input = d_net_input
        self.build_model()

        self.model_name = "DCGAN.model"

    def build_model(self):
        self.D, self.D_logits = self.discriminator(self.input) # Feed in ground truth images
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,
                                                    tf.zeros_like(self.D))) # TODO: This was previously "ones_like" -> make sure the loss is bigger the more unrealistic the image is looking!!!
        # TODO: One could also try to compute the loss on the occluded part only
        self.d_loss_sum = tf.summary.scalar("d_loss_real", self.d_loss)
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

    def get_loss(self):
        return self.d_loss
        
    def get_vars(self):
        return self.d_vars     
        
    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        # h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin') # 64x64 --> 4x4x512 = 8192
        h4 = linear(tf.reshape(h3, [-1, 4608]), 1, 'd_h3_lin') # 41x41 --> 4608

        return tf.nn.sigmoid(h4), h4

