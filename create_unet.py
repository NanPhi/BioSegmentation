import tensorflow as tf
import numpy as np

class Unet(object):
    def  __init__(self, image, label, n_class):
        ### TODO ###
        self.image = image
        self.label = label
        self.n_class = n_class

    '''
    these functions all use the methods written in tf.layers where weights and bias are not visible
    '''

    # downwards
    def downlayers(self, image):
        # down path image lists
        img_dlist = []
        img_dlist.append(image)
        for i = xrange(4):
            tmp = img_list[i]
            tmp = tf.layers.conv2d(inputs=tmp, filters=2 ** (i+6), kernel_size=3, padding="same", activation=tf.nn.relu)
            tmp = tf.layers.conv2d(inputs=tmp, filters=2 ** (i+6), kernel_size=3, padding="same",  activation=tf.nn.relu)
            tmp = tf.layers.max_pooling2d(inputs=tmp, pool_size=2, strides=2)
            img_dlist.append(tmp)
        tmp = img_dlist[3]
        tmp = tf.layers.conv2d(inputs=tmp, filters=1024, kernel_size=3, padding="same", activation=tf.nn.relu)
        tmp = tf.layers.conv2d(inputs=tmp, filters=1024, kernel_size=3, padding="same", activation=tf.nn.relu)
        img_dlist.append(tmp)
        return img_dlist

    # upwards
    def uplayers(self, img_dist):
        tmp = img_dist[4]
        for i in xrange(4, 0, -1):
            tmp = tf.layers.conv2d_transpose(inputs=tmp, filters=2 ** (i+5), kernel_size=3, padding="same", strides=2)
            tmp = tf.concat([img_dist[i-1], tmp], axis=3)
            tmp = tf.layers.conv2d(inputs=tmp, filters=2 ** (i+5), kernel_size=3, padding="same", activation=tf.nn.relu)
            tmp = tf.layers.conv2d(inputs=tmp, filters=2 ** (i+5), kernel_size=3, padding="same", activation=tf.nn.relu)
        img_u = tmp
        return img_u

    # final segmentation layer
    def outputlayers_segmentation(self, img_u):
        img_seg = tf.layers.conv2d(inputs=img_u, filters=self.n_class, kernel_size=1, padding="same")
        return img_seg

    def layers_segmentation(self):
        img_dlist = self.downlayers(self.image)
        img_u = self.uplayers(img_dlist)
        img_seg = self.outputlayers_segmentation(img_u)
        return img_seg

    '''
    these functions all use the attributes written under tf.nn where weights and bias are visible 
    '''
    def downnetwork(self, weights, bias, image):
        ### TODO ###
        return img_dlist

    def upnetwork(self, weights, bias, img_dlist):
        ### TODO ###
        return img_u

    def outputnetwork_segmentation(self, weights, bias, img_u):
        ### TODO ###
        return img_seg

    def network_segmentation(self, weights, bias):
        ### TODO ###
        return img_seg
