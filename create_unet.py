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
        for i in xrange(4):
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
    def downnetwork(self, weights, image):
        ### TODO ###
        img_list = []
        img_list.append(image)
        for i in xrange(5):
            tmp = img_list[i]
            tmp = tf.nn.conv2d(tmp, weights, strides=[1,1,1,1], padding="SAME")
            tmp = tf.nn.relu(tmp)
            tmp = tf.nn.conv2d(tmp, weights, strides=[1,1,1,1], padding="SAME")
            tmp = tf.nn.relu(tmp)
            tmp = tf.nn.max_pool(tmp, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            img_dist.append(tmp)
        tmp = img_dist[3]
        tmp = tf.nn.conv2d(tmp, weights, strides=[1,1,1,1], padding="SAME")
        tmp = tf.nn.relu(tmp)
        tmp = tf.nn.conv2d(tmp, weights, strides=[1,1,1,1], padding="SAME")
        tmp = tf.nn.relu(tmp)
        img_dist.append(tmp)
        return img_dlist

    def upnetwork(self, weights, img_dlist):
        ### TODO ###
        tmp = img_dlist[4]
        for i in xrange(4,0,-1):
            tmp_shape = tf.shape(tmp)
            tmp = tf.nn.conv2d_transpose(tmp, weights, tf.stack([tmp_shape[0], 2*tmp_shape[1], 2*tmp_shape[2], tmp_shape[3]/2]), strides = [1, 2, 2, 1])
            tmp = tf.concat([img_dist[i-1], tmp]. axis=3)
            tmp = tf.nn.conv2d(tmp, weights, strides=[1,1,1,1], padding="SAME")                                                                                                                                    
            tmp = tf.nn.relu(tmp)                                                                                                                                                                                  
            tmp = tf.nn.conv2d(tmp, weights, strides=[1,1,1,1], padding="SAME")                                                                                                                                    
            tmp = tf.nn.relu(tmp) 
        img_u = tmp
        return img_u

    def outputnetwork_segmentation(self, weights, img_u):
        ### TODO ###
        img_seg = tf.nn.conv2d(img_u, weights, strides=[1,1,1,1], padding="SAME")
        return img_seg

    def network_segmentation(self, weights):
        ### TODO ###
        img_dist = self.downnetwork(weights, self.image)
        image_u = self.upnetwork(weights, img_dlist)
        img_seg = self.outputnetwork_segmentation(weights, img_u)
        return img_seg






















