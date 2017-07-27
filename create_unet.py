import tensorflow as tf
import numpy as np

class Unet(object):
    def  __init__(self, image, n_class):
        self.batch_size = image[0]
        self.img_size_x = image[1]
        self.img_size_y = image[2]
        self.image = tf.placeholder("float", shape=[self.batch_size, self.img_size_x, self.img_size_y, 1])
        self.n_class = n_class

    '''
    these functions all use the methods written in tf.layers where weights and bias are not visible
    '''

    # initialize the weights
    # create a variable w = tf.Variable(<initail value>, name=<optional-name>)
    def weights_create(self, shape, stddev=0.2, name=None):
        initial = tf.truncated_normal(shape, stddev=stddev)
        if name is None:
            return tf.Variable(initial)
        else:
            return tf.get_variable(name, initializer=initial)

    def bias_create(self, shape, name=None):
        initial = tf.constant(0.0, shape)
        if name is None:
            return tf.Variable(initial)
        else:
            return tf.get_variable(name, initializer=initial)
    
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
    def downnetwork(self, image):
        img_list = []
        img_list.append(image)
        with tf.variable_scope("downwards"):
            for i in xrange(5):
                if i == 0:
                    weight1 = self.weights_create(shape=[3,3,1,64], name="dw01")
                else:
                    weight1 = self.weights_create(shape=[3,3,2**(i+5), 2**(i+6)], name="dw"+str(i)+"1")
                weight2 = self.weights_create(shape=[3,3,2**(i+6), 2**(i+6)], name="dw"+str(i)+"2")
                bias1 = self.bias_create(shape=[2**(i+6)], name="db"+str(i)+"1")
                bias2 = self.bias_create(shape=[2**(i+6)], name="db"+str(i)+"2")

                tmp = img_list[i]
                tmp = tf.nn.conv2d(tmp, weight1, strides=[1,1,1,1], padding="SAME")
                tmp = tf.nn.relu(tmp+bias1)
                tmp = tf.nn.conv2d(tmp, weight2, strides=[1,1,1,1], padding="SAME")
                tmp = tf.nn.relu(tmp+bias2)
                if i != 4:
                    tmp = tf.nn.max_pool(tmp, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
                img_dist.append(tmp)
            
        return img_dlist

    def upnetwork(self, img_dlist):
        tmp = img_dlist[4]
        with tf.variable_scope("upwards"):
            for i in xrange(4,0,-1):
                tmp_shape = tf.shape(tmp)
                upfilter = self.weights_create(shape=[3,3,tmp_shape[3]/2,tmp_shape[3]], name="uf"+str(i))
                upb = self.bias_create(shape=[tmp_shape[3]/2], name="upb"+str(i))
                tmp = tf.nn.conv2d_transpose(tmp, upfilter, tf.stack([tmp_shape[0], 2*tmp_shape[1], 2*tmp_shape[2], tmp_shape[3]/2]), strides = [1, 2, 2, 1])
                tmp = tf.nn.relu(tmp+upb)
                tmp = tf.concat([img_dist[i-1], tmp]. axis=3)
                weight1 = self.weights_create(shape=[3,3,2**(i+6),2**(i+5)], name="uw1"+str(i))
                bias1 = self.bias_create(shape=[2**(i+5)], name="ub1"+str(i))
                tmp = tf.nn.conv2d(tmp, weight1, strides=[1,1,1,1], padding="SAME") 
                tmp = tf.nn.relu(tmp + bias1)                                                        
                weight2 = self.weights_create(shape=[3,3,2**(i+5),2**(i+5)], name="uw2"+str(i))
                bias2 = self.bias_create(shape=[2**(i+5)], name="ub2"+str(i))
                tmp = tf.nn.conv2d(tmp, weight2, strides=[1,1,1,1], padding="SAME")                  
                tmp = tf.nn.relu(tmp + bias2) 
        img_u = tmp
        return img_u

    def outputnetwork_segmentation(self, img_u):
        # it is the operation of conv 1x1
        with tf.variable_scope("output"):
            weights = self.weights_create(shape=[1,1,64,self.n_class], name="ow")
            img_seg = tf.nn.conv2d(img_u, weights, strides=[1,1,1,1], padding="SAME")
        return img_seg

    def network_segmentation(self):
        img_dist = self.downnetwork(self.image)
        image_u = self.upnetwork(img_dlist)
        img_seg = self.outputnetwork_segmentation(img_u)
        return img_seg



class Trainer(object):
    def __init__(self, img_seg, n_class, learning_rate, global_step):
        self.img_seg = tf.reshape(img_seg, [-1, n_class])
        self.label = tf.placeholder("float", shape=[None,None,None,n_class])
        self.loss = self.loss()
        self.optimizer = self.train(learning_rate, global_step)

    def loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.img_seg)
        return loss

    def train(self, learning_rate, global_step):
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = trainer.minimize(self.loss, global_step=gloabl_step)
        return optimizer

