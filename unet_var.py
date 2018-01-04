from __future__ import print_function, division, absolute_import, unicode_literals
from collections import OrderedDict

import sys
import os
import shutil
import logging
import numpy as np
import tensorflow as tf
import scipy.io as sio

import util


def unet_inference(x, keep_prob, channels, n_class):
    '''
    Inference via a standard unet with 5 layers from https://arxiv.org/abs/1505.04597
    ideas from https://github.com/jakeret/tf_unet

    :param x: input image tensor, shape [?,h,w,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of input image channels
    :param n_class: number of output labels
    :return: unet inferenced output map
    '''
    img = x
    dw_layers = OrderedDict()

    # downward layers
    for layer in xrange(5):
        features = 2**(layer + 6)
        stddev = np.sqrt(2 / (9 * features))
        if layer == 0:
            shape = [3, 3, channels, features]
            w1 = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
        else:
            shape = [3, 3, features // 2, features]
            w1 = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))

        shape = [3, 3, features, features]
        w2 = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
        b1 = tf.Variable(tf.constant(0.1, shape=[features]))
        b2 = tf.Variable(tf.constant(0.1, shape=[features]))

        strides = [1, 1, 1, 1]
        # convolution
        conv1 = tf.nn.conv2d(img, w1, strides=strides, padding='VALID')
        # dropout
        drop1 = tf.nn.dropout(conv1, keep_prob)
        # activation
        act = tf.nn.relu(drop1 + b1)
        conv2 = tf.nn.conv2d(act, w2, strides=strides, padding='VALID')
        drop2 = tf.nn.dropout(conv2, keep_prob)
        dw_layers[layer] = tf.nn.relu(drop2 + b2)

        # max pooling down sampling
        p_size = [1, 2, 2, 1]
        if layer < 4:
            pool = tf.nn.max_pool(dw_layers[layer], ksize=p_size, strides=p_size, padding='VALID')
            img = pool

    img = dw_layers[4]

    # upward layers
    for layer in xrange(3, -1, -1):
        features = 2 ** (layer + 7)
        stddev = np.sqrt(2 / (9 * features))
        shape = [2, 2, features // 2, features]
        wd = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
        bd = tf.Variable(tf.constant(0.1, shape=[features // 2]))
        deshape = tf.stack([tf.shape(img)[0], tf.shape(img)[1]*2, tf.shape(img)[2]*2, tf.shape(img)[3]//2])
        deconv = tf.nn.conv2d_transpose(img, wd, deshape, strides=[1, 2, 2, 1], padding='VALID')
        act = tf.nn.relu(deconv + bd)
        # cropping the downward output map into the same size as the upward map
        offset = (tf.shape(dw_layers[layer])[1] - tf.shape(act)[1]) // 2
        size = [-1, tf.shape(act)[1], tf.shape(act)[2], -1]
        crop = tf.slice(dw_layers[layer], [0, offset, offset, 0], size)
        # concatenating two maps
        concat = tf.concat([crop, act], 3)

        w1 = tf.Variable(tf.truncated_normal(shape=[3, 3, features, features // 2]), stddev=stddev)
        w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, features // 2, features // 2]), stddev=stddev)
        b1 = tf.Variable(tf.constant(0.1, shape=[features // 2]))
        b2 = tf.Variable(tf.constant(0.1, shape=[features // 2]))

        conv1 = tf.nn.conv2d(concat, w1, strides=[1, 1, 1, 1], padding='VALID')
        drop1 = tf.nn.dropout(conv1, keep_prob)
        act1 = tf.nn.relu(drop1 + b1)
        conv2 = tf.nn.conv2d(act1, w2, strides=[1, 1, 1, 1], padding='VALID')
        drop2 = tf.nn.dropout(conv2, keep_prob)
        img = tf.nn.relu(drop2 + b2)


    # fully convolutional layer
    shape = [1, 1, 64, n_class]
    w = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
    b = tf.Variable(tf.constant(0.1, shape=[n_class]))
    conv = tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='VALID')
    output_map = tf.nn.relu(conv + b)

    # recording downward layer convolutional outputs
    for layer in xrange(5):
        tf.summary.histogram("dw_convolution_%02d" % layer + '/activations', dw_layers[layer])
    return output_map

class UNet(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=1, n_class=6, cost="cross_entropy", cost_kwargs={}, **kwargs):
        tf.reset_default_graph()

        self.n_class = n_class

        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, n_class])
        # one-hot labels
        self.z = tf.placeholder(tf.int64, shape=[None, None, None, 1])
        self.weight = tf.placeholder(tf.float32, shape=[n_class])
        # coefficient to choose one class
        self.choose_coef = tf.placeholder("float", shape=[n_class])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


        logits = unet_inference(self.x, self.keep_prob, channels, n_class)

        self.predicter = tf.nn.softmax(logits)
        # theta is all the variables
        theta = tf.trainable_variables()

        self.cost = self._get_cost(logits, "cross_entropy", cost_kwargs={})

        self.gradients_node = tf.gradients(self.cost, theta)

        self.unweighted_cost = self._get_unweighted_cross_entropy(logits, self.y)
        self.J_2 = self._get_cost(logits, "balanced_cross_entropy", cost_kwargs={})

        self.gradients_w_0 = tf.gradients(self.unweighted_cost[0], theta)
        self.gradients_w_1 = tf.gradients(self.unweighted_cost[1], theta)
        self.gradients_w_2 = tf.gradients(self.unweighted_cost[2], theta)
        self.gradients_w_3 = tf.gradients(self.unweighted_cost[3], theta)
        self.gradients_w_4 = tf.gradients(self.unweighted_cost[4], theta)
        self.gradients_w_5 = tf.gradients(self.unweighted_cost[5], theta)
        self.gradients_J_2 = tf.gradients(self.J_2, theta)

        # output the trained result directly
        self.predict_res = tf.argmax(self.predicter, axis=3)

        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, axis=3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_unweighted_cross_entropy(self, prediction, truthtable):
        unweighted_cross_entropy = []
        flat_prediction = tf.reshape(prediction, [-1, self.n_class])
        flat_tt = tf.reshape(truthtable, [-1, self.n_class])
        choose_coefs = [np.array([1., 0., 0., 0., 0., 0.]), np.array([0., 1., 0., 0., 0., 0.]),
                        np.array([0., 0., 1., 0., 0., 0.]), np.array([0., 0., 0., 1., 0., 0.]),
                        np.array([0., 0., 0., 0., 1., 0.]), np.array([0., 0., 0., 0., 0., 1.])]
        for choose_coef in choose_coefs:
            flat_labels = flat_tt * choose_coef
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_prediction, labels=flat_labels))
            unweighted_cross_entropy.append(loss)

        return unweighted_cross_entropy


    def _get_cost(self, logits, cost_name, cost_kwargs):

        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])
        flat_z = tf.reshape(self.z, [-1])
        flat_prediction = tf.reshape(self.predicter, [-1, self.n_class])

        if cost_name == "unweighted_cross_entropy":
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_logits,labels=flat_z))

        elif cost_name == "weighted_cross_entropy":
            weight_map = tf.nn.embedding_lookup(tf.constant(self.weight), flat_z)
            loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_logits,labels=flat_z)
            loss = tf.reduce_mean(tf.multiply(loss_map, weight_map))

        # balanced cross entropy getting weights from each batch
        elif cost_name == "balanced_cross_entropy":
            eps = 1.0
            loss = 0
            len_label = 0

            choose_coefs = [np.array([1., 0., 0., 0., 0., 0.]), np.array([0., 1., 0., 0., 0., 0.]),
                            np.array([0., 0., 1., 0., 0., 0.]), np.array([0., 0., 0., 1., 0., 0.]),
                            np.array([0., 0., 0., 0., 1., 0.]), np.array([0., 0., 0., 0., 0., 1.])]
            for choose_coef in choose_coefs:
                choose_label = tf.multiply(flat_labels, choose_coef)
                sum_loss = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=choose_label))
                sub_loss = tf.div(sum_loss, tf.reduce_sum(choose_label) + eps)
                len_label += tf.clip_by_value(tf.reduce_sum(choose_label), 0, 1)
                loss += sub_loss

            loss /= tf.cast(len_label, tf.float32)

        elif cost_name == "jaccard_distance":
            eps = 1e-10
            flat_slabels = flat_labels * self.choose_coef
            flat_sprediction = flat_prediction * self.choose_coef
            pred_positive = tf.reduce_sum(tf.square(flat_sprediction), axis=0)
            ref_positive = tf.reduce_sum(tf.square(flat_slabels), axis=0) + eps
            truth_postive = tf.reduce_sum(tf.multiply(flat_sprediction, flat_slabels), axis=0)
            dice_coef = truth_postive / (pred_positive + ref_positive - truth_postive)
            loss = 1 - tf.reduce_sum(dice_coef)

        elif cost_name == "dice_coef":
            eps = 1e-10
            flat_slabels = flat_labels * self.choose_coef
            flat_sprediction = flat_prediction * self.choose_coef
            pred_positive = tf.reduce_sum(tf.square(flat_sprediction)) + eps
            ref_positive = tf.reduce_sum(tf.square(flat_slabels)) + eps
            truth_postive = tf.reduce_sum(tf.multiply(flat_sprediction, flat_slabels)) + eps
            dice_coef = 2 * truth_postive / (pred_positive + ref_positive)
            loss = 1 - dice_coef

        return loss


    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

        return prediction

    def vote_strategy(self, model_paths, x_test, y_test):
        """
        it is the strategy for voting in the field of one-vs.-all method

        :param model_paths: five model paths for each class
        :param x_test: evaluation testing origin images
        :param y_test: evaluation testing labels
        :return: the confusion table
        """
        num_cls = len(model_paths)

        label_maps = []
        dict_label = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        evaluation_table = np.zeros((self.n_class, self.n_class), dtype=np.int32)

        for i in xrange(num_cls):
            model_path = model_paths[i]
            label_map = []
            with tf.Session() as sess:
                self.restore(sess, model_path)
                eval_size = x_test.shape[0]
                for j in xrange(eval_size):
                    y_dummy = np.empty([1,x_test.shape[1],x_test.shape[2], self.n_class])
                    x_test_feed = np.expand_dims(x_test[j], axis=0)
                    prediction = sess.run(self.predicter, feed_dict={self.x:x_test_feed, self.y:y_dummy, self.keep_prob:1.})
                    prediction = np.squeeze(prediction) # size [388,388,6]
                    label_map.append(prediction) # size  [27,388,388,6]
            label_maps.append(label_map) # list has 5 sublists each sublist has 27 subsublists and each subsublist is np.array

        for m in xrange(27):
            for n in xrange(5):
                x_0 = np.empty([388,388,6])
                x_0 = np.concatenate((x_0, label_maps[n][m]), axis=2)
            y = np.argmax(x_0, axis=2)
            z = np.mod(y,6)
            y_test_feed = np.expand_dims(y_test[m], axis=0)
            y_test_crop = util.crop_to_shape(y_test_feed, np.array([1,388,388,6]))
            elems, counts = np.unique(y_test_crop, return_counts=True)
            dict_y = dict(zip(elems, counts))
            y_test_crop_cmp = np.squeeze(y_test_crop)
            for elems_y, counts_y in dict_y.iteritems():
                dict_label[elems_y] += counts_y
                # find the predicted result of the groundtruth
                prediction_tmp = z[y_test_crop_cmp==elems_y]
                keys_label, counts_label = np.unique(prediction_tmp, return_counts=True)
                dict_prediction = dict(zip(keys_label, counts_label))
                # add the result to teh evaluation table
                for key_label, count_label in dict_prediction.iteritems():
                    evaluation_table[elems_y, key_label] += count_label
        label_table = np.zeros((self.n_class,1), dtype=np.int32)
        for dict_label, dict_count in dict_label.iteritems():
            label_table[dict_label] = dict_count

        return label_table, evaluation_table


    def evaluation(self, model_path, x_test, y_test):
        dict_label = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        evaluation_table = np.zeros((self.n_class, self.n_class), dtype=np.int32)
        with tf.Session() as sess:
            self.restore(sess, model_path)
            eval_size = x_test.shape[0]
            for i in xrange(eval_size):
                y_dummy = np.empty((1, x_test.shape[1], x_test.shape[2], self.n_class))
                x_test_feed = np.expand_dims(x_test[i], axis=0)
                prediction = sess.run(self.predict_res, feed_dict={self.x: x_test_feed, self.y: y_dummy, self.keep_prob: 1.})
                y_test_feed = np.expand_dims(y_test[i], axis=0)
                y_test_crop = util.crop_to_shape(y_test_feed, prediction.shape)
                elems, counts = np.unique(y_test_crop, return_counts=True)
                dict_y = dict(zip(elems, counts))
                prediction_cmp = np.squeeze(prediction)
                y_test_crop_cmp = np.squeeze(y_test_crop)
                for elems_y, counts_y in dict_y.iteritems():
                    dict_label[elems_y] += counts_y
                    # find the predicted result of the groundtruth
                    prediction_tmp = prediction_cmp[y_test_crop_cmp==elems_y]
                    keys_label, counts_label = np.unique(prediction_tmp, return_counts=True)
                    dict_prediction = dict(zip(keys_label, counts_label))
                    # add the result to teh evaluation table
                    for key_label, count_label in dict_prediction.iteritems():
                        evaluation_table[elems_y, key_label] += count_label
        label_table = np.zeros((self.n_class,1), dtype=np.int32)
        for dict_label, dict_count in dict_label.iteritems():
            label_table[dict_label] = dict_count

        return label_table, evaluation_table

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logger.info("Model restored from file: %s" % model_path)

class Trainer(object):
    """
    Trains a unet instance

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """

    def __init__(self, net, prediction_path, batch_size=2, optimizer="adam", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.prediction_path = prediction_path

    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=training_iters,
                                                        decay_rate=decay_rate,
                                                        staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost,
                                                                                global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.0001)
            self.learning_rate_node = tf.Variable(learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                     global_step=global_step)

        return optimizer

    def _get_upsilon_optimizer(self, training_iters, global_step):
        learning_rate = self.opt_kwargs.pop("learning_rate", 1e-7)
        self.upsilon_learning_rate_node = tf.Variable(learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.upsilon_learning_rate_node, **self.opt_kwargs).minimize(self.net.ban_cost,
                                                                                                                      global_step=global_step)
        return optimizer

    def _initialize(self, training_iters, output_path, restore):
        global_step = tf.Variable(0)

        self.gradients_0 = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_w_0)]))
        self.gradients_1 = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_w_0)]))
        self.gradients_2 = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_w_0)]))
        self.gradients_3 = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_w_0)]))
        self.gradients_4 = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_w_0)]))
        self.gradients_5 = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_w_0)]))

        if self.net.summaries:
            tf.summary.histogram('grads_class0', self.gradients_0)
            tf.summary.histogram('grads_class1', self.gradients_1)
            tf.summary.histogram('grads_class2', self.gradients_2)
            tf.summary.histogram('grads_class3', self.gradients_3)
            tf.summary.histogram('grads_class4', self.gradients_4)
            tf.summary.histogram('grads_class5', self.gradients_5)


        self.optimizer = self._get_optimizer(training_iters, global_step)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        prediction_path = self.prediction_path

        if not restore:
            logger.info("Removing '{:}'".format(prediction_path))
            shutil.rmtree(prediction_path, ignore_errors=True)
            logger.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(prediction_path):

            logger.info("Allocating '{:}'".format(prediction_path))
            os.makedirs(prediction_path)

        if not os.path.exists(output_path):
            logger.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, training_iters, epochs=100, dropout=0.9, restore=False, write_graph=False):
        """
        Lauches the training process
        training standard unet not including delta-net, all-in-one structure

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        """
        save_path = os.path.join(output_path, "model.ckpt")
        if epochs == 0:
            return save_path

        init = self._initialize(training_iters, output_path, restore)

        choose_dict = {1: np.array([0, 1, 0, 0, 0, 0]), 2: np.array([0, 0, 1, 0, 0, 0]),
                       3: np.array([0, 0, 0, 1, 0, 0]), 4: np.array([0, 0, 0, 0, 1, 0]),
                       5: np.array([0, 0, 0, 0, 0, 1])}
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            weight = np.array([1., 1., 1., 1., 1., 1.])

            pred_shape = np.array([2, 388, 388, 6])
            choose_coef = choose_dict[cls_idx]

            logger.info("Start optimization")

            for epoch in range(epochs):
                total_loss = 0

                for step in range(training_iters):
                    batch_y = data_provider.get_batchonehotlabel()
                    batch_x, batch_z = data_provider.get_batchdata()

                    _, loss, learning_rate = sess.run((self.optimizer, self.net.cost, self.learning_rate_node),
                                                      feed_dict={
                                                          self.net.x: batch_x,
                                                          self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                          self.net.weight: weight, self.net.choose_coef: choose_coef,
                                                          self.net.keep_prob: dropout
                                                      })
                    total_loss += loss
                logger.info(
                    "epoch {epoch} mean_dice_coefficient {J_2}".format(epoch=epoch, J_2=total_loss / training_iters))
                if (epoch + 1) % 10 == 0:
                    save_path = self.net.save(sess, save_path)

            logger.info("Optimization Finished!")

            return save_path

    def record_balanced_train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.9,
                              display_step=1, restore=False, write_graph=False):
        """
        Recording changes of gradients via balanced training

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        """
        save_path = os.path.join(output_path, "model.ckpt")
        if epochs == 0:
            return save_path
        global_upsilon_step = tf.Variable(0, trainable=False)
        self.upsilon_optimizer = self._get_upsilon_optimizer(training_iters, global_upsilon_step)
        init = self._initialize(training_iters, output_path, restore)

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            test_x, test_y = data_provider.get_batchdata()
            pred_shape = self.store_prediction(sess, test_x, test_y, "_init")

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)

            logger.info("Start optimization")

            ng0 = None
            ng1 = None
            ng2 = None
            ng3 = None
            ng4 = None
            ng5 = None

            for epoch in xrange(epochs):
                total_loss = 0
                for step in xrange(epoch * training_iters, (epoch + 1) * training_iters):
                    onehot_y = data_provider.get_batchonehotlabel()
                    batch_x, batch_y = data_provider.get_batchdata()
                    _, lr = sess.run((self.upsilon_optimizer, self.upsilon_learning_rate_node),
                                     feed_dict={self.net.x: batch_x,
                                                self.net.one_hot_y: util.crop_to_shape(onehot_y, pred_shape),
                                                self.net.keep_prob: dropout})
                    loss, g0, g1, g2, g3, g4, g5 = sess.run((self.net.cost, self.net.gradients_w_0,
                                                             self.net.gradients_w_1, self.net.gradients_w_2,
                                                             self.net.gradients_w_3, self.net.gradients_w_4,
                                                             self.net.gradients_w_5), feed_dict={self.net.x: batch_x,
                                                                                                 self.net.y: util.crop_to_shape(
                                                                                                     batch_y,
                                                                                                     pred_shape),
                                                                                                 self.net.one_hot_y: util.crop_to_shape(
                                                                                                     onehot_y,
                                                                                                     pred_shape),
                                                                                                 self.net.keep_prob: dropout})

                    if ng0 is None:
                        ng0 = [np.zeros_like(gradient) for gradient in g0]
                    for i in range(len(g0)):
                        ng0[i] = (ng0[i] * (1.0 - (1.0 / (step + 1)))) + (g0[i] / (step + 1))

                    norm_gradients = [np.linalg.norm(gradient) for gradient in ng0]
                    self.gradients_0.assign(norm_gradients)

                    if ng1 is None:
                        ng1 = [np.zeros_like(gradient) for gradient in g1]
                    for i in range(len(g1)):
                        ng1[i] = (ng1[i] * (1.0 - (1.0 / (step + 1)))) + (g1[i] / (step + 1))

                    norm_gradients = [np.linalg.norm(gradient) for gradient in ng1]
                    self.gradients_1.assign(norm_gradients)

                    if ng2 is None:
                        ng2 = [np.zeros_like(gradient) for gradient in g2]
                    for i in range(len(g2)):
                        ng2[i] = (ng2[i] * (1.0 - (1.0 / (step + 1)))) + (g2[i] / (step + 1))

                    norm_gradients = [np.linalg.norm(gradient) for gradient in ng2]
                    self.gradients_2.assign(norm_gradients)

                    if ng3 is None:
                        ng3 = [np.zeros_like(gradient) for gradient in g3]
                    for i in range(len(g3)):
                        ng3[i] = (ng3[i] * (1.0 - (1.0 / (step + 1)))) + (g3[i] / (step + 1))

                    norm_gradients = [np.linalg.norm(gradient) for gradient in ng3]
                    self.gradients_3.assign(norm_gradients)

                    if ng4 is None:
                        ng4 = [np.zeros_like(gradient) for gradient in g4]
                    for i in range(len(g4)):
                        ng4[i] = (ng4[i] * (1.0 - (1.0 / (step + 1)))) + (g4[i] / (step + 1))

                    norm_gradients = [np.linalg.norm(gradient) for gradient in ng4]
                    self.gradients_4.assign(norm_gradients)

                    if ng5 is None:
                        ng5 = [np.zeros_like(gradient) for gradient in g5]
                    for i in range(len(g5)):
                        ng5[i] = (ng5[i] * (1.0 - (1.0 / (step + 1)))) + (g5[i] / (step + 1))

                    norm_gradients = [np.linalg.norm(gradient) for gradient in ng5]
                    self.gradients_5.assign(norm_gradients)

                    total_loss += loss
                    summary_str = sess.run(self.summary_op, feed_dict={self.net.x: batch_x,
                                                                       self.net.y: util.crop_to_shape(batch_y,
                                                                                                      pred_shape),
                                                                       self.net.one_hot_y: util.crop_to_shape(onehot_y,
                                                                                                              pred_shape),
                                                                       self.net.keep_prob: 1})
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
            save_path = self.net.save(sess, save_path)
            logger.info("Optimization Finished!")

            return save_path

    def balanced_train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1, restore=False, write_graph=False):
        """
        Lauches the training process

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        """
        save_path = os.path.join(output_path, "model.ckpt")
        if epochs == 0:
            return save_path
        global_upsilon_step = tf.Variable(0, trainable=False)
        self.upsilon_optimizer = self._get_upsilon_optimizer(training_iters, global_upsilon_step)
        init = self._initialize(training_iters, output_path, restore)

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            test_x, test_y = data_provider.get_batchdata()
            pred_shape = self.store_prediction(sess, test_x, test_y, "_init")

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)

            logger.info("Start optimization")

            avg_gradients = None
            for epoch in range(130):
                total_loss = 0
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    batch_x, batch_y = data_provider.get_batchdata()

                    # Run optimization op (backprop)
                    _, loss, lr, gradients = sess.run((self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                                 self.net.keep_prob: dropout})

                    if avg_gradients is None:
                        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
                    for i in range(len(gradients)):
                        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step+1)))) + (gradients[i] / (step+1))

                    norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                    self.norm_gradients_node.assign(norm_gradients).eval()


                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)

                if epoch+1 == epochs:
                    save_path = self.net.save(sess, save_path)

            for epoch in xrange(epochs):
                total_loss = 0
                for step in xrange(epoch*training_iters, (epoch+1)*training_iters):
                    onehot_y = data_provider.get_batchonehotlabel()
                    batch_x, batch_y = data_provider.get_batchdata()
                    _, lr = sess.run((self.upsilon_optimizer,self.upsilon_learning_rate_node), feed_dict={self.net.x:batch_x, self.net.one_hot_y:util.crop_to_shape(onehot_y, pred_shape),
                                                                                                          self.net.keep_prob:dropout})
                    loss = sess.run(self.net.cost, feed_dict={self.net.x:batch_x, self.net.y:util.crop_to_shape(batch_y, pred_shape),
                                                              self.net.keep_prob:dropout})
                    total_loss += loss
                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
            save_path = self.net.save(sess, save_path)
            logger.info("Optimization Finished!")

            return save_path



    ### delta-net structure and training procedure ###

    def get_gd_optimizer(self, training_iters, global_step):
        self.nice_learning_rate_node = tf.convert_to_tensor(0.01)
        nice_gd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=
                                                              self.nice_learning_rate_node).minimize(self.net.cost,
                                                                                                     global_step=global_step)

        return nice_gd_optimizer

    def get_adam_optimizer(self, training_iters, global_step):
        learning_rate_node = tf.Variable(1e-4)
        return tf.train.AdamOptimizer(learning_rate=learning_rate_node).minimize(self.net.cost, global_step=global_step)

    def delta_initialize(self, training_iters, output_path, restore):
        """
        Initializer for delta-net
        """

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('accuracy', self.net.accuracy)


        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        prediction_path = self.prediction_path

        if not restore:

            logger.info("Removing '{:}'".format(prediction_path))
            shutil.rmtree(prediction_path, ignore_errors=True)

            logger.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(prediction_path):

            logger.info("Allocating '{:}'".format(prediction_path))
            os.makedirs(prediction_path)

        if not os.path.exists(output_path):

            logger.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def delta_train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.9, restore=False, write_graph=False):
        """
        Training delta-net

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        """
        save_path = os.path.join(output_path, "model.ckpt")
        global_step = tf.Variable(0, trainable=False)
        global_steps = tf.Variable(0, trainable=False)
        self.nice_gd_optimizer = self.get_gd_optimizer(training_iters, global_step)
        self.optimizer = self.get_adam_optimizer(training_iters, global_steps)
        if epochs == 0:
            return save_path
        init = self.delta_initialize(training_iters, output_path, restore)

        plt_weights = [[],[],[],[],[],[]]

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)
            sess.run(init)
            logger.info("Finishing Initialization and Starting Optimization")

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            delta_weight = np.array([1.,1.,1.,1.,1.,1.])

            shape_train = np.array([2,388,388,6])


            epochs_ = 5
            for epoch in xrange(epochs_):
                for step in xrange(epoch*training_iters, (epoch+1)*training_iters):
                    batch_y = data_provider.get_batchonehotlabel()
                    batch_x, batch_z = data_provider.get_batchdata()

                    [minimize, delta_gradients_w_0, delta_gradients_w_1, delta_gradients_w_2, delta_gradients_w_3,
                     delta_gradients_w_4, delta_gradients_w_5, rate] = sess.run((self.nice_gd_optimizer, self.net.gradients_w_0,
                                                                                 self.net.gradients_w_1, self.net.gradients_w_2,
                                                                                 self.net.gradients_w_3, self.net.gradients_w_4,
                                                                                 self.net.gradients_w_5, self.nice_learning_rate_node), feed_dict={
                        self.net.x:batch_x, self.net.y:util.crop_to_shape(batch_y, shape_train), self.net.weight:delta_weight, self.net.keep_prob:dropout
                    })
                    delta_gradients_w = [delta_gradients_w_0, delta_gradients_w_1, delta_gradients_w_2, delta_gradients_w_3, delta_gradients_w_4, delta_gradients_w_5]

                    feed_dict={self.net.x:batch_x, self.net.y:util.crop_to_shape(batch_y, shape_train), self.net.weight:delta_weight, self.net.keep_prob:dropout}

                    [J1_loss, J2_loss, delta_gradients_J2_theta] = sess.run((self.net.cost, self.net.J_2, self.net.gradients_J_2), feed_dict=feed_dict)
                    logger.info("step {step}: learning rate {rate}, weight {delta_weight}, J1_loss {J1_loss}, J2_loss {J2_loss}".format(step=step,
                                                                                                                                        rate=rate,
                                                                                                                                        delta_weight=delta_weight,
                                                                                                                                        J1_loss=J1_loss,
                                                                                                                                        J2_loss=J2_loss))
                    delta_weight = self.learning_delta_weight(delta_weight,delta_gradients_w, delta_gradients_J2_theta, rate)

                    plt_weight = 6*delta_weight/np.sum(delta_weight)
                    for plt_index in xrange(6):
                        plt_weights[plt_index].append(plt_weight[plt_index])

                    if J2_loss <= 0.03:
                        logger.info("Reaching optimization goal!")
                        save_path = self.net.save(sess, save_path)
                        return save_path
                if epoch+1 == epochs:
                    logger.info("Finishing training")
                    save_path = self.net.save(sess, save_path)
            # normalization of weights
            weight = 6*delta_weight/np.sum(delta_weight)
            logger.info("the final weight is {weight}".format(weight=weight))

            # training using the weights from delta-net
            sess.run(tf.global_variables_initializer())
            for epoch in xrange(epochs):
                total_loss = 0
                for step in xrange(epoch*training_iters, (epoch+1)*training_iters):
                    batch_y = data_provider.get_batchonehotlabel()
                    batch_x, batch_z = data_provider.get_batchdata()
                    feed_dict={self.net.x:batch_x, self.net.y:util.crop_to_shape(batch_y, shape_train),
                               self.net.weight:weight, self.net.keep_prob:dropout}
                    _, loss = sess.run((self.optimizer, self.net.cost), feed_dict=feed_dict)
                    total_loss += loss
                logger.info("epoch {epoch}: average loss:{mean_loss}".format(epoch=epoch, mean_loss=total_loss/training_iters))

            logger.info("finishing training!")

            """
            ### plot the weights
            plt.title('Weights update in Delta-Net')
            plt.xlabel('Steps')
            plt.ylabel('Weights')
            line_k, = plt.plot(plt_weights[0],'k')
            line_g, = plt.plot(plt_weights[1],'g')
            line_y, = plt.plot(plt_weights[2],'y')
            line_b, = plt.plot(plt_weights[3],'b')
            line_m, = plt.plot(plt_weights[4],'m')
            line_r, = plt.plot(plt_weights[5],'r')

            plt.legend([line_k,line_g,line_y,line_b,line_m,line_r],['background','not activated','partially activated',
                                                                    'fully activated','aggregate','questionable'])
            plt.show()
            plt.savefig("/home/students/nanjiang/Latex/images/fig-42.png")
            """
            save_path = self.net.save(sess, save_path)
        return save_path

    def learning_delta_weight(self, weight, gradients_w, gradients_J, rate):
        len_theta = len(gradients_J)
        for w_idx in xrange(self.net.n_class):
            for theta_idx in xrange(len_theta):
                weight[w_idx] += rate*rate*np.sum(gradients_J[theta_idx] * gradients_w[w_idx][theta_idx])
        return weight


    ### all-in-one training ###
    def allocate_data(self, data_provider, training_iters=54):
        """
        Allocating mini-batch data into five buckets according to their labels

        """
        cls_1 = []
        cls_2 = []
        cls_3 = []
        cls_4 = []
        cls_5 = []
        cls_all = [1,2,3,4,5]
        for step in xrange(training_iters):
            onehot = data_provider.get_batchonehotlabel()
            img, label = data_provider.get_batchdata()
            in_cls = np.unique(label)
            for cls_index in xrange(in_cls.shape[0]):
                if in_cls[cls_index] == 0:
                    continue
                elif in_cls[cls_index] == 1:
                    cls_1.append([img, onehot])
                elif in_cls[cls_index] == 2:
                    cls_2.append([img, onehot])
                elif in_cls[cls_index] == 3:
                    cls_3.append([img, onehot])
                elif in_cls[cls_index] == 4:
                    cls_4.append([img, onehot])
                elif in_cls[cls_index] == 5:
                    cls_5.append((img, onehot))
        if len(cls_1) == 0:
            logger.error("no class 1 training data")
            cls_all.remove(1)
        elif len(cls_2) == 0:
            logger.error("no class 2 training data")
            cls_all.remove(2)
        elif len(cls_3) == 0:
            logger.error("no class 3 training data")
            cls_all.remove(3)
        elif len(cls_4) == 0:
            logger.error("no class 4 training data")
            cls_all.remove(4)
        elif len(cls_5) == 0:
            logger.error("no class 5 training data")
            cls_all.remove(5)

        self.cls = [[], cls_1, cls_2, cls_3, cls_4, cls_5]
        return  max(len(cls_1), len(cls_2), len(cls_3), len(cls_4), len(cls_5)), cls_all

    def allinone_train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.9, restore=False, index_num=0, write_graph=False):
        training_steps, cls_all = self.allocate_data(data_provider, training_iters)
        save_path = os.path.join(output_path, "model.ckpt")
        if epochs == 0:
            return save_path

        init = self._initialize(training_iters, output_path, restore)

        choose_dict ={1:np.array([0,1,0,0,0,0]),2:np.array([0,0,1,0,0,0]),3:np.array([0,0,0,1,0,0]),4:np.array([0,0,0,0,1,0]),5:np.array([0,0,0,0,0,1])}
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            weight = np.array([1.,1.,1.,1.,1.,1.])
            pred_shape = [2, 388, 388, 6]

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)

            logger.info("Start optimization")

            for epoch in range(epochs):
                total_loss = 0

                for step in range(training_steps):
                    for cls_index in cls_all:
                        batch_x, batch_y = self.get_data_from_list(self.cls[cls_index], step)
                        choose_coef = choose_dict[cls_index].astype(np.float32)

                        _,loss, learning_rate = sess.run((self.optimizer, self.net.cost,self.learning_rate_node), feed_dict={
                            self.net.x: batch_x, self.net.y: util.crop_to_shape(batch_y, pred_shape), self.net.weight: weight, self.net.choose_coef:choose_coef, self.net.keep_prob: dropout
                        })
                        total_loss += loss
                logger.info("epoch {epoch} mean_dice_coefficient {J_2}".format(epoch=epoch, J_2=total_loss/(len(cls_all)*training_steps)))
                if (epoch+1) % 10 == 0:
                    save_path = self.net.save(sess, save_path)

            logger.info("Optimization Finished!")

            return save_path

    def get_data_from_list(self, datalist, index):
        len_list = len(datalist)
        if index < len_list:
            batch_x, batch_y = datalist[index]
        else:
            batch_x, batch_y = datalist[randint(0, len_list - 1)]
        return batch_x, batch_y


    def store_prediction(self, sess, batch_x, batch_y, name):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x,
                                                             self.net.y: batch_y,
                                                             self.net.keep_prob: 1.})
        # print(np.unique(np.argmax(prediction,axis=3)))
        pred_shape = prediction.shape
        loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x,
                                                       self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                       self.net.keep_prob: 1.})


        logger.info("Verification error= {:.1f}%, loss= {:.4f}".format(error_rate(prediction,
                                                                          util.crop_to_shape(batch_y,prediction.shape)),
                                                                          loss))

        img = util.combine_img_prediction(batch_x, batch_y, prediction)
        util.save_image(img, "%s/%s.jpg"%(self.prediction_path, name))

        return pred_shape

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logger.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                            self.net.cost,
                                                            self.net.accuracy,
                                                            self.net.predicter],
                                                           feed_dict={self.net.x: batch_x,
                                                                      self.net.y: batch_y,
                                                                      self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

        logger.info("Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                            loss,
                                                                                                            acc,
                                                                                                            error_rate(predictions, batch_y)))


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.squeeze(labels, axis=3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
