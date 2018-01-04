import os
from PIL import Image
import numpy as np
import glob
# just for the test whether I can use tensorflow to replace numpy
import tensorflow as tf

class DataReader(object):
    # for the specific grayscale images
    n_channels = 1
    n_classes = 6
    # path should be annotated with *_label.png
    def __init__(self, path, batchsize):
        # the filename is absolute filepath name such as '/home/Data/subdirectories/%name%.png'
        self.filenames = self.get_filenames(path)
        self.batchsize = batchsize
        self.offset = 0
        self.offset_eval = 0
        self.labels_all = self.get_labels()
        self.images_all = self.get_images(method="Normalization")
        self.get_benchmark()
        # self.split_data()
        # retrieve the data from all images to attain groundtruth
        # self.retrieve()

    def get_origin(self, fold_index):
        """
        this is for 5-fold cross-validation of origin data with 135 pairs
        :param fold_index: the index of fold
        :return: split of training and validation set
        """
        if fold_index == 0:
            self.labels = self.labels_all[27:135]
            self.labels_validate = self.labels_all[0:27]
            self.one_hot_labels = self.one_hot_labels_all[27:135]
            self.one_hot_labels_validate = self.one_hot_labels_all[0:27]
            # self.maps = self.maps_all[perm[:split_index]]
            self.images = self.images_all[27:135]
            self.images_validate = self.images_all[0:27]
            self.train_num = self.labels.shape[0]
            self.eval_num = self.labels_validate.shape[0]
            self.images_fcn = self.images_fcn_all[27:135]
            self.images_fcn_validate = self.images_fcn_all[0:27]

        if fold_index == 1:
            self.labels = np.concatenate((self.labels_all[0:27], self.labels_all[54:135]), axis=0)
            self.labels_validate = self.labels_all[27:54]
            self.one_hot_labels = np.concatenate((self.one_hot_labels_all[0:27], self.one_hot_labels_all[54:135]), axis=0)
            self.one_hot_labels_validate = self.one_hot_labels_all[27:54]
            # self.maps = self.maps_all[perm[:split_index]]
            self.images = np.concatenate((self.images_all[0:27], self.images_all[54:135]), axis=0)
            self.images_validate = self.images_all[27:54]
            self.train_num = self.labels.shape[0]
            self.eval_num = self.labels_validate.shape[0]
            self.images_fcn = np.concatenate((self.images_fcn_all[0:27], self.images_fcn_all[54:135]), axis=0)
            self.images_fcn_validate = self.images_fcn_all[27:54]

        if fold_index == 2:
            self.labels = np.concatenate((self.labels_all[0:54], self.labels_all[81:135]), axis=0)
            self.labels_validate = self.labels_all[54:81]
            self.one_hot_labels = np.concatenate((self.one_hot_labels_all[0:54], self.one_hot_labels_all[81:135]), axis=0)
            self.one_hot_labels_validate = self.one_hot_labels_all[54:81]
            # self.maps = self.maps_all[perm[:split_index]]
            self.images = np.concatenate((self.images_all[0:54], self.images_all[81:135]), axis=0)
            self.images_validate = self.images_all[54:81]
            self.train_num = self.labels.shape[0]
            self.eval_num = self.labels_validate.shape[0]
            self.images_fcn = np.concatenate((self.images_fcn_all[0:54], self.images_fcn_all[81:135]), axis=0)
            self.images_fcn_validate = self.images_fcn_all[54:81]

        if fold_index == 3:
            self.labels = np.concatenate((self.labels_all[0:81], self.labels_all[108:135]), axis=0)
            self.labels_validate = self.labels_all[81:108]
            self.one_hot_labels = np.concatenate((self.one_hot_labels_all[0:81], self.one_hot_labels_all[108:135]), axis=0)
            self.one_hot_labels_validate = self.one_hot_labels_all[81:108]
            # self.maps = self.maps_all[perm[:split_index]]
            self.images = np.concatenate((self.images_all[0:81], self.images_all[108:135]), axis=0)
            self.images_validate = self.images_all[81:108]
            self.train_num = self.labels.shape[0]
            self.eval_num = self.labels_validate.shape[0]
            self.images_fcn = np.concatenate((self.images_fcn_all[0:81], self.images_fcn_all[108:135]), axis=0)
            self.images_fcn_validate = self.images_fcn_all[81:108]

        if fold_index == 4:
            self.labels = self.labels_all[0:108]
            self.labels_validate = self.labels_all[108:135]
            self.one_hot_labels = self.one_hot_labels_all[0:108]
            self.one_hot_labels_validate = self.one_hot_labels_all[108:135]
            # self.maps = self.maps_all[perm[:split_index]]
            self.images = self.images_all[0:108]
            self.images_validate = self.images_all[108:135]
            self.train_num = self.labels.shape[0]
            self.eval_num = self.labels_validate.shape[0]
            self.images_fcn = self.images_fcn_all[0:108]
            self.images_fcn_validate = self.images_fcn_all[108:135]

    def get_fold(self, fold_index):
        """
        this is for 5-fold cross-validation
        :param fold_index: the index of fold
        :return: split of training and validation set
        """
        if fold_index == 0:
            self.labels = self.labels_all[47:235]
            self.labels_validate = self.labels_all[0:47]
            self.one_hot_labels = self.one_hot_labels_all[47:235]
            self.one_hot_labels_validate = self.one_hot_labels_all[0:47]
            # self.maps = self.maps_all[perm[:split_index]]
            self.images = self.images_all[47:235]
            self.images_validate = self.images_all[0:47]
            self.train_num = self.labels.shape[0]
            self.eval_num = self.labels_validate.shape[0]
            self.images_fcn = self.images_fcn_all[47:235]
            self.images_fcn_validate = self.images_fcn_all[0:47]

        if fold_index == 1:
            self.labels = np.concatenate((self.labels_all[0:47], self.labels_all[94:235]), axis=0)
            self.labels_validate = self.labels_all[47:94]
            self.one_hot_labels = np.concatenate((self.one_hot_labels_all[0:47], self.one_hot_labels_all[94:235]), axis=0)
            self.one_hot_labels_validate = self.one_hot_labels_all[47:94]
            # self.maps = self.maps_all[perm[:split_index]]
            self.images = np.concatenate((self.images_all[0:47], self.images_all[94:235]), axis=0)
            self.images_validate = self.images_all[47:94]
            self.train_num = self.labels.shape[0]
            self.eval_num = self.labels_validate.shape[0]
            self.images_fcn = np.concatenate((self.images_fcn_all[0:47], self.images_fcn_all[94:235]), axis=0)
            self.images_fcn_validate = self.images_fcn_all[47:94]

        if fold_index == 2:
            self.labels = np.concatenate((self.labels_all[0:94], self.labels_all[141:235]), axis=0)
            self.labels_validate = self.labels_all[94:141]
            self.one_hot_labels = np.concatenate((self.one_hot_labels_all[0:94], self.one_hot_labels_all[141:235]), axis=0)
            self.one_hot_labels_validate = self.one_hot_labels_all[94:141]
            # self.maps = self.maps_all[perm[:split_index]]
            self.images = np.concatenate((self.images_all[0:94], self.images_all[141:235]), axis=0)
            self.images_validate = self.images_all[94:141]
            self.train_num = self.labels.shape[0]
            self.eval_num = self.labels_validate.shape[0]
            self.images_fcn = np.concatenate((self.images_fcn_all[0:94], self.images_fcn_all[141:235]), axis=0)
            self.images_fcn_validate = self.images_fcn_all[94:141]

        if fold_index == 3:
            self.labels = np.concatenate((self.labels_all[0:141], self.labels_all[188:235]), axis=0)
            self.labels_validate = self.labels_all[141:188]
            self.one_hot_labels = np.concatenate((self.one_hot_labels_all[0:141], self.one_hot_labels_all[188:235]), axis=0)
            self.one_hot_labels_validate = self.one_hot_labels_all[141:188]
            # self.maps = self.maps_all[perm[:split_index]]
            self.images = np.concatenate((self.images_all[0:141], self.images_all[188:235]), axis=0)
            self.images_validate = self.images_all[141:188]
            self.train_num = self.labels.shape[0]
            self.eval_num = self.labels_validate.shape[0]
            self.images_fcn = np.concatenate((self.images_fcn_all[0:141], self.images_fcn_all[188:235]), axis=0)
            self.images_fcn_validate = self.images_fcn_all[141:188]

        if fold_index == 4:
            self.labels = self.labels_all[0:188]
            self.labels_validate = self.labels_all[188:235]
            self.one_hot_labels = self.one_hot_labels_all[0:188]
            self.one_hot_labels_validate = self.one_hot_labels_all[188:235]
            # self.maps = self.maps_all[perm[:split_index]]
            self.images = self.images_all[0:188]
            self.images_validate = self.images_all[188:235]
            self.train_num = self.labels.shape[0]
            self.eval_num = self.labels_validate.shape[0]
            self.images_fcn = self.images_fcn_all[0:188]
            self.images_fcn_validate = self.images_fcn_all[188:235]

    def get_filenames(self, path):
        # params
        # path can be either path to image 
        # path should have a form like /home/data/*.tif
        filenames = glob.glob(path)
        return filenames

    def update_offset(self):
        self.offset += 1
        if self.offset == self.train_num:
            self.offset = 0
            ### no shuffle for U-Net Benchmark###
            """
            perm = np.arange(self.train_num)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            """

    def update_offset_eval(self):
        self.offset_eval += 1
        if self.offset_eval == self.eval_num:
            self.offset_eval = 0
            ### no shuffle for U-Net Benchmark###
            """
            perm = np.arange(self.eval_num)
            np.random.shuffle(perm)
            self.images_validate = self.images_validate[perm]
            self.labels_validate = self.labels_validate[perm]
            """

    def update_offset_fcn(self):
        self.offset += 1
        if self.offset == self.train_num:
            self.offset = 0
            ### no shuffle for U-Net Benchmark ###
            """
            perm = np.arange(self.train_num)
            np.random.shuffle(perm)
            self.images_fcn = self.images_fcn[perm]
            self.labels = self.labels[perm]
            """

    def update_offset_fcn_eval(self):
        self.offset_eval += 1
        if self.offset_eval == self.eval_num:
            self.offset_eval = 0
            ### no shuffle for U-Net Benchmark###
            """
            perm = np.arange(self.eval_num)
            np.random.shuffle(perm)
            self.images_fcn_validate = self.images_fcn_validate[perm]
            self.labels_validate = self.labels_validate[perm]
            """

    def get_nextonehotlabel(self):
        offset = self.offset
        onehotlabel = self.one_hot_labels[offset]
        return onehotlabel

    def get_nextmap(self):
        offset = self.offset
        map = self.maps[offset]
        return map

    def get_nextdata(self):
        offset = self.offset
        image = self.images[offset]
        label = self.labels[offset]
        self.update_offset()
        return image, label

    def get_nextdata_eval(self):
        offset = self.offset_eval
        image_eval = self.images_validate[offset]
        label_eval = self.labels_validate[offset]
        self.update_offset_eval()
        return image_eval, label_eval

    def get_nextdata_fcn(self):
        offset = self.offset
        image_fcn = self.images_fcn[offset]
        label = self.labels[offset]
        self.update_offset_fcn()
        return image_fcn, label

    def get_nextdata_fcn_eval(self):
        offset = self.offset_eval
        image_fcn_eval = self.images_fcn_validate[offset]
        label = self.labels[offset]
        self.update_offset_fcn_eval()
        return image_fcn_eval, label

    def get_benchmark(self):
        labels = []
        images = []
        datapath = "/home/students/nanjiang/Benchmark_Bonanza/*label.png"
        for labelname in glob.glob(datapath):
            label_RGB = np.array(Image.open(labelname), dtype=np.uint8)
            label = self.get_label_from_rgb(label_RGB)
            labels.append(label)
            image = np.array(Image.open(labelname.replace("label", "image")),dtype=np.float32)
            image_min = np.amin(image)
            image_max = np.amax(image)
            image_scale = image_max - image_min
            image -= image_min
            image /= image_scale
            images.append(image)
        labels = np.array(labels)
        labels = labels.reshape(5,188,188,1)
        one_hot_labels = tf.one_hot(indices=labels, depth=self.n_classes)
        with tf.Session() as sess:
            one_hot_labels = sess.run(one_hot_labels)
        self.labels_benchmark = np.squeeze(one_hot_labels)
        images = np.array(images)
        images = images.reshape(5,188,188,1)
        self.images_benchmark = images


    def get_images(self, method="Standardization", img_type=np.float32):
        images = []
        images_names = [name.replace('labelled_JOC', 'origin') for name in self.filenames]

        for filename in images_names:
            image = np.array(Image.open(filename), dtype=img_type)
            # normalize the image in order to scale the loss in a acceptable limit
            if method == "Standardization":
                image_mean = np.mean(image)
                image_std = np.std(image)
                image -= image_mean
                image /= image_std
            elif method == "Normalization":
                image_min = np.amin(image)
                image_max = np.amax(image)
                image_scale = image_max - image_min
                image -= image_min
                image /= image_scale
            images.append(image)

        images = np.array(images)
        images = images.reshape((self.datasize, self.img_size_x, self.img_size_y, self.n_channels))
        self.images_fcn_all = np.repeat(images, 3, axis=3)
        return images

    def get_labels(self, label_type=np.uint8):
        labels = []
        datasize = 0
        for filename in self.filenames:
            label_RGB = np.array(Image.open(filename), dtype=label_type)
            label = self.get_label_from_rgb(label_RGB)
            labels.append(label)
            datasize += 1
        self.datasize = datasize
        self.img_size_x = labels[0].shape[0]
        self.img_size_y = labels[0].shape[1]
        # using the sparse entropy later
        labels = np.array(labels)
        labels = labels.reshape(self.datasize, self.img_size_x, self.img_size_y, 1)
        self.one_hot_encoding(labels)
        # self.maps_all = self.get_all_maps(self.one_hot_labels_all)
        return labels

    def get_batchmap(self):
        batchmap = np.zeros((self.batchsize, self.n_classes-1, self.img_size_x, self.img_size_y, self.n_classes))
        for i in xrange(self.batchsize):
            batchmap[i] = self.get_nextmap()
        return batchmap

    def get_batchonehotlabel(self):
        batchonehotlabel = np.zeros((self.batchsize, self.img_size_x, self.img_size_y, self.n_classes))
        for i in xrange(self.batchsize):
            batchonehotlabel[i] = self.get_nextonehotlabel()
        return batchonehotlabel

    def get_batchdata(self):
        batchimg = np.zeros((self.batchsize, self.img_size_x, self.img_size_y, self.n_channels))
        batchlabel = np.zeros((self.batchsize, self.img_size_x, self.img_size_y, 1))
        for i in xrange(self.batchsize):
            batchimg[i], batchlabel[i] = self.get_nextdata()
        return batchimg, batchlabel

    def get_batchdata_eval(self):
        batchimgeval = np.zeros((self.batchsize, self.img_size_x, self.img_size_y, self.n_channels))
        batchlabeleval = np.zeros((self.batchsize, self.img_size_x, self.img_size_y, 1))
        for i in xrange(self.batchsize):
            batchimgeval[i], batchlabeleval[i] = self.get_nextdata_eval()
        return batchimgeval, batchlabeleavl

    def get_batchdata_fcn(self):
        batchimgfcn = np.zeros((self.batchsize, self.img_size_x, self.img_size_y, 3))
        batchlabel = np.zeros((self.batchsize, self.img_size_x, self.img_size_y, 1))
        for i in xrange(self.batchsize):
            batchimgfcn[i], batchlabel[i] = self.get_nextdata_fcn()
        return batchimgfcn, batchlabel

    def get_batchdata_fcn_eval(self):
        batchimgfcneval = np.zeros((self.batchsize, self.img_size_x, self.img_size_y, 3))
        batchlabeleval = np.zeros((self.batchsize, self.img_size_x, self.img_size_y, 1))
        for i in xrange(self.batchsize):
            batchimgfcneval[i], batchlabeleval[i] = self.get_nextdata_fcn_eval()
        return batchimgfcneval, batchlabeleval

    def get_label_from_rgb(self, label_RGB):
        sparse_label = np.zeros((label_RGB.shape[0], label_RGB.shape[1], 1), dtype=np.uint8)

        # get red, green and blue from their feature
        index_red = label_RGB[:,:,0] == 255
        index_green = label_RGB[:,:,1] == 255
        index_blue = label_RGB[:,:,2] == 255
        sparse_label[index_red] = 5
        sparse_label[index_green] = 1
        sparse_label[index_blue] = 3

        # get yellow and purple from sum of pixels
        label_RGB_sum = np.sum(label_RGB, axis=2)
        index_yellow = label_RGB_sum==510 # yellow (255,255,0)
        sparse_label[index_yellow] = 2
        index_purple = label_RGB_sum==254 # purple (127,0,127)
        sparse_label[index_purple] = 4

        return sparse_label

    def get_random_batch_fcn_eval(self):
        index = np.random.randint(0, self.eval_num, size=[self.batchsize]).tolist()
        return self.images_fcn_validate[index], self.labels_validate[index]

    def split_data(self):
        perm = np.arange(self.datasize)
        np.random.shuffle(perm)
        # split the data into 0.8:0.2 to train and validate
        split_index = int(self.datasize*0.8)
        self.labels = self.labels_all[perm[:split_index]]
        self.labels_validate = self.labels_all[perm[split_index:]]
        self.one_hot_labels = self.one_hot_labels_all[perm[:split_index]]
        self.one_hot_labels_validate = self.one_hot_labels_all[perm[split_index:]]
        # self.maps = self.maps_all[perm[:split_index]]
        self.images = self.images_all[perm[:split_index]]
        self.images_validate = self.images_all[perm[split_index:]]
        self.train_num = self.labels.shape[0]
        self.eval_num = self.labels_validate.shape[0]
        self.images_fcn = self.images_fcn_all[perm[:split_index]]
        self.images_fcn_validate = self.images_fcn_all[perm[split_index:]]

    def retrieve(self):
        self.labels = self.labels_all
        self.one_hot_labels = self.one_hot_labels_all
        # self.maps = self.maps_all
        self.images = self.images_all
        self.images_fcn = self.images_fcn_all
        self.train_num = self.labels.shape[0]
        self.eval_num = 0

    def one_hot_encoding(self, labels):
        one_hot_labels = tf.one_hot(indices=labels, depth=self.n_classes)
        with tf.Session() as sess:
            one_hot_labels_all = sess.run(one_hot_labels)
        self.one_hot_labels_all = np.squeeze(one_hot_labels_all)

    def get_all_maps(self, one_hot_labels):
        all_maps = np.zeros([one_hot_labels.shape[0], self.n_classes-1, one_hot_labels.shape[1], one_hot_labels.shape[2], self.n_classes])
        for cls_index in xrange(self.n_classes-1):
            all_maps[:,cls_index,:,:,cls_index+1] = one_hot_labels[...,cls_index+1]
        return all_maps

class TestdataReader(object):
    def __init__(self, test_path, save_path):
        self.x_test = self.get_data(test_path, save_path)

    def get_data(self, test_path, save_path):
        file_names = glob.glob(test_path)
        images = []
        imagenames = []
        for filename in file_names:
            imagename = os.path.basename(filename)
            imagenames.append(imagename)
            image = np.array(Image.open(filename), dtype=np.float32)
            """
            image_crop = image[92:480, 92:480]
            Image.fromarray(image_crop.round().astype(np.int32)).save(save_path+imagename)
            """
            # normalize the image in order to scale the loss in a acceptable limit
            image_min = np.amin(image)
            image_max = np.amax(image)
            image_scale = image_max - image_min
            image -= image_min
            image /= image_scale

            images.append(image)


        self.imagenames = imagenames
        img_size = len(images)
        images = np.array(images)
        images = images.reshape((img_size, 572, 572, 1))
        self.images_fcn_all = np.repeat(images, 3, axis=3)
        return images
