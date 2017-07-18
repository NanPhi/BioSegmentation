from PIL import Image
import numpy as np
import glob
# just for the test whether I can use tensorflow to replace numpy
import tensorflow as tf

class ImageReader(object):
    image_mean = 1509
    image_std = 1297
    # for the specific greyscale images
    n_channels = 1 
    def __init__(self, path, batchsize):
        # the filename is absolute filepath name such as '/home/Data/%name%.png'
        self.filename = self.get_filename(path)
        self.batchsize = batchsize
        self.offset = 0

    def get_filename(self, path):
        # params
        # path can be either path to image 
        # path should have a form like /home/data/*.png
        filename = glob.glob(path)
        return filename

    def read_batchimages(self):
        offset = self.offset
        batchsize = self.batchsize
        filename = self.filename[offset:offset+batchsize]
        firstimage = np.array(Image.open(filename[0]), dtype=np.float32)
        nx = firstimage.shape[0]
        ny = firstimage.shape[1]
        assert nx > 0 and ny > 0, "wrong images" 
        # batch_buffer is for images of different sizes
        batch_buffer = [firstimage]
        for offset in xrange(1, batchsize):
            image = np.array(Image.open(filename[offset]), dtype=np.float32)
            ix = image.shape[0]
            iy = image.shape[1]
            nx = max(ix, nx)
            ny = max(iy, ny)
            batch_buffer.append((image-self.mean)/self.image_std)
        # calibration size is the maximum of the both axis of batch images
        cal_size = [nx, ny]
        batch_images = resize_images(batch_buffer, cal_size)
        self.offset = offset
        return batch_images


    def resize_images(self, batch_buffer, cal_size):
        # params 
        # image: original input image with smaller size prepared to add zeros to be calibrated
        # cal_size:  the largest size in one batch to be recognized as calibrated size
        batchsize = self.batchsize
        batch_images = np.zeros([batchsize, cal_size[0], cal_size[1], self.n_channels])        
        for i in xrange(batchsize):
            image = batch_buffer[i]
            nx = image.shape[0]
            ny = image.shape[1]
            # zero padding to the margin
            batch_images[i, 0:nx, 0:ny, 0] =  image
        return batch_images

class LabelReader(ImageReader):
    n_class = 7
    def read_batchlabels(self):
        offset = self.offset
        batchsize = self.batchsize
        filename = self.filename[offset:offset+batchsize]
        firstlabel = np.array(Image.open(filename[0]), dtype=np.float32)
        nx = firstlabel.shape[0]
        ny = firstlabel.shape[1]
        assert nx > 0 and ny > 0, "wrong labels"
        batch_buffer = [firstlabel]
        for offset in xrange(1, batchsize):
            label = np.array(Image.open(filename[offset]), dtype=np.float32)
            ix = label.shape[0]
            iy = label.shape[1]
            nx = max(ix, nx)
            ny = max(iy, ny)
            batch_buffer.append(label)
        cal_size = [nx. ny]
        # batch_labels has a size of [batchsize, width, height]
        batch_labels = resize_labels(batch_buffer, cal_size)
        ### CODE TO DO ###
        batch_labels = act_onehotkey(batch_labels)
        return batch_labels

    def resize_labels(self, batch_buffer, cal_size):
        # resize the label inputs
        batchsize = self.batchsize
        batch_labels = np.zeros([batchsize, cal_size[0], cal_size[1]])
        for i in xrange(batchsize):
            label = batch_buffer[i]
            nx = label.shape[0]
            ny = label.shape[1]
            batch_labels[i, 0:nx, 0:ny] = label/40
        return batch_labels

    def act_onehotkey(self, labels):
        # activate the one hot key function
        return tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.n_class)

















































