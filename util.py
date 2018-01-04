# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 10, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from PIL import Image

def plot_prediction(x_test, y_test, prediction, save=False):
    import matplotlib
    import matplotlib.pyplot as plt
    
    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 3, figsize=(12,12), sharey=True, sharex=True)
    
    x_test = crop_to_shape(x_test, prediction.shape)
    y_test = crop_to_shape(y_test, prediction.shape)
    
    ax = np.atleast_2d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i])
        plt.colorbar(cax, ax=ax[i,0])
        cax = ax[i, 1].imshow(y_test[i, ..., 1])
        plt.colorbar(cax, ax=ax[i,1])
        pred = prediction[i, ..., 1]
        pred -= np.amin(pred)
        pred /= np.amax(pred)
        cax = ax[i, 2].imshow(pred)
        plt.colorbar(cax, ax=ax[i,2])
        if i==0:
            ax[i, 0].set_title("x")
            ax[i, 1].set_title("y")
            ax[i, 2].set_title("pred")
    fig.tight_layout()
    
    if save:
        fig.savefig(save)
    else:
        fig.show()
        plt.show()

def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    '''
    img -= np.amin(img)
    # img /= np.amax(img)

    np.divide(img, np.amax(img), out=img)
    
    img *= 255
    '''
    img *= 40
    return img

def to_rgb_nan(img):
    rgb_img = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
    rgb_img[img==1] = np.array([0,255,0])
    rgb_img[img==2] = np.array([255,255,0])
    rgb_img[img==3] = np.array([0,0,255])
    rgb_img[img==4] = np.array([127,0,127])
    rgb_img[img==5] = np.array([255,0,0])
    rgb_img = np.squeeze(rgb_img)
    return rgb_img


def to_rgb_1(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    # img /= np.amax(img)

    np.divide(img, np.amax(img), out=img)

    img *= 255

    return img

def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    # print("it is the crop_to_shape")
    # print(data.shape)
    # print(shape)
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    # print(offset0, offset1)
    return data[:, offset0:(-offset0), offset1:(-offset1)]

def crop_to_shape_map(data, shape):
    offset0 = (data.shape[2] - shape[1])//2
    offset1 = (data.shape[3] - shape[2])//2
    return data[:,:, offset0:-offset0, offset1:-offset1]

def combine_img_prediction(data, gt, pred):
    """
    Combines the data, ground truth and the prediction into one rgb image
    
    :param data: the data tensor
    :param gt: the ground truth tensor
    :param pred: the prediction tensor
    
    :returns img: the concatenated rgb image 
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    # print(to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)).shape)
    ### TODO SAME ###
    '''
    img = np.concatenate((to_rgb(data.reshape(-1, ny, ch)), 
                          to_rgb(gt[..., 1].reshape(-1, ny, 1)), 
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)

    img = np.concatenate((to_rgb_1(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)),
                          to_rgb(crop_to_shape(np.squeeze(gt, axis=3), pred.shape).reshape(-1, ny, 1)),
                          to_rgb(np.argmax(pred, axis=3).reshape(-1, ny, 1))), axis=1)
    '''
    img = np.concatenate((to_rgb_1(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)),
                          to_rgb_nan(crop_to_shape(np.squeeze(gt, axis=3), pred.shape).reshape(-1, ny, 1)),
                          to_rgb_nan(np.argmax(pred, axis=3).reshape(-1, ny, 1))), axis=1)

    return img

def combine_img(data):
    """
    Combines the data, ground truth and the prediction into one rgb image

    :param data: the data tensor
    :param gt: the ground truth tensor
    :param pred: the prediction tensor

    :returns img: the concatenated rgb image
    """
    '''
    ny = pred.shape[2]
    ch = data.shape[3]
    # print(to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)).shape)
    ### TODO SAME ###

    img = np.concatenate((to_rgb(data.reshape(-1, ny, ch)),
                          to_rgb(gt[..., 1].reshape(-1, ny, 1)),
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)

    img = np.concatenate((to_rgb_1(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)),
                          to_rgb(crop_to_shape(np.squeeze(gt, axis=3), pred.shape).reshape(-1, ny, 1)),
                          to_rgb(np.argmax(pred, axis=3).reshape(-1, ny, 1))), axis=1)

    img = np.concatenate((to_rgb_nan(crop_to_shape(np.squeeze(data, axis=3), pred.shape).reshape(-1, ny, ch)),
                          to_rgb_nan(pred.reshape(-1, ny, 1))), axis=1)
    '''
    img = to_rgb_1(crop_to_shape(data, np.array([1,388,388,1])).reshape(-1, 388, 1))

    return img

def save_image(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300,300], quality=90)

def save_image_Nan(img, path):
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'PNG', dpi=[300,300], quality=90)

