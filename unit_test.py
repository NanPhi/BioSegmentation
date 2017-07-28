# it is for the test of the functions written
from read_images import ImageReader, LabelReader
from create_unet import Unet, Trainer
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "training", "Mode read_image read_labels training")

if FLAGS.mode == "read_image":
    # testing read_images
    batchsize = 4
    path = "/home/students/nanjiang/Data/unit_test/label/*.tif"
    labelreader = LabelReader(path, batchsize)
    label = labelreader.read_batchlabels()
    print(label.shape)
    print(label[0].shape)
    print(label[1].shape)

if FLAGS.mode == "training":
    # raw data preparation
    batchsize = 4
    img_path = "/home/students/nanjiang/Data/unit_test/image/*.tif"
    imagereader = ImageReader(img_path, batchsize)
    image = imagereader.read_batchimages()
    label_path = "/home/students/nanjiang/Data/unit_test/label/*.tif"
    labelreader = LabelReader(label_path, batchsize)
    label = labelreader.read_batchlabels()

    # structure of NN generation
    unet = Unet(image.shape, 7)
    trainer = Trainer(unet.img_seg, n_class=7, learning_rate=1e-6)

    # starting to train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(trainer.train, feed_dict={unet.image:image, trainer.label:label})











































