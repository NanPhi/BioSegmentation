# it is for the test of the functions written
from read_images import ImageReader, LabelReader

# testing read_images
batchsize = 4
path = "/home/students/nanjiang/Data/unit_test/label/*.tif"
labelreader = LabelReader(path, batchsize)
label = labelreader.read_batchlabels()
print(label.shape)
print(label[0].shape)
print(label[1].shape)











































