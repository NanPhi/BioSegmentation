from tf_unet import unet, util, image_util

# preparing data loading
data_provider = image_util.ImageDataProvider("/home/students/nanjiang/Data/Data_Image_1.0/*.tif")

# setup & training
output_path = "/home/students/nanjiang/PycharmProjects/Nan_unet/prediction"
net = unet.Unet(layers=3, features_root=64, channels=1, n_class=7)
trainer = unet.Trainer(net)
path = trainer.train(data_provider, output_path, training_iters=32, epochs=100)

# verification

prediction = net.predict(path, data)
unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))

img = util.combine_img_prediction(data, label, prediction)
util.save_image(img, "prediction.jpg")


