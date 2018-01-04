import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from read_images import DataReader
from read_images import TestdataReader
import U_Net
import util
import numpy as np
import logging



data_path = "/home/students/nanjiang/Origin_Bonanza/*_labelled_JOC.png"

logging.basicConfig(filename='/home/students/nanjiang/Log_Bonanza/Alpha/myapp1.log', level=logging.INFO, filemode='w', format='%(asctime)s %(message)s')
prediction_path = "/home/students/nanjiang/Pred_Bonanza/Alpha_1"
output_path = "/home/students/nanjiang/Model_Bonanza/Alpha_1"
txt_path = "/home/students/nanjiang/Log_Bonanza/Alpha/Alpha_1.out"

batchsize = 2
# batchsize = 1
model_path = output_path + "/model.ckpt"

data_provider =  DataReader(data_path, batchsize)
data_provider.get_origin(0)
# here is different configurations of experiment
unet = U_Net.Unet()

trainer = U_Net.Trainer(unet, prediction_path, optimizer="adam")


path = trainer.train(data_provider, output_path, dropout=0.9, training_iters=54, epochs=180)

IoU_unet = np.zeros([6,1])
label_table_unet, evaluation_table_unet = unet.evaluation(model_path, data_provider.images_validate, data_provider.labels_validate)
for cls_index in xrange(6):
    tp = evaluation_table_unet[cls_index,cls_index]
    fp = np.sum(evaluation_table_unet[:, cls_index]) - tp
    fn = np.sum(evaluation_table_unet[cls_index, :]) - tp
    IoU_unet[cls_index] = 1.*tp/(tp+fp+fn)
print(IoU_unet)


with open(txt_path, 'w') as f:
    f.write("conventional unet\n")
    np.savetxt(f, label_table_unet, fmt='%-12d', newline='')
    f.write("\n")
    f.write("------------------------------------------------------------------\n")
    np.savetxt(f, evaluation_table_unet, fmt='%-12d', footer='================================================================')
    np.savetxt(f, IoU_unet, fmt='%-12.5f', newline='')

