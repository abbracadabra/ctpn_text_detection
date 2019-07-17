import os
import numpy as np

rnn_units = 128
epochs = 999999
base_dir = os.getcwd()
log_dir = os.path.join(base_dir,'log')
train_img_dir = r'C:\xxx\mlt\mlt\image'
train_label_dir = r'C:\xxx\mlt\mlt\label'
model_dir = os.path.join(base_dir,'model','ctpn')
y_anchors = np.array([11, 16, 22, 32, 46, 66, 93, 134, 191, 273])
