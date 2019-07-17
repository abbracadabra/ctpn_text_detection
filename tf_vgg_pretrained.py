import tensorflow as tf
import keras
from config import *

from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet', include_top=False)
tf.train.export_meta_graph(filename='./vgg16/vgg16.meta')
saver = tf.train.Saver()
saver.save(keras.backend.get_session(), save_path='./vgg16/vgg16',write_meta_graph=False)






