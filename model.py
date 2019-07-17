import tensorflow as tf
from config import *
import numpy as np

tf.train.import_meta_graph("./vgg16/vgg16.meta")
input_ph = tf.get_default_graph().get_tensor_by_name("input_1:0")
vgg16_output = tf.get_default_graph().get_tensor_by_name("block5_conv3/Relu:0")
vgg16_output = tf.stop_gradient(vgg16_output)
vggsaver = tf.train.Saver()

vgg16_output = tf.layers.dropout(vgg16_output,rate=0.75)
#vgg16_output = tf.layers.batch_normalization(vgg16_output,fused=False)
conv6 = tf.layers.conv2d(inputs=vgg16_output,filters=256,kernel_size=3,padding='SAME',use_bias=False)
conv6 = tf.layers.batch_normalization(conv6,fused=False)
conv6 = tf.contrib.layers.bias_add(conv6)
conv6 = tf.nn.leaky_relu(conv6)
conv6 = tf.layers.dropout(conv6)
featuremapshape = tf.shape(conv6)
N = featuremapshape[0]
H = featuremapshape[1]
W = featuremapshape[2]
C = featuremapshape[3]
# blstm_input = tf.reshape(conv6,[N*H,W,256])

# lstm_fw_cell = tf.contrib.rnn.LSTMCell(rnn_units, state_is_tuple=True)
# lstm_bw_cell = tf.contrib.rnn.LSTMCell(rnn_units, state_is_tuple=True)
# lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, blstm_input, dtype=tf.float32)
# lstm_out = tf.concat(lstm_out, axis=-1)
# lstm_out = tf.reshape(lstm_out,[N,H,W,rnn_units*2])

ctpn_output = tf.layers.dense(inputs=conv6,units=10*3,kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))#[N,H,W,10*3]
ctpn_output = tf.reshape(ctpn_output,[N,H,W,10,3],name='ctpn_output')
textprob_pred = tf.sigmoid(ctpn_output[...,0:1],name='textprob_pred')#[N,H,W,10,1] ϵ{0,1}
y_pred = ctpn_output[...,1:3]/10#[N,H,W,10,2] ϵ{-∞,+∞}
y_pred = tf.identity(y_pred,name='y_pred')#[N,H,W,10,2] ϵ{-∞,+∞}
#xside_pred = ctpn_output[...,3:4]

positive_mask = tf.placeholder(dtype=tf.float32,shape=[None,None,None,10,1])
sampling_mask = tf.placeholder(dtype=tf.float32,shape=[None,None,None,10,1])
verti_gt = tf.placeholder(dtype=tf.float32,shape=[None,None,None,10,2])
#xside_gt = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1])
#sideanchor_mask = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1])
#num_sideanchor = tf.reduce_sum(sideanchor_mask)
num_positives = tf.reduce_sum(positive_mask)
conf_loss = tf.reduce_sum((textprob_pred-positive_mask)**2*sampling_mask)/tf.reduce_sum(sampling_mask)

def l1smooth(x,a=1):
    return tf.where(tf.abs(x)>a,tf.abs(x),tf.square(x)/a)

y_loss = tf.reduce_sum(l1smooth(verti_gt-y_pred)*positive_mask)/num_positives
#xside_loss = tf.reduce_sum((xside_gt-xside_pred)**2*sideanchor_mask)/num_sideanchor
total_loss = conf_loss+y_loss

tf.summary.scalar('conf_loss',conf_loss)
tf.summary.scalar('y_loss',y_loss)
#tf.summary.scalar('xside_loss',xside_loss)
logging = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_dir)






