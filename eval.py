import numpy as np
from config import *
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input

tf.train.import_meta_graph(model_dir+'.meta')
input_ph = tf.get_default_graph().get_tensor_by_name("input_1:0")
textprob = tf.get_default_graph().get_tensor_by_name("textprob_pred:0")#[N,H,W,10,1] ϵ{0,1}
ypred = tf.get_default_graph().get_tensor_by_name("y_pred:0")#[N,H,W,10,2] ϵ{-∞,+∞}

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,model_dir)
#D:\Users\xxx\Desktop\
impath = r'D:\Users\xxx\Desktop\20190716104633.jpg'
imagepil = Image.open(impath)
im = load_img(impath)
im = img_to_array(im)
im = preprocess_input(im)
im = np.expand_dims(np.array(im),axis=0)

probs,ypreds = sess.run([textprob,ypred],feed_dict={input_ph:im})

matrix_shape = probs.shape[1:]#tuple (H,W,10,1)
probs=np.reshape(probs[0],[-1])
ypreds=np.reshape(ypreds[0],[-1,2])

flat_indices = np.flatnonzero(probs>0.3)
ypreds_map={}
for flat_index in flat_indices:
    ypred = ypreds[flat_index]
    index = np.unravel_index(flat_index,matrix_shape)
    center_y_grid = index[0]*16+8
    anchor_height = y_anchors[index[2]]
    center_y = center_y_grid + ypred[0]*anchor_height
    half_height = anchor_height * np.exp(ypred[1]) / 2
    ypreds_map[flat_index] = (center_y-half_height,center_y+half_height)

#nms
winner_indices=[]
colgroups={}
for flat_index in flat_indices:
    index = np.unravel_index(flat_index, matrix_shape)
    w_off = index[1]
    if w_off not in colgroups:
        colgroups[w_off]=[]
    colgroups[w_off].append(flat_index)

for col,ixgroup in colgroups.items():
    ixgroup = np.array(ixgroup)
    groupix_ordered = ixgroup[np.argsort(probs[ixgroup])[::-1]]
    groupix_ordered = list(groupix_ordered)
    pos=0
    while pos<len(groupix_ordered)-1:
        ix1 = groupix_ordered[pos]
        top1,bottom1 = ypreds_map[ix1]
        pos2=pos+1
        while pos2<len(groupix_ordered):
            ix2 = groupix_ordered[pos2]
            top2, bottom2 = ypreds_map[ix2]
            if top2 >= top1 and bottom2 <= bottom1:
                del groupix_ordered[pos2]
                continue
            sect_top = max(top1, top2)
            sect_bottom = min(bottom1, bottom2)
            sect = (sect_bottom - sect_top)
            union = (bottom1 - top1) + (bottom2 - top2) - sect
            iou = sect / union
            if iou > 0:
                del groupix_ordered[pos2]
                continue
            pos2+=1
        pos+=1
    winner_indices+=groupix_ordered

rectangles=[]
for flat_index in winner_indices:
    top,bottom = ypreds_map[flat_index]
    index = np.unravel_index(flat_index, matrix_shape)
    rectangles.append((index[1]*16,top,(index[1]+1)*16,bottom))

draw = ImageDraw.Draw(imagepil)
for rect in rectangles:
    draw.rectangle(rect, outline='green', width=2)
imagepil.show()















