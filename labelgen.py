import numpy as np
from config import *
import os
import traceback
from matplotlib import pyplot as plt
from PIL import ImageDraw
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from PIL import Image


num_anchors = len(y_anchors)
stride = 16

def computeiou(boxa,boxb):
    top_a = boxa[0]
    bottom_a = boxa[2]
    x_a = boxa[1]
    top_b = boxb[0]
    bottom_b = boxb[2]
    x_b = boxb[1]

    if x_a!=x_b:
        return 0

    top_sect = max(top_a,top_b)
    bottom_sect = min(bottom_a, bottom_b)
    height_sect = bottom_sect-top_sect
    if height_sect<=0:
        return 0
    height_boxa = bottom_a-top_a
    height_boxb = bottom_b-top_b
    return height_sect/(height_boxa+height_boxb-height_sect)

def getone(imp, lbp):
    lbf = open(lbp,encoding='utf-8')
    gt_boxes = np.array([[int(spl) for spl in line.strip().split(',')] for line in lbf.readlines()])#[[x1,y1,x2,y2]]
    widthcenterx = (gt_boxes[..., 0:1] + gt_boxes[..., 2:3]) / 2
    gt_boxes = np.concatenate((gt_boxes[...,1:2],widthcenterx,gt_boxes[...,3:4],widthcenterx),axis=-1)#topy,widthcenterx,bottomy,widthcenterx
    gt_boxes = gt_boxes-[1,0,0,0]
    num_gtbox = len(gt_boxes)
    im = load_img(imp)
    im = img_to_array(im)
    im = preprocess_input(im)
    imh,imw = im.shape[0],im.shape[1]
    fmw,fmh = (imw//stride,imh//stride)
    stridew_precise,strideh_precise = imw/fmw,imh/fmh
    gt_boxes = gt_boxes/[strideh_precise,stridew_precise,strideh_precise,stridew_precise]
    gt_boxes[:,[1,3]] = np.floor(gt_boxes[:,[1,3]])
    yarr, xarr = np.meshgrid(np.arange(fmh)+0.5, np.arange(fmw), indexing='ij')
    centroids = np.tile(np.expand_dims(np.stack((yarr,xarr),axis=-1),axis=-2),[1,1,num_anchors,1])#[fmh,fmw,10,2]
    fh_anchors = y_anchors/strideh_precise#[10]
    half_fh_anchor = np.stack((fh_anchors/2,np.zeros([10])),axis=-1)#[10,2]
    top = centroids - half_fh_anchor#[fmh,fmw,10,2]
    bottom = centroids + half_fh_anchor#[fmh,fmw,10,2]
    anchor_boxes = np.concatenate((top,bottom),axis=-1)#[fmh,fmw,10,4]
    anchor_boxes = np.reshape(anchor_boxes,[-1,4])#[fmh*fmw*10,4]
    iou_matrix = np.array([computeiou(anchor_box,gt_box) for anchor_box in anchor_boxes for gt_box in gt_boxes]).reshape([len(anchor_boxes),len(gt_boxes)])#[fw*fh*10,num_gt]
    ##---------construct postive anchors mask
    positives = np.nonzero(np.max(iou_matrix,axis=1,keepdims=False)>0.7)[0]#[?] indices of anchorboxes having a iou>0.7 with any gt box
    bests = np.argmax(iou_matrix,axis=0)#[num_gt_box] indices of anchorboxes having the best iou with each gt box
    iou_matrix[(bests,np.arange(num_gtbox))] = 1.1#so that at least one anchor is assigned to each gt box
    positives = list(set(positives.tolist()+bests.tolist()))#[?]
    positive_idx = np.unravel_index(positives,(fmh,fmw,num_anchors,1))#tuple of arrays
    positive_mask = np.zeros([fmh,fmw,num_anchors,1])#[fmh,fmw,10,1]
    positive_mask[positive_idx] = 1#[fmh,fmw,10,1]
    # plt.matshow(np.max(positive_mask,(2,3)))
    # plt.pause(888)
    #----------construct sampling mask
    negatives = np.nonzero(np.max(iou_matrix, axis=1, keepdims=False) < 0.5)[0]
    # num_pos = len(positives)
    # num_neg = len(negatives)
    # if num_neg>num_pos:
    #     negatives = np.random.choice(negatives,num_pos,replace=False)
    # else:
    #     positives = np.random.choice(positives,num_neg,replace=False)
    sampling_mask = np.zeros([fmh,fmw,num_anchors,1])#[fmh,fmw,10,1]
    sampling_idx = np.unravel_index(np.concatenate((positives,negatives),-1), (fmh, fmw, num_anchors,1))
    sampling_mask[sampling_idx] = 1#[fmh,fmw,10,1]
    # plt.matshow(np.max(sampling_mask,(2,3)))
    # plt.pause(9999)
    ##--------construct localization label
    idx_gt_target = np.argmax(iou_matrix,axis=1)#[fmh*fmw*10]
    gt_targets = gt_boxes[idx_gt_target]#[fmh*fmw*10,4]
    gt_y_center = (gt_targets[..., 2:3]+gt_targets[...,0:1])/2#[fmh*fmw*10,1]
    gt_height = gt_targets[..., 2:3]-gt_targets[...,0:1]#[fmh*fmw*10,1]
    anchor_y_center = (anchor_boxes[..., 2:3]+anchor_boxes[...,0:1])/2#[fmh*fmw*10,1]
    anchor_height = anchor_boxes[..., 2:3]-anchor_boxes[...,0:1]#[fmh*fmw*10,1]
    vc = (gt_y_center-anchor_y_center)/anchor_height#[fmh*fmw*10,1]
    vh = np.log(np.maximum(gt_height/anchor_height,1e-3))#[fmh*fmw*10,1]

    # aa = anchor_height*vc
    # bb = np.exp(vh)*anchor_height/2
    # cc = np.reshape(positive_mask,[-1])
    # centroids = centroids+[0,0.5]
    # dd = np.reshape(centroids,[-1,2])
    # ee = np.nonzero(cc)[0]
    #
    # imm = Image.open(imp)
    # draw = ImageDraw.Draw(imm)
    # for i in ee:
    #     ac = dd[i]
    #     coff = aa[i]
    #     tc = ac+[coff,0]#yx
    #     hhh = bb[i]#half
    #     lf = tc-[hhh,0.5]
    #     ri = tc + [hhh, 0.5]
    #     lf = lf * [strideh_precise,stridew_precise]
    #     ri = ri * [strideh_precise,stridew_precise]
    #     draw.rectangle(((lf[1],lf[0]),(ri[1],ri[0])),outline="black",width=1)
    # imm.show()

    verti_label = np.concatenate((vc,vh),axis=-1).reshape([fmh,fmw,num_anchors,2])#[fmh,fmw,10,2]
    return im,positive_mask,sampling_mask,verti_label


def traindatagen():
    fns = os.listdir(train_img_dir)
    fns = np.random.permutation(fns)
    for fn in fns:
        try:
            fnp = fn.split('.')[0]
            im, positivemask, samplingmask, verticallabel = getone(os.path.join(train_img_dir, fn),os.path.join(train_label_dir, fnp + '.txt'))
            yield np.expand_dims(im,0), np.expand_dims(positivemask,0), np.expand_dims(samplingmask,0), np.expand_dims(verticallabel,0)
        except Exception:
            traceback.print_exc()
