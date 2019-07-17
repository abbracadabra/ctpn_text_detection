## references
- <a src='https://arxiv.org/abs/1609.03605'>paper</a>
- <a src='https://github.com/eragonruan/text-detection-ctpn'>a github repo where i get train images and its processed labels</a>

## difference of mine from the paper
- no side refinement
- no recurrent network

## data source
<a src='https://pan.baidu.com/s/1nbbCZwlHdgAI20_P9uw9LQ'>baidu cloud drive</a>

## dependency
- tensorflow(1.12.0)
- PIL
- numpy

## scripts
- train.py	#for training
- eval.py	#for evaluation
- config.py	#all configs are congregated here
- model.py	#build the model
- tf_vgg_pretrained.py	#convert a keras vgg16 model to tensorflow

## use
edit eval.py at line 19 to put your image path,then run ```python ./eval.py```.You will see result image pops up with bounding boxes drawn on it.

## demo
<div>
<img src='https://user-images.githubusercontent.com/35487258/61369544-ecb64200-a8c2-11e9-9a76-44fd978d5382.png' width='30%' display='inline' style='vertical-align: top;'>
<img src='https://user-images.githubusercontent.com/35487258/61369545-ed4ed880-a8c2-11e9-8ee0-d82b965031a4.png' width='30%' display='inline' style='vertical-align: top;'>
<img src='https://user-images.githubusercontent.com/35487258/61369546-ed4ed880-a8c2-11e9-9182-65e7b6feede5.png' width='30%' display='inline' style='vertical-align: top;'>
  </div>
