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