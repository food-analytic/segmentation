import os
import mmcv
from mmcv import Config
from mmseg.apis import init_segmentor

config = os.path.abspath('segmentation/configs/SETR_MLA_768x768_80k_base_on_FoodSeg73.py')
checkpoint = os.path.abspath('segmentation/checkpoints/iter_80000.pth')
model = init_segmentor(config, checkpoint)

