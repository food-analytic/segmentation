# Check Pytorch installation
import torch, torchvision
# Check MMSegmentation installation
import mmseg
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

config_file = "./configs/test/SETR_MLA_768x768_80k_base.py"
checkpoint_file = "./checkpoints/iter_80000.pth"

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

directory = './data/FoodSeg103/Images/img_dir/test/'

mIOU = 0

for path in tqdm(os.listdir(directory)):
    img = os.path.join('./data/FoodSeg103/Images/img_dir/test', path)
    mask = os.path.join('./data/FoodSeg103/Images/ann_dir/test', path[:-3]+'png')
    result = inference_segmentor(model, img)
    mask = Image.open(mask)
    label = np.array(mask)

    condition1 = label == result
    condition2 = label != 0

    intersection = np.count_nonzero(condition1 & condition2)
    union = np.count_nonzero(label + result)

    mIOU += intersection / union

mIOU /= len(os.listdir(directory))

print(mIOU)

