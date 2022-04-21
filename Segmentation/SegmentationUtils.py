from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation, BeitConfig, DetrFeatureExtractor, DetrForSegmentation
from PIL import Image
import requests
import torch
import os
import numpy as np
from torch import nn
import torch
import torchvision
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import random_split
import pickle
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd

# root_path_image = 'C:/Users/Kombangkoe Dias/Downloads/FoodSeg103/FoodSeg103/Images'
# root_path_type = 'C:/Users/Kombangkoe Dias/Downloads/FoodSeg103/FoodSeg103/types'
# root_path = 'C:/Users/Kombangkoe Dias/Downloads/FoodSeg103'
root_path = '/content/FoodSeg103'
root_path73 = '/content/FoodSeg73'

data_path = root_path

root_path_image = f'{data_path}/Images'
root_path_type = f'{data_path}/types'
drive_path = '/content/drive/MyDrive/Senior Project'
shared_drive_path = '/content/drive/Shareddrives/Food Analytic'
# root_path_image = '/home/kbd/Desktop/data/FoodSeg103/FoodSeg103/Images'
# root_path_type = '/home/kbd/Desktop/data/FoodSeg103/FoodSeg103/types'
# root_path = '/home/kbd/Desktop/data/FoodSeg103/FoodSeg103'

class Utils:
    def __init__(self, root_path):
        self.root_path = root_path
        self.ingredients = self.getIngredients()
        self.id2label = self.getid2label()
        self.label2id = self.getlabel2id()
        self.numPicsTrain = self.getClassificationNumPics(train=True)
        self.numPicsTest = self.getClassificationNumPics(train=False)
    def getIngredients(self):
        ingredients = dict()
        with open(f"{self.root_path}/category_id.txt") as f:
            for line in f:
                data = line.strip().split()
                Id = data[0]
                food_type = ' '.join(data[1:])
                ingredients[int(Id)] = food_type
        return ingredients
        return False
    def getid2label(self):
        id2label = dict()
        for id in self.ingredients.keys():
          id2label[id] = self.ingredients[id]
        return id2label
    def getlabel2id(self):
        label2id = dict()
        for id, val in self.ingredients.items():
          label2id[val] = id
        return label2id
    def getClassificationNumPics(self, train=True):
      path = '/content/FoodSeg103Classification/'
      path =  path + "train" if train else path + 'test'
      numPics = []
      for class_name in os.listdir(path):
        try:
          numPics.append((class_name, len(os.listdir(os.path.join(path, class_name)))))
        except:
          pass
      return numPics

class ConfigClass():
  def __init__(self, num_labels, id2label, label2id, image_size=640, patch_size=640, auxiliary_loss_weight=0.4, use_return_dict=False, output_hidden_states=False):
    self.num_labels = num_labels
    self.id2label = id2label
    self.label2id = label2id
    self.auxiliary_loss_weight = auxiliary_loss_weight
    self.image_size = image_size
    self.use_return_dict = use_return_dict
    self.output_hidden_states = output_hidden_states

utils = Utils(data_path)
config = ConfigClass(num_labels=len(utils.id2label.keys()), id2label=utils.id2label, label2id=utils.label2id)

# class Transforms():
#   def __init__(self):
#     self.transforms = [self.randomCrop]
#     self.randomCrop = None 
  
#   def __call__(self,img,mask):
#     pass 

class TransformsGenerator():
  def __init__(self, tfms):
    self.tfms = tfms
  def __call__(self, img, mask):
    for t in self.tfms:
      img, mask = t(img, mask)
    return img, mask

class randomCrop():
    def __init__(self,cropw, croph):
      self.cropw = cropw
      self.croph = croph
    def __call__(self, img, mask):
      w, h = img.size
      startw = 0
      starth = 0
      if self.cropw < w:
        startw = np.random.randint(0, w-self.cropw)
      if self.croph < h:
        starth = np.random.randint(0, h-self.croph)
      endw = startw + min(self.cropw,w)
      endh = starth + min(self.croph,h)
      pos = (startw, starth, endw, endh)

      img = img.crop(pos)
      mask = mask.crop(pos)
      return img, mask

class randomHorizontalFlip():
  def __init__(self, p):
    self.p = p
  def __call__(self, img, mask):
    from PIL import ImageOps
    p = np.random.rand()
    if p > self.p :
      img = ImageOps.mirror(img)
      mask = ImageOps.mirror(mask)
    return img, mask

class randomVerticalFlip():
  def __init__(self, p):
    self.p = p
  def __call__(self, img, mask):
    from PIL import ImageOps
    p = np.random.rand()
    if p > self.p:
      img = ImageOps.flip(img)
      mask = ImageOps.flip(mask)
    return img, mask

import albumentations as A
import cv2

transforms = A.Compose([
    A.Resize(1025, 2049, cv2.INTER_NEAREST),
    A.RandomCrop(640,640),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.PadIfNeeded(min_height=640, min_width=640)
])

class FoodSeg103Segmentation(torch.utils.data.Dataset):
    def __init__(self,root, train=True, feature_extractor=None, transforms=None):
        if train:
          path = 'train'
        else:
          path = 'test'
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root,f'img_dir/{path}'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, f'ann_dir/{path}'))))
        self.train = train
        self.feature_extractor = feature_extractor
        # filtered = [1078, 3642, 4909]
        # filtered.sort(reverse=True)
        # if train:
        #   for idx in filtered:
        #     self.imgs.pop(idx)
        #     self.masks.pop(idx)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'img_dir/train' if self.train else 'img_dir/test', self.imgs[idx])
        mask_path = os.path.join(self.root, 'ann_dir/train' if self.train else 'ann_dir/test', self.masks[idx])
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        # import cv2
        # img = cv2.resize(np.array(img), self.imageSize, interpolation=cv2.INTER_NEAREST)
        # mask = cv2.resize(np.array(mask), self.imageSize, interpolation=cv2.INTER_NEAREST)
        # mask = np.array(mask)
        # Ids = np.unique(mask)
        # Ids = Ids[1:]
        # masks = []
        # for id in Ids:
        #     masks.append(mask == id)
        # num_mask = len(Ids)
        # boxes = []
        # for i in range(num_mask):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])
        
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.as_tensor(Ids, dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        # image_id = torch.tensor([idx])
        # area = (boxes[:,3] - boxes[:, 1]) * (boxes[:,2] - boxes[:, 0])
        # # iscrowd = torch.zeros((num_mask,), dtype=torch.int64)
        # for a in area:
        #   if a == 0:
        #     return self.__getitem__(idx-1)

        if not self.train:
          return {'img_tensor': torch.tensor(np.array(img)).permute((2,0,1)),'img_np': np.array(img), 
                  'mask_tensor': torch.tensor([np.array(mask)]), 'mask_np': np.array(mask),
                  'img_pil': img, 'mask_pil': mask, 'img_path': img_path, 'mask_path': mask_path} # for testing on test dataset

        if self.transforms is not None:
          img = cv2.imread(img_path)
          mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
          res = transforms(image=img, mask=mask)
          img = res['image']
          mask = res['mask']
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          img = Image.fromarray(img)
          mask = Image.fromarray(mask)

        data = dict()
        data['image'] = img_path
        if self.feature_extractor is not None:
          encoded_inputs = self.feature_extractor(img, mask, return_tensors='pt')
        else:
          return {'pixel_values': np.array(img), 'labels': np.array(mask)}
        # else:
        #   data['target'] = {'boxes': boxes, 'labels': labels, 
        #                   'masks': masks, 'image_id': image_id, 'area': area}
        return {'pixel_values': encoded_inputs['pixel_values'][0,:,:,:], 'labels': encoded_inputs['labels']}
    def __len__(self):
        return len(self.imgs)
    
    def set_feature_extractor(self, feature_extractor):
      self.feature_extractor = feature_extractor
  
def create_datasets(root_path_image=root_path_image, feature_extractor=None, split = 301, transforms=None):
  if feature_extractor is None:
    feature_extractor = BeitFeatureExtractor(do_resize=True, size=640, resample=Image.NEAREST,align=False, do_center_crop=False)
  dataset = FoodSeg103Segmentation(root_path_image, feature_extractor=feature_extractor, transforms=transforms)
  dataset_test = FoodSeg103Segmentation(root_path_image, train=False)
  dataset_size = len(dataset)
  indices = list(range(dataset_size))

  np.random.seed(42)
  np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]

  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)

  train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, 
                                            sampler=train_sampler)
  val_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  sampler=valid_sampler)
  test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1)
  return dataset, dataset_test, train_loader, val_loader, test_loader

os.system("pip install git+http://github.com/ildoonet/pytorch-gradual-warmup-lr.git")
os.system("pip install madgrad")

import seaborn as sns
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches

def confusion_matrix_decorator(func):
    def inner(*args, **kwargs):
        segmenter = args[0].segmenter
        x = segmenter.evaluation.confusionMatrix
        if x is None:
          print(f"no confusion matrix found, function {func.__name__} not available")
          return
        func(*args, **kwargs)
    return inner
  
def accuracies_decorator(func):
  def inner(*args, **kwargs):
    segmenter = args[0].segmenter
    try:
      x = segmenter.evaluation.accuracies
      if x is None:
        print(f"no accuracies array found, function {func.__name__} not available")
        return
    except:
        print(f"no accuracies array found, function {func.__name__} not available")
        return
    func(*args, **kwargs)
  return inner

class SegmentationVisualization():
  def __init__(self, model, evaluation_img_type):
    self.segmenter = model
    self.palettes = self.generateRandomPalettes()
    self.evaluation_img_type = evaluation_img_type

    self.calculateColumnsAndValuesSortedByTruePositives()
    print("Visualization class initialized")

  def generateRandomPalettes(self):
    palettes = [255,255,255]
    for i in range(self.segmenter.model.config.num_labels):
      palette = list(np.random.choice(range(1,255), size=3, replace=False))
      palettes += palette
    return palettes

  def predictAndVisualize(self, idx, real_img, real_mask, filter_small_classes=False):
    self.segmenter.model.eval()
    if real_mask is not None:
      img = real_img
      mask = np.array(real_mask)
      predicted = self.segmenter.predict(img, filter_small_classes=filter_small_classes)
      if type(img) == str:
        img = np.array(Image.open(img))
      accuracy = self.segmenter.calculateIOU(predicted, mask)
      f, axarr = plt.subplots(1,3, figsize=(20,6))
      plt.subplots_adjust(wspace=0.4)
      f.suptitle(f"image No.{idx} IOU: {accuracy}",fontsize=16)

      predicted_labels = list(np.unique(predicted))
      predicted_labels.sort(key=lambda x: np.count_nonzero(predicted == x), reverse=True)
      predicted_labels_color = [self.palettes[label*3:label*3+3] for label in predicted_labels]
      # print(predicted_labels)
      # print(predicted_labels_color)
      # print(predicted.shape)

      predicted = Image.fromarray(predicted)
      predicted.putpalette(self.palettes)
      predicted = predicted.convert('RGBA')

      mask_labels = list(np.unique(mask))
      mask_labels.sort(key=lambda x: np.count_nonzero(mask == x), reverse=True)
      mask_labels_color = [self.palettes[label*3:label*3+3] for label in mask_labels]
      #print(mask_labels)
      #print(mask_labels_color)

      mask = Image.fromarray(mask, mode='L')
      mask.putpalette(self.palettes)
      mask = mask.convert('RGBA')

      axarr[0].imshow(img)
      axarr[0].title.set_text('test image')
      box = axarr[0].get_position()
      axarr[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
      axarr[1].imshow(mask)
      axarr[1].title.set_text('ground truth')
      real_patches = []
      for idx, color in enumerate(mask_labels_color):
        color = list(np.array(color)/255)
        patch = mpatches.Patch(color=color, label=self.segmenter.model.config.id2label[mask_labels[idx]])
        real_patches.append(patch)
      box = axarr[1].get_position()
      axarr[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
      axarr[1].legend(handles=real_patches,loc='center left', bbox_to_anchor=(1, 0.5))

      axarr[2].imshow(predicted, cmap='gray')
      axarr[2].title.set_text('prediction')
      predicted_patches = []
      for idx, color in enumerate(predicted_labels_color):
        color = list(np.array(color)/255)
        patch = mpatches.Patch(color=color, label=self.segmenter.model.config.id2label[predicted_labels[idx]])
        predicted_patches.append(patch)
      box = axarr[2].get_position()
      axarr[2].set_position([box.x0, box.y0, box.width * 0.8, box.height])
      axarr[2].legend(handles=predicted_patches,loc='center left', bbox_to_anchor=(1, 0.5))
      plt.show()
      print()
    else:
      img = real_img
      predicted = self.segmenter.predict(img,filter_small_classes=filter_small_classes)
      f, axarr = plt.subplots(1,2, figsize=(10,6))
      plt.subplots_adjust(wspace=0.4)

      predicted_labels = list(np.unique(predicted))
      predicted_labels.sort(key=lambda x: np.count_nonzero(predicted == x), reverse=True)
      predicted_labels_color = [self.palettes[label*3:label*3+3] for label in predicted_labels]
      
      axarr[0].imshow(img)
      axarr[0].title.set_text('input image')
      box = axarr[0].get_position()
      axarr[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])

      predicted = Image.fromarray(predicted)
      predicted.putpalette(self.palettes)
      predicted = predicted.convert('RGBA')
      predicted_patches = []
      axarr[1].imshow(predicted, cmap='gray')
      axarr[1].title.set_text('prediction')
      predicted_patches = []
      for idx, color in enumerate(predicted_labels_color):
        color = list(np.array(color)/255)
        patch = mpatches.Patch(color=color, label=self.segmenter.model.config.id2label[predicted_labels[idx]])
        predicted_patches.append(patch)
      box = axarr[1].get_position()
      axarr[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
      axarr[1].legend(handles=predicted_patches,loc='center left', bbox_to_anchor=(1, 0.5))
      plt.show()
      print()

  def predictTestSetAndVisualize(self, random_n=0, indices=[], filter_small_classes=False):
    # randomly predict
    if random_n != 0:
      for i in range(random_n):
        idx = np.random.randint(0, len(dataset_test))
        self.predictAndVisualize(idx, self.segmenter.dataset_test[idx][self.evaluation_img_type], self.segmenter.dataset_test[idx]['mask_np'],filter_small_classes)
    # predict some specific image
    elif len(indices) != 0:
      for i in indices:
        self.predictAndVisualize(i, self.segmenter.dataset_test[i][self.evaluation_img_type], self.segmenter.dataset_test[i]['mask_np'],filter_small_classes)
    # predict all
    else:
      for i in range(len(dataset_test)):
        self.predictAndVisualize(i, self.segmenter.dataset_test[i][self.evaluation_img_type], self.segmenter.dataset_test[i]['mask_np'],filter_small_classes)
    
  
  @confusion_matrix_decorator
  def calculateColumnsAndValuesSortedByTruePositives(self):
    RsptR = []
    RsptC = []
    for column in self.segmenter.evaluation.confusionMatrixRsptR.columns[:-1]:
      valR = self.segmenter.evaluation.confusionMatrixRsptR[column].loc[column]
      valC = self.segmenter.evaluation.confusionMatrixRsptC[column].loc[column]
      if np.isnan(valR):
        valR = 0
      if np.isnan(valC):
        valC = 0
      RsptR.append((column, valR))
      RsptC.append((column, valC))

    self.confusionMatrixRsptRSort = sorted(RsptR, key=lambda x: x[1], reverse=True)
    self.confusionMatrixRsptCSort = sorted(RsptC, key=lambda x: x[1], reverse=True)

    self.confusionMatrixRsptRSortColumn = list(map(lambda x: x[0], self.confusionMatrixRsptRSort))
    self.confusionMatrixRsptCSortColumn = list(map(lambda x: x[0], self.confusionMatrixRsptCSort))

    self.confusionMatrixRsptRSortValue = list(map(lambda x: x[1], self.confusionMatrixRsptRSort))
    self.confusionMatrixRsptCSortValue = list(map(lambda x: x[1], self.confusionMatrixRsptCSort))
    
    


  @confusion_matrix_decorator
  def plotTruePositiveClassAccuracy(self, compareWithPath : str = ""):

    if (compareWithPath != ""):
      compare_confusionMatrix = pd.read_csv(compareWithPath, index_col=0)
      confusionMatrixRsptR = compare_confusionMatrix.divide(compare_confusionMatrix.loc['totalR'],axis='columns')
      confusionMatrixRsptC = compare_confusionMatrix.divide(compare_confusionMatrix['totalC'], axis='index')
      data = [(self.confusionMatrixRsptRSortValue, self.confusionMatrixRsptRSortColumn, confusionMatrixRsptR, "True Positive Accuracy compared between two results"), 
              (self.confusionMatrixRsptCSortValue, self.confusionMatrixRsptCSortColumn, confusionMatrixRsptC, "True Positive Accuracy compared between two results")]
      
      for values, columns, compareMatrix, title in data:
        differences = list()
        for idx, c in enumerate(columns):
          if (c in compareMatrix.columns):
            val_compare = compareMatrix[c].loc[c];
            print(val_compare, values[idx], c)
            differences.append(values[idx] - val_compare)
        figure(figsize=(17, 10), dpi=80)
        ax = plt.gca()
        bar_plot = plt.bar([idx for idx in range(len(differences))], differences, color='m')
        for idx, rect in enumerate(bar_plot):
          height = rect.get_height()
          if height > 1e-3:
            height = float(format(height,".3f"))
          ax.text(rect.get_x() + rect.get_width()/2., height+0.02,
                  height,
                  ha='center', va='bottom', size='small', rotation=90)
        plt.xticks([idx for idx in range(len(columns))], columns, rotation='vertical')
        plt.ylabel("true positive accuracy")
        plt.xlabel("classes")
        plt.title(title)
        plt.show()
        print()
        
    else:
      data = [(self.confusionMatrixRsptRSortValue, self.confusionMatrixRsptRSortColumn, "True Positive Accuracy of classes with respect to number of predicted labels"), 
              (self.confusionMatrixRsptCSortValue, self.confusionMatrixRsptCSortColumn, "True Positive Accuracy of classes with respect to number of ground truth labels")]

      for values, columns, title in data:
        figure(figsize=(17, 10), dpi=80)
        ax = plt.gca()
        bar_plot = plt.bar([idx for idx in range(len(values))], values, color='m')
        for idx, rect in enumerate(bar_plot):
          height = rect.get_height()
          if height > 1e-3:
            height = float(format(height,".3f"))
          ax.text(rect.get_x() + rect.get_width()/2., height+0.02,
                  height,
                  ha='center', va='bottom', size='small', rotation=90)
        plt.xticks([idx for idx in range(len(columns))], columns, rotation='vertical')
        plt.ylabel("true positive accuracy")
        plt.xlabel("classes")
        plt.title(title)
        plt.show()
        print()

  @confusion_matrix_decorator
  def plotHeatMapOfConfusionMatrix(self):
      data = [(self.segmenter.evaluation.confusionMatrixRsptR, "Heatmap of confusion matrix with respect to number of prediction labels"), 
              (self.segmenter.evaluation.confusionMatrixRsptC, "Heatmap of confusion matrix with respect to number of ground truth labels")]
      for frame, title in data:
        with sns.axes_style("white"):
          f, ax = plt.subplots(figsize=(30, 30))
          frame = frame.fillna(0)
          frame = frame[frame.columns[:-1]]
          frame = frame.loc[frame.index[:-1]]
          ax = sns.heatmap(frame,xticklabels=True, yticklabels=True, cmap="Blues", square=True)
          ax.xaxis.set_ticks_position('top')
          ax.xaxis.set_label_position('top')
          ax.tick_params(labelsize=8)
          plt.xticks(rotation="vertical")
          plt.title(title, fontsize=20)
          plt.xlabel("prediction", fontsize=16)
          plt.ylabel("ground truth", fontsize=16)
          plt.show()
          print()

  @confusion_matrix_decorator
  def plotCorrespondingClassNumPics(self):
  
    numPicsTrain = utils.numPicsTrain

    values1 = []
    values2 = []

    for column in self.confusionMatrixRsptRSortColumn:
      for label, val in numPicsTrain:
        if label == column:
          values1.append(val)
          break
    
    for column in self.confusionMatrixRsptCSortColumn:
      for label, val in numPicsTrain:
        if label == column:
          values2.append(val)

    corrR = np.corrcoef(values1, self.confusionMatrixRsptRSortValue)
    corrC = np.corrcoef(values2, self.confusionMatrixRsptCSortValue)
    
    R_patch = mpatches.Patch(label=f"correlation: {corrR[0][1]}")
    C_patch = mpatches.Patch(label=f"correlation: {corrC[0][1]}")

    data = [(self.confusionMatrixRsptRSortColumn, values1, R_patch, "Number of pictures by class ranked by its true positive accuracy with respect to predicted labels"), 
            (self.confusionMatrixRsptCSortColumn, values2, C_patch, "Number of pictures by class ranked by its true positive accuracy with respect to ground truth labels")]

    for columns, values, handle,  title in data:
      figure(figsize=(17, 10), dpi=80)
      ax = plt.gca()
      bar_plot = plt.bar([idx for idx in range(len(values))], values, color='b')
      for idx, rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height+5,
                str(height),
                ha='center', va='bottom', size='small', rotation=90)
      plt.xticks([idx for idx in range(len(columns))], columns, rotation='vertical')
      plt.ylabel("Number Of Pictures")
      plt.xlabel("classes")
      plt.title(title)
      plt.legend(handles=[handle])
      plt.show()
      print()

  @accuracies_decorator
  def plotAccuracies(self):
    accuracies = self.segmenter.evaluation.accuracies
    num_bins = 20
    xmin = 0
    xmax = 1
    plt.hist(accuracies, bins=num_bins, range=[xmin,xmax], color='g')
    plt.title("Histogram of picture-wise accuracy (IOU)")
    plt.xlabel("IOU")
    plt.ylabel("Number of pictures")
    plt.show()

class SegmentationEvaluation():
  def __init__(self, model, confusionMatrixPath, accuraciesPath, evaluation_img_type):
    self.segmenter = model
    self.confusionMatrixPath = confusionMatrixPath
    self.accuraciesPath = accuraciesPath
    self.confusionMatrix = None
    self.accuracies = None
    self.loadConfusionMatrix()
    self.loadAccuraciesArray()
    self.evaluation_img_type = evaluation_img_type

    print("Evaluation class initialized")

  def evaluate(self, try_for_test=False):
    self.segmenter.model.eval()
    sum_accuracy = 0
    if try_for_test:
      dataloader = self.segmenter.test_loader
    else:
      dataloader = self.segmenter.val_loader
    for batch in tqdm(dataloader):
      pixel_values = batch['pixel_values'].to(self.segmenter.device)
      labels = batch['labels'][:,0,:,:]
      labels = labels.to(self.segmenter.device)
      # print(pixel_values.shape)
      # print(labels.shape)
      outputs = self.segmenter.model(pixel_values=pixel_values, labels=labels)
      # interpolate the result
      upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
      predicted = upsampled_logits.argmax(dim=1)

      mask = (labels != 255) # we don't include the background class in the accuracy calculation
      pred_labels = predicted[mask].detach().cpu().numpy()
      true_labels = labels[mask].detach().cpu().numpy()
      #print(np.unique(pred_labels), pred_labels.shape)
      #print(np.unique(true_labels), true_labels.shape)
      accuracy = self.segmenter.calculateIOU(pred_labels, true_labels)
      sum_accuracy += accuracy
    print("mIOU:", sum_accuracy / len(dataloader))
    return sum_accuracy / len(dataloader)

  def evaluate_on_test_dataset(self):
    self.segmenter.model.eval()
    sum_accuracy = 0
    accuracies = []
    with tqdm(self.segmenter.dataset_test) as pbar_object:
      for batch in pbar_object:
        img = batch[self.evaluation_img_type]
        mask = batch['mask_np']
        predicted = self.segmenter.predict(img)
        accuracy = self.segmenter.calculateIOU(predicted, mask)
        pbar_object.set_postfix(accuracy=accuracy)
        accuracies.append(accuracy)  
        sum_accuracy += accuracy
      self.test_dataset_mIOU = sum_accuracy / len(self.segmenter.dataset_test)
      self.accuracies = accuracies
      print("mIOU:", sum_accuracy / len(self.segmenter.dataset_test))
      return sum_accuracy / len(self.segmenter.dataset_test), accuracies
  
  def calculatemIOUAndmAcc(self):
      replacedConfusionMatrix = self.confusionMatrix.replace(np.nan, 0)
      replacedConfusionMatrix.drop(columns='totalC', inplace=True)
      replacedConfusionMatrix.drop(index='totalR', inplace=True)
      self.confusionMatrixVal = dict()
      for i in range(1,self.segmenter.model.config.num_labels):
        self.confusionMatrixVal[self.segmenter.model.config.id2label[i]] = {"TP": 0, "FN": 0, "FP": 0}
      
      val = replacedConfusionMatrix.to_numpy()
      
      IOUs = np.diag(val) / (val.sum(axis=1) + val.sum(axis=0) - np.diag(val))
      Accs = np.diag(val) / (val.sum(axis=1))
      #print(len(IOUs))
      
      print("mIOU: ", np.mean(IOUs))
      print("mAcc: ", np.mean(Accs))

      mIOU = np.mean(IOUs)
      mAcc = np.mean(Accs)

      return mIOU ,mAcc

      #print(IOUs)
  
  def getaAcc(self):
    self.segmenter.model.eval()
    total_correct = 0
    total_pixels = 0
    for batch in tqdm(self.segmenter.dataset_test):
      img = batch[self.evaluation_img_type]
      mask = batch['mask_np']
      predicted = self.segmenter.predict(img)
      correct = np.count_nonzero(mask == predicted)
      pixels = mask.shape[0] * mask.shape[1]
      total_correct += correct
      total_pixels += pixels
    self.aAcc = total_correct/ total_pixels
    print("aAcc :", self.aAcc)


  def generateConfusionMatrix(self):
    self.segmenter.model.eval()
    num_labels = self.segmenter.model.config.num_labels
    accuracies = []
    sum_accuracy = 0
    labels = [self.segmenter.model.config.id2label[i] for i in range(num_labels)]
    self.confusionMatrix = [[np.nan for i in range(num_labels)] for j in range(num_labels)]
    with tqdm(self.segmenter.dataset_test) as pbar_object:
      for batch in pbar_object:
        img = batch[self.evaluation_img_type]
        mask = batch['mask_np']
        predicted = self.segmenter.predict(img)
        accuracy = self.segmenter.calculateIOU(predicted, mask)
        pbar_object.set_postfix(accuracy=accuracy)
        accuracies.append(accuracy)
        sum_accuracy += accuracy
        for i in range(predicted.shape[0]):
          for j in range(predicted.shape[1]):
            predict_label = predicted[i][j]
            real_label = mask[i][j]
            if np.isnan(self.confusionMatrix[real_label][predict_label]):
              self.confusionMatrix[real_label][predict_label] = 1
            else:
              self.confusionMatrix[real_label][predict_label] += 1
    self.test_dataset_mIOU = sum_accuracy / len(self.segmenter.dataset_test)
    print("mIOU:",self.test_dataset_mIOU)
    self.accuracies = accuracies
    labels = [self.segmenter.model.config.id2label[idx] for idx in range(num_labels)]
    self.confusionMatrix = pd.DataFrame(self.confusionMatrix, index=labels, columns=labels)
    totalC = self.confusionMatrix.sum(axis=1)
    totalR = self.confusionMatrix.sum(axis=0)
    self.confusionMatrix.loc['totalR'] = totalR
    totalC.loc['totalR'] = totalR.sum()
    self.confusionMatrix['totalC'] = totalC
    self.confusionMatrix.to_csv(self.confusionMatrixPath)
    
  def loadConfusionMatrix(self):
    try:
      self.confusionMatrix = pd.read_csv(self.confusionMatrixPath, index_col=0)
      print("Loaded the confusion matrix")
      self.calculateTwoConfusionMatrix()
    except:
      print("No pickle file for confusion matrix")

  def calculateTwoConfusionMatrix(self):
    if self.confusionMatrix is not None:
      self.confusionMatrixRsptR = self.confusionMatrix.divide(self.confusionMatrix.loc['totalR'],axis='columns')
      self.confusionMatrixRsptC = self.confusionMatrix.divide(self.confusionMatrix['totalC'], axis='index')
      print("created the respective confusion matrix")

  def loadAccuraciesArray(self):
    try:
      with open(self.accuraciesPath, 'rb') as f:
        self.accuracies = pickle.load(f)
        print("Loaded accuracies")
    except:
      print("no accuracies file found")
  
  def saveAccuraciesArray(self):
    with open(self.accuraciesPath, 'wb') as f:
        pickle.dump(self.accuracies, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved Accuracies array")
  
class SegmentationLoss(nn.Module):
  import enum
  class LossType(enum.Enum):
      FOCAL_LOSS = 'FOCAL_LOSS'
      DICE_LOSS = 'DICE_LOSS'
      TVERSKY_LOSS = 'TVERSKY_LOSS'
      DEFAULT = 'DEFAULT'
  types = LossType

  def __init__(self, segmenter, loss_type, use_auxiliary_loss=False, **kwargs):
    super().__init__()
    self.segmenter = segmenter
    self.loss_type = loss_type
    self.use_auxiliary_loss = use_auxiliary_loss
    if (self.use_auxiliary_loss):
      print("using auxiliary loss")
    else:
      print("not using auxiliary loss")
    self.addLossFunctions = {self.types.FOCAL_LOSS: self.addFocalLoss, 
                             self.types.DICE_LOSS: self.addDiceLoss,
                             self.types.TVERSKY_LOSS: self.addTverskyLoss}
    self.loss = None
    self.addLoss(loss_type, **kwargs);

  def forward(self, inputs, targets, auxiliary_outputs=None):
    main_loss = self.loss(inputs, targets)
    if self.use_auxiliary_loss:
      auxiliary_loss = self.loss(auxiliary_outputs, targets)
      return main_loss + self.segmenter.model.config.auxiliary_loss_weight * auxiliary_loss
    else:
      return main_loss
  
  def addLoss(self, losstype, **kwargs):
    if (len(kwargs.keys()) == 0):
      print(f"using loss '{losstype.value}' with default arguments")
    else:
      print(f"using loss '{losstype.value}' with arguments {kwargs}")
    if losstype == self.types.DEFAULT:
      return
    self.addLossFunctions[losstype](**kwargs)

  def addFocalLoss(self, alpha=0.25, gamma=2):
    class FocalLoss(nn.Module):
      def __init__(self, gamma=gamma, alpha=alpha, size_average=True):
          super(FocalLoss, self).__init__()
          self.gamma = gamma
          if (alpha is None):
            self.alpha = None
          else:
            self.alpha = torch.tensor([alpha] + [1-alpha] * 103)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.alpha = self.alpha.to(self.device)

      def forward(self, input, labels):
          print(input.shape)
          print(labels.shape)
          BCE_loss = F.cross_entropy(input, labels, reduction='none')
          targets = labels.type(torch.long)
          pt = torch.exp(-BCE_loss)
          if (self.alpha is not None):
            at = torch.gather(self.alpha, 0, targets.data.view(-1))
            at = torch.reshape(at, (2,640,640))
            F_loss = at*(1-pt)**self.gamma * BCE_loss
          else:
            F_loss = (1-pt)**self.gamma * BCE_loss
          return F_loss.mean()
    self.loss = FocalLoss(gamma=gamma, alpha=alpha)
  
  def addDiceLoss(self):
    class DiceLoss(nn.Module):
      def __init__(self, weight=None, size_average=True):
          super(DiceLoss, self).__init__()

      def forward(self, inputs, targets, smooth=1):
          inputs = inputs.flatten()
          targets = F.one_hot(targets,num_classes=104)
          targets = targets.permute(0,3,1,2)
          targets = targets.flatten()
          
          
          intersection = (inputs * targets).sum()                            
          dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
          
          return 1 - dice
    self.loss = DiceLoss()
  
  def addTverskyLoss(self, alpha=0.5, beta=0.5):
    ALPHA = alpha
    BETA = beta

    class TverskyLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(TverskyLoss, self).__init__()

        def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
            
            inputs = inputs.flatten()
            targets = F.one_hot(targets)
            targets = targets.permute(0,3,1,2)
            targets = targets.flatten()
            
            TP = (inputs * targets).sum()    
            FP = ((1-targets) * inputs).sum()
            FN = (targets * (1-inputs)).sum()

            Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
            
            return 1 - Tversky
    self.loss = TverskyLoss()

from transformers import AdamW
import torch
from torch import nn 
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter

class SegmentationTrainWrapper():
  def __init__(self,feature_extractor, train_loader, val_loader, test_loader, dataset_test, evaluation_img_type='img_pil'):
    
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.dataset_test = dataset_test

    self.highest_accuracy = 0

    self.evaluation = SegmentationEvaluation(self, self.confusionMatrixPath, self.accuraciesPath, evaluation_img_type)
    self.visualization = SegmentationVisualization(self, evaluation_img_type)
    self.lossHandler = SegmentationLoss

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.prepare_for_continue_training()
    self.freezeModelLayer()
    self.countTrainableParameters()
    self.model.to(self.device)
    self.writer = None

  def countTrainableParameters(self):
    model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"total trainable parameters : {params}")

  def prepare_for_continue_training(self):
    try:
      with open(self.iterationPath, 'r') as f:
        self.iteration = int(f.readline().strip())
        print("current iteration:", self.iteration)
      self.model.load_state_dict(torch.load(self.model_path))
      print("loaded the best model's state dict")
    except Exception as e:
      print("using the default Model because", e)
    
  def setup_training(self, epochs=10, **kwargs):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    self.lossHandler = SegmentationLoss
    self.epochs = epochs
    self.optimizer = madgrad.MADGRAD(self.model.parameters(), lr=0.001)
    # lr = 0.0005
    #AdamW(self.model.parameters(), lr=0.00006)
    self.scheduler_after = CosineAnnealingLR(self.optimizer,10)
    self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=4, after_scheduler=self.scheduler_after)
    print(f"epochs set to {epochs}")

    try:
      self.lossHandler = self.lossHandler(self, **kwargs)
    except:
      print("must specify loss type (see SegmentationLoss.types for options) ")
    
    try:
      self.optimizer.load_state_dict(torch.load(f"{shared_drive_path}/{self.store_root_path}/optimizer.pth"))
      print("loaded the optimizer's state dict")
      self.scheduler.load_state_dict(torch.load(f"{shared_drive_path}/{self.store_root_path}/scheduler.pth"))
      print("loaded the scheduler's state dict")
    except:
      print("using the default (starter) optimizer and scheduler")

  def freezeModelLayer(self):
    print("No layer freezed")
    pass

  def finetune(self, test=False):
    # create the writer on the same log_dir 
    # in case the iteration is continuing, then might be that the same log file will be used.
    if self.writer is None:
      self.writer = SummaryWriter(log_dir=self.tensorboard_path)
    print("Generating the previous highest accuracy.")
    if not test:
      self.highest_accuracy, _ = self.evaluation.evaluate_on_test_dataset() # generate the previous highest accuracy.
    for epoch in range(self.epochs):  # loop over the dataset multiple times
      self.model.train()
      print("Epoch:", epoch)
      with tqdm(self.train_loader) as pbar_object:
        for batch in pbar_object:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch['labels'][:,0,:,:]
            labels = labels.to(self.device)
            #print(pixel_values.shape, labels.shape)
            self.optimizer.zero_grad()

            if self.lossHandler.use_auxiliary_loss:
              upsampled_logits, upsampled_auxiliary_logits = self.getMockForwardForAuxiliaryOutputs(pixel_values)
              loss = self.lossHandler(inputs=upsampled_logits, targets=labels, auxiliary_outputs=upsampled_auxiliary_logits)
            else:
              outputs = self.forwardForFineTune(pixel_values=pixel_values, labels=labels)
              logits = outputs.logits
              if (self.lossHandler.loss_type == SegmentationLoss.types.DEFAULT):
                loss = outputs.loss
              else:
                upsampled_logits = nn.functional.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
                loss = self.lossHandler(inputs=upsampled_logits, targets=labels)

            pbar_object.set_postfix(loss=loss.item(), iteration=self.iteration)
            # TODO : add tensorboard
            loss.backward()
            self.optimizer.step()
            self.iteration += 1

            # logging
            if not test:
              with open(self.iterationPath, 'w') as f:
                  f.write(str(self.iteration))
                  f.close()
              self.writer.add_scalar('Loss/train', loss.item(), self.iteration)
      self.scheduler.step()
      self.saveTraining()
      accuracy, _ = self.evaluation.evaluate_on_test_dataset()
      if not test:
        self.writer.add_scalar('Accuracy/test', accuracy, self.iteration)
      if accuracy > self.highest_accuracy:
        print("New highest accuracy achieved at", accuracy, f"(previous: {self.highest_accuracy})")
        self.highest_accuracy = accuracy
        torch.save(self.model.state_dict(), self.model_path)
        print("Saved the highest-accuracy model's state dict")

  def predict(self, img, filter_small_classes=False):
    pass

  def forwardForFineTune(self, pixel_values, labels=None):
    pass
  
  def calculateIOU(self, pred_labels, true_labels):
    condition1 = pred_labels != 0
    condition2 = pred_labels == true_labels
    Intersection = np.count_nonzero(condition1 & condition2)
    Union = np.count_nonzero(pred_labels | true_labels)
    #print("Intersection:",Intersection)
    #print("Union:",Union)
    IOU = Intersection / Union
    #print("IOU:",IOU)
    return IOU
      
  def resetIterations(self, itr = 0):
    with open(self.iterationPath, 'w') as f:
      f.write(str(itr))
      f.close()
      self.iteration = itr
  
  def postTraining(self):
    self.prepare_for_continue_training()
    self.evaluation.generateConfusionMatrix()
    self.evaluation.saveAccuraciesArray()
    return self.evaluation.calculatemIOUAndmAcc()

  def saveTraining(self):
    torch.save(BEiT.optimizer.state_dict(), f"{shared_drive_path}/{self.store_root_path}/optimizer.pth")
    print("saved the optimizer's state dict")
    torch.save(BEiT.scheduler.state_dict(), f"{shared_drive_path}/{self.store_root_path}/scheduler.pth")
    print("saved the scheduler's state dict")

class StaticUtils():
  def __init__(self):
    pass
  def doComprehensiveEvaluation(self,itr_start, itr_end, itr_step, store_root_path, ModelClass : SegmentationTrainWrapper,train_loader, validation_loader, test_loader, dataset_test ):
    best_mIOU = 0
    best_mAcc = 0
    best_itr_mIOU = 0
    best_itr_mAcc = 0
    for itr in range(itr_start,itr_end+itr_step, itr_step):
      print(f"evaluating the {itr} iteration model")
      checkpoint_file = f"{shared_drive_path}/{store_root_path}/iter_{itr}.pth"
      folder = f"{store_root_path}/{itr}"
      try:
        os.mkdir(f"{shared_drive_path}/{folder}")
      except Exception as e:
        if not isinstance(e, FileExistsError):
          raise e
      model = ModelClass(None, train_loader, validation_loader, test_loader, dataset_test, folder, checkpoint_file)
      if model.evaluation.confusionMatrix is not None:
        mIOU, mAcc = model.evaluation.calculatemIOUAndmAcc()
      else:
        mIOU, mAcc = model.postTraining()
      if (mIOU > best_mIOU):
        best_itr_mIOU = itr
        print(f"best mIOU achieved at {mIOU} of iteration {itr} (previous : {best_mIOU})")
        best_mIOU = mIOU
      
      if (mAcc > best_mAcc):
        best_itr_mAcc = itr
        print(f"best mAcc achieved at {mAcc} of iteration {itr} (previous : {best_mAcc})")
        best_mAcc = mAcc
  def mmsegToOnnx(self, config_path, checkpoint_path, example_img_path, work_dir):
    os.chdir("/content")
    os.system("pip install openmim")
    os.system("pip install onnx onnxruntime")
    os.system("mim install mmdeploy")
    os.system("git clone https://github.com/open-mmlab/mmdeploy/")
    os.system(f"""!python mmdeploy/tools/deploy.py 
    mmdeploy/configs/mmseg/segmentation_onnxruntime_dynamic.py
    {config_path}
    {checkpoint_path} 
    {example_img_path}
    --work-dir {work_dir} 
    --show""")
    os.chdir("/content/CapstoneFinal/SETR_MLA")


