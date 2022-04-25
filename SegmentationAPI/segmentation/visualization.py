import numpy as np
from PIL import Image
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
  def __init__(self, model):
    self.segmenter = model
    self.palettes = self.generateRandomPalettes()

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