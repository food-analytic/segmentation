import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

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

  def getVisualization(self, prediction, base64=False):
    predicted = np.array(prediction).astype(np.uint8)
    fig = plt.figure(figsize=(10,6))

    predicted_labels = list(np.unique(predicted))
    predicted_labels.sort(key=lambda x: np.count_nonzero(predicted == x), reverse=True)
    predicted_labels_color = [self.palettes[label*3:label*3+3] for label in predicted_labels]

    predicted = Image.fromarray(predicted)
    predicted.putpalette(self.palettes)
    predicted = predicted.convert('RGBA')
    predicted_patches = []
    plt.imshow(predicted, cmap='gray')
    #fig.imshow(predicted, cmap='gray')
    plt.title('prediction', fontsize=30)
    predicted_patches = []
    for idx, color in enumerate(predicted_labels_color):
      color = list(np.array(color)/255)
      patch = mpatches.Patch(color=color, label=self.segmenter.model.config.id2label[predicted_labels[idx]])
      predicted_patches.append(patch)
    axis = fig.axes[0]
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    plt.gca().set_position([0.03, 0.05, 0.64, 0.90])
    plt.legend(handles=predicted_patches,loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 20})
    if not base64:
      return fig
    else:
      import io
      import base64
      IOBytes = io.BytesIO()
      plt.savefig(IOBytes, format='png')
      IOBytes.seek(0)
      base64Image = base64.encodebytes(IOBytes.getvalue()).decode('ascii')
      return base64Image