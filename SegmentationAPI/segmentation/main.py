import os
from .visualization import SegmentationVisualization
import torch
from mmseg.apis import inference_segmentor
from .model import model
from .utils import config

class SegmentationInferenceWrapper():
    def __init__(self, model):
        self.model = model
        self.model.config = config
        self.visualization = SegmentationVisualization(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, img):
        pass

class SETR_MLA_InferenceWrapper(SegmentationInferenceWrapper):
    def __init__(self, model):
        super().__init__(model)    

    # override
    def predict(self, img):
        prediction = inference_segmentor(self.model, img)
        prediction = prediction[0].tolist()
        return prediction

SETR_MLA = SETR_MLA_InferenceWrapper(model)




