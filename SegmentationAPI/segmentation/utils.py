import os
class ConfigClass():
  def __init__(self, num_labels, id2label, label2id, image_size=640, patch_size=640, auxiliary_loss_weight=0.4, use_return_dict=False, output_hidden_states=False):
    self.num_labels = num_labels
    self.id2label = id2label
    self.label2id = label2id
    self.auxiliary_loss_weight = auxiliary_loss_weight
    self.image_size = image_size
    self.use_return_dict = use_return_dict
    self.output_hidden_states = output_hidden_states

class Utils:
    def __init__(self):
        self.root_path = 'segmentation'
        self.ingredients = self.getIngredients()
        self.id2label = self.getid2label()
        self.label2id = self.getlabel2id()
    def getIngredients(self):
        ingredients = dict()
        with open(os.path.abspath(f"{self.root_path}/category_id.txt")) as f:
            for line in f:
                data = line.strip().split()
                Id = data[0]
                food_type = ' '.join(data[1:])
                ingredients[int(Id)] = food_type
        return ingredients
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

utils = Utils()
config = ConfigClass(104, utils.id2label, utils.label2id)
    