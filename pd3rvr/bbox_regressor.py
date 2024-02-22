from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid

class ObjectDetector(Module):
    """docstring for ObjectDetector."""
    #object detector model with two arguments: basemodel and number of classes
    def __init__(self, baseModel, numClasses):
        super(ObjectDetector, self).__init__()
        self.baseModel = baseModel
        self.numClasses = numClasses
        #regrssor head for assigning bbox coordinates
        #first linear layer of regressor inputs fully connected basemodel layer of size 128
        self.regressor = Sequential(
            Linear(baseModel.fc.in_features, 128),
            ReLU(),
			Linear(128, 64),
			ReLU(),
			Linear(64, 32),
			ReLU(),
			Linear(32, 4),
			Sigmoid()
        )
        #classifier for object label
        #first linear layer of classifier inputs fully connected basemodel layer of size == feature size
        self.classifier = Sequential(
			Linear(baseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, self.numClasses)
		)
        #make base model’s fc layer into Identity layer, which means classifier to outputs same results as previous convolution block
        self.baseModel.fc = Identity()

    def forward(self, x):
		# pass the inputs through  base model to get features as output. Feed features into regressor and classifier
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)
        # return the outputs as a tuple
        return (bboxes, classLogits)
