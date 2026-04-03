import torch.nn as nn
import torchvision.models as models


def get_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 14)
    return model