import torch.nn as nn
import torchvision.models as models

class FilmGenreClassifier18(nn.Module):
    def __init__(self, num_classes=10):
        super(FilmGenreClassifier18, self).__init__()
        # Charger le modèle pré-entraîné (ici ResNet18)
        self.base_model = models.resnet18(weights='DEFAULT')

        # Remplacer la dernière couche entièrement connectée
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
