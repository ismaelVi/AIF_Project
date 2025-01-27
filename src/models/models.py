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

# Classe du modèle
class AnomalieClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(AnomalieClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        # Remplacer la dernière couche pour la classification binaire
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        return self.sigmoid(x)  # Pour une sortie de probabilité entre 0 et 1