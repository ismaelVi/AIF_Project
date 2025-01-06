import torch.nn as nn
import torchvision.models as models

class FilmGenreClassifier18(nn.Module):
    def __init__(self, num_classes=10):
        super(FilmGenreClassifier18, self).__init__()
        # Charger un modèle pré-entraîné (par exemple ResNet18)
        self.base_model = models.resnet18(weights='DEFAULT')  # 'DEFAULT' pour utiliser les poids pré-entraînés

        # Remplacer la dernière couche entièrement connectée
        # Assurez-vous que le nombre de classes correspond à votre jeu de données
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)



class FilmGenreClassifier50(nn.Module):
    def __init__(self, num_classes=10):
        super(FilmGenreClassifier18, self).__init__()
        # Charger un modèle pré-entraîné (par exemple ResNet18)
        self.base_model = models.resnet50(weights='DEFAULT')  # 'DEFAULT' pour utiliser les poids pré-entraînés

        # Remplacer la dernière couche entièrement connectée
        # Assurez-vous que le nombre de classes correspond à votre jeu de données
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
