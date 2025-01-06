import torch.nn as nn
import torch.optim as optim
import torch

from param.param import DATA_PATH, WEIGHTS_PATH
from param.param import BATCH_SIZE, LR, NUM_EPOCHS, NUM_CLASSES

from data.data_loader import get_data_loaders
from models.models import FilmGenreClassifier18
from models.train import test_model, train_model


if __name__ == "__main__":

    # Déterminer le device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Chargement des données
    train_loader, test_loader = get_data_loaders(DATA_PATH, batch_size=BATCH_SIZE)

    # Définition du modèle, de la fonction de perte et de l'optimiseur
    model = FilmGenreClassifier18(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()  # Fonction de perte pour la classification multi-classe
    optimizer = optim.Adam(model.parameters(), lr=LR)


    # Entraînement du modèle
    trained_model = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, device)


    # Évaluation du modèle sur l'ensemble de test
    test_accuracy = test_model(trained_model, test_loader, device)


    # Sauvegarde du modèle entraîné
    torch.save(trained_model.state_dict(), WEIGHTS_PATH['RESNET18'])



    print("Modèle sauvegardé avec succès.")
