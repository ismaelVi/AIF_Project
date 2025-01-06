import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split


def get_data_loaders(data_folder, img_size=(224, 224), batch_size=32, test_split=0.2):

    # Transformations à appliquer aux images
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),  # Convertit les images en tenseurs PyTorch et les normalise entre 0 et 1
    ])
    
    # Utilisation d'ImageFolder pour charger les images et leurs étiquettes
    dataset = datasets.ImageFolder(root=data_folder, transform=transform)
    
    # Séparation des indices pour l'entraînement et le test
    train_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        test_size=test_split,
        stratify=[label for _, label in dataset.samples],
        random_state=42
    )
    
    # Sous-ensembles d'entraînement et de test
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Création des DataLoader pour l'entraînement et le test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
