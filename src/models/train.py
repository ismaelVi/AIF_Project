import torch
import sys

##### Fonction classique pour entrainer et tester un modèle #####

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):

    print("Start training")

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            print("."*((i%3)+1),'     ' ,end='\r')
            sys.stdout.flush()  # Forcer l'affichage


            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions.double() / total_samples
        print(f"Époque [{epoch+1}/{num_epochs}], Perte: {epoch_loss:.4f}, Précision: {epoch_accuracy:.4f}")
    
    print("Entraînement terminé")
    return model

def test_model(model, test_loader, device='cuda'):
    
    model.to(device)
    model.eval()
    
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
    
    accuracy = correct_predictions.double() / total_samples
    print(f"Précision sur l'ensemble de test : {accuracy:.4f}")
    return accuracy