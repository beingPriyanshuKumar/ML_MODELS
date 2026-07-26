import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def run_training():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 10
    num_workers = 4 

    # 2. Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Datasets
    train_data = datasets.ImageFolder(root='./dataset/training_set', transform=transform)
    test_data = datasets.ImageFolder(root='./dataset/test_set', transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)

    # 4. Model Setup
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 5. Training and Testing Loops
    print(f"Training on: {device}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Testing
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images).data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {running_loss/len(train_loader):.4f} - Accuracy: {100 * correct / total:.2f}%")

    print("Training Complete.")

if __name__ == '__main__':
    run_training()