import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import ChineseCharDataset
from model import CNN

def train():
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    dataset = ChineseCharDataset()
    print(f"Classes: {dataset.classes}")
    print(f"Total samples: {len(dataset)}")
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    # Model, loss, optimizer
    model = CNN(num_classes=len(dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} Val Acc: {val_acc:.2f}%")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
    }, 'models/cnn.pth')
    print("Model saved!")

if __name__ == '__main__':
    train()