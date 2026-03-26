import torch
import torch.nn as nn
import os
import argparse
from torch.utils.data import DataLoader, random_split
from dataset import ChineseCharDataset
from model import CNN

def train(resume=False):
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
    os.makedirs('models', exist_ok=True)
    checkpoint_path = 'models/cnn.pth'
    start_epoch = 0

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}.")
    elif resume:
        print("No checkpoint found. Starting fresh training.")
    
    # Training loop
    for epoch in range(start_epoch, EPOCHS):
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

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'classes': dataset.classes,
            'class_to_idx': dataset.class_to_idx,
        }

        # Keep an epoch-specific file and update the latest checkpoint every epoch.
        torch.save(checkpoint, f"models/cnn_epoch_{epoch+1}.pth")
        torch.save(checkpoint, 'models/cnn.pth')
        print(f"Saved checkpoint: models/cnn_epoch_{epoch+1}.pth")
    
    print("Training finished. Latest model: models/cnn.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Chinese character classifier')
    parser.add_argument('--resume', action='store_true', help='Resume from models/cnn.pth if available')
    args = parser.parse_args()
    train(resume=args.resume)