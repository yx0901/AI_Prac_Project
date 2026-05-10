import torch
import torch.nn as nn
import os
import argparse
from torch.utils.data import DataLoader, random_split, Subset
from dataset import ChineseCharDataset, build_default_transform, build_training_transform
from model import CNN


def train(resume=False, epochs=5, patience=5, batch_size=32, learning_rate=0.001):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    dataset = ChineseCharDataset(transform=build_default_transform())
    print(f"Classes: {dataset.classes}")
    print(f"Total samples: {len(dataset)}")

    # Split into train/validation (80/20) deterministically
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    train_dataset = ChineseCharDataset(transform=build_training_transform())
    val_dataset = ChineseCharDataset(transform=build_default_transform())
    train_set = Subset(train_dataset, train_subset.indices)
    val_set = Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Model, loss, optimizer
    model = CNN(num_classes=len(dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    os.makedirs('models', exist_ok=True)
    checkpoint_path = 'models/cnn.pth'
    start_epoch = 0
    best_val_acc = 0.0
    epochs_without_improvement = 0

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                pass
        start_epoch = checkpoint.get('epoch', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        print(f"Resuming from epoch {start_epoch}.")
    elif resume:
        print("No checkpoint found. Starting fresh training.")

    # Training loop
    for epoch in range(start_epoch, epochs):
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
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += loss.item()

        if total == 0:
            raise ValueError(f"Validation set is empty. Check your data split.")

        val_acc = 100 * correct / total
        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.2f}%")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'classes': dataset.classes,
            'class_to_idx': dataset.class_to_idx,
        }

        # Save best checkpoint on improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            checkpoint['best_val_acc'] = best_val_acc
            checkpoint['epochs_without_improvement'] = epochs_without_improvement
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, f"models/cnn_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint: models/cnn_epoch_{epoch+1}.pth")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs. Best: {best_val_acc:.2f}%")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    print("Training finished. Latest model: models/cnn.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Chinese character classifier')
    parser.add_argument('--resume', action='store_true', help='Resume from models/cnn.pth if available')
    parser.add_argument('--epochs', type=int, default=5, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()
    train(
        resume=args.resume,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )