import torch.nn as nn

class CNN(nn.Module):
    """
    Simple CNN model for 64x64 images
    Input: (batch, 1, 64, 64)
    Output: (batch, num_classes)
    """
    def __init__(self, num_classes=15):
        super().__init__()
        
        # Feature extraction: learn visual patterns
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Classification: map features to class scores
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x  # Raw scores (logits)