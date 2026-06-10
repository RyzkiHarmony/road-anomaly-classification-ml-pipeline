import torch
import torch.nn as nn

class Lightweight1DCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, conv1_filters=16, conv2_filters=32, dropout_rate=0.3):
        super(Lightweight1DCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, conv1_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.relu2 = nn.ReLU()
        
        self.pool2 = nn.MaxPool1d(100)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(conv2_filters, 16)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Test parameter count
    model = Lightweight1DCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 200)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")
