import torch
import torch.nn as nn

class Lightweight1DCNN(nn.Module):
    def __init__(self, in_channels=10, num_classes=3, conv1_filters=32, conv2_filters=64, dropout_rate=0.3):
        super(Lightweight1DCNN, self).__init__()
        
        # 1. Branch 1: Local / Sharp Features (Pothole - Kernel 3)
        self.branch1_conv1 = nn.Conv1d(in_channels, conv1_filters, kernel_size=3, padding=1)
        self.branch1_bn1 = nn.BatchNorm1d(conv1_filters)
        self.branch1_relu1 = nn.ReLU()
        self.branch1_pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.branch1_conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=3, padding=1)
        self.branch1_bn2 = nn.BatchNorm1d(conv2_filters)
        self.branch1_relu2 = nn.ReLU()
        self.branch1_pool2 = nn.MaxPool1d(kernel_size=12, stride=12, padding=2) # Output length: 8
        
        # 2. Branch 2: Medium Features (General - Kernel 7)
        self.branch2_conv1 = nn.Conv1d(in_channels, conv1_filters, kernel_size=7, padding=3)
        self.branch2_bn1 = nn.BatchNorm1d(conv1_filters)
        self.branch2_relu1 = nn.ReLU()
        self.branch2_pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.branch2_conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=5, padding=2)
        self.branch2_bn2 = nn.BatchNorm1d(conv2_filters)
        self.branch2_relu2 = nn.ReLU()
        self.branch2_pool2 = nn.MaxPool1d(kernel_size=12, stride=12, padding=2) # Output length: 8
        
        # 3. Branch 3: Global / Long Features (Speed Bump / Slope - Kernel 15)
        self.branch3_conv1 = nn.Conv1d(in_channels, conv1_filters, kernel_size=15, padding=7)
        self.branch3_bn1 = nn.BatchNorm1d(conv1_filters)
        self.branch3_relu1 = nn.ReLU()
        self.branch3_pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.branch3_conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=7, padding=3)
        self.branch3_bn2 = nn.BatchNorm1d(conv2_filters)
        self.branch3_relu2 = nn.ReLU()
        self.branch3_pool2 = nn.MaxPool1d(kernel_size=12, stride=12, padding=2) # Output length: 8
        
        self.flatten = nn.Flatten()
        
        # Gabungan dari 3 cabang paralel: conv2_filters * 8 * 3 cabang
        self.fc1 = nn.Linear(conv2_filters * 8 * 3, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Forward Branch 1
        x1 = self.branch1_pool1(self.branch1_relu1(self.branch1_bn1(self.branch1_conv1(x))))
        x1 = self.branch1_pool2(self.branch1_relu2(self.branch1_bn2(self.branch1_conv2(x1))))
        x1 = self.flatten(x1)
        
        # Forward Branch 2
        x2 = self.branch2_pool1(self.branch2_relu1(self.branch2_bn1(self.branch2_conv1(x))))
        x2 = self.branch2_pool2(self.branch2_relu2(self.branch2_bn2(self.branch2_conv2(x2))))
        x2 = self.flatten(x2)
        
        # Forward Branch 3
        x3 = self.branch3_pool1(self.branch3_relu1(self.branch3_bn1(self.branch3_conv1(x))))
        x3 = self.branch3_pool2(self.branch3_relu2(self.branch3_bn2(self.branch3_conv2(x3))))
        x3 = self.flatten(x3)
        
        # Gabungkan fitur dari 3 cabang
        out = torch.cat((x1, x2, x3), dim=1)
        
        out = self.dropout(self.relu3(self.fc1(out)))
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    # Test parameter count
    model = Lightweight1DCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 10, 200)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")
