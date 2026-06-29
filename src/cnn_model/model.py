import torch
import torch.nn as nn

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[5, 11, 21], bottleneck_channels=16):
        super(InceptionBlock1D, self).__init__()
        
        # 1. Bottleneck layer
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        
        # 2. Parallel convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, out_channels // 4, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        
        # 3. MaxPool branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, bias=False)
        
        # 4. BatchNorm and Activation
        # Total output channels: (out_channels // 4) * 3 (convs) + (out_channels // 4) (pool) = out_channels
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, in_channels, seq_len)
        x_bottleneck = self.bottleneck(x)
        
        # Parallel conv outputs
        conv_outs = [conv(x_bottleneck) for conv in self.convs]
        
        # MaxPool branch
        x_pool = self.conv_pool(self.maxpool(x))
        conv_outs.append(x_pool)
        
        # Concatenate along channel dimension
        out = torch.cat(conv_outs, dim=1)
        out = self.bn(out)
        return self.relu(out)

class InceptionTime1D(nn.Module):
    def __init__(self, in_channels=14, num_classes=3, num_blocks=2, channels=64, bottleneck_channels=16, dropout_rate=0.2):
        super(InceptionTime1D, self).__init__()
        
        self.blocks = nn.ModuleList()
        # First block maps in_channels to channels
        self.blocks.append(InceptionBlock1D(in_channels, channels, bottleneck_channels=bottleneck_channels))
        
        # Remaining blocks map channels to channels
        for _ in range(1, num_blocks):
            self.blocks.append(InceptionBlock1D(channels, channels, bottleneck_channels=bottleneck_channels))
            
        # Residual connections
        self.shortcuts = nn.ModuleList()
        for i in range(num_blocks):
            # Shortcut mapping channels
            if i == 0:
                self.shortcuts.append(nn.Conv1d(in_channels, channels, kernel_size=1, bias=False))
            else:
                self.shortcuts.append(nn.Identity())
                
        self.bns = nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(num_blocks)])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_blocks)])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(channels, num_classes)
        
    def forward(self, x):
        res = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Add shortcut connection
            x = x + self.shortcuts[i](res)
            x = self.bns[i](x)
            x = self.relus[i](x)
            res = x
            
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Test parameter count
    model = InceptionTime1D()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 14, 200)
    out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output shape: {out.shape}")
