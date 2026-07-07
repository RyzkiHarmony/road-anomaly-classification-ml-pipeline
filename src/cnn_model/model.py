import torch
import torch.nn as nn

class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=16):
        super(InceptionBlock1D, self).__init__()
        
        # 1. Bottleneck layer
        self.use_bottleneck = in_channels > bottleneck_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        conv_in_channels = bottleneck_channels if self.use_bottleneck else in_channels
        
        # 2. Parallel convolutions
        num_branches = len(kernel_sizes) + 1
        base_channels = out_channels // num_branches
        pool_channels = out_channels - (base_channels * len(kernel_sizes))
        
        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in_channels, base_channels, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        
        # 3. MaxPool branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(conv_in_channels, pool_channels, kernel_size=1, bias=False)
        
        # 4. BatchNorm and SE Block (ReLU moved to InceptionTime1D)
        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels, reduction=8)
        
    def forward(self, x):
        x_bottleneck = self.bottleneck(x) if self.use_bottleneck else x
        
        # Parallel conv outputs
        conv_outs = [conv(x_bottleneck) for conv in self.convs]
        
        # MaxPool branch
        x_pool = self.conv_pool(self.maxpool(x_bottleneck))
        conv_outs.append(x_pool)
        
        # Concatenate along channel dimension
        out = torch.cat(conv_outs, dim=1)
        out = self.bn(out)
        return self.se(out)

class InceptionTime1D(nn.Module):
    def __init__(self, in_channels=7, num_classes=3, num_blocks=3, channels=128, bottleneck_channels=32, dropout_rate=0.5):
        super(InceptionTime1D, self).__init__()
        
        self.blocks = nn.ModuleList()
        # First block maps in_channels to channels
        self.blocks.append(InceptionBlock1D(in_channels, channels, bottleneck_channels=bottleneck_channels))
        
        # Remaining blocks map channels to channels
        for _ in range(1, num_blocks):
            self.blocks.append(InceptionBlock1D(channels, channels, bottleneck_channels=bottleneck_channels))
            
        # Residual connections (added every 3 blocks)
        self.shortcuts = nn.ModuleList()
        for i in range(num_blocks // 3):
            # Shortcut mapping channels
            if i == 0:
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(in_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(channels)
                ))
            else:
                self.shortcuts.append(nn.BatchNorm1d(channels))
                
        self.relu = nn.ReLU()
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(channels, num_classes)
        
    def forward(self, x):
        res = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            if i % 3 == 2:
                # Add shortcut connection every 3 blocks
                x = x + self.shortcuts[i // 3](res)
                x = self.relu(x)
                res = x
            else:
                x = self.relu(x)
            
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
    dummy_input = torch.randn(2, 7, 200)
    out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output shape: {out.shape}")
