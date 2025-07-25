"""
Improved neural network architecture for stronger Go play.
Key improvements:
1. Deeper network with residual connections
2. Better feature extraction
3. Stronger pattern recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for deeper networks without vanishing gradients."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class StrongGoNNet(nn.Module):
    """
    Improved neural network for Go with deeper architecture.
    Based on AlphaGo/AlphaZero principles but smaller for 5x5 board.
    """
    def __init__(self, board_size: int):
        super(StrongGoNNet, self).__init__()
        
        action_size = board_size * board_size + 1
        
        # 🔧 Enhanced input processing
        self.input_conv = nn.Conv2d(4, 128, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(128)
        
        # 🔧 Residual tower (deeper network)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(6)  # 6 residual blocks
        ])
        
        # 🔧 Enhanced policy head
        self.policy_conv = nn.Conv2d(128, 8, kernel_size=1)  # More features
        self.policy_bn = nn.BatchNorm2d(8)
        self.policy_fc = nn.Linear(8 * board_size * board_size, action_size)
        
        # 🔧 Enhanced value head  
        self.value_conv = nn.Conv2d(128, 4, kernel_size=1)  # More features
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 64)
        self.value_fc3 = nn.Linear(64, 1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Better weight initialization for deeper networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier initialization for better gradient flow
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for output layers
        nn.init.normal_(self.policy_fc.weight, 0, 0.01)
        nn.init.normal_(self.value_fc3.weight, 0, 0.01)

    def forward(self, x):
        # Input processing
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        pi = F.relu(self.policy_bn(self.policy_conv(x)))
        pi = pi.view(pi.size(0), -1)
        pi = F.log_softmax(self.policy_fc(pi), dim=1)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = F.relu(self.value_fc2(v))
        v = torch.tanh(self.value_fc3(v))
        
        return pi, v

# Keep original for compatibility
GoNNet = StrongGoNNet 