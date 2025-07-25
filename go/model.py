import torch
import torch.nn as nn
import torch.nn.functional as F

class GoNNet(nn.Module):
    """
    Neural Network for a 7x7 Go board.
    Input shape: (batch_size, 8, 7, 7)
    Outputs:
    - Policy: A log probability distribution over all possible moves (7x7 + 1 pass = 50 moves).
    - Value: A single scalar value between -1 and 1, estimating the win probability.
    """
    def __init__(self, board_size: int):
        super(GoNNet, self).__init__()
        
        action_size = board_size * board_size + 1 # size x size board positions + 1 pass move
        
        # Shared Convolutional Body
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        b = board_size
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        # Policy Head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * b*b, action_size)

        # Value Head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * b*b, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Initialize weights to reduce bias
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights to reduce initial bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for policy output layer to start more uniform
        # This helps prevent bias toward specific actions (like passing)
        nn.init.normal_(self.policy_fc.weight, 0, 0.01)
        nn.init.constant_(self.policy_fc.bias, 0)

    def forward(self, x):
        # Pass through shared convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # --- Policy Head ---
        # Reduce feature map depth
        pi = F.relu(self.policy_bn(self.policy_conv(x)))
        # Flatten
        pi = pi.view(pi.size(0), -1)
        # Get log probabilities for each move
        pi = F.log_softmax(self.policy_fc(pi), dim=1)

        # --- Value Head ---
        # Reduce feature map depth
        v = F.relu(self.value_bn(self.value_conv(x)))
        # Flatten
        v = v.view(v.size(0), -1)
        # Pass through fully connected layers
        v = F.relu(self.value_fc1(v))
        # Get the final [-1, 1] value prediction
        v = torch.tanh(self.value_fc2(v))

        return pi, v
