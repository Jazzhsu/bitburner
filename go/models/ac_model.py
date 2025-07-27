import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and skip connection.
    """
    
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x):
        """Forward pass with residual connection."""
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += residual
        out = F.relu(out)
        
        return out


class ACModel(nn.Module):
    """
    Actor-Critic model for Go that takes a base encoder as input.
    Uses residual blocks for improved feature extraction.
    
    The model consists of:
    1. Initial convolution layer
    2. Multiple residual blocks for deep feature extraction
    3. Actor head that outputs action probabilities
    4. Critic head that outputs state value estimation
    """
    
    def __init__(self, encoder, hidden_dim=512, num_filters=256, num_residual_blocks=8):
        """
        Initialize the Actor-Critic model with residual blocks.
        
        Args:
            encoder: Base encoder instance (e.g., Oneplane, Sevenplane, etc.)
            hidden_dim: Dimension of hidden layers
            num_filters: Number of convolutional filters
            num_residual_blocks: Number of residual blocks (deeper = better but slower)
        """
        super(ACModel, self).__init__()
        
        self.encoder = encoder
        self.board_size = encoder.num_points()
        self.num_residual_blocks = num_residual_blocks
        
        # Get input shape from encoder
        input_shape = encoder.shape()  # (num_planes, height, width)
        self.num_input_planes = input_shape[0]
        self.board_height = input_shape[1] 
        self.board_width = input_shape[2]
        
        # Initial convolution to project input to desired number of filters
        self.initial_conv = nn.Conv2d(self.num_input_planes, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        
        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Actor head (policy network)
        # Output: probability for each board position + pass + resign
        self.actor_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)  # Reduce to 2 channels
        self.actor_bn = nn.BatchNorm2d(2)
        self.actor_fc = nn.Linear(2 * self.board_height * self.board_width, self.board_size)
        
        # Critic head (value network) 
        self.critic_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)  # Reduce to 1 channel
        self.critic_bn = nn.BatchNorm2d(1)
        self.critic_fc1 = nn.Linear(self.board_height * self.board_width, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using He initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def shared_features(self, x):
        """Extract shared features using initial conv + residual blocks."""
        # x shape: (batch_size, num_input_planes, height, width)
        
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Pass through residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        return x
    
    def actor_head(self, shared_features):
        """
        Actor head that outputs action probabilities.
        
        Returns:
            action_probs: Tensor of shape (batch_size, board_size + 2)
                         Last 2 positions are for pass and resign actions
        """
        x = F.relu(self.actor_bn(self.actor_conv(shared_features)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        action_logits = self.actor_fc(x)
        action_probs = F.softmax(action_logits, dim=1)
        
        return action_probs
    
    def critic_head(self, shared_features):
        """
        Critic head that outputs state value estimate.
        
        Returns:
            value: Tensor of shape (batch_size, 1) with value estimates
        """
        x = F.relu(self.critic_bn(self.critic_conv(shared_features)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.critic_fc1(x))
        x = self.dropout(x)
        value = torch.tanh(self.critic_fc2(x))  # Value between -1 and 1
        
        return value
    
    def forward(self, game_state):
        """
        Forward pass of the model.
        
        Args:
            game_state: Go game state or pre-encoded tensor
            
        Returns:
            action_probs: Action probability distribution
            value: State value estimate
        """
        # If input is a game state, encode it first
        if not isinstance(game_state, torch.Tensor):
            # get current device of this mode
            device = next(self.parameters()).device
            encoded_state = self.encoder.encode(game_state)
            x = torch.FloatTensor(encoded_state).unsqueeze(0).to(device)  # Add batch dimension
        else:
            x = game_state
            
        # Ensure proper shape: (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Extract shared features
        shared_features = self.shared_features(x)
        
        # Get actor and critic outputs
        action_probs = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        
        return action_probs, value
    
    def predict_move(self, game_state):
        """
        Predict the best move for a given game state.
        
        Returns:
            best_action_idx: Index of the best action
            action_probs: Action probability distribution
            value: State value estimate
        """
        self.eval()
        with torch.no_grad():
            action_probs, _, value = self.forward(game_state)
            best_action_idx = torch.argmax(action_probs, dim=1).item()
            
        return best_action_idx, action_probs.squeeze(), value.item()
    
    def get_move_from_action_idx(self, action_idx):
        """
        Convert action index to a Go move.
        
        Args:
            action_idx: Integer index of the action
            
        Returns:
            Move object (play, pass, or resign)
        """
        from go.goboard_fast import Move
        
        if action_idx == self.board_size:  # Pass
            return Move.pass_turn()
        elif action_idx == self.board_size + 1:  # Resign
            return Move.resign()
        else:  # Regular move
            point = self.encoder.decode_point_index(action_idx)
            return Move.play(point)


# Convenience function to create different model sizes
def create_small_model(encoder):
    """Create a small model with 4 residual blocks."""
    return ACModel(encoder, hidden_dim=256, num_filters=128, num_residual_blocks=4)


def create_medium_model(encoder):
    """Create a medium model with 8 residual blocks."""
    return ACModel(encoder, hidden_dim=512, num_filters=256, num_residual_blocks=8)


def create_large_model(encoder):
    """Create a large model with 16 residual blocks."""
    return ACModel(encoder, hidden_dim=1024, num_filters=384, num_residual_blocks=16)


def create_alphago_zero_style_model(encoder):
    """Create a model similar to AlphaGo Zero architecture."""
    return ACModel(encoder, hidden_dim=512, num_filters=256, num_residual_blocks=20)

