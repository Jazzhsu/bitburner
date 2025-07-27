import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


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


class PolicyModel(nn.Module):
    """
    Policy-only model for Go that takes 4-channel one-hot input.
    
    Input channels:
    0: Empty positions (1 if empty, 0 otherwise)
    1: Wall positions (1 if wall, 0 otherwise)  
    2: Self stones (1 if current player's stone, 0 otherwise)
    3: Opponent stones (1 if opponent's stone, 0 otherwise)
    
    Output: Softmax probabilities over all board positions + pass move
    """
    
    def __init__(self, board_size, num_filters=256, num_residual_blocks=8):
        """
        Initialize the Policy model.
        
        Args:
            board_size: Size of the Go board (e.g., 9, 13, 19)
            num_filters: Number of convolutional filters
            num_residual_blocks: Number of residual blocks
        """
        super(PolicyModel, self).__init__()
        
        self.board_size = board_size
        self.board_height = board_size
        self.board_width = board_size
        self.num_positions = board_size * board_size
        self.num_actions = self.num_positions + 1  # board positions + pass
        self.num_residual_blocks = num_residual_blocks
        
        # Input is always 4 channels (empty, wall, self, opponent)
        self.num_input_channels = 4
        
        # Initial convolution to project 4-channel input to desired number of filters
        self.initial_conv = nn.Conv2d(self.num_input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        
        # Stack of residual blocks for feature extraction
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Policy head - outputs move probabilities
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)  # Reduce to 2 channels
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_height * self.board_width, self.num_actions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
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
    
    def encode_game_state(self, game_state, current_player):
        """
        Encode game state into 4-channel one-hot representation.
        
        Args:
            game_state: GameState object
            current_player: Player object (current player to move)
            
        Returns:
            torch.Tensor: Shape (4, board_size, board_size)
        """
        from goboard_fast import Player, Point
        
        board = game_state.board
        encoded = torch.zeros(4, self.board_size, self.board_size, dtype=torch.float32)
        
        for row in range(1, self.board_size + 1):
            for col in range(1, self.board_size + 1):
                point = Point(row, col)
                
                # Convert to 0-indexed for tensor
                r_idx = row - 1
                c_idx = col - 1
                
                if board.is_wall(point):
                    encoded[1, r_idx, c_idx] = 1.0  # Wall channel
                else:
                    stone = board.get(point)
                    if stone is None:
                        encoded[0, r_idx, c_idx] = 1.0  # Empty channel
                    elif stone == current_player:
                        encoded[2, r_idx, c_idx] = 1.0  # Self channel
                    else:
                        encoded[3, r_idx, c_idx] = 1.0  # Opponent channel
        
        return encoded
    
    def forward(self, x):
        """
        Forward pass of the policy model.
        
        Args:
            x: Input tensor of shape (batch_size, 4, board_size, board_size)
            
        Returns:
            action_probs: Softmax probabilities over actions, shape (batch_size, num_actions)
        """
        # Ensure proper shape: (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Pass through residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Policy head
        x = F.relu(self.policy_bn(self.policy_conv(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        action_logits = self.policy_fc(x)
        action_probs = F.softmax(action_logits, dim=1)
        
        return action_probs
    
    def predict_move(self, game_state, current_player, temperature=1.0):
        """
        Predict move probabilities for a game state.
        
        Args:
            game_state: GameState object
            current_player: Player object
            temperature: Temperature for softmax (lower = more deterministic)
            
        Returns:
            action_probs: Move probabilities
            best_action: Index of highest probability action
        """
        self.eval()
        
        with torch.no_grad():
            # Encode the game state
            encoded_state = self.encode_game_state(game_state, current_player)
            
            # Move to same device as model
            device = next(self.parameters()).device
            encoded_state = encoded_state.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get action probabilities
            action_probs = self.forward(encoded_state)
            
            # Apply temperature
            if temperature != 1.0:
                action_logits = torch.log(action_probs + 1e-8)  # Avoid log(0)
                action_logits = action_logits / temperature
                action_probs = F.softmax(action_logits, dim=1)
            
            action_probs = action_probs.squeeze(0)  # Remove batch dimension
            best_action = torch.argmax(action_probs).item()
            
        return action_probs, best_action
    
    def action_to_point(self, action_idx):
        """
        Convert action index to Point on the board.
        
        Args:
            action_idx: Action index (0 to board_size^2 - 1 for board moves, board_size^2 for pass)
            
        Returns:
            Point object or None for pass
        """
        from goboard_fast import Point
        
        if action_idx >= self.num_positions:  # Pass move
            return None
        
        # Convert 1D index to 2D coordinates (1-indexed for Point)
        row = (action_idx // self.board_size) + 1
        col = (action_idx % self.board_size) + 1
        
        return Point(row, col)
    
    def point_to_action(self, point):
        """
        Convert Point to action index.
        
        Args:
            point: Point object or None for pass
            
        Returns:
            Action index
        """
        if point is None:  # Pass move
            return self.num_positions
        
        # Convert 2D coordinates to 1D index (Point is 1-indexed)
        row_idx = point.row - 1
        col_idx = point.col - 1
        
        return row_idx * self.board_size + col_idx
    
    def get_legal_actions(self, game_state):
        """
        Get legal action indices for the current game state.
        
        Args:
            game_state: GameState object
            
        Returns:
            List of legal action indices
        """
        from goboard_fast import Move
        
        legal_actions = []
        
        # Check all board positions
        for action_idx in range(self.num_positions):
            point = self.action_to_point(action_idx)
            move = Move.play(point)
            if game_state.is_valid_move(move):
                legal_actions.append(action_idx)
        
        # Pass is always legal
        legal_actions.append(self.num_positions)
        
        return legal_actions
    
    def mask_illegal_moves(self, action_probs, game_state):
        """
        Mask illegal moves by setting their probabilities to 0 and renormalizing.
        
        Args:
            action_probs: Tensor of action probabilities
            game_state: GameState object
            
        Returns:
            Masked and renormalized action probabilities
        """
        legal_actions = self.get_legal_actions(game_state)
        
        # Create mask
        mask = torch.zeros_like(action_probs)
        for action_idx in legal_actions:
            mask[action_idx] = 1.0
        
        # Apply mask and renormalize
        masked_probs = action_probs * mask
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)  # Avoid division by 0
        
        return masked_probs


# Convenience functions to create different model sizes
def create_small_policy_model(board_size):
    """Create a small policy model with 4 residual blocks."""
    return PolicyModel(board_size, num_filters=128, num_residual_blocks=4)


def create_medium_policy_model(board_size):
    """Create a medium policy model with 8 residual blocks."""
    return PolicyModel(board_size, num_filters=256, num_residual_blocks=8)


def create_large_policy_model(board_size):
    """Create a large policy model with 16 residual blocks."""
    return PolicyModel(board_size, num_filters=384, num_residual_blocks=16)


def create_deep_policy_model(board_size):
    """Create a deep policy model similar to AlphaGo Zero."""
    return PolicyModel(board_size, num_filters=256, num_residual_blocks=20)


# Example usage and testing
def test_policy_model():
    """Test the policy model with a sample game state."""
    from goboard_fast import GameState, Player
    
    print("Testing PolicyModel...")
    
    # Create model
    board_size = 9
    model = create_medium_policy_model(board_size)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create sample game state
    game_state = GameState.new_game(board_size)
    current_player = Player.black
    
    # Test prediction
    action_probs, best_action = model.predict_move(game_state, current_player)
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Best action index: {best_action}")
    
    # Convert to point
    best_point = model.action_to_point(best_action)
    print(f"Best point: {best_point}")
    
    # Test legal move masking
    legal_actions = model.get_legal_actions(game_state)
    print(f"Number of legal actions: {len(legal_actions)}")
    
    masked_probs = model.mask_illegal_moves(action_probs, game_state)
    print(f"Masked probabilities sum: {masked_probs.sum():.6f}")
    
    print("PolicyModel test completed successfully!")


if __name__ == "__main__":
    test_policy_model() 