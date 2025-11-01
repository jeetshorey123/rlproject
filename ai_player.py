import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple, Optional
from go_game import GoGame
import copy

class GoNeuralNetwork(nn.Module):
    """
    Neural network for Go move prediction and position evaluation
    Simplified version of AlphaGo's architecture
    """
    
    def __init__(self, board_size: int = 19, num_channels: int = 128):
        super(GoNeuralNetwork, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        
        # Input processing
        self.input_conv = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(num_channels) for _ in range(8)
        ])
        
        # Policy head (move prediction)
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)  # +1 for pass move
        
        # Value head (position evaluation)
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def _make_residual_block(self, channels: int):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        # Input processing
        x = F.relu(self.input_conv(x))
        
        # Residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)
        
        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class MonteCarloTreeSearch:
    """
    Monte Carlo Tree Search implementation for Go AI
    """
    
    def __init__(self, neural_network: GoNeuralNetwork, num_simulations: int = 100, c_puct: float = 1.0):
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.tree = {}  # Dictionary to store tree nodes
        
    def search(self, game_state: GoGame) -> Tuple[int, int]:
        """Perform MCTS and return best move"""
        root_key = self._get_state_key(game_state)
        
        for _ in range(self.num_simulations):
            game_copy = copy.deepcopy(game_state)
            self._simulate(game_copy, root_key)
        
        # Select move with highest visit count
        if root_key not in self.tree:
            return self._get_random_valid_move(game_state)
        
        best_move = None
        best_visits = -1
        
        for move, stats in self.tree[root_key]['children'].items():
            if stats['visits'] > best_visits:
                best_visits = stats['visits']
                best_move = move
        
        return best_move if best_move and best_move != 'pass' else self._get_random_valid_move(game_state)
    
    def _simulate(self, game_state: GoGame, state_key: str) -> float:
        """Single MCTS simulation"""
        if game_state.is_game_over():
            black_score, white_score = game_state.calculate_score()
            return 1.0 if black_score > white_score else -1.0
        
        if state_key not in self.tree:
            # Leaf node - expand and evaluate
            self._expand_node(game_state, state_key)
            return self._evaluate_position(game_state)
        
        # Select best child using UCB1
        move = self._select_move(game_state, state_key)
        
        if move == 'pass':
            game_state.pass_turn()
        else:
            row, col = move
            game_state.make_move(row, col)
        
        child_key = self._get_state_key(game_state)
        value = -self._simulate(game_state, child_key)  # Negate for opponent's perspective
        
        # Backpropagate
        self._backpropagate(state_key, move, value)
        
        return value
    
    def _expand_node(self, game_state: GoGame, state_key: str):
        """Expand tree node with all possible moves"""
        valid_moves = game_state.get_valid_moves()
        valid_moves.append('pass')  # Add pass move
        
        self.tree[state_key] = {
            'children': {},
            'visits': 0,
            'value_sum': 0.0
        }
        
        # Get neural network predictions
        policy_probs, _ = self._get_nn_predictions(game_state)
        
        for move in valid_moves:
            if move == 'pass':
                prior = policy_probs[-1] if len(policy_probs) > game_state.board_size ** 2 else 0.1
            else:
                row, col = move
                move_index = row * game_state.board_size + col
                prior = policy_probs[move_index] if move_index < len(policy_probs) else 0.1
            
            self.tree[state_key]['children'][move] = {
                'visits': 0,
                'value_sum': 0.0,
                'prior': prior
            }
    
    def _select_move(self, game_state: GoGame, state_key: str):
        """Select move using UCB1 formula"""
        node = self.tree[state_key]
        best_move = None
        best_ucb = float('-inf')
        
        total_visits = max(1, node['visits'])
        
        for move, child_stats in node['children'].items():
            if child_stats['visits'] == 0:
                ucb_value = float('inf')
            else:
                q_value = child_stats['value_sum'] / child_stats['visits']
                u_value = (self.c_puct * child_stats['prior'] * 
                          np.sqrt(total_visits) / (1 + child_stats['visits']))
                ucb_value = q_value + u_value
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_move = move
        
        return best_move
    
    def _backpropagate(self, state_key: str, move, value: float):
        """Update statistics back up the tree"""
        if state_key in self.tree:
            node = self.tree[state_key]
            node['visits'] += 1
            node['value_sum'] += value
            
            if move in node['children']:
                child = node['children'][move]
                child['visits'] += 1
                child['value_sum'] += value
    
    def _evaluate_position(self, game_state: GoGame) -> float:
        """Evaluate position using neural network"""
        _, value = self._get_nn_predictions(game_state)
        return value
    
    def _get_nn_predictions(self, game_state: GoGame) -> Tuple[np.ndarray, float]:
        """Get neural network predictions for current position"""
        input_tensor = self._game_to_tensor(game_state)
        
        with torch.no_grad():
            policy, value = self.neural_network(input_tensor.unsqueeze(0))
            policy = policy.squeeze(0).numpy()
            value = value.item()
        
        return policy, value
    
    def _game_to_tensor(self, game_state: GoGame) -> torch.Tensor:
        """Convert game state to neural network input tensor"""
        board = game_state.get_board_state()
        
        # Create 3-channel input: [current_player_stones, opponent_stones, current_player_turn]
        input_tensor = torch.zeros(3, game_state.board_size, game_state.board_size)
        
        # Channel 0: Current player's stones
        input_tensor[0] = torch.tensor((board == game_state.current_player).astype(float))
        
        # Channel 1: Opponent's stones
        opponent = 3 - game_state.current_player
        input_tensor[1] = torch.tensor((board == opponent).astype(float))
        
        # Channel 2: Current player indicator (all 1s if black's turn, all 0s if white's turn)
        if game_state.current_player == 1:
            input_tensor[2] = torch.ones(game_state.board_size, game_state.board_size)
        
        return input_tensor
    
    def _get_state_key(self, game_state: GoGame) -> str:
        """Generate unique key for game state"""
        board_str = ''.join(map(str, game_state.board.flatten()))
        return f"{board_str}_{game_state.current_player}"
    
    def _get_random_valid_move(self, game_state: GoGame) -> Tuple[int, int]:
        """Get random valid move as fallback"""
        valid_moves = game_state.get_valid_moves()
        if valid_moves:
            return random.choice(valid_moves)
        return (-1, -1)  # Pass move

class GoAI:
    """
    Main AI class that combines neural network and MCTS
    """
    
    def __init__(self, board_size: int = 19, difficulty: str = "medium"):
        self.board_size = board_size
        self.difficulty = difficulty
        self.neural_network = GoNeuralNetwork(board_size)
        
        # Adjust MCTS parameters based on difficulty
        simulation_counts = {
            "easy": 50,
            "medium": 200,
            "hard": 500,
            "expert": 1000
        }
        
        self.mcts = MonteCarloTreeSearch(
            self.neural_network,
            num_simulations=simulation_counts.get(difficulty, 200)
        )
        
        # Initialize with random weights (in a real implementation, you would load pre-trained weights)
        self._initialize_random_weights()
    
    def _initialize_random_weights(self):
        """Initialize neural network with random weights"""
        # In a real implementation, you would load pre-trained weights here
        # For now, we'll use random initialization
        for module in self.neural_network.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def get_move(self, game_state: GoGame) -> Tuple[int, int]:
        """Get AI move for current position"""
        if self.difficulty == "easy":
            # For easy mode, use more random moves
            if random.random() < 0.3:
                return self._get_random_move(game_state)
        
        # Use MCTS for move selection
        move = self.mcts.search(game_state)
        
        # Fallback to random move if MCTS fails
        if move == (-1, -1) or not game_state.is_valid_move(move[0], move[1]):
            return self._get_random_move(game_state)
        
        return move
    
    def _get_random_move(self, game_state: GoGame) -> Tuple[int, int]:
        """Get random valid move"""
        valid_moves = game_state.get_valid_moves()
        if valid_moves:
            return random.choice(valid_moves)
        return (-1, -1)  # Pass move
    
    def should_pass(self, game_state: GoGame) -> bool:
        """Determine if AI should pass"""
        valid_moves = game_state.get_valid_moves()
        
        # Pass if no valid moves
        if not valid_moves:
            return True
        
        # For higher difficulties, use neural network to decide
        if self.difficulty in ["hard", "expert"]:
            _, value = self.mcts._get_nn_predictions(game_state)
            # Pass if position evaluation is very negative
            return value < -0.8
        
        # Random pass for lower difficulties (small probability)
        return random.random() < 0.05