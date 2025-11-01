import numpy as np
import random
from collections import defaultdict, deque
import json
import os
from typing import Tuple, List, Optional, Dict
from go_game import GoGame
import copy

class DeepQLearningAgent:
    """
    Deep Q-Learning Agent for Go using neural networks
    Implements true reinforcement learning with experience replay
    """
    
    def __init__(self, board_size: int = 9, learning_rate: float = 0.001, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995,
                 memory_size: int = 10000, batch_size: int = 32):
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = 0.95  # Discount factor
        
        # Neural network weights (simplified implementation)
        self.input_size = board_size * board_size * 3 + 4  # board state + game info
        self.hidden_size = 256
        self.output_size = board_size * board_size + 1  # all moves + pass
        
        # Initialize network weights
        self._initialize_network()
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Training statistics
        self.training_stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_reward': 0,
            'average_game_length': 0,
            'epsilon_history': [],
            'loss_history': []
        }
        
        # Q-value matrix for visualization
        self.q_matrix = np.zeros((board_size, board_size))
        self.thinking_process = []
        
    def _initialize_network(self):
        """Initialize neural network weights"""
        # Input layer to hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros(self.hidden_size)
        
        # Hidden layer to output layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(self.output_size)
        
        # Target network (for stable training)
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def _forward_pass(self, state, use_target=False):
        """Forward pass through the network"""
        W1 = self.target_W1 if use_target else self.W1
        b1 = self.target_b1 if use_target else self.b1
        W2 = self.target_W2 if use_target else self.W2
        b2 = self.target_b2 if use_target else self.b2
        
        # Hidden layer
        z1 = np.dot(state, W1) + b1
        a1 = self._relu(z1)
        
        # Output layer
        z2 = np.dot(a1, W2) + b2
        
        return z2, a1, z1
    
    def _get_state_representation(self, game: GoGame) -> np.ndarray:
        """Convert game state to neural network input"""
        board = game.get_board_state()
        current_player = game.current_player
        opponent = 3 - current_player
        
        # Create 3-channel representation
        state = np.zeros((self.board_size, self.board_size, 3))
        
        # Channel 0: Current player's stones
        state[:, :, 0] = (board == current_player).astype(float)
        
        # Channel 1: Opponent's stones
        state[:, :, 1] = (board == opponent).astype(float)
        
        # Channel 2: Empty positions
        state[:, :, 2] = (board == 0).astype(float)
        
        # Flatten and add game metadata
        flat_state = state.flatten()
        
        # Add game information
        game_info = np.array([
            current_player / 2.0,  # Normalize to [0.5, 1]
            len(game.game_history) / 100.0,  # Normalize move count
            game.captured_stones[1] / 10.0,  # Normalize captures
            game.captured_stones[2] / 10.0
        ])
        
        return np.concatenate([flat_state, game_info])
    
    def _get_q_values(self, state, use_target=False):
        """Get Q-values for all actions"""
        q_values, _, _ = self._forward_pass(state, use_target)
        return q_values
    
    def _get_valid_actions(self, game: GoGame) -> List[int]:
        """Get valid action indices"""
        valid_actions = []
        
        # Check all board positions
        for i in range(self.board_size):
            for j in range(self.board_size):
                if game.is_valid_move(i, j):
                    action_idx = i * self.board_size + j
                    valid_actions.append(action_idx)
        
        # Add pass action
        valid_actions.append(self.board_size * self.board_size)
        
        return valid_actions
    
    def _action_to_move(self, action_idx: int) -> Tuple[int, int]:
        """Convert action index to board move"""
        if action_idx == self.board_size * self.board_size:
            return (-1, -1)  # Pass move
        
        row = action_idx // self.board_size
        col = action_idx % self.board_size
        return (row, col)
    
    def _move_to_action(self, row: int, col: int) -> int:
        """Convert board move to action index"""
        if row == -1 and col == -1:
            return self.board_size * self.board_size  # Pass move
        
        return row * self.board_size + col
    
    def get_move_with_thinking(self, game: GoGame, show_thinking: bool = True) -> Tuple[Tuple[int, int], Dict]:
        """Get move with detailed thinking process for UI"""
        state = self._get_state_representation(game)
        q_values = self._get_q_values(state)
        valid_actions = self._get_valid_actions(game)
        
        # Update Q-matrix for visualization
        self.q_matrix = np.zeros((self.board_size, self.board_size))
        for i in range(self.board_size):
            for j in range(self.board_size):
                action_idx = i * self.board_size + j
                if action_idx in valid_actions:
                    self.q_matrix[i, j] = q_values[action_idx]
                else:
                    self.q_matrix[i, j] = np.nan
        
        # Thinking process
        thinking_data = {
            'q_matrix': self.q_matrix.copy(),
            'valid_moves': [],
            'top_moves': [],
            'strategy_explanation': '',
            'move_values': {}
        }
        
        # Analyze valid moves
        move_analysis = []
        for action_idx in valid_actions:
            row, col = self._action_to_move(action_idx)
            q_value = q_values[action_idx]
            
            if row != -1:  # Not a pass move
                move_analysis.append({
                    'position': (row, col),
                    'q_value': float(q_value),
                    'action_idx': action_idx
                })
                thinking_data['valid_moves'].append((row, col))
                thinking_data['move_values'][(row, col)] = float(q_value)
        
        # Sort by Q-value
        move_analysis.sort(key=lambda x: x['q_value'], reverse=True)
        
        # Top 3 moves
        thinking_data['top_moves'] = move_analysis[:3]
        
        # Strategy explanation
        if move_analysis:
            best_move = move_analysis[0]
            thinking_data['strategy_explanation'] = self._explain_move_strategy(
                game, best_move['position'], best_move['q_value']
            )
        
        # Choose action (epsilon-greedy for training, greedy for play)
        if random.random() < self.epsilon and len(valid_actions) > 1:
            # Exploration
            action_idx = random.choice(valid_actions)
            thinking_data['decision_type'] = 'exploration'
        else:
            # Exploitation
            valid_q_values = [q_values[idx] for idx in valid_actions]
            best_idx = np.argmax(valid_q_values)
            action_idx = valid_actions[best_idx]
            thinking_data['decision_type'] = 'exploitation'
        
        move = self._action_to_move(action_idx)
        thinking_data['chosen_move'] = move
        thinking_data['chosen_q_value'] = float(q_values[action_idx])
        
        return move, thinking_data
    
    def _explain_move_strategy(self, game: GoGame, position: Tuple[int, int], q_value: float) -> str:
        """Generate human-readable explanation of move strategy"""
        if position == (-1, -1):
            return "Passing turn - no beneficial moves available"
        
        row, col = position
        board = game.get_board_state()
        
        explanations = []
        
        # Check for captures
        temp_game = copy.deepcopy(game)
        captures_before = temp_game.captured_stones[temp_game.current_player]
        if temp_game.make_move(row, col):
            captures_after = temp_game.captured_stones[temp_game.current_player]
            if captures_after > captures_before:
                captured = captures_after - captures_before
                explanations.append(f"Captures {captured} opponent stone(s)")
        
        # Check position type
        center = self.board_size // 2
        distance_from_center = abs(row - center) + abs(col - center)
        
        if distance_from_center <= 2:
            explanations.append("Controls center territory")
        elif row == 0 or row == self.board_size - 1 or col == 0 or col == self.board_size - 1:
            explanations.append("Edge play for territory")
        
        # Check connections
        friendly_neighbors = 0
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.board_size and 0 <= nc < self.board_size and
                board[nr, nc] == game.current_player):
                friendly_neighbors += 1
        
        if friendly_neighbors > 0:
            explanations.append(f"Connects to {friendly_neighbors} friendly stone(s)")
        
        # Q-value interpretation
        if q_value > 0.5:
            explanations.append("High strategic value")
        elif q_value > 0:
            explanations.append("Positive strategic value")
        else:
            explanations.append("Defensive or tactical move")
        
        return "; ".join(explanations) if explanations else "Strategic positioning"
    
    def get_move(self, game: GoGame) -> Tuple[int, int]:
        """Get move for gameplay (without detailed thinking)"""
        move, _ = self.get_move_with_thinking(game, show_thinking=False)
        return move
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Get current Q-values
        current_q_values = np.array([self._get_q_values(state) for state in states])
        
        # Get next Q-values from target network
        next_q_values = np.array([self._get_q_values(state, use_target=True) for state in next_states])
        
        # Calculate targets
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train network using gradient descent
        total_loss = 0
        for i in range(self.batch_size):
            loss = self._backward_pass(states[i], targets[i])
            total_loss += loss
        
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        
        average_loss = total_loss / self.batch_size
        self.training_stats['loss_history'].append(average_loss)
        
        return average_loss
    
    def _backward_pass(self, state, target_q_values):
        """Backward pass for gradient descent"""
        # Forward pass
        q_values, hidden_activations, hidden_pre_activations = self._forward_pass(state)
        
        # Calculate loss (mean squared error)
        loss = np.mean((q_values - target_q_values) ** 2)
        
        # Backward pass
        # Output layer gradients
        dq = 2 * (q_values - target_q_values) / len(q_values)
        
        dW2 = np.outer(hidden_activations, dq)
        db2 = dq
        
        # Hidden layer gradients
        dhidden = np.dot(dq, self.W2.T)
        dhidden_pre = dhidden * self._relu_derivative(hidden_pre_activations)
        
        dW1 = np.outer(state, dhidden_pre)
        db1 = dhidden_pre
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
        return loss
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
    
    def calculate_reward(self, game: GoGame, action: int, game_ended: bool = False) -> float:
        """Calculate reward for an action"""
        if game_ended:
            # Game end reward
            black_score, white_score = game.calculate_score()
            if game.current_player == 1:  # Agent is black
                return 10 if black_score > white_score else -10 if white_score > black_score else 0
            else:  # Agent is white
                return 10 if white_score > black_score else -10 if black_score > white_score else 0
        
        # Immediate rewards
        row, col = self._action_to_move(action)
        
        if row == -1:  # Pass move
            return -0.1
        
        # Small positive reward for valid moves
        reward = 0.01
        
        # Reward for captures (calculated by checking game state change)
        # This would need to be implemented based on actual game state changes
        
        return reward
    
    def train_episode(self, opponent=None):
        """Train for one episode"""
        game = GoGame(self.board_size)
        episode_memory = []
        total_reward = 0
        moves_made = 0
        
        while not game.is_game_over() and moves_made < self.board_size * self.board_size * 2:
            # Get current state
            state = self._get_state_representation(game)
            
            # Choose action
            valid_actions = self._get_valid_actions(game)
            if not valid_actions:
                break
            
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                q_values = self._get_q_values(state)
                valid_q_values = [q_values[idx] for idx in valid_actions]
                best_idx = np.argmax(valid_q_values)
                action = valid_actions[best_idx]
            
            # Execute action
            row, col = self._action_to_move(action)
            if row == -1:
                game.pass_turn()
            else:
                if not game.make_move(row, col):
                    # Invalid move penalty
                    reward = -1
                    episode_memory.append((state, action, reward, state, True))
                    break
            
            # Get next state
            next_state = self._get_state_representation(game)
            
            # Calculate reward
            game_ended = game.is_game_over()
            reward = self.calculate_reward(game, action, game_ended)
            total_reward += reward
            
            # Store experience
            episode_memory.append((state, action, reward, next_state, game_ended))
            
            moves_made += 1
            
            # Opponent move (if provided)
            if opponent and not game.is_game_over():
                opponent_move = opponent.get_move(game)
                if opponent_move == (-1, -1):
                    game.pass_turn()
                else:
                    game.make_move(opponent_move[0], opponent_move[1])
        
        # Add all experiences to memory
        for experience in episode_memory:
            self.remember(*experience)
        
        # Train network
        loss = self.replay()
        
        # Update statistics
        self.training_stats['games_played'] += 1
        self.training_stats['total_reward'] += total_reward
        
        # Determine winner for statistics
        if game.is_game_over():
            black_score, white_score = game.calculate_score()
            if black_score > white_score:
                self.training_stats['wins'] += 1
            elif white_score > black_score:
                self.training_stats['losses'] += 1
            else:
                self.training_stats['draws'] += 1
        
        return loss, total_reward, moves_made
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'target_W1': self.target_W1.tolist(),
            'target_b1': self.target_b1.tolist(),
            'target_W2': self.target_W2.tolist(),
            'target_b2': self.target_b2.tolist(),
            'epsilon': self.epsilon,
            'training_stats': self.training_stats,
            'board_size': self.board_size,
            'learning_rate': self.learning_rate
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.W1 = np.array(model_data['W1'])
            self.b1 = np.array(model_data['b1'])
            self.W2 = np.array(model_data['W2'])
            self.b2 = np.array(model_data['b2'])
            self.target_W1 = np.array(model_data['target_W1'])
            self.target_b1 = np.array(model_data['target_b1'])
            self.target_W2 = np.array(model_data['target_W2'])
            self.target_b2 = np.array(model_data['target_b2'])
            self.epsilon = model_data['epsilon']
            self.training_stats = model_data['training_stats']
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def train_rl_agent(board_size: int = 9, episodes: int = 1000, save_interval: int = 100) -> DeepQLearningAgent:
    """Train a Deep Q-Learning agent"""
    print(f"Training Deep Q-Learning agent on {board_size}x{board_size} board...")
    
    agent = DeepQLearningAgent(board_size=board_size)
    
    # Load existing model if available
    model_path = f"dql_agent_{board_size}x{board_size}.json"
    if agent.load_model(model_path):
        print("Loaded existing model, continuing training...")
    
    best_win_rate = 0
    
    for episode in range(episodes):
        loss, reward, moves = agent.train_episode()
        
        # Update target network periodically
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Print progress
        if (episode + 1) % save_interval == 0:
            games = agent.training_stats['games_played']
            wins = agent.training_stats['wins']
            win_rate = wins / max(games, 1) * 100
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Win rate: {win_rate:.1f}% ({wins}/{games})")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Avg reward: {reward:.3f}")
            print(f"Loss: {loss:.6f}" if loss > 0 else "Loss: N/A")
            print("-" * 40)
            
            # Save model
            agent.save_model(model_path)
            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_model_path = f"best_dql_agent_{board_size}x{board_size}.json"
                agent.save_model(best_model_path)
    
    # Final save
    agent.save_model(model_path)
    print(f"Training completed! Model saved to {model_path}")
    
    return agent

# For backward compatibility
class GoAI:
    """Wrapper class for backward compatibility"""
    
    def __init__(self, board_size: int = 19, difficulty: str = "medium"):
        self.agent = DeepQLearningAgent(board_size=board_size)
        self.board_size = board_size
        self.difficulty = difficulty
        
        # Load pre-trained model if available
        model_path = f"dql_agent_{board_size}x{board_size}.json"
        if not self.agent.load_model(model_path):
            # Train a quick model if none exists
            print(f"No trained model found for {board_size}x{board_size}. Training...")
            train_rl_agent(board_size=board_size, episodes=100)
            self.agent.load_model(model_path)
    
    def get_move(self, game_state):
        """Get move for compatibility"""
        return self.agent.get_move(game_state)
    
    def get_move_with_thinking(self, game_state):
        """Get move with thinking process"""
        return self.agent.get_move_with_thinking(game_state)
    
    def should_pass(self, game_state):
        """Determine if should pass"""
        move, _ = self.agent.get_move_with_thinking(game_state)
        return move == (-1, -1)

if __name__ == "__main__":
    # Example training
    agent = train_rl_agent(board_size=9, episodes=500)
    print("Training completed!")