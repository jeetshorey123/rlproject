import numpy as np
import random
from typing import List, Tuple, Optional
from go_game import GoGame
import copy

class SimpleGoAI:
    """
    Simplified Go AI that doesn't require PyTorch
    Uses basic heuristics and random selection for gameplay
    """
    
    def __init__(self, board_size: int = 19, difficulty: str = "medium"):
        self.board_size = board_size
        self.difficulty = difficulty
        
        # Adjust parameters based on difficulty
        self.evaluation_depth = {
            "easy": 1,
            "medium": 2,
            "hard": 3,
            "expert": 4
        }.get(difficulty, 2)
        
        self.random_move_probability = {
            "easy": 0.4,
            "medium": 0.2,
            "hard": 0.1,
            "expert": 0.05
        }.get(difficulty, 0.2)
    
    def get_move(self, game_state: GoGame) -> Tuple[int, int]:
        """Get AI move for current position"""
        valid_moves = game_state.get_valid_moves()
        
        if not valid_moves:
            return (-1, -1)  # Pass move
        
        # Use random move for easy difficulty or occasionally for variety
        if (self.difficulty == "easy" and random.random() < self.random_move_probability) or \
           (random.random() < self.random_move_probability):
            return random.choice(valid_moves)
        
        # Use strategic move selection
        return self._select_strategic_move(game_state, valid_moves)
    
    def _select_strategic_move(self, game_state: GoGame, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Select move using basic Go heuristics"""
        board = game_state.get_board_state()
        current_player = game_state.current_player
        opponent = 3 - current_player
        
        # Score each valid move
        move_scores = []
        
        for row, col in valid_moves:
            score = 0
            
            # 1. Capture opponent stones (high priority)
            capture_score = self._evaluate_captures(game_state, row, col)
            score += capture_score * 100
            
            # 2. Defend own groups (high priority)
            defense_score = self._evaluate_defense(game_state, row, col)
            score += defense_score * 80
            
            # 3. Expand territory (medium priority)
            territory_score = self._evaluate_territory(board, row, col, current_player)
            score += territory_score * 30
            
            # 4. Control center and key points (medium priority)
            center_score = self._evaluate_center_control(row, col)
            score += center_score * 20
            
            # 5. Connect friendly stones (medium priority)
            connection_score = self._evaluate_connections(board, row, col, current_player)
            score += connection_score * 25
            
            # 6. Attack opponent groups (lower priority)
            attack_score = self._evaluate_attacks(game_state, row, col)
            score += attack_score * 15
            
            # Add small random factor for unpredictability
            score += random.uniform(-5, 5)
            
            move_scores.append((score, row, col))
        
        # Sort by score and return best move
        move_scores.sort(reverse=True, key=lambda x: x[0])
        
        # For higher difficulties, always pick the best move
        # For lower difficulties, sometimes pick from top few moves
        if self.difficulty in ["hard", "expert"]:
            return (move_scores[0][1], move_scores[0][2])
        else:
            # Pick from top 3 moves with some randomness
            top_moves = move_scores[:min(3, len(move_scores))]
            chosen_move = random.choice(top_moves)
            return (chosen_move[1], chosen_move[2])
    
    def _evaluate_captures(self, game_state: GoGame, row: int, col: int) -> int:
        """Evaluate potential captures from this move"""
        # Create temporary game state to test the move
        temp_game = copy.deepcopy(game_state)
        if not temp_game.make_move(row, col):
            return 0
        
        # Count captured stones
        captured_before = game_state.captured_stones[game_state.current_player]
        captured_after = temp_game.captured_stones[temp_game.current_player]
        
        return captured_after - captured_before
    
    def _evaluate_defense(self, game_state: GoGame, row: int, col: int) -> int:
        """Evaluate defensive value of this move"""
        board = game_state.get_board_state()
        current_player = game_state.current_player
        score = 0
        
        # Check if this move saves friendly groups in atari
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.board_size and 0 <= nc < self.board_size and
                board[nr, nc] == current_player):
                
                group = game_state._get_group(board, nr, nc)
                liberties = game_state._count_liberties(board, group)
                
                if liberties == 1:  # Group in atari
                    score += len(group) * 2  # Saving larger groups is more valuable
        
        return score
    
    def _evaluate_territory(self, board: np.ndarray, row: int, col: int, player: int) -> int:
        """Evaluate territorial value of this move"""
        score = 0
        
        # Count empty points around this position
        empty_neighbors = 0
        friendly_neighbors = 0
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if board[nr, nc] == 0:
                    empty_neighbors += 1
                elif board[nr, nc] == player:
                    friendly_neighbors += 1
        
        # Prefer moves with more empty space (potential territory)
        score += empty_neighbors * 2
        score += friendly_neighbors
        
        return score
    
    def _evaluate_center_control(self, row: int, col: int) -> int:
        """Evaluate how much this move controls the center"""
        center = self.board_size // 2
        distance_from_center = abs(row - center) + abs(col - center)
        
        # Closer to center is better (but not always the best strategy)
        max_distance = self.board_size
        return max_distance - distance_from_center
    
    def _evaluate_connections(self, board: np.ndarray, row: int, col: int, player: int) -> int:
        """Evaluate how well this move connects friendly stones"""
        friendly_neighbors = 0
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.board_size and 0 <= nc < self.board_size and
                board[nr, nc] == player):
                friendly_neighbors += 1
        
        return friendly_neighbors * 3
    
    def _evaluate_attacks(self, game_state: GoGame, row: int, col: int) -> int:
        """Evaluate attacking potential of this move"""
        board = game_state.get_board_state()
        opponent = 3 - game_state.current_player
        score = 0
        
        # Check if this move threatens opponent groups
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.board_size and 0 <= nc < self.board_size and
                board[nr, nc] == opponent):
                
                group = game_state._get_group(board, nr, nc)
                liberties = game_state._count_liberties(board, group)
                
                if liberties <= 2:  # Threatening group with few liberties
                    score += len(group)
        
        return score
    
    def should_pass(self, game_state: GoGame) -> bool:
        """Determine if AI should pass"""
        valid_moves = game_state.get_valid_moves()
        
        # Pass if no valid moves
        if not valid_moves:
            return True
        
        # For expert difficulty, sometimes pass strategically
        if self.difficulty == "expert":
            # Very basic endgame detection - pass if very few moves left
            if len(valid_moves) < 5:
                return random.random() < 0.3
        
        # Random pass for variety (small probability)
        return random.random() < 0.02

# For backward compatibility, create an alias
GoAI = SimpleGoAI