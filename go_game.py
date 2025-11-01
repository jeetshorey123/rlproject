import numpy as np
from typing import Tuple, Optional, List
import copy

class GoGame:
    """
    Core Go game logic implementation
    """
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)  # 0: empty, 1: black, 2: white
        self.current_player = 1  # 1: black, 2: white
        self.captured_stones = {1: 0, 2: 0}  # captured stones count
        self.game_history = []
        self.ko_rule_board = None  # for ko rule implementation
        self.territory_points = {1: 0, 2: 0}
        
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid"""
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        
        if self.board[row, col] != 0:
            return False
        
        # Create a temporary board to test the move
        temp_board = copy.deepcopy(self.board)
        temp_board[row, col] = self.current_player
        
        # Check if the move captures opponent stones
        opponent = 3 - self.current_player
        captured_groups = []
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.board_size and 0 <= nc < self.board_size and 
                temp_board[nr, nc] == opponent):
                group = self._get_group(temp_board, nr, nc)
                if self._count_liberties(temp_board, group) == 0:
                    captured_groups.extend(group)
        
        # Remove captured stones
        for cr, cc in captured_groups:
            temp_board[cr, cc] = 0
        
        # Check if our own group has liberties (suicide rule)
        our_group = self._get_group(temp_board, row, col)
        if self._count_liberties(temp_board, our_group) == 0:
            # Suicide is only allowed if it captures opponent stones
            return len(captured_groups) > 0
        
        # Check ko rule
        if self.ko_rule_board is not None and np.array_equal(temp_board, self.ko_rule_board):
            return False
        
        return True
    
    def make_move(self, row: int, col: int) -> bool:
        """Make a move on the board"""
        if not self.is_valid_move(row, col):
            return False
        
        # Store current board state for ko rule
        previous_board = copy.deepcopy(self.board)
        
        # Place the stone
        self.board[row, col] = self.current_player
        
        # Capture opponent stones
        opponent = 3 - self.current_player
        captured_count = 0
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.board_size and 0 <= nc < self.board_size and 
                self.board[nr, nc] == opponent):
                group = self._get_group(self.board, nr, nc)
                if self._count_liberties(self.board, group) == 0:
                    # Capture the group
                    for cr, cc in group:
                        self.board[cr, cc] = 0
                        captured_count += 1
        
        self.captured_stones[self.current_player] += captured_count
        
        # Update ko rule board
        self.ko_rule_board = previous_board if captured_count == 1 else None
        
        # Add to game history
        self.game_history.append((row, col, self.current_player))
        
        # Switch players
        self.current_player = 3 - self.current_player
        
        return True
    
    def _get_group(self, board: np.ndarray, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all stones in the same group (connected component)"""
        if board[row, col] == 0:
            return []
        
        color = board[row, col]
        group = []
        visited = set()
        stack = [(row, col)]
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            
            visited.add((r, c))
            group.append((r, c))
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.board_size and 0 <= nc < self.board_size and
                    board[nr, nc] == color and (nr, nc) not in visited):
                    stack.append((nr, nc))
        
        return group
    
    def _count_liberties(self, board: np.ndarray, group: List[Tuple[int, int]]) -> int:
        """Count liberties (empty adjacent points) for a group"""
        liberties = set()
        
        for row, col in group:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.board_size and 0 <= nc < self.board_size and
                    board[nr, nc] == 0):
                    liberties.add((nr, nc))
        
        return len(liberties)
    
    def pass_turn(self):
        """Pass the current turn"""
        self.game_history.append((-1, -1, self.current_player))  # -1, -1 represents pass
        self.current_player = 3 - self.current_player
    
    def is_game_over(self) -> bool:
        """Check if game is over (two consecutive passes)"""
        if len(self.game_history) < 2:
            return False
        
        last_two_moves = self.game_history[-2:]
        return (last_two_moves[0][0] == -1 and last_two_moves[0][1] == -1 and
                last_two_moves[1][0] == -1 and last_two_moves[1][1] == -1)
    
    def calculate_score(self) -> Tuple[int, int]:
        """Calculate final score using area scoring"""
        # Create a copy of the board for territory calculation
        score_board = copy.deepcopy(self.board)
        
        # Find territories
        visited = np.zeros((self.board_size, self.board_size), dtype=bool)
        black_territory = 0
        white_territory = 0
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if not visited[row, col] and score_board[row, col] == 0:
                    territory, surrounding_colors = self._get_territory(score_board, row, col, visited)
                    
                    # If territory is surrounded by only one color, award points
                    if len(surrounding_colors) == 1:
                        color = list(surrounding_colors)[0]
                        if color == 1:  # Black territory
                            black_territory += len(territory)
                        elif color == 2:  # White territory
                            white_territory += len(territory)
        
        # Count stones on board
        black_stones = np.sum(self.board == 1)
        white_stones = np.sum(self.board == 2)
        
        # Add captured stones
        black_score = black_stones + black_territory + self.captured_stones[1]
        white_score = white_stones + white_territory + self.captured_stones[2]
        
        return black_score, white_score
    
    def _get_territory(self, board: np.ndarray, row: int, col: int, visited: np.ndarray) -> Tuple[List[Tuple[int, int]], set]:
        """Get territory and surrounding colors"""
        territory = []
        surrounding_colors = set()
        stack = [(row, col)]
        local_visited = set()
        
        while stack:
            r, c = stack.pop()
            if (r, c) in local_visited or visited[r, c]:
                continue
            
            local_visited.add((r, c))
            visited[r, c] = True
            
            if board[r, c] == 0:
                territory.append((r, c))
                
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        if board[nr, nc] == 0 and (nr, nc) not in local_visited:
                            stack.append((nr, nc))
                        elif board[nr, nc] != 0:
                            surrounding_colors.add(board[nr, nc])
        
        return territory, surrounding_colors
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves for current player"""
        valid_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_valid_move(row, col):
                    valid_moves.append((row, col))
        return valid_moves
    
    def get_board_state(self) -> np.ndarray:
        """Get current board state"""
        return copy.deepcopy(self.board)
    
    def reset_game(self):
        """Reset the game to initial state"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.captured_stones = {1: 0, 2: 0}
        self.game_history = []
        self.ko_rule_board = None
        self.territory_points = {1: 0, 2: 0}