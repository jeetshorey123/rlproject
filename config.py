# Alpha Go Game Configuration

# Neural Network Settings
NEURAL_NETWORK_CONFIG = {
    "num_channels": 128,
    "num_residual_blocks": 8,
    "learning_rate": 0.001,
    "batch_size": 32
}

# Monte Carlo Tree Search Settings
MCTS_CONFIG = {
    "easy": {
        "simulations": 50,
        "c_puct": 1.0,
        "random_move_probability": 0.3
    },
    "medium": {
        "simulations": 200,
        "c_puct": 1.0,
        "random_move_probability": 0.1
    },
    "hard": {
        "simulations": 500,
        "c_puct": 1.4,
        "random_move_probability": 0.05
    },
    "expert": {
        "simulations": 1000,
        "c_puct": 1.4,
        "random_move_probability": 0.02
    }
}

# Game Settings
GAME_CONFIG = {
    "default_board_size": 19,
    "available_board_sizes": [9, 13, 19],
    "default_difficulty": "medium",
    "komi": 6.5,  # Compensation points for white player
    "enable_ko_rule": True,
    "enable_suicide_rule": False,
    "scoring_method": "area"  # "area" or "territory"
}

# UI Settings
UI_CONFIG = {
    "theme": "dark",
    "board_colors": {
        "background": "#DEB887",
        "lines": "#8B4513",
        "star_points": "#654321"
    },
    "stone_colors": {
        "black": "#000000",
        "white": "#FFFFFF",
        "black_border": "#333333",
        "white_border": "#CCCCCC"
    },
    "animation_speed": 0.3,
    "show_coordinates": True,
    "show_last_move": True,
    "highlight_valid_moves": False
}

# Performance Settings
PERFORMANCE_CONFIG = {
    "max_game_history": 1000,
    "cache_neural_network_predictions": True,
    "parallel_mcts_simulations": False,
    "memory_limit_mb": 512
}

# Training Settings (for future enhancements)
TRAINING_CONFIG = {
    "save_game_data": False,
    "training_data_path": "./training_data/",
    "model_save_path": "./models/",
    "auto_save_interval": 100,  # games
    "validation_split": 0.2
}