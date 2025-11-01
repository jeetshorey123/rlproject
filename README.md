# Alpha Go Game

A sophisticated Go game implementation with AI opponent using reinforcement learning concepts, built with Streamlit for an interactive web interface.

## Features

### 🎮 Game Features
- **Full Go Rules Implementation**: Complete rule set including capture, ko rule, territory scoring
- **Multiple Board Sizes**: Choose from 9×9, 13×13, or 19×19 boards
- **Player Customization**: Enter your name and choose your color (Black or White)
- **AI Difficulty Levels**: Easy, Medium, Hard, and Expert levels
- **Interactive UI**: Beautiful, responsive web interface built with Streamlit

### 🤖 AI Features
- **Neural Network**: Simplified AlphaGo-style neural network architecture
- **Monte Carlo Tree Search (MCTS)**: Intelligent move selection using MCTS algorithm
- **Reinforcement Learning Concepts**: Position evaluation and policy prediction
- **Adaptive Difficulty**: AI strength scales with selected difficulty level

### 📊 Game Interface
- **Real-time Board Visualization**: Interactive board with Plotly graphics
- **Score Tracking**: Live score calculation and display
- **Move History**: Complete game move log
- **Game Statistics**: Captured stones, valid moves, and more

## Installation

1. **Clone or download the project files**
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Game

1. **Start the Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your web browser** and navigate to the displayed URL (usually `http://localhost:8501`)

3. **Set up your game**:
   - Enter your player name
   - Choose your color (Black goes first)
   - Select board size (9×9 for quick games, 19×19 for full experience)
   - Pick AI difficulty level

4. **Start playing**:
   - Click on board intersections to place stones
   - Use "Pass" button to skip turns
   - Use "Resign" to concede the game

## Game Rules

### Basic Rules
- **Objective**: Control more territory than your opponent
- **Turns**: Players alternate placing stones on board intersections
- **Capture**: Stones with no liberties (empty adjacent points) are captured
- **Ko Rule**: Cannot immediately recapture in the same position
- **End Game**: Game ends when both players pass consecutively

### Scoring
- **Territory**: Empty areas surrounded by your stones
- **Stones**: Stones remaining on the board
- **Captures**: Opponent stones you've captured
- **Winner**: Player with the highest total score

## AI Implementation

### Neural Network Architecture
- **Input**: 3-channel board representation (player stones, opponent stones, turn indicator)
- **Residual Blocks**: 8 residual convolutional layers for feature extraction
- **Policy Head**: Predicts move probabilities for all board positions
- **Value Head**: Evaluates position strength (-1 to +1)

### Monte Carlo Tree Search
- **Simulation Count**: Varies by difficulty (50-1000 simulations)
- **UCB1 Selection**: Balances exploration and exploitation
- **Neural Network Integration**: Uses NN for position evaluation and move priors
- **Tree Expansion**: Dynamically expands game tree during search

### Difficulty Levels
- **Easy**: 50 simulations, 30% random moves
- **Medium**: 200 simulations, strategic play
- **Hard**: 500 simulations, strong tactical play
- **Expert**: 1000 simulations, near-optimal play

## File Structure

```
go/
├── requirements.txt      # Python dependencies
├── streamlit_app.py     # Main Streamlit application
├── go_game.py          # Core Go game logic
├── ai_player.py        # AI implementation (Neural Network + MCTS)
└── README.md           # This file
```

## Technical Details

### Technologies Used
- **Streamlit**: Web interface framework
- **PyTorch**: Neural network implementation
- **NumPy**: Numerical computations
- **Plotly**: Interactive board visualization
- **Python**: Core programming language

### Performance Considerations
- Neural network uses random weights (in production, would use pre-trained weights)
- MCTS tree is reset between moves for memory efficiency
- Board state caching for faster game rule evaluation
- Difficulty scaling balances performance vs. playing strength

## Future Enhancements

### Potential Improvements
- **Pre-trained Weights**: Load actual AlphaGo-style trained neural networks
- **Opening Book**: Database of professional opening sequences
- **Persistent Game Storage**: Save and load games
- **Multiplayer Support**: Human vs. human gameplay
- **Advanced Analytics**: Move analysis and suggestions
- **Mobile Optimization**: Responsive design for mobile devices

### Advanced Features
- **Handicap System**: Stone advantage for weaker players
- **Time Controls**: Add game clocks and time pressure
- **Review Mode**: Analyze completed games
- **Training Mode**: Guided tutorials and puzzles

## Contributing

Feel free to enhance the game by:
- Improving the neural network architecture
- Adding new features to the UI
- Optimizing the MCTS algorithm
- Adding more game variants

## License

This project is open-source and available for educational and personal use.

## Acknowledgments

- Inspired by DeepMind's AlphaGo research
- Go rules implementation based on traditional Weiqi/Baduk
- Neural network architecture simplified from AlphaGo Zero paper
- Monte Carlo Tree Search algorithm adapted for Go gameplay

---

**Enjoy playing Go against the AI!** 🎮⚫⚪