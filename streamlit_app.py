import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from go_game import GoGame
import time
from typing import Optional, Tuple, Dict
import pandas as pd

# Try to import the RL AI first, then fall back to others
AI_TYPE = "Simple Heuristic"
GoAI = None

try:
    from rl_agent import DeepQLearningAgent, GoAI, train_rl_agent
    AI_TYPE = "Deep Q-Learning (Reinforcement Learning)"
    RL_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    RL_AVAILABLE = False
    try:
        from ai_player import GoAI
        AI_TYPE = "Advanced (Neural Network + MCTS)"
    except (ImportError, OSError, Exception):
        from simple_ai import GoAI
        AI_TYPE = "Simplified (Heuristic-based)"

# Page configuration
st.set_page_config(
    page_title="Alpha Go Game",
    page_icon="⚫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: gradient 3s ease-in-out infinite;
    }
    
    @keyframes gradient {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(180deg); }
    }
    
    .game-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .score-display {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        margin: 0.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    .score-display:hover {
        transform: translateY(-5px);
    }
    
    .black-score {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        color: white;
    }
    
    .white-score {
        background: linear-gradient(135deg, #ecf0f1, #bdc3c7);
        color: #2c3e50;
    }
    
    .thinking-panel {
        background: linear-gradient(135deg, #FF9A8B, #FF6A88);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .matrix-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .hint-card {
        background: linear-gradient(135deg, #A8EDEA, #A8E6CF);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #2c3e50;
        border-left: 4px solid #16a085;
    }
    
    .rule-card {
        background: linear-gradient(135deg, #FFD93D, #FF9A8B);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #2c3e50;
        border-left: 4px solid #f39c12;
    }
    
    .move-history {
        max-height: 400px;
        overflow-y: auto;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .ai-thinking {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

class GoGameUI:
    """
    Streamlit UI for the Go Game
    """
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'game' not in st.session_state:
            st.session_state.game = None
        if 'ai_player' not in st.session_state:
            st.session_state.ai_player = None
        if 'player_name' not in st.session_state:
            st.session_state.player_name = ""
        if 'player_color' not in st.session_state:
            st.session_state.player_color = 1  # 1 for black, 2 for white
        if 'board_size' not in st.session_state:
            st.session_state.board_size = 19
        if 'difficulty' not in st.session_state:
            st.session_state.difficulty = "medium"
        if 'game_started' not in st.session_state:
            st.session_state.game_started = False
        if 'ai_thinking' not in st.session_state:
            st.session_state.ai_thinking = False
        if 'game_over' not in st.session_state:
            st.session_state.game_over = False
        if 'winner' not in st.session_state:
            st.session_state.winner = None
        if 'thinking_data' not in st.session_state:
            st.session_state.thinking_data = None
        if 'show_hints' not in st.session_state:
            st.session_state.show_hints = True
        if 'show_matrix' not in st.session_state:
            st.session_state.show_matrix = True
        if 'training_mode' not in st.session_state:
            st.session_state.training_mode = False
        if 'ai_type' not in st.session_state:
            st.session_state.ai_type = "rl" if RL_AVAILABLE else "simple"
    
    def render_main_page(self):
        """Render the main game page"""
        st.markdown('<h1 class="main-header">🔴⚫ Alpha Go Game ⚪🔴</h1>', unsafe_allow_html=True)
        
        if not st.session_state.game_started:
            self.render_setup_page()
        else:
            self.render_game_page()
    
    def render_setup_page(self):
        """Render the game setup page"""
        st.markdown("## 🎮 Game Setup")
        
        # Modern setup interface
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### 👤 Player Configuration")
            
            # Player name input
            player_name = st.text_input(
                "Enter your name:",
                value=st.session_state.player_name,
                placeholder="Your name here...",
                help="This will be displayed during the game"
            )
            st.session_state.player_name = player_name
            
            # Color selection
            color_options = {"⚫ Black (goes first)": 1, "⚪ White (goes second)": 2}
            selected_color = st.selectbox(
                "Choose your color:",
                options=list(color_options.keys()),
                index=0 if st.session_state.player_color == 1 else 1
            )
            st.session_state.player_color = color_options[selected_color]
            
            # Display color preview
            if st.session_state.player_color == 1:
                st.success("**You:** ⚫ Black stones | **AI:** ⚪ White stones")
            else:
                st.info("**You:** ⚪ White stones | **AI:** ⚫ Black stones")
        
        with col2:
            st.markdown("### ⚙️ Game Configuration")
            
            # Board size selection
            board_size_options = [9, 13, 19]
            board_size = st.selectbox(
                "Choose board size:",
                options=board_size_options,
                index=board_size_options.index(st.session_state.board_size),
                format_func=lambda x: f"{x}×{x} ({'Beginner' if x==9 else 'Intermediate' if x==13 else 'Professional'})",
                help="9×9: Quick games (15-30 min)\n13×13: Medium games (30-60 min)\n19×19: Full games (60+ min)"
            )
            st.session_state.board_size = board_size
            
            # AI Type selection (if RL is available)
            if RL_AVAILABLE:
                ai_type_options = {
                    "🧠 Deep Q-Learning": "rl",
                    "🎯 Simple Heuristic": "simple"
                }
                selected_ai_type = st.selectbox(
                    "AI Type:",
                    options=list(ai_type_options.keys()),
                    index=0 if st.session_state.ai_type == "rl" else 1,
                    help="Deep Q-Learning: Uses reinforcement learning\nSimple: Uses heuristic rules"
                )
                st.session_state.ai_type = ai_type_options[selected_ai_type]
            
            # Difficulty selection
            difficulty_options = {
                "🟢 Easy": "easy",
                "🟡 Medium": "medium", 
                "🟠 Hard": "hard",
                "🔴 Expert": "expert"
            }
            
            selected_difficulty = st.selectbox(
                "Choose AI difficulty:",
                options=list(difficulty_options.keys()),
                index=list(difficulty_options.values()).index(st.session_state.difficulty)
            )
            st.session_state.difficulty = difficulty_options[selected_difficulty]
        
        with col3:
            st.markdown("### 🎯 Game Features")
            
            # Feature toggles
            st.session_state.show_hints = st.checkbox(
                "🧩 Show Hints & Tips",
                value=st.session_state.show_hints,
                help="Display helpful hints and Go rules during gameplay"
            )
            
            st.session_state.show_matrix = st.checkbox(
                "📊 Show AI Thinking Matrix",
                value=st.session_state.show_matrix,
                help="Visualize AI's decision-making process"
            )
            
            if RL_AVAILABLE:
                st.session_state.training_mode = st.checkbox(
                    "🏋️ Training Mode",
                    value=st.session_state.training_mode,
                    help="AI learns from this game (may be slower)"
                )
            
            # AI Type description
            if st.session_state.ai_type == "rl":
                st.markdown(f"""
                <div class="hint-card">
                <strong>🧠 Deep Q-Learning AI</strong><br>
                Uses reinforcement learning with neural networks to make decisions.
                Shows Q-value matrix for move analysis.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="hint-card">
                <strong>🎯 Heuristic AI</strong><br>
                Uses traditional game analysis with strategic rules.
                Fast and reliable gameplay.
                </div>
                """, unsafe_allow_html=True)
        
        # Game rules and information
        with st.expander("📖 Go Rules & Strategy Guide", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 🎯 Basic Rules:
                - **Objective**: Control more territory than your opponent
                - **Turns**: Players alternate placing stones on intersections
                - **Capture**: Stones with no liberties (empty adjacent points) are captured
                - **Ko Rule**: Cannot immediately recapture in the same position
                - **End**: Game ends when both players pass consecutively
                
                ### 📊 Scoring:
                - **Territory**: Empty areas surrounded by your stones
                - **Stones**: Stones remaining on the board  
                - **Captures**: Opponent stones you've captured
                """)
            
            with col2:
                st.markdown("""
                ### 💡 Strategy Tips:
                - **Control corners** early - easier to secure territory
                - **Connect your stones** to form strong groups
                - **Attack weak groups** with few liberties
                - **Don't play inside opponent's territory** unless capturing
                - **Balance** between attack and defense
                
                ### 🎮 Controls:
                - **Click** on intersections to place stones
                - **Pass** button to skip your turn
                - **Resign** to concede the game
                """)
        
        # Start game button
        st.markdown("---")
        start_col1, start_col2, start_col3 = st.columns([1, 2, 1])
        
        with start_col2:
            if st.button("🚀 Start Game", type="primary", use_container_width=True):
                if not player_name.strip():
                    st.error("Please enter your name to start the game!")
                else:
                    self.start_new_game()
                    st.rerun()
    
    def start_new_game(self):
        """Start a new game with current settings"""
        st.session_state.game = GoGame(st.session_state.board_size)
        
        # Initialize AI based on type
        if RL_AVAILABLE and st.session_state.ai_type == "rl":
            ai_player = DeepQLearningAgent(
                board_size=st.session_state.board_size,
                epsilon=0.05 if not st.session_state.training_mode else 0.1
            )
            # Try to load existing model
            model_path = f"dql_agent_{st.session_state.board_size}x{st.session_state.board_size}.json"
            if not ai_player.load_model(model_path):
                st.warning("No trained model found. Using freshly initialized AI.")
            st.session_state.ai_player = ai_player
        else:
            st.session_state.ai_player = GoAI(st.session_state.board_size, st.session_state.difficulty)
        
        st.session_state.game_started = True
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.ai_thinking = False
        st.session_state.thinking_data = None
        
        # If player chose white, AI (black) goes first
        if st.session_state.player_color == 2:
            self.make_ai_move()
    
    def render_game_page(self):
        """Render the main game interface"""
        # Modern game header
        header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
        
        with header_col1:
            if st.button("🔄 New Game", type="secondary"):
                st.session_state.game_started = False
                st.rerun()
        
        with header_col2:
            current_player = "You" if st.session_state.game.current_player == st.session_state.player_color else "AI"
            player_symbol = "⚫" if st.session_state.game.current_player == 1 else "⚪"
            
            if st.session_state.ai_thinking:
                st.markdown(f"""
                <div class="ai-thinking">
                    <h3 style="text-align: center;">🤖 AI is thinking... {player_symbol}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"### Current Turn: {current_player} {player_symbol}")
        
        with header_col3:
            st.markdown(f"**Player:** {st.session_state.player_name}")
            st.markdown(f"**AI Type:** {AI_TYPE}")
            st.markdown(f"**Board:** {st.session_state.board_size}×{st.session_state.board_size}")
        
        # Main game layout
        if st.session_state.game_over:
            self.render_game_over()
        else:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🎮 Game Board", "🧠 AI Analysis", "📚 Learning Center"])
            
            with tab1:
                game_col, info_col = st.columns([3, 1])
                
                with game_col:
                    self.render_game_board()
                    self.render_game_controls()
                
                with info_col:
                    self.render_game_info()
                    self.render_move_history()
            
            with tab2:
                self.render_ai_analysis()
            
            with tab3:
                self.render_learning_center()
    
    def render_game_board(self):
        """Render the Go board using Plotly"""
        board = st.session_state.game.get_board_state()
        size = st.session_state.board_size
        
        # Create the board visualization
        fig = go.Figure()
        
        # Draw grid lines
        for i in range(size):
            # Horizontal lines
            fig.add_trace(go.Scatter(
                x=[0, size-1], y=[i, i],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Vertical lines
            fig.add_trace(go.Scatter(
                x=[i, i], y=[0, size-1],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add star points for larger boards
        if size == 19:
            star_points = [(3,3), (3,9), (3,15), (9,3), (9,9), (9,15), (15,3), (15,9), (15,15)]
            for x, y in star_points:
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(color='black', size=8),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add stones
        black_stones_x, black_stones_y = [], []
        white_stones_x, white_stones_y = [], []
        
        for row in range(size):
            for col in range(size):
                if board[row, col] == 1:  # Black stone
                    black_stones_x.append(col)
                    black_stones_y.append(size - 1 - row)  # Flip y-axis
                elif board[row, col] == 2:  # White stone
                    white_stones_x.append(col)
                    white_stones_y.append(size - 1 - row)  # Flip y-axis
        
        # Add black stones
        if black_stones_x:
            fig.add_trace(go.Scatter(
                x=black_stones_x, y=black_stones_y,
                mode='markers',
                marker=dict(color='black', size=20, line=dict(color='gray', width=1)),
                name='Black Stones',
                hovertemplate='Black Stone<br>Position: %{x}, %{y}<extra></extra>'
            ))
        
        # Add white stones
        if white_stones_x:
            fig.add_trace(go.Scatter(
                x=white_stones_x, y=white_stones_y,
                mode='markers',
                marker=dict(color='white', size=20, line=dict(color='black', width=2)),
                name='White Stones',
                hovertemplate='White Stone<br>Position: %{x}, %{y}<extra></extra>'
            ))
        
        # Configure layout
        fig.update_layout(
            title=f"Go Board ({size}×{size})",
            xaxis=dict(
                range=[-0.5, size-0.5],
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                dtick=1,
                fixedrange=True
            ),
            yaxis=dict(
                range=[-0.5, size-0.5],
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                dtick=1,
                fixedrange=True,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='#DEB887',
            paper_bgcolor='white',
            width=600,
            height=600,
            showlegend=True
        )
        
        # Handle board clicks
        clicked_point = st.plotly_chart(fig, use_container_width=True, key="go_board")
        
        # Process clicks if it's player's turn
        if (not st.session_state.ai_thinking and 
            not st.session_state.game_over and 
            st.session_state.game.current_player == st.session_state.player_color):
            
            # Note: Streamlit plotly doesn't support click events directly
            # We'll use a grid of buttons as an alternative
            self.render_move_buttons()
    
    def render_move_buttons(self):
        """Render clickable buttons for making moves"""
        st.markdown("**Click on the board position to make your move:**")
        
        board = st.session_state.game.get_board_state()
        size = st.session_state.board_size
        
        # Create a grid of buttons for moves
        for row in range(size):
            cols = st.columns(size)
            for col in range(size):
                with cols[col]:
                    # Create button label based on current state
                    if board[row, col] == 0:  # Empty
                        label = "+"
                        disabled = False
                    elif board[row, col] == 1:  # Black
                        label = "⚫"
                        disabled = True
                    else:  # White
                        label = "⚪"
                        disabled = True
                    
                    # Button for making move
                    if st.button(
                        label, 
                        key=f"move_{row}_{col}",
                        disabled=disabled or st.session_state.ai_thinking,
                        help=f"Position ({row}, {col})"
                    ):
                        self.make_player_move(row, col)
    
    def render_game_controls(self):
        """Render game control buttons"""
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(
                "⏭️ Pass", 
                disabled=st.session_state.ai_thinking or st.session_state.game_over,
                help="Skip your turn"
            ):
                self.pass_turn()
        
        with col2:
            if st.button(
                "🏳️ Resign", 
                disabled=st.session_state.ai_thinking or st.session_state.game_over,
                help="Concede the game"
            ):
                self.resign_game()
        
        with col3:
            if st.button(
                "📊 Score", 
                help="Calculate current score"
            ):
                self.show_current_score()
        
        # AI thinking indicator
        if st.session_state.ai_thinking:
            st.info("🤖 AI is thinking...")
            time.sleep(0.1)  # Small delay for visual feedback
            self.make_ai_move()
            st.rerun()
    
    def render_game_info(self):
        """Render game information panel"""
        st.markdown("### 📊 Game Information")
        
        # Current scores
        black_score, white_score = st.session_state.game.calculate_score()
        
        st.markdown(
            f'<div class="score-display black-score">⚫ Black: {black_score}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="score-display white-score">⚪ White: {white_score}</div>',
            unsafe_allow_html=True
        )
        
        # Captured stones
        captured = st.session_state.game.captured_stones
        st.markdown(f"**Captured Stones:**")
        st.markdown(f"⚫ by Black: {captured[1]}")
        st.markdown(f"⚪ by White: {captured[2]}")
        
        # Game statistics
        st.markdown(f"**Total Moves:** {len(st.session_state.game.game_history)}")
        
        # Valid moves count
        valid_moves = len(st.session_state.game.get_valid_moves())
        st.markdown(f"**Valid Moves:** {valid_moves}")
    
    def render_move_history(self):
        """Render move history"""
        st.markdown("### 📜 Move History")
        
        history = st.session_state.game.game_history
        if not history:
            st.markdown("*No moves yet*")
            return
        
        # Display last 10 moves
        recent_moves = history[-10:]
        
        move_text = []
        for i, (row, col, player) in enumerate(recent_moves):
            move_num = len(history) - len(recent_moves) + i + 1
            player_symbol = "⚫" if player == 1 else "⚪"
            
            if row == -1 and col == -1:
                move_description = "Pass"
            else:
                move_description = f"({row}, {col})"
            
            move_text.append(f"{move_num}. {player_symbol} {move_description}")
        
        st.markdown(
            f'<div class="move-history">{"<br>".join(move_text)}</div>',
            unsafe_allow_html=True
        )
        
    def render_ai_analysis(self):
        """Render AI thinking analysis and matrix visualization"""
        st.markdown("## 🧠 AI Decision Analysis")
        
        if not st.session_state.show_matrix:
            st.info("Enable 'Show AI Thinking Matrix' in game setup to see AI analysis.")
            return
        
        if st.session_state.thinking_data is None:
            st.info("AI thinking data will appear here after AI makes a move.")
            return
        
        thinking_data = st.session_state.thinking_data
        
        # AI Decision Summary
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="thinking-panel">
                <h4>🎯 AI Decision Process</h4>
            </div>
            """, unsafe_allow_html=True)
            
            decision_type = thinking_data.get('decision_type', 'unknown')
            chosen_move = thinking_data.get('chosen_move', (-1, -1))
            chosen_q_value = thinking_data.get('chosen_q_value', 0)
            
            if chosen_move == (-1, -1):
                st.write("**Decision:** Pass turn")
            else:
                st.write(f"**Chosen Move:** ({chosen_move[0]}, {chosen_move[1]})")
            
            st.write(f"**Decision Type:** {decision_type.title()}")
            st.write(f"**Q-Value:** {chosen_q_value:.4f}")
            
            # Strategy explanation
            strategy = thinking_data.get('strategy_explanation', 'No explanation available')
            st.markdown(f"""
            <div class="hint-card">
                <strong>🧩 Strategy Explanation:</strong><br>
                {strategy}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Top moves analysis
            top_moves = thinking_data.get('top_moves', [])
            if top_moves:
                st.markdown("""
                <div class="thinking-panel">
                    <h4>🏆 Top Move Candidates</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for i, move_data in enumerate(top_moves[:3]):
                    pos = move_data['position']
                    q_val = move_data['q_value']
                    rank_emoji = ["🥇", "🥈", "🥉"][i]
                    
                    st.write(f"{rank_emoji} Position ({pos[0]}, {pos[1]}): Q={q_val:.4f}")
        
        # Q-Value Matrix Visualization
        if 'q_matrix' in thinking_data:
            st.markdown("### 📊 Q-Value Matrix Heatmap")
            
            q_matrix = thinking_data['q_matrix']
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=q_matrix,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Q-Value"),
                hovertemplate='Position: (%{x}, %{y})<br>Q-Value: %{z:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="AI's Q-Value Assessment for Each Board Position",
                xaxis_title="Column",
                yaxis_title="Row",
                width=600,
                height=600
            )
            
            # Overlay valid moves
            valid_moves = thinking_data.get('valid_moves', [])
            for row, col in valid_moves:
                fig.add_scatter(
                    x=[col], y=[row],
                    mode='markers',
                    marker=dict(
                        symbol='circle-open',
                        size=15,
                        color='white',
                        line=dict(width=3, color='black')
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Matrix interpretation
            st.markdown("""
            <div class="hint-card">
                <strong>🔍 How to Read the Matrix:</strong><br>
                • <strong>Red areas:</strong> Low Q-values - AI thinks these moves are poor<br>
                • <strong>Yellow areas:</strong> Medium Q-values - Neutral moves<br>
                • <strong>Blue areas:</strong> High Q-values - AI thinks these moves are excellent<br>
                • <strong>White circles:</strong> Valid moves the AI can make<br>
                • <strong>Gray areas:</strong> Invalid positions (occupied or illegal)
            </div>
            """, unsafe_allow_html=True)
    
    def render_learning_center(self):
        """Render learning center with hints and rules"""
        st.markdown("## 📚 Learning Center")
        
        if not st.session_state.show_hints:
            st.info("Enable 'Show Hints & Tips' in game setup to see learning content.")
            return
        
        # Dynamic hints based on game state
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 💡 Current Game Hints")
            
            # Analyze current position and give hints
            game = st.session_state.game
            if game:
                hints = self._generate_dynamic_hints(game)
                
                for i, hint in enumerate(hints):
                    st.markdown(f"""
                    <div class="hint-card">
                        <strong>Hint {i+1}:</strong> {hint}
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📖 Go Rules Reference")
            
            rules = [
                "**Liberties:** Empty points adjacent to a stone or group",
                "**Atari:** When a group has only one liberty remaining",
                "**Capture:** Remove stones with zero liberties from the board",
                "**Ko Rule:** Prevents infinite capture-recapture loops",
                "**Territory:** Empty areas completely surrounded by one color",
                "**Seki:** Mutual life where neither player can capture",
                "**Dead Stones:** Stones that cannot avoid capture",
                "**Eyes:** Empty points surrounded by one color that cannot be filled"
            ]
            
            for rule in rules:
                st.markdown(f"""
                <div class="rule-card">
                    {rule}
                </div>
                """, unsafe_allow_html=True)
        
        # Strategy tips based on game phase
        st.markdown("### 🎯 Strategic Concepts")
        
        game_phase = self._determine_game_phase()
        phase_strategies = {
            "opening": [
                "Focus on corner play - corners are easiest to secure",
                "Play on the third and fourth lines for territory",
                "Don't play too close to opponent stones early",
                "Develop multiple areas rather than focusing on one"
            ],
            "middle": [
                "Look for weak groups to attack",
                "Connect your stones and separate opponent's",
                "Fight for the center when groups are stable",
                "Consider the direction of play for each move"
            ],
            "endgame": [
                "Count territory to determine if you're ahead",
                "Play the largest endgame moves first",
                "Be careful not to play inside your own territory",
                "Look for tesuji (tactical plays) to gain points"
            ]
        }
        
        strategies = phase_strategies.get(game_phase, ["Keep learning and practicing!"])
        
        st.markdown(f"**Current Game Phase: {game_phase.title()}**")
        for strategy in strategies:
            st.markdown(f"• {strategy}")
        
        # Training exercises
        if RL_AVAILABLE and st.session_state.ai_type == "rl":
            st.markdown("### 🏋️ AI Training Information")
            
            if hasattr(st.session_state.ai_player, 'training_stats'):
                stats = st.session_state.ai_player.training_stats
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>{stats.get('games_played', 0)}</h4>
                        <p>Games Played</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    wins = stats.get('wins', 0)
                    games = max(stats.get('games_played', 1), 1)
                    win_rate = (wins / games) * 100
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>{win_rate:.1f}%</h4>
                        <p>Win Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    epsilon = getattr(st.session_state.ai_player, 'epsilon', 0)
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>{epsilon:.3f}</h4>
                        <p>Exploration Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    total_reward = stats.get('total_reward', 0)
                    avg_reward = total_reward / max(games, 1)
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>{avg_reward:.2f}</h4>
                        <p>Avg Reward</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    def _generate_dynamic_hints(self, game):
        """Generate hints based on current game position"""
        hints = []
        board = game.get_board_state()
        size = game.board_size
        current_player = game.current_player
        
        # Check for stones in atari
        groups_in_atari = []
        for row in range(size):
            for col in range(size):
                if board[row, col] != 0:
                    group = game._get_group(board, row, col)
                    liberties = game._count_liberties(board, group)
                    if liberties == 1 and (row, col, board[row, col]) not in [(g[0], g[1], g[2]) for g in groups_in_atari]:
                        groups_in_atari.append((row, col, board[row, col]))
        
        if groups_in_atari:
            for row, col, color in groups_in_atari:
                if color == current_player:
                    hints.append(f"⚠️ Your group at ({row}, {col}) is in atari! Defend it or it will be captured.")
                else:
                    hints.append(f"🎯 Enemy group at ({row}, {col}) is in atari! You can capture it.")
        
        # Check for good opening moves
        if len(game.game_history) < 10:
            corners = [(3, 3), (3, size-4), (size-4, 3), (size-4, size-4)]
            empty_corners = [pos for pos in corners if board[pos[0], pos[1]] == 0]
            if empty_corners:
                hints.append(f"🏠 Consider playing in the corners: {empty_corners[0]} for solid territory.")
        
        # Check for connection opportunities
        friendly_stones = []
        for row in range(size):
            for col in range(size):
                if board[row, col] == current_player:
                    friendly_stones.append((row, col))
        
        if len(friendly_stones) >= 2:
            hints.append("🔗 Look for ways to connect your stones to form stronger groups.")
        
        # Territory advice
        if len(game.game_history) > size:
            hints.append("🏛️ Start thinking about territory - surround empty areas to claim them.")
        
        # Default hints if no specific advice
        if not hints:
            hints.extend([
                "👀 Look for opponent weaknesses before making your move.",
                "⚖️ Balance between attack and defense in your strategy.",
                "🎯 Every move should have a clear purpose - attack, defend, or build territory."
            ])
        
        return hints[:3]  # Return max 3 hints
    
    def _determine_game_phase(self):
        """Determine current phase of the game"""
        if not st.session_state.game:
            return "opening"
        
        moves_played = len(st.session_state.game.game_history)
        board_size = st.session_state.game.board_size
        
        if moves_played < board_size:
            return "opening"
        elif moves_played < board_size * 2:
            return "middle"
        else:
            return "endgame"
    
    def make_player_move(self, row: int, col: int):
        """Handle player move"""
        if st.session_state.game.make_move(row, col):
            st.success(f"Move played at ({row}, {col})")
            
            # Check if game is over
            if st.session_state.game.is_game_over():
                self.end_game()
            else:
                # Trigger AI move
                st.session_state.ai_thinking = True
            
            st.rerun()
        else:
            st.error("Invalid move! Please try a different position.")
    
    def make_ai_move(self):
        """Handle AI move with thinking process"""
        try:
            # Check if AI should pass
            if hasattr(st.session_state.ai_player, 'should_pass') and st.session_state.ai_player.should_pass(st.session_state.game):
                st.session_state.game.pass_turn()
                st.info("AI passes")
                st.session_state.thinking_data = {
                    'chosen_move': (-1, -1),
                    'decision_type': 'strategic_pass',
                    'strategy_explanation': 'AI decided to pass - no beneficial moves available'
                }
            else:
                # Get AI move with thinking process if available
                if hasattr(st.session_state.ai_player, 'get_move_with_thinking'):
                    ai_move, thinking_data = st.session_state.ai_player.get_move_with_thinking(st.session_state.game)
                    st.session_state.thinking_data = thinking_data
                else:
                    ai_move = st.session_state.ai_player.get_move(st.session_state.game)
                    st.session_state.thinking_data = {
                        'chosen_move': ai_move,
                        'decision_type': 'heuristic',
                        'strategy_explanation': 'Move selected using heuristic analysis'
                    }
                
                if ai_move == (-1, -1):
                    st.session_state.game.pass_turn()
                    st.info("AI passes")
                else:
                    row, col = ai_move
                    if st.session_state.game.make_move(row, col):
                        st.success(f"AI played at ({row}, {col})")
                        
                        # Training mode: remember the experience
                        if (RL_AVAILABLE and st.session_state.ai_type == "rl" and 
                            st.session_state.training_mode and 
                            hasattr(st.session_state.ai_player, 'remember')):
                            
                            # This would need proper state tracking for training
                            pass
                    else:
                        # Fallback to pass if AI move is invalid
                        st.session_state.game.pass_turn()
                        st.warning("AI move was invalid, AI passes")
                        st.session_state.thinking_data = {
                            'chosen_move': (-1, -1),
                            'decision_type': 'forced_pass',
                            'strategy_explanation': 'Invalid move detected, forced to pass'
                        }
        
        except Exception as e:
            st.error(f"AI error: {e}")
            st.session_state.game.pass_turn()
            st.session_state.thinking_data = {
                'chosen_move': (-1, -1),
                'decision_type': 'error_pass',
                'strategy_explanation': f'Error occurred: {e}'
            }
        
        st.session_state.ai_thinking = False
        
        # Check if game is over
        if st.session_state.game.is_game_over():
            self.end_game()
    
    def pass_turn(self):
        """Handle pass turn"""
        st.session_state.game.pass_turn()
        st.info("You passed your turn")
        
        if st.session_state.game.is_game_over():
            self.end_game()
        else:
            st.session_state.ai_thinking = True
        
        st.rerun()
    
    def resign_game(self):
        """Handle game resignation"""
        st.session_state.game_over = True
        opponent_color = 3 - st.session_state.player_color
        st.session_state.winner = "AI" if opponent_color != st.session_state.player_color else "You"
        st.warning("You resigned the game!")
        st.rerun()
    
    def show_current_score(self):
        """Show current score calculation"""
        black_score, white_score = st.session_state.game.calculate_score()
        
        st.info(f"""
        **Current Score:**
        
        ⚫ Black: {black_score} points
        ⚪ White: {white_score} points
        
        **Lead:** {abs(black_score - white_score)} points for {'Black' if black_score > white_score else 'White'}
        """)
    
    def end_game(self):
        """Handle game end"""
        st.session_state.game_over = True
        black_score, white_score = st.session_state.game.calculate_score()
        
        if black_score > white_score:
            winner_color = 1
            winner_name = st.session_state.player_name if st.session_state.player_color == 1 else "AI"
        elif white_score > black_score:
            winner_color = 2
            winner_name = st.session_state.player_name if st.session_state.player_color == 2 else "AI"
        else:
            winner_color = 0
            winner_name = "Draw"
        
        st.session_state.winner = winner_name
    
    def render_game_over(self):
        """Render game over screen"""
        black_score, white_score = st.session_state.game.calculate_score()
        
        st.markdown("## 🎉 Game Over!")
        
        # Winner announcement
        if st.session_state.winner == "Draw":
            st.balloons()
            st.success("It's a draw! Great game!")
        elif st.session_state.winner == st.session_state.player_name:
            st.balloons()
            st.success(f"🎊 Congratulations {st.session_state.player_name}! You won!")
        else:
            st.info("AI wins! Better luck next time!")
        
        # Final scores
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="score-display black-score">⚫ Black: {black_score}</div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="score-display white-score">⚪ White: {white_score}</div>',
                unsafe_allow_html=True
            )
        
        # Game statistics
        st.markdown("### 📈 Game Statistics")
        total_moves = len(st.session_state.game.game_history)
        captured_by_black = st.session_state.game.captured_stones[1]
        captured_by_white = st.session_state.game.captured_stones[2]
        
        st.markdown(f"""
        - **Total Moves:** {total_moves}
        - **Stones Captured by Black:** {captured_by_black}
        - **Stones Captured by White:** {captured_by_white}
        - **Final Score Difference:** {abs(black_score - white_score)} points
        """)
        
        # Play again button
        if st.button("🔄 Play Again", type="primary", use_container_width=True):
            st.session_state.game_started = False
            st.rerun()

def main():
    """Main function to run the Go game"""
    game_ui = GoGameUI()
    game_ui.render_main_page()

if __name__ == "__main__":
    main()