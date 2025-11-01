import streamlit as st
import time
import threading
from rl_agent import train_rl_agent, DeepQLearningAgent
import json
import os

def render_training_interface():
    """Render RL training interface"""
    st.title("🧠 Reinforcement Learning Training Center")
    
    st.markdown("""
    Train your own Deep Q-Learning agent to play Go! The AI will learn through 
    self-play and improve its strategy over time.
    """)
    
    # Training configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ⚙️ Training Configuration")
        
        board_size = st.selectbox(
            "Board Size:",
            options=[9, 13, 19],
            index=0,
            help="Smaller boards train faster"
        )
        
        episodes = st.number_input(
            "Training Episodes:",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="More episodes = better AI, but takes longer"
        )
        
        learning_rate = st.number_input(
            "Learning Rate:",
            min_value=0.001,
            max_value=0.1,
            value=0.001,
            step=0.001,
            format="%.4f",
            help="How fast the AI learns (0.001 is usually good)"
        )
        
        epsilon = st.number_input(
            "Exploration Rate:",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="How often AI tries random moves (exploration vs exploitation)"
        )
    
    with col2:
        st.markdown("### 📊 Current Model Status")
        
        model_path = f"dql_agent_{board_size}x{board_size}.json"
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                
                stats = model_data.get('training_stats', {})
                
                st.success("✅ Trained model found!")
                st.write(f"**Games Played:** {stats.get('games_played', 0)}")
                st.write(f"**Wins:** {stats.get('wins', 0)}")
                st.write(f"**Win Rate:** {stats.get('wins', 0) / max(stats.get('games_played', 1), 1) * 100:.1f}%")
                st.write(f"**Current Epsilon:** {model_data.get('epsilon', 0):.3f}")
                
            except Exception as e:
                st.warning(f"Model file exists but couldn't read stats: {e}")
        else:
            st.info("No trained model found for this board size.")
            st.write("Training will create a new model from scratch.")
    
    # Training controls
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.session_state.training_active = True
            st.session_state.training_progress = 0
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Training", use_container_width=True):
            st.session_state.training_active = False
            st.rerun()
    
    with col3:
        if st.button("🗑️ Reset Model", use_container_width=True):
            if os.path.exists(model_path):
                os.remove(model_path)
                st.success("Model reset!")
                st.rerun()
            else:
                st.info("No model to reset.")
    
    # Training progress
    if getattr(st.session_state, 'training_active', False):
        st.markdown("### 🏋️ Training in Progress...")
        
        # Progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Run training
        try:
            agent = train_rl_agent_with_progress(
                board_size=board_size,
                episodes=episodes,
                learning_rate=learning_rate,
                epsilon=epsilon,
                progress_placeholder=progress_placeholder,
                status_placeholder=status_placeholder
            )
            
            st.success("🎉 Training completed successfully!")
            st.session_state.training_active = False
            
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.session_state.training_active = False
    
    # Quick test section
    st.markdown("---")
    st.markdown("### 🎮 Quick Test")
    
    if st.button("Test Current Model", use_container_width=True):
        if os.path.exists(model_path):
            with st.spinner("Testing model..."):
                test_results = quick_test_model(board_size)
                
                st.markdown("**Test Results:**")
                for result in test_results:
                    st.write(f"• {result}")
        else:
            st.warning("No model found to test. Train a model first.")

def train_rl_agent_with_progress(board_size, episodes, learning_rate, epsilon, 
                                progress_placeholder, status_placeholder):
    """Train RL agent with real-time progress updates"""
    
    agent = DeepQLearningAgent(
        board_size=board_size,
        learning_rate=learning_rate,
        epsilon=epsilon
    )
    
    # Load existing model if available
    model_path = f"dql_agent_{board_size}x{board_size}.json"
    if agent.load_model(model_path):
        status_placeholder.info("Continuing training from existing model...")
    else:
        status_placeholder.info("Starting training from scratch...")
    
    for episode in range(episodes):
        # Check if training should stop
        if not getattr(st.session_state, 'training_active', False):
            break
        
        # Train one episode
        loss, reward, moves = agent.train_episode()
        
        # Update progress
        progress = (episode + 1) / episodes
        progress_placeholder.progress(progress)
        
        # Update status every 10 episodes
        if (episode + 1) % 10 == 0:
            games = agent.training_stats['games_played']
            wins = agent.training_stats['wins']
            win_rate = wins / max(games, 1) * 100
            
            status_placeholder.write(f"""
            **Episode {episode + 1}/{episodes}**
            - Win Rate: {win_rate:.1f}% ({wins}/{games})
            - Epsilon: {agent.epsilon:.3f}
            - Last Reward: {reward:.3f}
            - Last Loss: {loss:.6f if loss > 0 else 'N/A'}
            """)
        
        # Save model every 100 episodes
        if (episode + 1) % 100 == 0:
            agent.save_model(model_path)
    
    # Final save
    agent.save_model(model_path)
    return agent

def quick_test_model(board_size):
    """Quick test of the trained model"""
    from go_game import GoGame
    
    results = []
    
    try:
        # Load the model
        agent = DeepQLearningAgent(board_size=board_size, epsilon=0)  # No exploration for testing
        model_path = f"dql_agent_{board_size}x{board_size}.json"
        
        if not agent.load_model(model_path):
            return ["No model found to test"]
        
        # Test game
        game = GoGame(board_size)
        moves_made = 0
        max_moves = board_size * board_size
        
        while not game.is_game_over() and moves_made < max_moves:
            move = agent.get_move(game)
            
            if move == (-1, -1):
                game.pass_turn()
                results.append(f"Move {moves_made + 1}: AI passed")
            else:
                if game.make_move(move[0], move[1]):
                    results.append(f"Move {moves_made + 1}: AI played at ({move[0]}, {move[1]})")
                else:
                    results.append(f"Move {moves_made + 1}: AI attempted invalid move, passed instead")
                    game.pass_turn()
            
            moves_made += 1
            
            if moves_made >= 10:  # Limit output for display
                results.append(f"... (continued for {moves_made} total moves)")
                break
        
        # Final game state
        black_score, white_score = game.calculate_score()
        results.append(f"Final Score - Black: {black_score}, White: {white_score}")
        
        if black_score > white_score:
            results.append("Result: Black (AI) wins!")
        elif white_score > black_score:
            results.append("Result: White wins!")
        else:
            results.append("Result: Draw!")
        
    except Exception as e:
        results.append(f"Test failed: {e}")
    
    return results

def main():
    """Main training app"""
    st.set_page_config(
        page_title="Go AI Training",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize session state
    if 'training_active' not in st.session_state:
        st.session_state.training_active = False
    
    render_training_interface()

if __name__ == "__main__":
    main()