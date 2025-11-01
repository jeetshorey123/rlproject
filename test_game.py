#!/usr/bin/env python3
"""
Test script for Alpha Go Game
Verifies that all components are working correctly
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import torch
        import torch.nn as nn
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    return True

def test_game_logic():
    """Test core game logic"""
    print("\nTesting game logic...")
    
    try:
        from go_game import GoGame
        
        # Test basic game creation
        game = GoGame(9)
        print("✅ Go game created successfully")
        
        # Test valid move
        if game.is_valid_move(4, 4):
            print("✅ Move validation working")
        else:
            print("❌ Move validation failed")
            return False
        
        # Test making a move
        if game.make_move(4, 4):
            print("✅ Move execution working")
        else:
            print("❌ Move execution failed")
            return False
        
        # Test board state
        board = game.get_board_state()
        if board[4, 4] == 1:
            print("✅ Board state tracking working")
        else:
            print("❌ Board state tracking failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Game logic test failed: {e}")
        traceback.print_exc()
        return False

def test_ai_player():
    """Test AI player functionality"""
    print("\nTesting AI player...")
    
    try:
        from ai_player import GoAI
        from go_game import GoGame
        
        # Create AI player
        ai = GoAI(board_size=9, difficulty="easy")
        print("✅ AI player created successfully")
        
        # Create game for AI to play
        game = GoGame(9)
        
        # Test AI move generation
        move = ai.get_move(game)
        if isinstance(move, tuple) and len(move) == 2:
            print("✅ AI move generation working")
        else:
            print("❌ AI move generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ AI player test failed: {e}")
        traceback.print_exc()
        return False

def test_neural_network():
    """Test neural network functionality"""
    print("\nTesting neural network...")
    
    try:
        from ai_player import GoNeuralNetwork
        import torch
        
        # Create neural network
        nn = GoNeuralNetwork(board_size=9)
        print("✅ Neural network created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 9, 9)
        policy, value = nn(dummy_input)
        
        if policy.shape[1] == 82 and value.shape[1] == 1:  # 9*9 + 1 for pass move
            print("✅ Neural network forward pass working")
        else:
            print("❌ Neural network output shape incorrect")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'streamlit_app.py',
        'go_game.py', 
        'ai_player.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def run_all_tests():
    """Run all test functions"""
    print("=" * 60)
    print("🔴⚫ ALPHA GO GAME TEST SUITE ⚪🔴")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Package Imports", test_imports),
        ("Game Logic", test_game_logic),
        ("Neural Network", test_neural_network),
        ("AI Player", test_ai_player),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"🎉 {test_name} PASSED")
                passed += 1
            else:
                print(f"💥 {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The game is ready to play!")
        print("Run 'streamlit run streamlit_app.py' to start the game.")
    else:
        print("💥 Some tests failed. Please check the installation.")
        print("Try running: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)