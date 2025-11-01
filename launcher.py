#!/usr/bin/env python3
"""
Alpha Go Game Launcher
Simple script to start the Streamlit Go game application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'numpy', 
        'torch',
        'matplotlib',
        'plotly',
        'pandas',
        'scikit-learn',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def launch_game():
    """Launch the Streamlit Go game"""
    print("🚀 Launching Alpha Go Game...")
    print("🌐 Opening game in your default web browser...")
    print("🎮 Enjoy playing Go against the AI!")
    print("-" * 50)
    
    try:
        # Run streamlit with the main app file
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.headless', 'false',
            '--server.runOnSave', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Game closed. Thanks for playing!")
    except FileNotFoundError:
        print("❌ Error: streamlit_app.py not found!")
        print("Make sure you're running this script from the correct directory.")
    except Exception as e:
        print(f"❌ Error launching game: {e}")

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🔴⚫ ALPHA GO GAME LAUNCHER ⚪🔴")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path('streamlit_app.py').exists():
        print("❌ Error: streamlit_app.py not found!")
        print("Please run this script from the game directory.")
        return
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"📦 Missing packages: {', '.join(missing)}")
        install_choice = input("Install missing dependencies? (y/n): ").lower().strip()
        
        if install_choice in ['y', 'yes']:
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install manually:")
                print("pip install -r requirements.txt")
                return
        else:
            print("❌ Cannot run game without required dependencies.")
            print("Please install manually: pip install -r requirements.txt")
            return
    
    # Launch the game
    launch_game()

if __name__ == "__main__":
    main()