#!/usr/bin/env python
"""
Train or test the Reinforcement Learning Sudoku solver.

Usage:
    python scripts/train_rl.py                      # Train with defaults
    python scripts/train_rl.py --episodes 50000    # Train for 50k episodes
    python scripts/train_rl.py --test-only         # Test existing model
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from RL module
sys.path.insert(0, str(Path(__file__).parent.parent / "RL"))

from train_sudoku_dqn import train, test


def main():
    """Parse arguments and run RL training/testing."""
    parser = argparse.ArgumentParser(
        description='Train or test the RL Sudoku solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_rl.py
  python scripts/train_rl.py --episodes 50000
  python scripts/train_rl.py --test-only
  python scripts/train_rl.py --episodes 10000 --test-only
        """
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=20000,
        help='Number of training episodes (default: 20000)'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only test existing model without training'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='RL',
        help='Path to training data files (default: RL/)'
    )
    
    args = parser.parse_args()
    
    # Set working directory to RL folder for data access
    rl_path = Path(__file__).parent.parent / "RL"
    import os
    original_cwd = os.getcwd()
    os.chdir(str(rl_path))
    
    try:
        if args.test_only:
            print("\n🧪 Testing RL model...\n")
            test(model_path="best_model.keras")
        else:
            print(f"\n🚀 Training RL model for {args.episodes} episodes...\n")
            agent = train(
                train_path="sudoku_train_9x9.csv",
                solution_path="sudoku_solution_9x9.csv",
                n_episodes=args.episodes,
                save_path="best_model.keras"
            )
            
            # Test after training
            print("\n🧪 Testing trained model...\n")
            test(model_path="best_model.keras")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
