#!/usr/bin/env python
"""
Run comprehensive tests on all Sudoku solvers.

Usage:
    python scripts/run_tests.py                                    # Test moyen difficulty
    python scripts/run_tests.py --difficulty facile --limit 10    # Test 10 easy puzzles
    python scripts/run_tests.py --difficulty expert               # Test expert difficulty

Available difficulties:
    facile, moyen, difficile, expert, impossible
"""

import sys
import argparse
import time
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Sudoku.backtrack import BacktrackSolver
from Sudoku.mcts import MCTS
from Sudoku.deep_iterative_solver import DeepIterativeSolver
from Sudoku.alpha_sudoku import AlphaSudoku
from Sudoku.SAT_Sudoku import SATSolver
from Sudoku.Reinforcement_Learning_Sudoku import RLSolver
from Sudoku.preprocessing import ConstraintSolver


def string_to_grid(s):
    """Convert string representation to 9x9 numpy array."""
    return np.array([int(c) if c.isdigit() else 0 for c in s.strip()]).reshape((9, 9))


def run_tests(difficulty='moyen', limit=None):
    """
    Run tests on all solvers for a specific difficulty.
    
    Args:
        difficulty (str): Puzzle difficulty level
        limit (int): Max number of puzzles to test (None = all)
    """
    # Load test data
    data_path = Path(__file__).parent.parent / "Test_grid" / f"data_{difficulty}.csv"
    
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        print(f"Available difficulties: facile, moyen, difficile, expert, impossible")
        return
    
    print(f"\n{'='*70}")
    print(f"Testing on {difficulty} difficulty")
    print(f"{'='*70}\n")
    
    df = pd.read_csv(data_path)
    
    data_X = [string_to_grid(puzzle) for puzzle in df['puzzle']]
    data_Y = [string_to_grid(solution) for solution in df['solution']]
    clues = [clue for clue in df['clues']]
    
    # Limit number of tests
    if limit:
        data_X = data_X[:limit]
        data_Y = data_Y[:limit]
        clues = clues[:limit]
    
    n_tests = len(data_X)
    print(f"Testing {n_tests} puzzles...\n")
    
    # Initialize result lists
    results = {
        'BT': {'time': [], 'it': [], 'succ': [], 'back': []},
        'MCTS': {'time': [], 'it': [], 'succ': [], 'rollouts': []},
        'DIS': {'time': [], 'it': [], 'succ': []},
        'AS': {'time': [], 'it': [], 'succ': []},
        'SAT': {'time': [], 'it': [], 'succ': []},
        'RL': {'time': [], 'it': [], 'succ': []},
    }
    
    # Load model
    model_path = Path(__file__).parent.parent / "Sudoku" / "policy_network"
    if model_path.exists():
        model = load_model(str(model_path))
    else:
        model = None
        print("⚠️  Warning: Policy network not found. Some solvers will be skipped.\n")
    
    # Run tests
    for i, (puzzle, solution) in enumerate(zip(data_X, data_Y)):
        print(f"Testing grid {i+1}/{n_tests}...", end='\r')
        
        # Preprocess
        preprocessed = ConstraintSolver(puzzle.copy())
        preprocessed.solve()
        preprocessed_grid = preprocessed.grid
        
        # Backtrack
        try:
            start = time.time()
            bt_solver = BacktrackSolver(preprocessed_grid.copy())
            bt_solver.solve()
            elapsed = round(time.time() - start, 3)
            results['BT']['time'].append(elapsed)
            results['BT']['it'].append(bt_solver.iterations)
            results['BT']['succ'].append(np.array_equal(bt_solver.sudoku_grid.grid.grid, solution))
            results['BT']['back'].append(bt_solver.bactracks)
        except Exception as e:
            print(f"\nBacktrack error on grid {i+1}: {e}")
        
        # MCTS (limited iterations for speed)
        try:
            start = time.time()
            mcts_solver = MCTS(preprocessed_grid.copy(), max_iterations=500)
            mcts_solver.solve()
            elapsed = round(time.time() - start, 3)
            results['MCTS']['time'].append(elapsed)
            results['MCTS']['it'].append(mcts_solver.iterations)
            results['MCTS']['succ'].append(np.array_equal(mcts_solver.sudoku_grid.grid.grid, solution))
            results['MCTS']['rollouts'].append(mcts_solver.rollouts)
        except Exception as e:
            print(f"\nMCTS error on grid {i+1}: {e}")
        
        # Deep Iterative Solver
        if model:
            try:
                start = time.time()
                dis_solver = DeepIterativeSolver(preprocessed_grid.copy(), model=model)
                dis_solver.solve()
                elapsed = round(time.time() - start, 3)
                results['DIS']['time'].append(elapsed)
                results['DIS']['it'].append(dis_solver.iterations)
                results['DIS']['succ'].append(np.array_equal(dis_solver.grid.grid, solution))
            except Exception as e:
                print(f"\nDeep Iterative error on grid {i+1}: {e}")
    
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    # Print results
    for method_name, method_results in results.items():
        if not method_results['time']:
            continue
        
        print(f"--- {method_name} ---")
        print(f"  Avg Time: {statistics.mean(method_results['time']):.4f}s")
        print(f"  Median Time: {statistics.median(method_results['time']):.4f}s")
        print(f"  Avg Iterations: {statistics.mean(method_results['it']):.2f}")
        if method_results['succ']:
            success_rate = 100 * sum(method_results['succ']) / len(method_results['succ'])
            print(f"  Success Rate: {success_rate:.2f}%")
        print()


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description='Test Sudoku solvers on puzzles of various difficulties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py
  python scripts/run_tests.py --difficulty facile
  python scripts/run_tests.py --difficulty expert --limit 5
        """
    )
    
    parser.add_argument(
        '--difficulty',
        default='moyen',
        choices=['facile', 'moyen', 'difficile', 'expert', 'impossible'],
        help='Puzzle difficulty level (default: moyen)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of puzzles to test'
    )
    
    args = parser.parse_args()
    
    run_tests(args.difficulty, args.limit)


if __name__ == "__main__":
    main()
