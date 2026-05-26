# Simulating Sudoku Solvers

A comprehensive comparative study of multiple Sudoku solving algorithms, implemented in Python. This project explores and benchmarks different approaches including classical algorithms, constraint satisfaction, and machine learning techniques.

## 🎯 Project Overview

This research project implements and compares various Sudoku solving strategies:
- **Backtracking**: Classic recursive constraint propagation
- **Deep Iterative Solver**: Constraint satisfaction with iterative deepening
- **SAT Solver**: Boolean satisfiability approach
- **Reinforcement Learning (DQN)**: Deep Q-Network trained model
- **Monte Carlo Tree Search (MCTS)**: Probabilistic tree exploration
- **GUI Interface**: Interactive Sudoku visualization and solving

Perfect for understanding algorithm efficiency, performance metrics, and AI/ML approaches to combinatorial problems.

## 📁 Project Structure

```
├── Sudoku/                          # Main Sudoku solver package
│   ├── backtrack/                   # Backtracking algorithm
│   │   └── backtrack.py
│   ├── deep_iterative_solver/       # Iterative constraint solver
│   │   └── deep_iterative_solver.py
│   ├── SAT_Sudoku/                  # SAT-based solver
│   │   └── SAT_Sudoku.py
│   ├── mcts/                        # Monte Carlo Tree Search
│   │   └── monte_carlo_tree_search.py
│   ├── Reinforcement_Learning_Sudoku/  # DQN-based solver
│   │   ├── RL_Sudoku.py
│   │   └── RL_Sudoku_model.h5       # Trained DQN model
│   ├── GUI_Sudoku/                  # Interactive GUI
│   │   └── GUI.py
│   ├── alpha_sudoku/                # Alpha solver variant
│   ├── preprocessing/               # Data preprocessing utilities
│   ├���─ utils/                       # Utility functions and helpers
│   ├── assets/                      # Sudoku puzzle datasets
│   │   ├── data_facile.csv          # Easy puzzles
│   │   ├── data_moyen.csv           # Medium puzzles
│   │   ├── data_difficile.csv       # Difficult puzzles
│   │   ├── data_expert.csv          # Expert puzzles
│   │   └── data_impossible.csv      # Impossible/Extreme puzzles
│   ├── policy_network/              # Pre-trained policy network
│   ├── main.py                      # Main entry point
│   └── requirements.txt
│
├── RL/                              # Reinforcement Learning experiments
│   ├── train_sudoku_dqn.py          # DQN training script
│   ├── create_sudoku.py             # Sudoku generation
│   ├── best_model.keras             # Best trained model
│   ├── sudoku_train_9x9.csv         # Training dataset
│   ├── sudoku_solution_9x9.csv      # Solutions
│   ├── data.csv                     # Experiment results
│   └── *.png                        # Performance plots
│
├── Test_grid/                       # Test grids and validation
│   ├── remplissage.py               # Grid filling utilities
│   └── data_*.csv                   # Test datasets by difficulty
│
└── Agenda.txt                       # Project timeline/notes
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Vsieso1/Simulating-Sudoku-Solvers.git
cd Simulating-Sudoku-Solvers
```

2. **Install dependencies**
```bash
cd Sudoku
pip install -r requirements.txt
```

### Quick Start

**Run the GUI**
```bash
python Sudoku/main.py
```

**Solve a puzzle with backtracking**
```python
from Sudoku.backtrack.backtrack import SudokuBacktrack
solver = SudokuBacktrack(grid)
solution = solver.solve()
```

**Use the SAT Solver**
```python
from Sudoku.SAT_Sudoku.SAT_Sudoku import SudokuSAT
solver = SudokuSAT(grid)
solution = solver.solve()
```

**Train/Use the RL Solver**
```python
from Sudoku.Reinforcement_Learning_Sudoku.RL_Sudoku import RLSudokuSolver
solver = RLSudokuSolver(model_path='RL_Sudoku_model.h5')
solution = solver.solve(grid)
```

## 📊 Algorithms Compared

| Algorithm | Type | Speed | Scalability | Notes |
|-----------|------|-------|-------------|-------|
| **Backtracking** | Classical | Fast | O(n^m) | Reliable baseline |
| **Deep Iterative** | Constraint-based | Medium | Good | Optimized propagation |
| **SAT Solver** | Boolean Logic | Fast | Excellent | Industry-standard approach |
| **MCTS** | Probabilistic | Slow | Medium | Explores solution space |
| **Reinforcement Learning** | ML-based | Variable | Good | Data-driven, trained on examples |

## 📈 Datasets

The project includes Sudoku puzzles of varying difficulty levels:
- **Easy (facile)**: ~30% pre-filled cells
- **Medium (moyen)**: ~25% pre-filled cells
- **Difficult (difficile)**: ~20% pre-filled cells
- **Expert (expert)**: ~15% pre-filled cells
- **Impossible**: Specially crafted extreme puzzles

Each dataset contains puzzles and their solutions in CSV format.

## 🧠 Machine Learning Components

### Reinforcement Learning (DQN)
- **Model**: Deep Q-Network trained with experience replay
- **Input**: Sudoku grid state (9×9 matrix)
- **Output**: Next best cell and value to fill
- **Training**: `RL/train_sudoku_dqn.py`
- **Evaluation**: Performance plots in `RL/` directory

### Policy Network
- Pre-trained neural network for move suggestion
- Located in `Sudoku/policy_network/`
- Used for intelligent move selection

## 📋 Usage Examples

### Example 1: Solve with Multiple Algorithms
```python
from Sudoku.backtrack.backtrack import SudokuBacktrack
from Sudoku.SAT_Sudoku.SAT_Sudoku import SudokuSAT
from Sudoku.mcts.monte_carlo_tree_search import SudokuMCTS

# Your Sudoku grid
grid = [[5,3,0,0,7,...], ...]

# Solve with different methods
backtrack_solver = SudokuBacktrack(grid)
sat_solver = SudokuSAT(grid)
mcts_solver = SudokuMCTS(grid)

# Compare solutions
print("Backtracking:", backtrack_solver.solve())
print("SAT:", sat_solver.solve())
print("MCTS:", mcts_solver.solve())
```

### Example 2: Interactive GUI
```bash
python Sudoku/main.py
```
The GUI provides:
- Visual grid representation
- Solver selection dropdown
- Step-by-step solution visualization
- Performance timing
- Multiple difficulty levels

## 🔬 Research Insights

This project explores:
1. **Classical vs Modern Approaches**: How traditional algorithms compare to ML-based solutions
2. **Trade-offs**: Speed vs. accuracy, training time vs. inference time
3. **Scalability**: Performance on increasingly difficult puzzles
4. **Hybrid Methods**: Combining constraint satisfaction with learning

## 📊 Results

Performance metrics and comparison plots are available in:
- `RL/`: Training curves, performance benchmarks
- `Sudoku/assets/`: Processed result data
- Root directory: `preprocessed_resultats_*.txt` files

## 🛠️ Tools & Technologies

- **Python 3.8+**
- **TensorFlow/Keras**: Neural network training
- **NumPy/Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **PyQt/Tkinter**: GUI framework
- **SAT Solvers**: Constraint satisfaction

## 📝 Key Files

- `Sudoku/main.py` - Main application entry point
- `Sudoku/readme.md` - Additional Sudoku package documentation
- `RL/train_sudoku_dqn.py` - DQN training pipeline
- `Sudoku/utils/sudoku_grid.py` - Grid manipulation utilities
- `Sudoku/preprocessing/preprocess.py` - Data preprocessing

## 🎓 Conference Presentation

This project is ideal for:
- **AI/ML Conferences**: Demonstrating ML approaches to classic problems
- **Algorithm Analysis Talks**: Comparing algorithm efficiency
- **Game Theory Symposiums**: Combinatorial problem solving
- **Software Engineering**: Practical implementation of multiple paradigms

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- [ ] Parallel processing for MCTS
- [ ] GPU acceleration for RL models
- [ ] Additional solver variants
- [ ] Performance optimization
- [ ] Extended documentation

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Vsieso1** - Research & Implementation
- GitHub: [@Vsieso1](https://github.com/Vsieso1)
- Project: [Simulating-Sudoku-Solvers](https://github.com/Vsieso1/Simulating-Sudoku-Solvers)

## 📚 References & Learning Resources

- **Backtracking**: Classic algorithm design
- **SAT Solving**: [SAT Solvers Wikipedia](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem)
- **Reinforcement Learning**: Deep Q-Networks (DQN) papers
- **MCTS**: Monte Carlo Tree Search in games

## 🔗 Quick Links

- [Sudoku Wikipedia](https://en.wikipedia.org/wiki/Sudoku)
- [Constraint Satisfaction Problems](https://en.wikipedia.org/wiki/Constraint_satisfaction_problem)
- [Deep Q-Networks](https://www.nature.com/articles/nature14236)

---

**Last Updated**: May 2026
**Status**: Active Research Project

For questions or discussion about the algorithms, implementations, or results, please open an issue or reach out!
