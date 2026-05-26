import numpy as np
import time
from ..utils import SudokuGrid

class BacktrackSolver:

    def __init__(self, sudoku_grid):
        if isinstance(sudoku_grid, np.ndarray):
            sudoku_grid = SudokuGrid(sudoku_grid)
        assert isinstance(sudoku_grid, SudokuGrid), \
            "Please enter an numpy array or a SudokuGrid object."
        self.iterations = 0
        self.bactracks = 0
        self.sudoku_grid = sudoku_grid
        self.history = [sudoku_grid]
        self.children = {}

    def solve_ui(self, update_callback=None):
        while not self.sudoku_grid.grid.is_complete():
            previous_grid = self.sudoku_grid.grid.grid.copy()
            self.sudoku_grid = self.choose_action()
            self.iterations += 1
            for i in range(9):
                for j in range(9):
                    old_val = previous_grid[i][j]
                    new_val = self.sudoku_grid.grid.grid[i][j]
                    if old_val != new_val and update_callback:
                        update_callback(i, j, new_val)
                        time.sleep(0.01)
        return self.sudoku_grid

    def solve(self):
        while not self.sudoku_grid.grid.is_complete():
            self.sudoku_grid = self.choose_action()

            self.iterations += 1
        return self.sudoku_grid

    def choose_action(self):

        if self.sudoku_grid not in self.children:
            self.children[self.sudoku_grid] = self.sudoku_grid.find_children()
            if len(self.children[self.sudoku_grid]) == 0:
                return self.sudoku_grid
            new_grid = self.children[self.sudoku_grid].pop()
            self.history.append(new_grid)
            return new_grid

        if len(self.children[self.sudoku_grid]) == 0:
            if len(self.history) == 0:
                raise RuntimeError("Solver failed")
            return self.history.pop()

        new_grid = self.children[self.sudoku_grid].pop()
        self.bactracks += 1
        self.history.append(new_grid)
        return new_grid
