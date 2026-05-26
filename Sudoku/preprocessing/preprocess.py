import numpy as np
from ..utils import SudokuGrid

class ConstraintSolver:

    def __init__(self, sudoku_grid):
        if isinstance(sudoku_grid, np.ndarray):
            sudoku_grid = SudokuGrid(sudoku_grid)
        assert isinstance(sudoku_grid, SudokuGrid), \
            "Please provide a numpy array or a SudokuGrid object."
        self.grid = sudoku_grid.grid.grid
        self.sudoku_grid = sudoku_grid
        self.fill = 0

    def solve(self):
        progress = True
        while progress:
            self.fill += 1
            progress = self.apply_naked_singles()
            progress |= self.apply_hidden_singles()

    def get_possibilities(self, row, col):
        """Retourne les valeurs possibles pour une case vide"""
        if self.grid[row, col] != 0:
            return set()

        used = set(self.grid[row, :]) | set(self.grid[:, col]) | set(self.get_block(row, col))
        return set(range(1, 10)) - used

    def get_block(self, row, col):
        """Retourne les valeurs du bloc 3x3 contenant (row, col)"""
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        return self.grid[start_row:start_row+3, start_col:start_col+3].flatten()

    def apply_naked_singles(self):
        """Remplit les cases où une seule valeur est possible (naked singles)"""
        progress = False
        for row in range(9):
            for col in range(9):
                if self.grid[row, col] == 0:
                    possibilities = self.get_possibilities(row, col)
                    if len(possibilities) == 1:
                        value = possibilities.pop()
                        self.grid[row, col] = value
                        progress = True
        return progress

    def apply_hidden_singles(self):
        """Remplit les cases qui sont la seule possibilité pour une valeur donnée (hidden singles)"""
        progress = False
        for unit in self.get_all_units():
            for digit in range(1, 10):
                positions = [(r, c) for r, c in unit if self.grid[r, c] == 0 and digit in self.get_possibilities(r, c)]
                if len(positions) == 1:
                    r, c = positions[0]
                    self.grid[r, c] = digit
                    progress = True
        return progress

    def get_all_units(self):
        """Retourne toutes les unités (lignes, colonnes, blocs) sous forme de listes de tuples (row, col)"""
        units = []

        # Lignes
        for r in range(9):
            units.append([(r, c) for c in range(9)])
        # Colonnes
        for c in range(9):
            units.append([(r, c) for r in range(9)])
        # Blocs
        for br in range(3):
            for bc in range(3):
                units.append([(r, c)
                              for r in range(br * 3, br * 3 + 3)
                              for c in range(bc * 3, bc * 3 + 3)])
        return units
