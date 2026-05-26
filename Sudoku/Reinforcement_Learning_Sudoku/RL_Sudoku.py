import numpy as np
import time
import os
from keras.models import load_model
from ..utils import SudokuGrid

class RLSolver:
    def __init__(self, sudoku_grid, model_path=None):
        if isinstance(sudoku_grid, np.ndarray):
            sudoku_grid = SudokuGrid(sudoku_grid)
        assert isinstance(sudoku_grid, SudokuGrid), \
            "Please enter a numpy array or a SudokuGrid object."

        self.sudoku_grid = sudoku_grid
        self.iterations = 0

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "RL_Sudoku_model.h5")

        self.model = load_model(model_path, compile=False)

    def prepare_input(self):
        # Transforme la grille en un vecteur plat normalisé entre 0 et 1
        grid_flat = np.array(self.sudoku_grid.grid.grid).flatten()
        # Normaliser : 0 reste 0, 1-9 deviennent 1/9 à 9/9
        normalized = grid_flat / 9.0
        return np.expand_dims(normalized, axis=0)  # shape (1, 81)

    def is_valid(self, row, col, num):
        # Vérifie ligne
        if num in self.sudoku_grid.grid.grid[row]:
            return False
        # Vérifie colonne
        if num in [self.sudoku_grid.grid.grid[i][col] for i in range(9)]:
            return False
        # Vérifie sous-grille 3x3
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if self.sudoku_grid.grid.grid[start_row + i][start_col + j] == num:
                    return False
        return True

    def get_valid_actions(self, grid):
        """
        Renvoie la liste des actions valides sous forme (index_case, valeur)
        """
        actions = []
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    for val in range(1, 10):
                        if self.is_valid_move(grid, i, j, val):
                            actions.append((i * 9 + j, val))
        return actions

    def is_valid_move(self, grid, row, col, val):
        # Vérifie ligne et colonne
        if val in grid[row] or val in [grid[i][col] for i in range(9)]:
            return False

        # Vérifie bloc 3x3
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if grid[i][j] == val:
                    return False

        return True

    def solve(self):
        max_steps = 100
        for step in range(max_steps):
            state = self.prepare_input()
            valid_actions = self.get_valid_actions(self.sudoku_grid.grid.grid)
            if not valid_actions:
                print("Plus aucune action valide.")
                break

            q_values = self.model.predict(state, verbose=0)[0]  # shape (729,)
            q_values = q_values.reshape(81, 9)

            # Masquer toutes les actions non valides en -inf
            mask = np.full((81, 9), -np.inf)
            for (cell_idx, val) in valid_actions:
                mask[cell_idx, val - 1] = q_values[cell_idx, val - 1]

            flat_mask = mask.flatten()
            best_action_idx = np.argmax(flat_mask)

            cell_idx = best_action_idx // 9
            number = (best_action_idx % 9) + 1
            row, col = divmod(cell_idx, 9)

            if self.sudoku_grid.grid.grid[row][col] == 0 and self.is_valid(row, col, number):
                self.sudoku_grid.grid.grid[row][col] = number
                self.iterations += 1
            else:
                print(f"Step {step}: Impossible de placer un chiffre valide.")
                break

            if all(all(cell != 0 for cell in row) for row in self.sudoku_grid.grid.grid):
                print(f"Résolu en {step + 1} étapes.")
                return True

        print("Grille non résolue.")
        return False

    def solve_ui(self, update_callback=None, delay=0.05):
        current_grid = self.sudoku_grid.grid.grid.copy()

        while not self.sudoku_grid.grid.is_complete():
            self.iterations += 1

            # Préparer l’état normalisé
            flat = current_grid.flatten().astype('float32') / 9.0
            state = np.array([flat])

            # Prédiction du modèle
            q_values = self.model.predict(state, verbose=0)[0]  # (729,)
            q_values = q_values.reshape(81, 9)

            # Récupérer les actions valides sur la grille actuelle
            valid_actions = self.get_valid_actions(current_grid)
            if not valid_actions:
                print("Plus aucune action valide.")
                break

            # Masque avec -inf sauf pour actions valides
            masked_q = np.full((81, 9), -np.inf)
            for (cell_idx, val) in valid_actions:
                masked_q[cell_idx, val - 1] = q_values[cell_idx, val - 1]

            # Choix de la meilleure action valide
            flat_masked = masked_q.flatten()
            best_action_idx = np.argmax(flat_masked)

            cell_idx = best_action_idx // 9
            number = (best_action_idx % 9) + 1
            row, col = divmod(cell_idx, 9)

            if current_grid[row][col] == 0 and self.is_valid_move(current_grid, row, col, number):
                current_grid[row][col] = number
                if update_callback:
                    update_callback(row, col, number)
                    time.sleep(delay)
            else:
                print(f"Impossible de placer {number} à ({row}, {col})")
                break

        self.sudoku_grid.grid.grid = current_grid
        return self.sudoku_grid
