import numpy as np
import pandas as pd
import random

# Fonction pour générer une grille 9x9 valide
def generate_full_grid():
    base = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                     [4, 5, 6, 7, 8, 9, 1, 2, 3],
                     [7, 8, 9, 1, 2, 3, 4, 5, 6],
                     [2, 3, 4, 5, 6, 7, 8, 9, 1],
                     [5, 6, 7, 8, 9, 1, 2, 3, 4],
                     [8, 9, 1, 2, 3, 4, 5, 6, 7],
                     [3, 4, 5, 6, 7, 8, 9, 1, 2],
                     [6, 7, 8, 9, 1, 2, 3, 4, 5],
                     [9, 1, 2, 3, 4, 5, 6, 7, 8]])

    # On mélange les lignes et colonnes à l'intérieur des blocs
    def shuffle(arr):
        np.random.shuffle(arr)
        return arr

    rows = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    cols = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    rows = shuffle(rows)
    cols = shuffle(cols)

    grid = base[rows][:, cols]
    return grid


# Vérifier la validité de la grille complète
def is_valid_grid(grid):
    # Vérifier les lignes
    for row in grid:
        if sorted(row) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return False

    # Vérifier les colonnes
    for col in range(9):
        if sorted(grid[row][col] for row in range(9)) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return False

    # Vérifier les sous-grilles 3x3
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            block = [grid[r + i][c + j] for i in range(3) for j in range(3)]
            if sorted(block) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                return False

    return True


# Fonction pour retirer des chiffres
def remove_numbers(grid):
    new_grid = grid.copy()
    cells = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(cells)
    num_to_remove = random.randint(30, 40)
    for i in range(num_to_remove):
        x, y = cells[i]
        new_grid[x][y] = 0
    return new_grid


# Vérification si une grille a une solution unique
def has_unique_solution(grid):
    def solve(grid):
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    for num in range(1, 10):
                        if is_safe(grid, r, c, num):
                            grid[r][c] = num
                            if solve(grid):
                                return True
                            grid[r][c] = 0
                    return False
        return True

    def is_safe(grid, r, c, num):
        # Vérifier la ligne
        if num in grid[r]:
            return False
        # Vérifier la colonne
        if num in [grid[i][c] for i in range(9)]:
            return False
        # Vérifier la sous-grille 3x3
        start_r, start_c = (r // 3) * 3, (c // 3) * 3
        for i in range(3):
            for j in range(3):
                if grid[start_r + i][start_c + j] == num:
                    return False
        return True

    grid_copy = [row[:] for row in grid]
    return solve(grid_copy)


# Génération des données
n_samples = 100
inputs = []
solutions = []

for _ in range(n_samples):
    while True:
        solution = generate_full_grid()

        if is_valid_grid(solution):
            break

    puzzle = remove_numbers(solution)
    puzzle2 = puzzle.copy()

    if has_unique_solution(puzzle):
        inputs.append(puzzle2.flatten())
        solutions.append(solution.flatten())
    else:
        continue

pd.DataFrame(inputs).to_csv('sudoku_train_9x9.csv', index=False, header=False)
pd.DataFrame(solutions).to_csv('sudoku_solution_9x9.csv', index=False, header=False)

print("Fichiers sudoku_train_9x9.csv et sudoku_solution_9x9.csv créés !")
