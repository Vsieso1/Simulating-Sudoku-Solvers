import tkinter as tk
from tkinter import ttk
import threading
import time
import csv
import random
from tkinter import StringVar

from ..backtrack import BacktrackSolver
from ..mcts import MCTS
from ..deep_iterative_solver import DeepIterativeSolver
from ..alpha_sudoku import AlphaSudoku
from ..SAT_Sudoku import SATSolver
from ..Reinforcement_Learning_Sudoku import RLSolver
import numpy as np

class SudokuApp:
    def __init__(self, racine):
        self.racine = racine
        self.racine.title("Résolution de Sudoku")
        self.racine.configure(bg="#2E2E2E")

        self.grid_size = 9
        self.cells = [[None]*self.grid_size for _ in range(self.grid_size)]
        self.sudoku_grid = [[0]*9 for _ in range(9)]
        self.solution_grid = [[0]*9 for _ in range(9)]
        self.initial_mask = [[False]*9 for _ in range(9)]
        self.current_puzzle_index = 0
        self.csv_puzzles = []
        self.selected_index = tk.IntVar(value=0)
        self.n_iterations = 0
        self.resolution_time = 0
        self.current_difficulty = ""
        self.rollouts = 0

        self.setup_ui()
        self.load_puzzle()

    def setup_ui(self):
        self.title_label = tk.Label(self.racine, text="", font=("Arial", 18, "bold"), bg="#2E2E2E", fg="white")
        self.title_label.pack(pady=(10, 0))

        frame = tk.Frame(self.racine, bg="#2E2E2E")
        frame.pack(pady=20)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                entry = tk.Entry(frame, width=3, font=("Arial", 14), justify='center', bg="#1E1E1E", fg="white", relief=tk.FLAT)
                padx, pady = (2, 2)
                if j % 3 == 0 and j != 0:
                    padx = (6, 2)
                if i % 3 == 0 and i != 0:
                    pady = (6, 2)
                entry.grid(row=i, column=j, padx=padx, pady=pady, ipadx=10, ipady=10)
                self.cells[i][j] = entry

        # Choix de la grille via menu déroulant
        dropdown_frame = tk.Frame(self.racine, bg="#2E2E2E")
        dropdown_frame.pack(pady=5)

        dropdown_label = tk.Label(dropdown_frame, text="Choisir une grille :", bg="#2E2E2E", fg="white")
        dropdown_label.pack(side=tk.LEFT, padx=(0, 10))

        self.combo_box = ttk.Combobox(dropdown_frame, state="readonly", width=30)
        self.combo_box['values'] = [f"Grille {i}" for i in range(700)]  # supposons max 699 grilles
        self.combo_box.set("Grille aléatoire")
        self.combo_box.pack(side=tk.LEFT)

        self.combo_box.bind("<<ComboboxSelected>>", self.on_puzzle_selected)

        button_frame = tk.Frame(self.racine, bg="#2E2E2E")
        button_frame.pack(pady=10)

        self.solve_buttons = {
            'backtrack': tk.Button(button_frame, text="Résoudre avec Backtrack", command=lambda: self.start_solving("backtrack"), bg="#444", fg="white", relief=tk.FLAT),
            'monte_carlo': tk.Button(button_frame, text="Résoudre avec Monte Carlo", command=lambda: self.start_solving("monte_carlo"), bg="#444", fg="white", relief=tk.FLAT),
            'deep_iterative': tk.Button(button_frame, text="Résoudre avec Deep Iterative Solver", command=lambda: self.start_solving("deep_iterative"), bg="#444", fg="white", relief=tk.FLAT),
            'alpha': tk.Button(button_frame, text="Résoudre avec AlphaSudoku", command=lambda: self.start_solving("alpha"), bg="#444", fg="white", relief=tk.FLAT),
            'sat': tk.Button(button_frame, text="Résoudre avec SAT Sudoku", command=lambda: self.start_solving("sat"), bg="#444", fg="white", relief=tk.FLAT),
            'rl_sudoku': tk.Button(button_frame, text="Résoudre avec RL Sudoku",command=lambda: self.start_solving("rl_sudoku"), bg="#444", fg="white",relief=tk.FLAT)

        }

        for btn in self.solve_buttons.values():
            btn.pack(side=tk.LEFT, padx=10)

        new_puzzle_btn = tk.Button(button_frame, text="Nouvelle Grille", command=self.load_puzzle, bg="#888", fg="white", relief=tk.FLAT)
        new_puzzle_btn.pack(side=tk.LEFT, padx=10)

        reset_puzzle_btn = tk.Button(button_frame, text="Réinitialiser la Grille", command=self.reset_current_puzzle, bg="#555", fg="white", relief=tk.FLAT)
        reset_puzzle_btn.pack(side=tk.LEFT, padx=10)

        check_btn = tk.Button(button_frame, text="Vérifier", command=self.check_user_input, bg="#444", fg="white",relief=tk.FLAT)
        check_btn.pack(side=tk.LEFT, padx=10)

        self.progress_bar = ttk.Progressbar(self.racine, mode='indeterminate')
        self.progress_bar.pack(pady=10, fill=tk.X, padx=20)
        self.progress_label = tk.Label(self.racine, text="Choisir une méthode de résolution.", bg="#2E2E2E", fg="white")
        self.progress_label.pack()

        self.iterations_label = tk.Label(self.racine, text="Nombre d'itérations : ", font=("Arial", 12), bg="#2e2e2e", fg="white")
        self.iterations_label.pack(pady=10)
        self.time_label = tk.Label(self.racine, text="Temps de résolution : ", font=("Arial", 12), bg="#2e2e2e", fg="white")
        self.time_label.pack(pady=10)
        self.rollouts_label = tk.Label(self.racine, text="Nombre de rollouts : ", font=("Arial", 12), bg="#2e2e2e",fg="white")
        self.rollouts_label.pack(pady=10)

    def update_timer(self, start_time, solving_flag):
        def _update():
            if solving_flag["running"]:
                elapsed = round(time.time() - start_time, 1)
                self.time_label.config(text=f"Temps de résolution : {elapsed} s")
                self.racine.after(100, _update)

        _update()

    def load_puzzle(self):
        with open("Sudoku/assets/data.csv", newline='') as csvfile:
            reader = list(csv.reader(csvfile))[1:]
            solution = list(csv.reader(csvfile))[2:]
            self.csv_puzzles = reader  # stocker pour sélection manuelle
            self.current_puzzle_index = random.randint(0, min(698, len(reader) - 1))

        self._load_puzzle_from_index(self.current_puzzle_index)
        if hasattr(self, "combo_box"):
            self.combo_box.set(f"Grille {self.current_puzzle_index}")

    def on_puzzle_selected(self, event):
        selected_label = self.combo_box.get()
        index = int(selected_label.split(" ")[-1])
        self.current_puzzle_index = index
        self._load_puzzle_from_index(index)

    def reset_current_puzzle(self):
        self._load_puzzle_from_index(self.current_puzzle_index)

    def is_valid(self, row, col, val):
        grid = np.array(self.sudoku_grid)
        if val in grid[row]: return False
        if val in grid[:, col]: return False
        block_row, block_col = 3 * (row // 3), 3 * (col // 3)
        if val in grid[block_row:block_row + 3, block_col:block_col + 3]: return False
        return True

    def _load_puzzle_from_index(self, index):
        puzzle_data = self.csv_puzzles[index]
        puzzle_str = puzzle_data[1]
        solution_str = puzzle_data[2]  # Supposons qu’elle soit à l’index 2
        self.current_difficulty = puzzle_data[4] if len(puzzle_data) > 4 else "Inconnue"

        self.sudoku_grid = [
            [int(puzzle_str[i * 9 + j]) if puzzle_str[i * 9 + j] != '.' else 0 for j in range(9)]
            for i in range(9)
        ]

        self.solution_grid = [
            [int(solution_str[i * 9 + j]) if solution_str[i * 9 + j] != '.' else 0 for j in range(9)]
            for i in range(9)
        ]

        for i in range(9):
            for j in range(9):
                self.cells[i][j].config(state='normal', bg="#1E1E1E", fg="white")
                self.cells[i][j].delete(0, tk.END)
                if self.sudoku_grid[i][j] != 0:
                    self.cells[i][j].insert(0, str(self.sudoku_grid[i][j]))
                    self.cells[i][j].config(state='disabled', disabledbackground="#666666", disabledforeground="white")
                self.initial_mask[i][j] = self.sudoku_grid[i][j] != 0

        self.title_label.config(text=f"Grille numéro {index} — Difficulté : {self.current_difficulty}")
        self.n_iterations = 0
        self.resolution_time = 0
        self.progress_label.config(text="Nouvelle grille chargée.")
        self.iterations_label.config(text="Nombre d'itérations : ")
        self.time_label.config(text="Temps de résolution : ")

    def start_solving(self, method):
        for btn in self.solve_buttons.values():
            btn.config(state=tk.DISABLED)

        self.progress_label.config(text="Résolution en cours...")
        self.progress_bar.start()

        method_names = {
            "backtrack": "Backtrack",
            "monte_carlo": "Monte Carlo",
            "deep_iterative": "Deep Iterative Solver",
            "alpha": "AlphaSudoku",
            "sat": "SAT Solver",
            "rl_sudoku": "RL Sudoku"
        }
        readable_method = method_names.get(method, method.capitalize())
        self.title_label.config(
            text=f"Grille numéro {self.current_puzzle_index} — Difficulté : {self.current_difficulty} — Méthode : {readable_method}")

        self.solving_flag = {"running": True}
        start_time = time.time()
        self.update_timer(start_time, self.solving_flag)

        thread = threading.Thread(target=self.solve, args=(method, start_time, self.solving_flag), daemon=True)
        thread.start()

    def solve(self, method, start_time, solving_flag):
        self.n_iterations = 0
        sudoku_array = np.array(self.sudoku_grid)

        if method == "backtrack":
            solver = BacktrackSolver(sudoku_array)
        elif method == "monte_carlo":
            solver = MCTS(sudoku_array)
        elif method == "deep_iterative":
            solver = DeepIterativeSolver(sudoku_array)
        elif method == "alpha":
            solver = AlphaSudoku(sudoku_array)
        elif method == "sat":
            solver = SATSolver(sudoku_array)
        elif method == "rl_sudoku":
            solver = RLSolver(sudoku_array)
        else:
            return

        solver.solve_ui(update_callback=lambda r, c, v: self.racine.after(0, self.update_cell, r, c, v))
        if method == "deep_iterative":
            grid_data = solver.grid.grid
        else:
            grid_data = solver.sudoku_grid.grid.grid

        if method == "backtrack":
            self.rollouts = solver.bactracks
        if method == "monte_carlo":
            self.rollouts = solver.rollouts

        try:
            self.sudoku_grid = np.array(grid_data).astype(int).tolist()
        except Exception as e:
            print(f"[ERREUR conversion grille] Type: {type(grid_data)}, Exception: {e}")
            self.sudoku_grid = [[0] * 9 for _ in range(9)]

        self.n_iterations = getattr(solver, 'iterations', 0)
        self.resolution_time = round(time.time() - start_time,2)
        solving_flag["running"] = False
        self.racine.after(0, self.update_ui)

    def check_user_input(self):
        for i in range(9):
            for j in range(9):
                if not self.initial_mask[i][j]:
                    entry_value = self.cells[i][j].get()
                    try:
                        value = int(entry_value)
                    except ValueError:
                        value = 0
                    if value == self.solution_grid[i][j]:
                        self.cells[i][j].config(bg="#007700", fg="white")  # vert
                    elif value != 0:
                        self.cells[i][j].config(bg="#aa0000", fg="white")  # rouge
                    else:
                        self.cells[i][j].config(bg="#1E1E1E", fg="white")  # vide

    def update_ui(self):
        for i in range(9):
            for j in range(9):
                self.cells[i][j].config(state='normal')
                self.cells[i][j].delete(0, tk.END)
                value = self.sudoku_grid[i][j]
                if value != 0:
                    self.cells[i][j].insert(0, str(value))

                if self.initial_mask[i][j]:
                    self.cells[i][j].config(state='disabled', disabledbackground="#666666", disabledforeground="white")
                else:
                    if value == self.solution_grid[i][j]:
                        self.cells[i][j].config(bg="#007700", fg="white")  # vert
                    elif value != 0:
                        self.cells[i][j].config(bg="#aa0000", fg="white")  # rouge
                    else:
                        self.cells[i][j].config(bg="#1E1E1E", fg="white")  # vide / neutre

        self.progress_bar.stop()
        self.progress_label.config(text="Résultat :")
        self.iterations_label.config(text=f"Nombre d'itérations : {self.n_iterations}")
        self.time_label.config(text=f"Temps de résolution : {self.resolution_time} secondes")
        self.rollouts_label.config(text=f"Nombre de rollouts : {self.rollouts}")

        for btn in self.solve_buttons.values():
            btn.config(state=tk.NORMAL)

    def update_cell(self, row, col, val, highlight=True):
        self.sudoku_grid[row][col] = val
        val = int(val)
        cell = self.cells[row][col]

        cell.config(state='normal')
        cell.delete(0, tk.END)

        if val != 0:
            cell.insert(0, str(val))
            if highlight:
                if val == self.solution_grid[row][col]:
                    cell.config(bg="#007700", fg="white")  # vert : correct
                else:
                    cell.config(bg="#aa0000", fg="white")  # rouge : incorrect
        else:
            cell.config(bg="#1E1E1E", fg="white")

        cell.update_idletasks()

if __name__ == '__main__':
    root = tk.Tk()
    app = SudokuApp(root)
    root.mainloop()