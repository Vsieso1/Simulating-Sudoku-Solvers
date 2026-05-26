import numpy as np
from ..utils import SudokuGrid
from itertools import combinations
from collections import Counter

class SATSolver:

    def __init__(self, sudoku_grid):
        # Vérifie que l'entrée est bien un Sudoku sous forme attendue
        if isinstance(sudoku_grid, np.ndarray):
            sudoku_grid = SudokuGrid(sudoku_grid)
        assert isinstance(sudoku_grid, SudokuGrid), \
            "Please enter a numpy array or a SudokuGrid object."

        self.iterations = 0  # Nombre d'itérations utilisées par le solveur
        self.variables = {}  # Dictionnaire des variables SAT
        self.clauses = []  # Liste des clauses logiques (CNF)
        self.sudoku_grid = sudoku_grid  # La grille Sudoku à résoudre

    def valeur_unique(self, i, j, k):
        # Encode une variable unique pour la case (i, j) contenant la valeur k
        return 81 * (i - 1) + 9 * (j - 1) + k

    def encode_clause(self):
        pairs_1_9 = list(combinations(range(1, 10), 2))

        # Sous-fonction : impose unicité parmi les variables données
        def clause_unique(vars_list):
            self.clauses.append(vars_list)  # Au moins une des variables doit être vraie
            for a, b in combinations(vars_list, 2):
                self.clauses.append([-a, -b])  # Pas deux variables vraies simultanément

        # Génération de toutes les variables possibles
        for i in range(1, 10):
            for j in range(1, 10):
                for k in range(1, 10):
                    var_id = self.valeur_unique(i, j, k)
                    self.variables[var_id] = None

        # Contraintes : respecter les valeurs initialement remplies dans la grille
        for i in range(9):
            for j in range(9):
                val = self.sudoku_grid.grid.grid[i][j]
                if val != 0:
                    var_id = self.valeur_unique(i + 1, j + 1, val)
                    self.clauses.append([var_id])

        # 1. Chaque case contient exactement une valeur
        for i in range(1, 10):
            for j in range(1, 10):
                vars_ = [self.valeur_unique(i, j, k) for k in range(1, 10)]
                clause_unique(vars_)

        # 2. Chaque valeur apparaît une seule fois par ligne
        for i in range(1, 10):
            for k in range(1, 10):
                vars_ = [self.valeur_unique(i, j, k) for j in range(1, 10)]
                clause_unique(vars_)

        # 3. Chaque valeur apparaît une seule fois par colonne
        for j in range(1, 10):
            for k in range(1, 10):
                vars_ = [self.valeur_unique(i, j, k) for i in range(1, 10)]
                clause_unique(vars_)

        # 4. Chaque valeur apparaît une seule fois par bloc 3x3
        for bi in range(0, 3):
            for bj in range(0, 3):
                for k in range(1, 10):
                    vars_ = [
                        self.valeur_unique(i, j, k)
                        for i in range(1 + 3 * bi, 4 + 3 * bi)
                        for j in range(1 + 3 * bj, 4 + 3 * bj)
                    ]
                    clause_unique(vars_)

    @staticmethod
    def resout_clause(clauses, assignment=None, solver=None):

        if assignment is None:
            assignment = {}

        # Simplifie les clauses après une affectation d'une variable
        def simplify(clauses, var, value):
            result = []
            for clause in clauses:
                if (var if value else -var) in clause:
                    continue  # Clause satisfaite, on l'ignore
                new_clause = [lit for lit in clause if lit != (-var if value else var)]
                if not new_clause:
                    return None  # Clause vide => conflit
                result.append(new_clause)
            return result

        if solver:
            solver.iterations += 1  # Compte une itération supplémentaire

        # === Propagation unitaire complète (résolution immédiate des variables évidentes) ===
        while True:
            unit_clause = next((c for c in clauses if len(c) == 1), None)
            if unit_clause is None:
                break
            lit = unit_clause[0]
            var = abs(lit)
            val = lit > 0

            if var in assignment:
                if assignment[var] != val:
                    return None  # Conflit détecté
            else:
                assignment[var] = val
                clauses = simplify(clauses, var, val)
                if clauses is None:
                    return None

        # Si toutes les clauses sont satisfaites, retour de l'affectation
        if not clauses:
            return assignment
        if [] in clauses:
            return None  # Conflit détecté

        # === Heuristique : choisir la variable la plus fréquente parmi celles non assignées ===
        lit_counts = Counter(abs(lit) for clause in clauses for lit in clause if abs(lit) not in assignment)
        if not lit_counts:
            return assignment  # Plus de variables à assigner

        var = lit_counts.most_common(1)[0][0]  # Variable la plus fréquente

        # Tentative avec la variable à True puis à False
        for val in [True, False]:
            new_assignment = assignment.copy()
            new_assignment[var] = val
            new_clauses = simplify(clauses, var, val)
            if new_clauses is not None:
                result = SATSolver.resout_clause(new_clauses, new_assignment, solver)
                if result is not None:
                    return result

        return None  # Échec

    def solve(self):
        self.encode_clause()  # Encode le sudoku sous forme SAT
        model = self.resout_clause(self.clauses, solver=self)

        if model is None:
            raise Exception("Le Sudoku n’a pas de solution.")  # Aucune solution trouvée

        self.interpret_model(model)  # Remplit la grille Sudoku avec la solution trouvée

    def interpret_model(self, model):
        # Interprète l'affectation du modèle pour remplir la grille Sudoku
        for var in model:
            if model[var]:
                v = var - 1
                i = v // 81
                j = (v % 81) // 9
                k = (v % 9) + 1
                self.sudoku_grid.grid.grid[i][j] = k

    def solve_ui(self, update_callback=None):
        self.solve()