import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
import statistics
import time
import pandas as pd
import ast
import matplotlib.pyplot as plt
import tensorflow as tf

from .Reinforcement_Learning_Sudoku import RLSolver
from .GUI_Sudoku.GUI import SudokuApp
from .utils import read_transform
from .backtrack import BacktrackSolver
from .mcts import MCTS
from .deep_iterative_solver import DeepIterativeSolver
from .alpha_sudoku import AlphaSudoku
from .SAT_Sudoku import SATSolver
from .preprocessing import ConstraintSolver

choix = int(input("1. Accès à l'interface \n 2. Test grilles multiples \n"))
if choix == 1:
    root = tk.Tk()
    app = SudokuApp(root)
    root.mainloop()
else :
    NROWS = 50
    #data_X, data_Y = read_transform(NROWS=NROWS)

    # Conversion des chaînes en numpy arrays
    def string_to_grid(s):
        return np.array([int(c) if c.isdigit() else 0 for c in s.strip()]).reshape((9, 9))

    difficultes = ['facile','moyen','difficile','expert','impossible']

    for difficulte in difficultes:
        print(f"\n===== Traitement de : {difficulte} =====")
        df = pd.read_csv(f"Sudoku/assets/data_{difficulte}.csv")

        data_X = [string_to_grid(puzzle) for puzzle in df['puzzle']]
        data_Y = [string_to_grid(solution) for solution in df['solution']]
        clues = [clue for clue in df['clues']]

        #Liste de résultats
        BT_time,MC_time,DIS_time,AS_time,SAT_time,RL_time = [],[],[],[],[],[]

        BT_it,MC_it,DIS_it,AS_it,SAT_it,RL_it = [],[],[],[],[],[]

        BT_succ,MC_succ,DIS_succ,AS_succ,SAT_succ,RL_succ = [],[],[],[],[],[]

        BT_back,MC_rollouts = [],[]

        nb_preprocessed_cells = []

        model = load_model('Sudoku/policy_network')

        for i in range(len(data_X)):   #Choisir le numéro du puzzle (in range 1, Nrows) pour tester toutes les grilles
            print('Grille numéro ', i)

            preprocessed = ConstraintSolver(data_X[i])
            preprocessed.solve()

            preprocessed_grid = preprocessed.grid
            # nb_preprocessed_cells.append(np.count_nonzero(preprocessed_grid) - clues[i])
            #
            # print('--- Backtrack ---')
            # start_time = time.time()
            # back_solver = BacktrackSolver(preprocessed_grid)
            # back_solver.solve()
            #
            # finish_time = round(time.time() - start_time, 2)
            # print('--- %s seconds ---' % finish_time)
            # print('--- %s iterations ---' % back_solver.iterations)
            # print('--- %s bactracks ---' % back_solver.bactracks)
            # BT_time.append(finish_time)
            # BT_it.append(back_solver.iterations)
            # BT_succ.append(np.array_equal(back_solver.sudoku_grid.grid.grid, data_Y[i]))
            # BT_back.append(back_solver.bactracks)
            #
            # print('--- MCTS ---')
            # start_time = time.time()
            # mcts_solver = MCTS(preprocessed_grid, max_iterations=2000)
            # mcts_solver.solve()
            #
            # finish_time = round(time.time() - start_time, 2)
            # print('--- %s seconds ---' % finish_time)
            # print('--- %s iterations ---' % mcts_solver.iterations)
            # print('--- %s rollouts ---' % mcts_solver.rollouts)
            # MC_time.append(finish_time)
            # MC_it.append(mcts_solver.iterations)
            # MC_succ.append(np.array_equal(mcts_solver.sudoku_grid.grid.grid, data_Y[i]))
            # MC_rollouts.append(mcts_solver.rollouts)
            #
            # print('--- DeepIterativeSolver ---')
            # start_time = time.time()
            # deep_solver = DeepIterativeSolver(preprocessed_grid, model=model)
            # deep_solver.solve()
            #
            # finish_time = round(time.time() - start_time, 2)
            # print('--- %s seconds ---' % finish_time)
            # print('--- %s iterations ---' % deep_solver.iterations)
            # DIS_time.append(finish_time)
            # DIS_it.append(deep_solver.iterations)
            # DIS_succ.append(np.array_equal(deep_solver.grid.grid, data_Y[i]))
            #
            # print('--- AlphaSudoku ---')
            # start_time = time.time()
            # alpha_solver = AlphaSudoku(preprocessed_grid, model=model)
            # alpha_solver.solve()
            #
            # finish_time = round(time.time() - start_time, 2)
            # print('--- %s seconds ---' % finish_time)
            # print('--- %s iterations ---' % alpha_solver.iterations)
            # AS_time.append(finish_time)
            # AS_it.append(alpha_solver.iterations)
            # AS_succ.append(np.array_equal(alpha_solver.sudoku_grid.grid.grid, data_Y[i]))
            #
            # print('--- SAT Sudoku ---')
            # start_time = time.time()
            # SAT_solver = SATSolver(preprocessed_grid)
            # SAT_solver.solve()
            #
            # finish_time = round(time.time() - start_time, 2)
            # print('--- %s seconds ---' % finish_time)
            # print('--- %s iterations ---' % SAT_solver.iterations)
            # SAT_time.append(finish_time)
            # SAT_it.append(SAT_solver.iterations)
            # SAT_succ.append(np.array_equal(SAT_solver.sudoku_grid.grid.grid, data_Y[i]))

            print('--- Reinforcement Learning Sudoku ---')
            start_time = time.time()
            RL_solver = RLSolver(data_X[i])
            RL_solver.solve()

            finish_time = round(time.time() - start_time, 2)
            print('--- %s seconds ---' % finish_time)
            print('--- %s iterations ---' % RL_solver.iterations)
            RL_time.append(finish_time)
            RL_it.append(RL_solver.iterations)
            RL_succ.append(np.array_equal(RL_solver.sudoku_grid.grid.grid, data_Y[i]))

        # #Résultat
        #
        # # Affichage des statistiques pour Backtracking (BT)
        # print("=== Backtracking (BT) ===")
        # print(f"Moyenne du temps d'exécution : {statistics.mean(BT_time):.4f} secondes")
        # print(f"Médiane du temps d'exécution : {statistics.median(BT_time):.4f} secondes\n")
        #
        # print(f"Moyenne du nombre d'itérations : {statistics.mean(BT_it):.2f}")
        # print(f"Médiane du nombre d'itérations : {statistics.median(BT_it):.2f}\n")
        #
        # print(f"Moyenne du nombre de bactracks : {statistics.mean(BT_back):.2f}")
        # print(f"Mediane du nombre de bactracks : {statistics.median(BT_back):.2f}")
        #
        # print(f"Taux de succès : {100 * sum(BT_succ)/len(BT_succ):.2f}%\n")
        #
        # # Affichage des statistiques pour Monte Carlo Tree Search (MC)
        # print("=== Monte Carlo Tree Search (MC) ===")
        # print(f"Moyenne du temps d'exécution : {statistics.mean(MC_time):.4f} secondes")
        # print(f"Médiane du temps d'exécution : {statistics.median(MC_time):.4f} secondes\n")
        #
        # print(f"Moyenne du nombre d'itérations : {statistics.mean(MC_it):.2f}")
        # print(f"Médiane du nombre d'itérations : {statistics.median(MC_it):.2f}\n")
        #
        # print(f"Moyenne du nombre de bactracks : {statistics.mean(MC_rollouts):.2f}")
        # print(f"Mediane du nombre de bactracks : {statistics.median(MC_rollouts):.2f}")
        #
        # print(f"Taux de succès : {100 * sum(MC_succ)/len(MC_succ):.2f}%\n")
        #
        # # Affichage des statistiques pour Deep Iterative Solver (DIS)
        # print("=== Deep Iterative Solver (DIS) ===")
        # print(f"Moyenne du temps d'exécution : {statistics.mean(DIS_time):.4f} secondes")
        # print(f"Médiane du temps d'exécution : {statistics.median(DIS_time):.4f} secondes\n")
        #
        # print(f"Moyenne du nombre d'itérations : {statistics.mean(DIS_it):.2f}")
        # print(f"Médiane du nombre d'itérations : {statistics.median(DIS_it):.2f}\n")
        #
        # print(f"Taux de succès : {100 * sum(DIS_succ)/len(DIS_succ):.2f}%\n")
        #
        # # Affichage des statistiques pour Alpha Sudoku (AS)
        # print("=== Alpha Sudoku (AS) ===")
        # print(f"Moyenne du temps d'exécution : {statistics.mean(AS_time):.4f} secondes")
        # print(f"Médiane du temps d'exécution : {statistics.median(AS_time):.4f} secondes\n")
        #
        # print(f"Moyenne du nombre d'itérations : {statistics.mean(AS_it):.2f}")
        # print(f"Médiane du nombre d'itérations : {statistics.median(AS_it):.2f}\n")
        #
        # print(f"Taux de succès : {100 * sum(AS_succ)/len(AS_succ):.2f}%\n")
        #
        # # Affichage des statistiques pour SAT Sudoku (SAT)
        # print("=== SAT Sudoku (SAT) ===")
        # print(f"Moyenne du temps d'exécution : {statistics.mean(SAT_time):.4f} secondes")
        # print(f"Médiane du temps d'exécution : {statistics.median(SAT_time):.4f} secondes\n")
        #
        # print(f"Moyenne du nombre d'itérations : {statistics.mean(SAT_it):.2f}")
        # print(f"Médiane du nombre d'itérations : {statistics.median(SAT_it):.2f}\n")
        #
        # print(f"Taux de succès : {100 * sum(SAT_succ)/len(SAT_succ):.2f}%\n")
        #
        # Affichage des statistiques pour SAT Sudoku (SAT)
        print("=== RL Sudoku (RL) ===")
        print(f"Moyenne du temps d'exécution : {statistics.mean(RL_time):.4f} secondes")
        print(f"Médiane du temps d'exécution : {statistics.median(RL_time):.4f} secondes\n")

        print(f"Moyenne du nombre d'itérations : {statistics.mean(RL_it):.2f}")
        print(f"Médiane du nombre d'itérations : {statistics.median(RL_it):.2f}\n")

        print(f"Taux de succès : {100 * sum(RL_succ)/len(RL_succ):.2f}%\n")
        #
        # moyennes = [
        #     statistics.mean(BT_time),
        #     statistics.mean(MC_time),
        #     statistics.mean(DIS_time),
        #     statistics.mean(AS_time),
        #     statistics.mean(SAT_time)
        # ]
        #
        # medians = [
        #     statistics.median(BT_time),
        #     statistics.median(MC_time),
        #     statistics.median(DIS_time),
        #     statistics.median(AS_time),
        #     statistics.median(SAT_time)
        # ]
        #
        # methods = ['Backtrack', 'MCTS', 'DeepIterative', 'AlphaSudoku', 'SAT', 'RL']
        #
        # # Plot
        # x = np.arange(len(methods))  # Position pour chaque groupe
        # width = 0.35
        #
        # fig, ax = plt.subplots()
        # rects1 = ax.bar(x - width / 2, moyennes, width, label='Moyenne')
        # rects2 = ax.bar(x + width / 2, medians, width, label='Médiane')
        #
        # ax.set_ylabel('Temps de résolution (secondes)')
        # ax.set_title('Temps moyen et médian de résolution par méthode')
        # ax.set_xticks(x)
        # ax.set_xticklabels(methods)
        # ax.legend()
        #
        # # Afficher les valeurs au-dessus des barres
        # def autolabel(rects):
        #     for rect in rects:
        #         height = rect.get_height()
        #         ax.annotate(f'{height:.2f}',
        #                     xy=(rect.get_x() + rect.get_width() / 2, height),
        #                     xytext=(0, 3),
        #                     textcoords="offset points",
        #                     ha='center', va='bottom')
        #
        #
        # autolabel(rects1)
        # autolabel(rects2)
        #
        # fig.tight_layout()
        #
        # plt.show()

        with open(f'preprocessed_resultats_{difficulte}.txt', 'w') as f:
            f.write(f"=== Résultats complets pour difficulté : {difficulte} ===\n\n")
        #
        #     f.write(f"Nombre de cases remplies par le prétraitement :\n{nb_preprocessed_cells}\n\n")
        #
        #     f.write("=== Backtracking (BT) ===\n")
        #     f.write(f"Temps : {BT_time}\n")
        #     f.write(f"Iterations : {BT_it}\n")
        #     f.write(f"Backtracks : {BT_back}\n")
        #     f.write(f"Succès : {BT_succ}\n\n")
        #
        #     f.write("=== Monte Carlo Tree Search (MC) ===\n")
        #     f.write(f"Temps : {MC_time}\n")
        #     f.write(f"Iterations : {MC_it}\n")
        #     f.write(f"Rollouts : {MC_rollouts}\n")
        #     f.write(f"Succès : {MC_succ}\n\n")
        #
        #     f.write("=== Deep Iterative Solver (DIS) ===\n")
        #     f.write(f"Temps : {DIS_time}\n")
        #     f.write(f"Iterations : {DIS_it}\n")
        #     f.write(f"Succès : {DIS_succ}\n\n")
        #
        #     f.write("=== Alpha Sudoku (AS) ===\n")
        #     f.write(f"Temps : {AS_time}\n")
        #     f.write(f"Iterations : {AS_it}\n")
        #     f.write(f"Succès : {AS_succ}\n\n")
        #
        #     f.write("=== SAT Sudoku (SAT) ===\n")
        #     f.write(f"Temps : {SAT_time}\n")
        #     f.write(f"Iterations : {SAT_it}\n")
        #     f.write(f"Succès : {SAT_succ}\n\n")

            f.write("=== Reinforcement Learning (RL) ===\n")
            f.write(f"Temps : {RL_time}\n")
            f.write(f"Iterations : {RL_it}\n")
            f.write(f"Succès : {RL_succ}\n\n")
