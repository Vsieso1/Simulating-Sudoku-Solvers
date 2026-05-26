"""
Entraînement DQN pour Sudoku 9x9
Données : sudoku_train_9x9.csv / sudoku_solution_9x9.csv  (ou data.csv)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import random
from collections import deque
import os

# ──────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────

def load_data(train_path="sudoku_train_9x9.csv", solution_path="sudoku_solution_9x9.csv"):
    """
    Charge sudoku_train_9x9.csv + sudoku_solution_9x9.csv (format : 81 colonnes, pas d'en-tête).
    Retourne deux tableaux (N, 9, 9).
    """
    train_df    = pd.read_csv(train_path,    header=None)
    solution_df = pd.read_csv(solution_path, header=None)

    grids     = train_df.values.reshape(-1, 9, 9).astype(int)
    solutions = solution_df.values.reshape(-1, 9, 9).astype(int)
    print(f"✅  {len(grids)} puzzles chargés depuis {train_path}")
    return grids, solutions


def load_data_single(filepath="data.csv"):
    """
    Charge data.csv (format Kaggle : colonne 'puzzle' + 'solution' en chaînes de 81 chars).
    Retourne deux tableaux (N, 9, 9).
    """
    df = pd.read_csv(filepath)

    # Détection automatique des colonnes
    puzzle_col   = [c for c in df.columns if "puzzle"   in c.lower() or "quiz"    in c.lower()][0]
    solution_col = [c for c in df.columns if "solution" in c.lower() or "answer"  in c.lower()][0]

    def parse(series):
        return np.array(
            series.apply(lambda x: [int(c) if c != '.' else 0 for c in str(x).strip()]).tolist()
        ).reshape(-1, 9, 9)

    grids     = parse(df[puzzle_col])
    solutions = parse(df[solution_col])
    print(f"✅  {len(grids)} puzzles chargés depuis {filepath}")
    return grids, solutions


# ──────────────────────────────────────────────
# 2. ENVIRONNEMENT SUDOKU
# ──────────────────────────────────────────────

class SudokuEnv:
    def reset(self, grid):
        self.grid     = grid.copy()
        self.original = grid.copy()   # cellules fixes (≠ 0) ne peuvent pas être écrasées
        return self.grid.flatten()

    # ---- validité ----
    def _is_valid(self, row, col, val):
        if val in self.grid[row]:                                        return False
        if val in self.grid[:, col]:                                     return False
        br, bc = 3 * (row // 3), 3 * (col // 3)
        if val in self.grid[br:br+3, bc:bc+3]:                          return False
        return True

    def _is_solved(self):
        for i in range(9):
            if len(set(self.grid[i]))    != 9: return False
            if len(set(self.grid[:, i])) != 9: return False
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                if len(set(self.grid[r:r+3, c:c+3].flatten())) != 9: return False
        return True

    # ---- actions valides ----
    def get_valid_actions(self):
        actions = []
        for r in range(9):
            for c in range(9):
                if self.grid[r, c] == 0:
                    for v in range(1, 10):
                        if self._is_valid(r, c, v):
                            actions.append(r * 81 + c * 9 + (v - 1))
        return actions

    # ---- step ----
    def step(self, action):
        row = action // 81
        col = (action % 81) // 9
        val = (action % 9) + 1

        if self.grid[row, col] != 0:           # case déjà remplie
            return self.grid.flatten(), -5, False

        if not self._is_valid(row, col, val):  # coup invalide
            self.grid[row, col] = val
            return self.grid.flatten(), -10, False

        # coup valide
        self.grid[row, col] = val
        reward = 1

        # bonus ligne / colonne / bloc complétés
        if 0 not in self.grid[row]:                                         reward += 10
        if 0 not in self.grid[:, col]:                                      reward += 10
        br, bc = 3 * (row // 3), 3 * (col // 3)
        if 0 not in self.grid[br:br+3, bc:bc+3]:                           reward += 10

        done = False
        if np.all(self.grid != 0) and self._is_solved():
            reward += 100
            done = True

        return self.grid.flatten(), reward, done

    # ---- encodage / décodage (utilitaires) ----
    @staticmethod
    def decode(action):
        return action // 81, (action % 81) // 9, (action % 9) + 1


# ──────────────────────────────────────────────
# 3. RÉSEAU DQN
# ──────────────────────────────────────────────

def build_model(n_features=81, n_actions=729):
    model = tf.keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_actions),
    ])
    return model


class DQNAgent:
    def __init__(self,
                 n_actions=729,
                 n_features=81,
                 lr=1e-3,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.9999,
                 memory_size=10_000,
                 batch_size=64,
                 target_update_freq=500):

        self.n_actions  = n_actions
        self.n_features = n_features
        self.gamma      = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory      = deque(maxlen=memory_size)
        self.step_count  = 0

        self.eval_net   = build_model(n_features, n_actions)
        self.target_net = build_model(n_features, n_actions)
        self.target_net.set_weights(self.eval_net.get_weights())

        self.optimizer     = optimizers.Adam(lr)
        self.loss_fn       = tf.keras.losses.Huber()

    # ── mémoire ──
    def remember(self, s, a, r, s_next):
        self.memory.append((s, a, r, s_next))

    # ── choix d'action (ε-greedy masqué) ──
    def act(self, state, valid_actions):
        if not valid_actions:
            return None
        if np.random.rand() < self.epsilon:          # exploration
            return random.choice(valid_actions)

        q = self.eval_net(state[np.newaxis], training=False).numpy()[0]
        masked = np.full(self.n_actions, -np.inf)
        masked[valid_actions] = q[valid_actions]
        return int(np.argmax(masked))

    # ── apprentissage ──
    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch      = random.sample(self.memory, self.batch_size)
        s, a, r, s_= zip(*batch)
        s          = np.array(s,   dtype=np.float32)
        s_         = np.array(s_,  dtype=np.float32)
        a          = np.array(a,   dtype=np.int32)
        r          = np.array(r,   dtype=np.float32)

        q_next   = self.target_net(s_, training=False).numpy()
        q_target = r + self.gamma * np.max(q_next, axis=1)

        with tf.GradientTape() as tape:
            q_eval   = self.eval_net(s, training=True)
            one_hot  = tf.one_hot(a, self.n_actions)
            q_taken  = tf.reduce_sum(q_eval * one_hot, axis=1)
            loss     = self.loss_fn(q_target, q_taken)

        grads = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.eval_net.trainable_variables))

        # màj ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # màj réseau cible
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.set_weights(self.eval_net.get_weights())

        return float(loss)


# ──────────────────────────────────────────────
# 4. BOUCLE D'ENTRAÎNEMENT
# ──────────────────────────────────────────────

def train(
    train_path    = "sudoku_train_9x9.csv",
    solution_path = "sudoku_solution_9x9.csv",
    use_single_csv= False,          # True → charge data.csv à la place
    data_csv      = "data.csv",
    n_episodes    = 20_000,
    max_steps     = 150,            # pas max par épisode
    save_path     = "best_model.keras",
    log_every     = 500,
):
    # -- données --
    if use_single_csv:
        grids, solutions = load_data_single(data_csv)
    else:
        grids, solutions = load_data(train_path, solution_path)

    env   = SudokuEnv()
    agent = DQNAgent()

    best_reward   = -np.inf
    reward_history= []

    print(f"\n🚀  Entraînement sur {n_episodes} épisodes — {len(grids)} puzzles disponibles\n")

    for ep in range(1, n_episodes + 1):
        idx   = np.random.randint(len(grids))
        state = env.reset(grids[idx]).astype(np.float32) / 9.0   # normalisation [0,1]
        total_reward = 0
        loss_sum     = 0
        steps        = 0

        for _ in range(max_steps):
            valid = env.get_valid_actions()
            if not valid:
                break

            action = agent.act(state, valid)
            if action is None:
                break

            next_state_raw, reward, done = env.step(action)
            next_state = next_state_raw.astype(np.float32) / 9.0

            agent.remember(state, action, reward, next_state)
            loss_sum += agent.learn()

            state         = next_state
            total_reward += reward
            steps        += 1

            if done:
                break

        reward_history.append(total_reward)

        # sauvegarde du meilleur modèle
        if total_reward > best_reward:
            best_reward = total_reward
            agent.eval_net.save(save_path)

        # log périodique
        if ep % log_every == 0:
            avg = np.mean(reward_history[-log_every:])
            print(f"Ep {ep:6d}/{n_episodes} | "
                  f"Reward moy: {avg:7.1f} | "
                  f"Meilleur: {best_reward:7.1f} | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Steps: {steps}")

    print(f"\n✅  Entraînement terminé. Meilleur modèle sauvegardé → {save_path}")
    return agent


# ──────────────────────────────────────────────
# 5. TEST RAPIDE
# ──────────────────────────────────────────────

def test(model_path="best_model.keras",
         train_path="sudoku_train_9x9.csv",
         solution_path="sudoku_solution_9x9.csv",
         n_test=200):

    grids, solutions = load_data(train_path, solution_path)
    model = tf.keras.models.load_model(model_path)
    env   = SudokuEnv()
    correct = 0

    for i in range(min(n_test, len(grids))):
        state = env.reset(grids[i]).astype(np.float32) / 9.0
        done  = False

        for _ in range(200):
            valid = env.get_valid_actions()
            if not valid:
                break
            q = model(state[np.newaxis], training=False).numpy()[0]
            masked = np.full(729, -np.inf)
            masked[valid] = q[valid]
            action = int(np.argmax(masked))
            state_raw, _, done = env.step(action)
            state = state_raw.astype(np.float32) / 9.0
            if done:
                break

        if np.array_equal(env.grid, solutions[i]):
            correct += 1

    print(f"Précision sur {n_test} grilles : {correct}/{n_test} = {100*correct/n_test:.1f}%")


# ──────────────────────────────────────────────
# 6. POINT D'ENTRÉE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # ── Choisissez votre source de données ──────────────────────────────
    #
    # Option A : sudoku_train_9x9.csv + sudoku_solution_9x9.csv (défaut)
    agent = train(
        train_path    = "sudoku_train_9x9.csv",
        solution_path = "sudoku_solution_9x9.csv",
        use_single_csv= False,
        n_episodes    = 20_000,
        save_path     = "best_model.keras",
    )
    #
    # Option B : data.csv (format Kaggle puzzle/solution en chaîne)
    # agent = train(
    #     use_single_csv = True,
    #     data_csv       = "data.csv",
    #     n_episodes     = 20_000,
    # )
    # ────────────────────────────────────────────────────────────────────

    # Test final
    if os.path.exists("best_model.keras"):
        test("best_model.keras")