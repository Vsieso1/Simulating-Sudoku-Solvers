import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import random
from collections import deque
import matplotlib.pyplot as plt

# === CHARGEMENT DES DONNÉES ===

def load_sudoku_data(filename="sudoku-3m.csv"):
    df = pd.read_csv(filename, header=None, skiprows=1)
    inputs = df.iloc[:, 1].apply(lambda x: [int(c) for c in x.strip().replace('.', '0')])
    solutions = df.iloc[:, 2].apply(lambda x: [int(c) for c in x.strip().replace('.', '0')])

    grids = np.array(inputs.tolist()).reshape(-1, 9, 9)
    solution_grids = np.array(solutions.tolist()).reshape(-1, 9, 9)

    return grids, solution_grids

train_grids, solution_grids = load_sudoku_data("data.csv")

# === ENVIRONNEMENT SUDOKU 9x9 ===
class Sudoku9x9Env:
    def __init__(self):
        self.grid = np.zeros((9, 9), dtype=int)
        self.done = False

    def reset(self, grid=None):
        self.grid = grid.copy() if grid is not None else np.zeros((9, 9), dtype=int)
        self.done = False
        return self.grid.copy()

    def is_valid(self, row, col, val):
        if val in self.grid[row]: return False
        if val in self.grid[:, col]: return False
        block_row, block_col = 3 * (row // 3), 3 * (col // 3)
        if val in self.grid[block_row:block_row+3, block_col:block_col+3]: return False
        return True

    def is_grid_valid(self):
        for i in range(9):
            if len(set(self.grid[i])) != 9 or len(set(self.grid[:, i])) != 9:
                return False
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                if len(set(self.grid[row:row+3, col:col+3].flatten())) != 9:
                    return False
        return True

    def step(self, action):
        row, col, val = self.decode_action(action)
        reward = 0

        if self.grid[row, col] != 0:
            reward = -5
        elif not self.is_valid(row, col, val):
            self.grid[row, col] = val
            reward = -10
        else:
            self.grid[row, col] = val
            reward = 1

            if 0 not in self.grid[row] or 0 not in self.grid[:, col]:
                reward += 10
            block_row, block_col = 3 * (row // 3), 3 * (col // 3)
            if 0 not in self.grid[block_row:block_row+3, block_col:block_col+3]:
                reward += 10

        if np.all(self.grid != 0):
            if self.is_grid_valid():
                self.done = True
                reward += 100

        return self.grid.copy(), reward, self.done, {}

    def get_valid_actions(self):
        actions = []
        for row in range(9):
            for col in range(9):
                if self.grid[row, col] == 0:
                    for val in range(1, 10):
                        if self.is_valid(row, col, val):
                            actions.append(self.encode_action(row, col, val))
        return actions

    @staticmethod
    def encode_action(row, col, val):
        return row * 81 + col * 9 + (val - 1)

    @staticmethod
    def decode_action(index):
        row = index // 81
        col = (index % 81) // 9
        val = (index % 9) + 1
        return row, col, val

# === DEEP Q NETWORK ===
class DeepQNetwork:
    def __init__(self, n_actions, n_features,
                 learning_rate=0.01, reward_decay=0.9,
                 epsilon_max=0.9, replace_target_iter=300,
                 memory_size=500, batch_size=32):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99985

        self.learn_step_counter = 0
        self.memory = deque(maxlen=self.memory_size)

        self.eval_net = self._build_net(name='eval_net')
        self.target_net = self._build_net(name='target_net')
        self.update_target_network()

        self.optimizer = optimizers.Adam(learning_rate=self.lr)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def _build_net(self, name):
        model = tf.keras.Sequential(name=name)
        model.add(layers.Input(shape=(self.n_features,)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.n_actions))
        return model

    def update_target_network(self):
        self.target_net.set_weights(self.eval_net.get_weights())

    def store_transition(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))

    def choose_action(self, observation, valid_actions):
        observation = np.array(observation).reshape(-1, self.n_features)

        if np.random.rand() < self.epsilon:
            q_values = self.eval_net.predict(observation, verbose=0)[0]
            masked_q = np.full_like(q_values, -np.inf)
            masked_q[valid_actions] = q_values[valid_actions]
            action = np.argmax(masked_q)
        else:
            action = random.choice(valid_actions)

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.update_target_network()

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        q_next = self.target_net.predict(next_states, verbose=0)
        q_target = rewards + self.gamma * np.max(q_next, axis=1)

        with tf.GradientTape() as tape:
            q_eval = self.eval_net(states)
            q_eval_action = tf.reduce_sum(tf.one_hot(actions, self.n_actions) * q_eval, axis=1)
            loss = self.loss_function(q_target, q_eval_action)

        grads = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.eval_net.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        self.learn_step_counter += 1

    @staticmethod
    def load_best_model(filepath, n_actions=729, n_features=81):
        dqn = DeepQNetwork(n_actions=n_actions, n_features=n_features)
        dqn.eval_net = tf.keras.models.load_model(filepath)
        dqn.update_target_network()
        return dqn

# === ENTRAÎNEMENT ===

# env = Sudoku9x9Env()
# dqn = DeepQNetwork(n_actions=729, n_features=81)
# rewards_per_episode = []
# best_reward = -float('inf')
#
# for episode in range(25000):
#     idx = np.random.randint(len(train_grids))
#     state = env.reset(train_grids[idx]).flatten()
#     done = False
#     total_reward = 0
#
#     while not done:
#         valid_actions = env.get_valid_actions()
#         if not valid_actions:
#             break
#         action = dqn.choose_action(state, valid_actions)
#         next_state, reward, done, _ = env.step(action)
#         next_state = next_state.flatten()
#
#         dqn.store_transition(state, action, reward, next_state)
#         dqn.learn()
#
#         total_reward += reward
#         state = next_state
#
#     rewards_per_episode.append(total_reward)
#
#     if total_reward > best_reward:
#         best_reward = total_reward
#         dqn.eval_net.save('best_model.h5')  # Format HDF5
#         print(f"Nouvelle meilleure récompense : {best_reward}")
#
#     if episode % 1000 == 0:
#         print(f"Épisode {episode}, Total Reward : {total_reward}, Epsilon : {dqn.epsilon:.3f}")

# === TEST DU MODÈLE ===

env = Sudoku9x9Env()
dqn = DeepQNetwork.load_best_model("RL_Sudoku_model.h5")
test_grids = pd.read_csv("sudoku_train_9x9.csv", header=None)
test_solutions = pd.read_csv("sudoku_solution_9x9.csv", header=None)

def test_model(env, dqn, test_grids, test_solutions):
    correct = 0
    total = 1000

    for idx in range(total):
        state = env.reset(test_grids.iloc[idx].values.reshape(9, 9)).flatten()
        done = False
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = dqn.choose_action(state, valid_actions)
            next_state, _, done, _ = env.step(action)
            state = next_state.flatten()

        if np.array_equal(env.grid, test_solutions.iloc[idx].values.reshape(9, 9)):
            print(f"Grille {idx+1} \n", env.grid)
            correct += 1
        else:
            print(f"Grille {idx+1} non résolue")

    accuracy = correct / total
    print(f"Précision du modèle sur {total} grilles de test : {accuracy * 100:.2f}%")

test_model(env, dqn, test_grids, test_solutions)