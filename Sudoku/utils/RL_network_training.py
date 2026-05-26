from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

class RLSolverTrainer:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(81,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(729, activation='softmax'))  # 9x9x9 actions
        model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy')
        return model

    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
        self.model.save("Sudoku/Reinforcement_Learning_Sudoku/RL_Sudoku_model.h5")
