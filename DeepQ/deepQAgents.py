import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from game import Directions

class PacmanDQNAgent:
    def __init__(self, numTraining=0, epsilon=0.05, alpha=0.2, gamma=0.8):
        self.numTraining = numTraining
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lastState = None
        self.lastAction = None
        self.lastScore = 0

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        # Modify the input_shape according to your state representation
        # Assuming the smallGrid is 7x7
        grid_size = 7

        # Features for each cell: [Wall, Pacman, Food, Ghost]
        num_features = 4

        # Input shape for the neural network
        state_shape = (grid_size * grid_size * num_features,)

        # Example usage in the build_model function
        model.add(Dense(64, input_shape=state_shape, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(Directions.ALL), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def getQValues(self, state):
        # Modify the state representation according to your needs
        q_values = self.model.predict(np.array([state]))
        return q_values[0]

    def getAction(self, state):
        legalActions = state.getLegalPacmanActions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(legalActions)
        q_values = self.getQValues(state)
        return legalActions[np.argmax(q_values[legalActions])]

    def update(self, nextState, reward):
        if self.lastState is not None:
            target = reward + self.gamma * np.max(self.getQValues(nextState))
            target_f = self.getQValues(self.lastState)
            target_f[self.lastAction] = target
            self.model.fit(np.array([self.lastState]), np.array([target_f]), epochs=1, verbose=0)

        if self.numTraining > 0:
            self.numTraining -= 1
            if self.numTraining == 0:
                self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
