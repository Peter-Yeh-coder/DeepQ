from game import *
from featureExtractors import *
from learningAgents import ReinforcementAgent
import util
import random
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense

class ConvolutionalDeepQAgent(ReinforcementAgent):
    """
    Convolutional Deep Q-Learning Agent using Keras

    Functions you should fill in:
        - getQValue
        - update

    Instance variables you have access to:
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

    Functions you should use:
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0, grid_size=(100, 100), channels=5, **args):
        """
        Initialize the Convolutional Deep Q-learning agent.

        :param epsilon: Exploration rate
        :param gamma: Discount factor
        :param alpha: Learning rate
        :param num_training: Number of training episodes
        :param grid_size: Tuple representing the grid size (height, width)
        :param channels: Number of channels for Pac-Man, active ghosts, scared ghosts, food, and power pellets
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # This is always Pacman

        # Set grid size and channels
        self.grid_size = grid_size
        self.channels = channels

        # Build the convolutional deep Q-network model
        self.model = self._build_model()

        # Initialize previous state and action
        self.prev_state = None
        self.prev_action = None

    def _build_model(self):
        """
        Build the convolutional deep Q-network model.

        :return: Keras Sequential model
        """
        model = Sequential()
        # Add convolutional layer for Pac-Man
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.grid_size[0], self.grid_size[1],
                                                                                 self.channels)))
        # Add convolutional layer for active ghosts
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        # Add convolutional layer for scared ghosts
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        # Add convolutional layer for food
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        # Add convolutional layer for power pellets
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        # Flatten the output for fully connected layers
        model.add(Flatten())
        # Add fully connected layers
        model.add(Dense(128, activation='relu'))
        model.add(Dense(5, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def getQValue(self, state, action):
        """
        Returns Q(state, action) using the convolutional deep Q-network.

        :param state: Current state
        :param action: Chosen action
        :return: Q-value for the given state-action pair
        """
        # Convert state to a format suitable for prediction
        state = self._format_state(state)
        # Predict Q-values for the given state
        q_values = self.model.predict(state)
        # Return the Q-value for the chosen action
        return q_values[0][action]

    def update(self, state, action, next_state, reward):
        """
        Update the convolutional deep Q-network based on the transition.

        :param state: Current state
        :param action: Chosen action
        :param next_state: Next state after taking the action
        :param reward: Reward obtained
        """
        # Convert states to a format suitable for prediction
        state = self._format_state(state)
        next_state = self._format_state(next_state)

        # Calculate target Q-value using Bellman equation
        target = reward + self.discount * max(self.model.predict(next_state)[0])

        # Predict Q-values for the current state
        q_values = self.model.predict(state)

        # Update the Q-value for the chosen action
        q_values[0][action] = (1 - self.alpha) * q_values[0][action] + self.alpha * target

        # Train the model with updated Q-values
        self.model.fit(state, q_values, epochs=1, verbose=0)

    def _format_state(self, state):
        """
        Format the state for prediction by the convolutional deep Q-network.

        :param state: Current state
        :return: Formatted state for prediction
        """
        # Convert state to a NumPy array and reshape it for the model
        return np.reshape(np.array(state), [1, self.grid_size[0], self.grid_size[1], self.channels])

    def getAction(self, state):
        """
        Compute the action to take in the current state. With probability self.epsilon,
        take a random action, and take the best policy action otherwise.

        :param state: Current state
        :return: Chosen action
        """
        legal_actions = self.getLegalActions(state)

        # Epsilon-greedy strategy
        if np.random.rand() <= self.epsilon:
            action = random.choice(legal_actions)
        else:
            # Choose the action with the highest Q-value
            action = np.argmax(self.model.predict(self._format_state(state))[0])

        # Store the current state and action for the next update
        self.prev_state = state
        self.prev_action = action

        return action

    def final(self, state):
        """
        Called at the end of each game.

        :param state: Current state
        """
        # Call the super-class final method
        ReinforcementAgent.final(self, state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to save your model here or perform other finalization steps
            pass


# Function to run the training for 1000 iterations
def train_convolutional_deep_q_agent(grid_size):
    # Create an instance of the Convolutional Deep Q-learning agent with the specified grid size
    conv_deep_q_agent = ConvolutionalDeepQAgent(grid_size=grid_size)

    # Number of training iterations
    num_iterations = 1000

    # Lists to store time and scores after each iteration
    iteration_times = []
    iteration_scores = []

    # Run training for 1000 iterations
    for iteration in range(1, num_iterations + 1):
        start_time = time.time()  # Record the start time

        # Run a game
        game = run_games(1, conv_deep_q_agent)[0]
        score = game.state.getScore()

        # Record the end time
        end_time = time.time()

        # Calculate the time taken for the iteration
        iteration_time = end_time - start_time

        # Print and store the results
        print(f"Iteration {iteration}: Time = {iteration_time:.2f}s, Score = {score}")
        iteration_times.append(iteration_time)
        iteration_scores.append(score)

    # Print average time and score
    avg_time = sum(iteration_times) / num_iterations
    avg_score = sum(iteration_scores) / num_iterations
    print(f"\nAverage Time = {avg_time:.2f}s, Average Score = {avg_score}")


# Run the training with a variable grid size, for example, (100, 100)
train_convolutional_deep_q_agent(grid_size=(100, 100))
