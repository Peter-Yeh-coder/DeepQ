import numpy as np
import random
import util
from util import manhattanDistance
import time
import sys

# Pacman game
from pacman import Directions
from game import Agent
import game

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad,Nadam
from tensorflow.keras.losses import Huber

params = {
    # Model backups
    'save_file': None,

    # Training parameters
    'train_start': 1000000,  # Episodes before training starts
    'batch_size': 32,  # Replay memory batch size
    'mem_size': 100,  # Replay memory size
    'gamma': 0.8,  # Discount rate (gamma value)

    # Epsilon value (epsilon-greedy)
    'eps': 0.05,  # Epsilon start value
    'eps_final': 0.01,  # Epsilon end value
    'eps_step': 1000,  # Epsilon steps between start and end (linear)

    'target_update_interval': 100  # Adjust the interval as needed

}


class PacmanDQN(game.Agent):
    def __init__(self, args):
        print("Initialize DQN Agent")

        # Load parameters from user-given arguments
        self.params = params  # Corrected from 'self.params = params'
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']

        # Create Q-network and target Q-network
        self.qnet = self._build_q_network()
        self.target_qnet = self._build_q_network()

        # Create Q-network
        self.qnet = self._build_q_network()

        # Additional code for tracking Q and cost
        self.Q_global = []
        self.cost_disp = 0

        # Stats
        self.cnt = 0
        self.local_cnt = 0
        self.total_training_time = 0
        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())

    def _build_q_network(self):
        model = Sequential()

        # Convolutional layers
        model.add(Conv2D(8, (2, 2), input_shape=(self.params['width'], self.params['height'], 5), activation='relu'))
        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(Flatten())

        # Additional Dense layer
        model.add(Dense(256, activation='relu'))

        # Dense layers
        fc_layer_params = (100, 50)
        for num_units in fc_layer_params:
            model.add(Dense(num_units, activation='relu'))

        # Output layer
        num_actions = 5  # Change this to the actual number of actions
        model.add(Dense(num_actions, activation=None,
                        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
                        bias_initializer=tf.keras.initializers.Constant(-0.2)))

        # Compiler details
        model.compile(optimizer=Adam(learning_rate=0.00001),
                      loss=tf.keras.losses.mean_squared_error,
                      metrics=['mse'])
        return model
    def getMove(self, state):
        # Explore with probability epsilon
        if np.random.rand() < self.params['eps']:
            move = self.get_direction(np.random.randint(0, 5))  # Random action
        else:
            # Exploit: Choose action with the highest estimated Q-value
            self.Q_pred = self.qnet.predict(np.reshape(self.current_state,
                                                       (1, self.params['width'], self.params['height'], 4)))[0]

            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))
            legal_moves = [self.get_direction(value[0]) for value in a_winner]

            # Filter out None values
            legal_moves = [move for move in legal_moves if move is not None]

            if legal_moves:
                move = legal_moves[np.random.randint(0, len(legal_moves))]
            else:
                # If no legal moves are available, choose a random action
                move = self.get_direction(np.random.randint(0, 5))

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    def get_value(self, direction):
        legal_moves = self.get_legal_directions()
        if direction == Directions.NORTH and Directions.NORTH in legal_moves:
            return 0.
        elif direction == Directions.EAST and Directions.EAST in legal_moves:
            return 1.
        elif direction == Directions.WEST and Directions.WEST in legal_moves:
            return 2.
        elif direction == Directions.SOUTH and Directions.SOUTH in legal_moves:
            return 3.
        else:
            return None  # Reclassify illegal moves as None

    def get_direction(self, value):
        legal_moves = self.get_legal_directions()
        if value == 0. and Directions.NORTH in legal_moves:
            return Directions.NORTH
        elif value == 1. and Directions.EAST in legal_moves:
            return Directions.EAST
        elif value == 2. and Directions.WEST in legal_moves:
            return Directions.WEST
        elif value == 3. and Directions.SOUTH in legal_moves:
            return Directions.SOUTH
        else:
            return None  # Reclassify illegal moves as None

    def get_legal_directions(self):
        # Get legal moves based on the current state
        return [Directions.NORTH, Directions.EAST, Directions.WEST, Directions.SOUTH]

    def observation_step(self, state):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)

            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            # Check if a legal action is taken and it's not "STOP"
            if reward < 0 and state.getLegalActions(0) and self.last_action != self.get_value(Directions.STOP):
                # Assuming state is an instance of GameState

                # Choose a legal action
                legal_actions = state.getLegalActions(0)
                best_action = random.choice(legal_actions)

                # Reward for taking a legal action, prioritizing moving away from ghosts
                self.last_reward = 1. if self.last_action == best_action else -1.
            else:
                self.last_reward = -50.  # Punish time
                self.won = False  # Pacman lost

            if self.terminal:
                self.last_reward = 100. if self.won else -100.  # Additional reward/penalty for episode termination
            self.ep_rew += self.last_reward

            # Store last experience into memory
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if params['save_file']:
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    self.qnet.save_ckpt(
                        'saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt) / float(self.params['eps_step']))

    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        # Record the start time
        start_time = time.time()

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Next
        self.ep_rew += self.last_reward

        # Get the true score from the state
        true_score = state.getScore()

        # Record the end time
        end_time = time.time()

        # Calculate the time taken for this episode
        time_taken = end_time - start_time

        # Update the total training time
        self.total_training_time += time_taken

        # Print true score, win status, and time taken
        print("True Score: {:.2f} | Won: {} | Time Taken: {:.2f} seconds | Total Training Time: {:.2f} seconds".format(
            true_score, self.won, time_taken, self.total_training_time))

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_states = [] # States (s)
            batch_rewards = [] # Rewards (r)
            batch_actions = [] # Actions (a)
            batch_next = [] # Next states (s')
            batch_terminal = [] # Terminal state (t)

            for i in batch:
                batch_states.append(i[0])
                batch_rewards.append(i[1])
                batch_actions.append(i[2])
                batch_next.append(i[3])
                batch_terminal.append(i[4])
            batch_states = np.array(batch_states)
            batch_rewards = np.array(batch_rewards)
            batch_actions = self.get_onehot(np.array(batch_actions))
            batch_next = np.array(batch_next)
            batch_terminal = np.array(batch_terminal)

            # self.qnet computes the temporal disturbance, training also implicitly calculates the TD error
            self.cnt, self.cost_disp = self.qnet.train(batch_states,
                                                       batch_actions,
                                                       batch_terminal,
                                                       batch_next,
                                                       batch_rewards)

            # Update the target network periodically
            if self.local_cnt % self.params['target_update_interval'] == 0:
                self.target_qnet.set_weights(self.qnet.get_weights())

    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        """ Return Pacman, food, ghosts, scared ghosts, and power pellets matrices separately """
        width, height = self.params['width'], self.params['height']

        pacman_matrix = np.zeros((height, width), dtype=np.int8)
        food_matrix = np.zeros((height, width), dtype=np.int8)
        ghosts_matrix = np.zeros((height, width), dtype=np.int8)
        scared_ghosts_matrix = np.zeros((height, width), dtype=np.int8)
        power_pellets_matrix = np.zeros((height, width), dtype=np.int8)

        for agentState in state.data.agentStates:
            pos = agentState.configuration.getPosition()

            if agentState.isPacman:
                pacman_matrix[-1 - int(pos[1])][int(pos[0])] = 1
            else:
                if agentState.scaredTimer > 0:
                    scared_ghosts_matrix[-1 - int(pos[1])][int(pos[0])] = 1
                else:
                    ghosts_matrix[-1 - int(pos[1])][int(pos[0])] = 1

        for i in state.data.layout.food.asList():
            food_matrix[-1 - i[1], i[0]] = 1

        for i in state.data.layout.capsules:
            power_pellets_matrix[-1 - i[1], i[0]] = 1

        return np.stack([pacman_matrix, food_matrix, ghosts_matrix, scared_ghosts_matrix, power_pellets_matrix],
                        axis=-1)

    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = False
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP

        return move
