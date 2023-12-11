# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import sys
from game import *
from featureExtractors import *
from learningAgents import ReinforcementAgent
import util
import random
import time
import layout

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.QValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QValues[state, action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Acquires the values for each legal action in a given state
        values = [self.getQValue(state, action) for action in self.getLegalActions(state)]

        # Returns the max value or 0
        if values:
            return max(values)
        else:
            return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Acquires the legal actions of the state and value of the state
        legal_actions = self.getLegalActions(state)
        value = self.getValue(state)

        # A for loop comparing the value with the value of the state and action. If equal, returns action
        for action in legal_actions:
            if value == self.getQValue(state, action):
                return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # Employs random choice based on probability epsilon
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Generates the new value
        newQValue = (1 - self.alpha) * self.getQValue(state, action)
        newQValue += self.alpha * (reward + (self.discount * self.getValue(nextState)))
        self.QValues[state, action] = newQValue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Acquire the feature vector and sets initial value to 0
        features = self.featExtractor.getFeatures(state,action)
        QValue = 0.0

        # Performs the dot product to acquire value
        for feature in features:
            QValue += self.weights[feature] * features[feature]

        # Returns the value
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Sets initial value as 0, calculates the difference, and extracts the features
        QValue = 0
        difference = reward + (self.discount * self.getValue(nextState) - self.getQValue(state, action))
        features = self.featExtractor.getFeatures(state, action)

        # Updates the weights based on transition
        for feature in features:
            self.weights[feature] += self.alpha * features[feature] * difference

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # Didn't need to alter?
            pass

# python pacman.py -p ApproximateQAgent -x 100 -n 110 -l smallGrid > approxQ100scores.text
# python pacman.py -p ApproximateQAgent -x 500 -n 510 -l smallGrid > approxQ500scores.text
# python pacman.py -p ApproximateQAgent -x 1000 -n 1010 -l smallGrid > approxQ1000scores.text
# python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid > approxQ2000scores.text

# import tensorflow as tf
# import numpy as np

# class SimpleNeuralNetwork(tf.Module):
#     def __init__(self, num_features, learning_rate=0.001, regularization_strength=0.01):
#         self.weights = tf.Variable(tf.random.normal(shape=(num_features, 1)))
#         self.bias = tf.Variable(tf.zeros(shape=(1,)))
#         self.learning_rate = learning_rate
#         self.regularization_strength = regularization_strength

#     def forward(self, inputs):
#         return tf.matmul(inputs, self.weights) + self.bias

#     def predict(self, features):
#         return self.forward(features)

#     def calculate_loss(self, features, targets):
#         predictions = self.predict(features)
#         loss = tf.reduce_mean(tf.square(targets - predictions))

#         # L2 regularization term
#         regularization_term = 0.5 * self.regularization_strength * tf.reduce_sum(tf.square(self.weights))
#         loss += regularization_term

#         return loss

#     def update_weights(self, features, targets):
#         with tf.GradientTape() as tape:
#             loss = self.calculate_loss(features, targets)

#         gradients = tape.gradient(loss, [self.weights, self.bias])
#         self.weights.assign_sub(self.learning_rate * gradients[0])
#         self.bias.assign_sub(self.learning_rate * gradients[1])

# Example usage:

# Assuming num_features is the number of features mentioned in the text
# num_features = 5  # Replace with the actual number of features

# Creating the neural network
# nn = SimpleNeuralNetwork(num_features)

# Example features from the text
# features = np.array([[distance_to_food, num_ghosts, bias, food_1_step_away, no_ghost_1_step_away] for _ in range(50)])
# targets = np.array([[q_value] for q_value in your_q_values_list])  # Replace with actual Q-values

# Training the neural network
# for epoch in range(num_epochs):
#     nn.update_weights(features, targets)

# Making predictions
# new_features = np.array([[new_distance_to_food, new_num_ghosts, new_bias, new_food_1_step_away, new_no_ghost_1_step_away]])
# predicted_q_value = nn.predict(new_features)
# print("Predicted Q-value:", predicted_q_value.numpy())
