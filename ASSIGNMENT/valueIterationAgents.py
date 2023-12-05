# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Initialize values for all states to zero
        for state in self.mdp.getStates():
            self.values[state] = 0.0

        # Perform a specified number of iterations
        for i in range(self.iterations):
            # Copy the current values to compute updates
            next_values = self.values.copy()

            # Update values for each state based on the current estimates
            for state in self.mdp.getStates():
                # Counter to store values for each possible action in the current state
                state_values = util.Counter()

                # Calculate values for all possible actions in the current state
                for action in self.mdp.getPossibleActions(state):
                    state_values[action] = self.getQValue(state, action)

                # Update the value of the current state using the maximum Q-value
                next_values[state] = state_values[state_values.argMax()]

            # Update the values for all states after processing all states
            self.values = next_values.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Get the transition probabilities for the given state-action pair, initialize value to 0
        trans_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0.0

        # Iterate over each possible transition
        for transition in trans_probs:
            trans_state, prob = transition

            # Update the Q-value using the Bellman equation
            QValue += prob * (
                    self.mdp.getReward(state, action, trans_state) + self.discount * self.getValue(trans_state))

        return QValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Check if the given state is terminal
        if self.mdp.isTerminal(state):
            return None
        else:
            # Initialize a counter to store values for each action
            QValues = util.Counter()

            # Get the list of possible actions for the given state
            actions = self.mdp.getPossibleActions(state)

            # Compute values for each action using the current value function
            for action in actions:
                QValues[action] = self.computeQValueFromValues(state, action)

            # Return the action with the highest Q-value (breaking ties arbitrarily)
            return QValues.argMax()
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Initialize values for all states to zero
        for state in self.mdp.getStates():
            self.values[state] = 0.0

        # Perform a specified number of iterations
        for iteration in range(self.iterations):
            # Select the next state in a cyclic manner
            state = self.mdp.getStates()[iteration % len(self.mdp.getStates())]

            # If the selected state is terminal, skip this iteration
            if self.mdp.isTerminal(state):
                continue

            # Compute the Q-value for each possible action in the selected state
            q_values = util.Counter()
            for action in self.mdp.getPossibleActions(state):
                q_values[action] = self.getQValue(state, action)

            # Update the value of the selected state using the maximum Q-value
            self.values[state] = q_values[q_values.argMax()]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for predecessor, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[predecessor].add(state)

        # Initialize an empty priority queue and a set to keep track of pushed states
        priority_queue = util.PriorityQueue()

        # For each non-terminal state s, push it into the priority queue
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                Qmax = max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))
                diff = abs(self.values[state] - Qmax)
                priority_queue.update(state, -diff)

        # Perform a specified number of iterations
        for iteration in range(self.iterations):
            # If the priority queue is empty, then terminate
            if priority_queue.isEmpty():
                break

            # Pop a state s off the priority queue
            state = priority_queue.pop()

            # Update the value of s (if it is not a terminal state)
            if not self.mdp.isTerminal(state):
                self.values[state] = max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))

            # For each predecessor p of s, push it into the priority queue if necessary
            for predecessor in predecessors[state]:
                if not self.mdp.isTerminal(predecessor):
                    diff = abs(self.values[predecessor] - max(self.getQValue(predecessor, action) for action in self.mdp.getPossibleActions(predecessor)))
                    if diff > self.theta:
                        priority_queue.update(predecessor, -diff)
