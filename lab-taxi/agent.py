import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 0.5
        self.n_step = 0
        self.alpha = 0.1
        self.gamma = 0.8

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.epsilon_greedy(state)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.n_step += 1
        if self.n_step < 1000:
            self.eps = 0.3
        else:
            self.eps = np.max([1 / (self.n_step - 995), 0.002])

        self.Q[state][action] = self.update_Q_sarsaexpected(
                                                            state,
                                                            action,
                                                            reward,
                                                            next_state)


    def update_Q_sarsaexpected(self, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""
        current = self.Q[state][action]  # estimate in Q-table (for current state, action pair)
        policy_s = np.ones(self.nA) * self.eps / self.nA  # current policy (for next state S')
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA) # greedy action


        Qsa_next = np.dot(self.Q[next_state], policy_s)         # get value of state at next time step
        target = reward + (self.gamma * Qsa_next)               # construct TD target
        new_value = current + (self.alpha * (target - current)) # get updated value
        return new_value



    def epsilon_greedy(self, state):
        """Selects epsilon-greedy action for supplied state.

        Params
        ======
            Q (dictionary): action-value function
            state (int): current state
            nA (int): number actions in the environment
            eps (float): epsilon
        """
        if random.random() > self.eps: # select greedy action with probability epsilon

            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return random.choice(np.arange(self.nA))
