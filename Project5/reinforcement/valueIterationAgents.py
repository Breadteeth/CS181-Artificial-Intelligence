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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        for i in range(self.iterations):
            values = {} #init
            states = self.mdp.getStates()
            for state in states:
                best_action = self.computeActionFromValues(state) #get the action
                if best_action: values[state] = self.computeQValueFromValues(state, best_action)
            for state in values: self.values[state] = values[state] # easy


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
        if action not in self.mdp.getPossibleActions(state): return 0  #  Make sure to handle the case when a state has no available actions in an MDP
        else:
            QValue = 0
            for transition_state, p in self.mdp.getTransitionStatesAndProbs(state, action):
                QValue += p * (self.mdp.getReward(state,  action,  transition_state) +  self.getValue(transition_state) * self.discount ) # the q value formula
            return QValue

        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state): return None
        else: return max(self.mdp.getPossibleActions(state), key=lambda action: self.getQValue(state, action)) # clean verison


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
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            index = iteration % len(states) # don't forget the iteration might be large
            state = states[index] # get the state
            best_action = self.computeActionFromValues(state) #get the action
            if best_action: self.values[state] = self.computeQValueFromValues(state, best_action)

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
        # Initialize an empty priority queue.
        # Please use util.PriorityQueue in your implementation. The update method in this class will likely be useful;
        priority_queue = util.PriorityQueue() 
        # Compute predecessors of all states.
        predecessors = {state: [s for s in self.mdp.getStates() 
                                for action in self.mdp.getPossibleActions(s) 
                                for transition_state, _ in self.mdp.getTransitionStatesAndProbs(s, action) if transition_state == state] 
                                if not self.mdp.isTerminal(state) else [] for state in self.mdp.getStates()}
        # For each non-terminal state s , do: 
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s) == "TERMINAL_STATE": continue
            else:
                # Find the absolute value of the difference
                diff = abs(self.values[s] - self.computeQValueFromValues(s, self.computeActionFromValues(s))   )
                # Push s into the priority queue with priority -diff
                priority_queue.push(s,  -diff)
        # For iteration in 0, 1, 2, ..., self.iterations - 1 , do:
        for iter in range(self.iterations):
            # If the priority queue is empty, then terminate
            if priority_queue.isEmpty(): 
                break
            # Pop a state s off the priority queue
            s = priority_queue.pop()
            # Update the value of s (if it is not a terminal state) in self.values
            if self.mdp.isTerminal(s) != "TERMINAL_STATE":
                self.values[s] = self.computeQValueFromValues(s,  self.computeActionFromValues(s))
            # For each predecessor p of s , do:

            for p in predecessors[s]:
                if self.mdp.isTerminal(s) == "TERMINAL_STATE":  continue
                else:
                    # Find the absolute value of the difference
                    diff = abs(self.values[p] - self.computeQValueFromValues(p, self.computeActionFromValues(p))   )
                    # If diff > theta , push p into the priority queue with priority -diff 
                    if diff > self.theta: 
                        priority_queue.update(p,  -diff) # same


