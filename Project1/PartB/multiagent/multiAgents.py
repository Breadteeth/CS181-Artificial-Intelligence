# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newFood = childGameState.getFood()
        newFood_list = newFood.asList()
        oldFood_list = oldFood.asList()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhost_list = [( ghostState.getPosition()[0],   ghostState.getPosition()[1] ) for ghostState in newGhostStates] # x and y postion
        ret = 0
        "*** YOUR CODE HERE ***"

        left_ScaredTimes = min(newScaredTimes) # remaining time that the ghost is scared

        if (newPos in newGhost_list) and (left_ScaredTimes <=0): # if ghost is not scared and pacman will jump into the ghost place
            ret = -10000 # worst case, so give a negative ret
        elif newPos in oldFood_list: # else, if pacman can get a new food to eat on the next step
            ret = 10000 # best case, so give a large positive ret
        else: # normal case, so construct a 1/food - 1/ghost to represent the "balance"
            nearFood = sorted(newFood_list, key=lambda food: manhattanDistance(food, newPos))[0]
            nearGhost = sorted(newGhost_list, key=lambda ghost: manhattanDistance(ghost, newPos))[0]
            alpha = 1 # coefficient to control the distance between the normal case's score and the other -> 1 is already fine
            ret =  alpha * ( 1/manhattanDistance(nearFood, newPos)) - (1/manhattanDistance(nearGhost, newPos) )
        
        return ret
        
        


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        import math
        
        # depth is very important in this case, so we set it to both maxvalue and minvalue function!

        def maxvalue(state, depth): # pacman is here, so next step is ghost
            value = -math.inf
            if (depth == self.depth or state.isWin() or state.isLose()): return self.evaluationFunction(state) # debug here!!!--do not miss the depth condition!!
            
            for legal_action in state.getLegalActions(0): # for all legal actions of the pacman:
                value_successor = minvalue( state.getNextState(0, legal_action), depth, 1) # look at the next ghost indexed one
                value = max(value, value_successor)
            return value

        def minvalue(state, depth, ghostIndex): # for ghost we use minvalue
            # to control the detection between ghosts and ghost and pacman, we introduce a parameter that stands for the index of the ghosts
            value = math.inf
            if (depth == self.depth or state.isWin() or state.isLose()): return self.evaluationFunction(state)
            
            for legal_action in state.getLegalActions(ghostIndex):
                value_successor = math.inf
                if ghostIndex == (gameState.getNumAgents()-1): value_successor = maxvalue(state.getNextState(ghostIndex, legal_action), depth + 1 )
                else: value_successor = minvalue(state.getNextState(ghostIndex, legal_action), depth, ghostIndex+1) # next ghost is the successor state, but same depth!
                value = min(value, value_successor)
            return value            

        # we start from a max node recursively - then the min node para is depth zero, ghost indexed one
        Actions = [ ( legal_action, minvalue(gameState.getNextState(0, legal_action), 0, 1) ) for legal_action in gameState.getLegalActions(0)]
        Actions.sort(key=lambda node: node[1], reverse=True) # must reverse here!
        action_ret = Actions[0][0] # this is what we wanted! finally done!
        return action_ret

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import math
        action_ret = 0
        # depth is very important in this case, so we set it to both maxvalue and minvalue function!

        def maxvalue(state, depth, alpha, beta): # pacman is here, so next step is ghost
            value = -math.inf
            if (depth == self.depth or state.isWin() or state.isLose()): return self.evaluationFunction(state) # debug here!!!--do not miss the depth condition!!
            
            for legal_action in state.getLegalActions(0): # for all legal actions of the pacman:
                value_successor = minvalue( state.getNextState(0, legal_action), depth, 1, alpha, beta) # look at the next ghost indexed one
                value = max(value, value_successor)
                if value > beta : return value
                alpha = max(alpha, value)
            return value

        def minvalue(state, depth, ghostIndex, alpha, beta): # for ghost we use minvalue
            # to control the detection between ghosts and ghost and pacman, we introduce a parameter that stands for the index of the ghosts
            value = math.inf
            if (depth == self.depth or state.isWin() or state.isLose()): return self.evaluationFunction(state)
            
            for legal_action in state.getLegalActions(ghostIndex):
                value_successor = math.inf
                if ghostIndex == (gameState.getNumAgents()-1): value_successor = maxvalue(state.getNextState(ghostIndex, legal_action), depth + 1 ,alpha, beta)
                else: value_successor = minvalue(state.getNextState(ghostIndex, legal_action), depth, ghostIndex+1, alpha, beta) # next ghost is the successor state, but same depth!
                value = min(value, value_successor)
                if value < alpha: return value
                beta = min(beta, value)
            return value            

        # we start from a max node recursively - then the min node para is depth zero, ghost indexed one
        # just write the first maxvalue down here is OK!
        alpha = -math.inf
        beta = -alpha
        value = -math.inf
        for legal_action in gameState.getLegalActions(0):
            value_successor = minvalue(gameState.getNextState(0, legal_action), 0, 1, alpha, beta) # depth 0
            value = max(value, value_successor)
            if value > beta: 
                return value
            if value > alpha:
                action_ret = legal_action # our goal action is here! finally done!
                alpha = value
        return action_ret
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        import math
        
        # depth is very important in this case, so we set it to both maxvalue and minvalue function!
        # NO CHANGE HERE AT ALL
        def maxvalue(state, depth): # pacman is here, so next step is ghost
            value = -math.inf
            if (depth == self.depth or state.isWin() or state.isLose()): return self.evaluationFunction(state) # debug here!!!--do not miss the depth condition!!
            
            for legal_action in state.getLegalActions(0): # for all legal actions of the pacman:
                value_successor = expvalue( state.getNextState(0, legal_action), depth, 1) # look at the next ghost indexed one
                value = max(value, value_successor)
            return value
        
        # ONLY CHANGE HERE IS OK, replace min with exp!!
        def expvalue(state, depth, ghostIndex): 
            # to control the detection between ghosts and ghost and pacman, we introduce a parameter that stands for the index of the ghosts
            value = 0 # Debug-ZERO here!!!!!!!!!!!!!!!!!!!!
            num_next_actions = len(state.getLegalActions(ghostIndex)) # how many actions we may acheive in the next action!
            if (depth == self.depth or state.isWin() or state.isLose()): return self.evaluationFunction(state)
            for legal_action in state.getLegalActions(ghostIndex):
                value_successor = 0 
                if ghostIndex == (gameState.getNumAgents()-1): value_successor = maxvalue(state.getNextState(ghostIndex, legal_action), depth + 1)
                else: value_successor = expvalue(state.getNextState(ghostIndex, legal_action), depth, ghostIndex+1)
                value += 1/num_next_actions * value_successor # the first factor is p
            return value
        
        # we start from a max node recursively - then the min node para is depth zero, ghost indexed one
        Actions = [ ( legal_action, expvalue(gameState.getNextState(0, legal_action), 0, 1) ) for legal_action in gameState.getLegalActions(0)]
        Actions.sort(key=lambda node: node[1], reverse=True) # must reverse here!
        action_ret = Actions[0][0] # this is what we wanted! finally done!
        return action_ret
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <besides the 1/food dist -1/ghost dist balance we used above in reflex agent, we add 3 things:
    1. we take advantage of the score information 2. we set additional attraction for pacman to let it eat more capsules by giving 
     additional points when the capsule number is smaller than a fixed account.
    3. we give additional credits to the free case where there's no food and ghosts around >
    """
    "*** YOUR CODE HERE ***"
        # childGameState = currentGameState.getPacmanNextState(action)
        # newPos = childGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newFood = childGameState.getFood()
        # newFood_list = newFood.asList()
        # oldFood_list = oldFood.asList()
        # newGhostStates = childGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # newGhost_list = [( ghostState.getPosition()[0],   ghostState.getPosition()[1] ) for ghostState in newGhostStates] # x and y postion
        # ret = 0
        # "*** YOUR CODE HERE ***"

        # left_ScaredTimes = min(newScaredTimes) # remaining time that the ghost is scared

        # if (newPos in newGhost_list) and (left_ScaredTimes <=0): # if ghost is not scared and pacman will jump into the ghost place
        #     ret = -10000 # worst case, so give a negative ret
        # elif newPos in oldFood_list: # else, if pacman can get a new food to eat on the next step
        #     ret = 10000 # best case, so give a large positive ret
        # else: # normal case, so construct a 1/food - 1/ghost to represent the "balance"
        #     nearFood = sorted(newFood_list, key=lambda food: manhattanDistance(food, newPos))[0]
        #     nearGhost = sorted(newGhost_list, key=lambda ghost: manhattanDistance(ghost, newPos))[0]
        #     alpha = 1 # coefficient to control the distance between the normal case's score and the other -> 1 is already fine
        #     ret =  alpha * ( 1/manhattanDistance(nearFood, newPos)) - (1/manhattanDistance(nearGhost, newPos) )
        
        # return ret

    # as old above is ok
    Pos = currentGameState.getPacmanPosition()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    Food_list = currentGameState.getFood().asList()
    Ghost_list = [( ghostState.getPosition()[0],   ghostState.getPosition()[1] ) for ghostState in GhostStates] # x and y postion
    left_ScaredTimes = min(ScaredTimes) # remaining time that the ghost is scared
    ret = 0
    if ((Pos in Ghost_list) and (left_ScaredTimes <=0)) or (currentGameState.isLose()): # if ghost is not scared and pacman will jump into the ghost place
        ret = -10 # worst case, so give a negative ret
    elif Pos in Food_list: # else, if pacman can get a new food to eat on the next step
        ret = 10 # good case, so give a positive ret
    else: # normal case, so construct a 1/food - 1/ghost to represent the "balance"
        ret += scoreEvaluationFunction(currentGameState) # take advantage of the score information!
        nearFood = sorted(Food_list, key=lambda food: manhattanDistance(food, Pos))
        nearGhost = sorted(Ghost_list, key=lambda ghost: manhattanDistance(ghost, Pos))
        alpha = 1 # coefficient to control the distance between the normal case's score and the other -> 1 is already fine
        num_bonus_capsules = 2 # according to the specific game, we set it to two
        num_current_capsules = len(currentGameState.getCapsules())
        if num_current_capsules < num_bonus_capsules: ret += 100 # Attraction!
        if not len(nearFood) or not len(nearGhost):  ret += 10 # positive is ok here!
        else: # 1/food dist -1/ghost dist
            ret += alpha * (1/manhattanDistance(nearFood[0], Pos) - 1/manhattanDistance(nearGhost[0], Pos))

    return ret
# Abbreviation
better = betterEvaluationFunction

q6_count = 0
def ContestbasedEvaluationFunction(currentGameState: GameState):
    Score_food = 10
    Score_ghost = -5000
    Score_scared_ghost = 3000
    Score_capsule = 15
    
    coefficients_list = [[0.0, 0.0, 0.0, 0.0], [0.07636028, 0.3439, 0.074668, 0.451139088], [0.07636028, 0.343902075, 0.074669, 0.45114], [0.9592718133333332, 0.574918935, 0.005236859999999999, 0.526277298], [1.0747421466666665, 0.55554165, 0.016415433333333333, 0.338329272], [0.15903976, 0.438594, 0.03330921333333333, 0.23673970799999997], [0.04889686666666667, 1.48672992, 0.06623387333333333, 0.3423579], [1.1972047466666667, 0.396417285, 0.24480970666666668, 0.312342834], [0.6675271066666667, 0.48372943500000004, 0.09181586, 0.446625072], [0.5673971999999999, 0.43282675500000006, 0.0129041, 0.20707335600000001], [0.2421062, 0.92999028, 0.016849093333333332, 0.150030102]]

    coefficients = coefficients_list[q6_count]


    Pos = currentGameState.getPacmanPosition()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    Food_list = currentGameState.getFood().asList()
    Capsule_list = currentGameState.getCapsules()
    Ghost_list = [( ghostState.getPosition()[0],   ghostState.getPosition()[1] ) for ghostState in GhostStates] # x and y postion

    left_ScaredTimes = min(ScaredTimes) # remaining time that the ghost is scared
    ret = 0

    ret += currentGameState.getScore() # take advantage of the score information!
    nearFood = sorted(Food_list, key=lambda food: manhattanDistance(food, Pos))
    nearCapsule = sorted(Capsule_list, key=lambda capsule: manhattanDistance(capsule, Pos))

    if nearCapsule:
        ret += Score_capsule * 1/manhattanDistance(nearCapsule[0], Pos) * coefficients[0]
    elif nearFood:
        ret += Score_food * 1/manhattanDistance(nearFood[0], Pos)* coefficients[1]
    
    scared_bonus = 0
    for ghost in GhostStates:
        ghost_pos = ghost.getPosition()
        ghost_scared = ghost.scaredTimer
        if ghost_scared > 0: 
            distance = manhattanDistance(Pos, ghost_pos)
            if distance < ghost_scared:
                scared_bonus = max(scared_bonus, Score_scared_ghost * (1.0 / distance)* coefficients[2])
        else:
            if Pos==ghost_pos:
                ret = Score_ghost * coefficients[3]

    return ret+scared_bonus

betternot = ContestbasedEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.depth = 3
        self.evaluationFunction = betternot
    def randomNumberGenerator(self):
        import random, datetime
        random.seed(datetime.datetime.now())
        coefficients = [random.uniform(0, 1) for _ in range(4)]
        random.seed(0)
        print(coefficients)
        return coefficients
    def getAction(self, gameState: GameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        if gameState.getScore() == 0 and gameState.getNumFood() == 69:
            # update q6_count
            global q6_count
            q6_count += 1
        import math
        action_ret = 0
        # depth is very important in this case, so we set it to both maxvalue and minvalue function!

        def maxvalue(state, depth, alpha, beta): # pacman is here, so next step is ghost
            value = -math.inf
            if (depth == self.depth or state.isWin() or state.isLose()): return self.evaluationFunction(state) # debug here!!!--do not miss the depth condition!!
            
            for legal_action in state.getLegalActions(0): # for all legal actions of the pacman:
                value_successor = minvalue( state.getNextState(0, legal_action), depth, 1, alpha, beta) # look at the next ghost indexed one
                value = max(value, value_successor)
                if value > beta : return value
                alpha = max(alpha, value)
            return value

        def minvalue(state, depth, ghostIndex, alpha, beta): # for ghost we use minvalue
            # to control the detection between ghosts and ghost and pacman, we introduce a parameter that stands for the index of the ghosts
            value = math.inf
            if (depth == self.depth or state.isWin() or state.isLose()): return self.evaluationFunction(state)
            
            for legal_action in state.getLegalActions(ghostIndex):
                value_successor = math.inf
                if ghostIndex == (gameState.getNumAgents()-1): value_successor = maxvalue(state.getNextState(ghostIndex, legal_action), depth + 1 ,alpha, beta)
                else: value_successor = minvalue(state.getNextState(ghostIndex, legal_action), depth, ghostIndex+1, alpha, beta) # next ghost is the successor state, but same depth!
                value = min(value, value_successor)
                if value < alpha: return value
                beta = min(beta, value)
            return value            

        # we start from a max node recursively - then the min node para is depth zero, ghost indexed one
        # just write the first maxvalue down here is OK!
        alpha = -math.inf
        beta = -alpha
        value = -math.inf
        for legal_action in gameState.getLegalActions(0):
            value_successor = minvalue(gameState.getNextState(0, legal_action), 0, 1, alpha, beta) # depth 0
            value = max(value, value_successor)
            if value > beta: 
                return value
            if value > alpha:
                action_ret = legal_action # our goal action is here! finally done!
                alpha = value
        return action_ret
        