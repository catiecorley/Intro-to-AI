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


from re import A
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # print("successorGameState")
        # print(successorGameState)
        newPos = successorGameState.getPacmanPosition()
        # print("successorGameState")
        # print(successorGameState)
        newFood = successorGameState.getFood()
        # print("newFood")
        # print(newFood)
        newGhostStates = successorGameState.getGhostStates()
        # print("newGhostStates")
        # print(newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print("newScaredTimes")
        # print(newScaredTimes)

        # print("score")
        # print(successorGameState.getScore())
        "*** YOUR CODE HERE ***"
        listOfNewFood = newFood.asList()
        minimumDistanceToFood = float("inf")
        minimumDistanceToGhost = float("inf")

        for food in listOfNewFood:
            if manhattanDistance(food, newPos) < minimumDistanceToFood:
                minimumDistanceToFood = manhattanDistance(food, newPos)
        for ghost in newGhostStates:
            position = ghost.getPosition()
            if manhattanDistance(newPos, position) < minimumDistanceToGhost:
                minimumDistanceToGhost = manhattanDistance(newPos, position)
        
        if successorGameState.isWin():
            return float("inf")
        if successorGameState.isLose() or minimumDistanceToGhost < 3:
            return -float("inf")
        return successorGameState.getScore() + 1.0/minimumDistanceToFood - 1.0/minimumDistanceToGhost

            

def scoreEvaluationFunction(currentGameState):
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
    def isTerminal(self, gameState, depth):
        if gameState.isWin() or gameState.isLose():
            return True
        if depth > self.depth:
            return True
        return False


  
    def maxValue(self, gameState, depth, agentIndex =0):
        v = -float('inf')
        legalActions = gameState.getLegalActions()

        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), len(legalActions)
    

        players = gameState.getNumAgents()
        actions = []
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            actions.append(self.minValue(successor, depth, 1, players)[0])
            v = max(v, self.minValue(successor, depth, 1, players)[0])
        bestScore = v
        bestScoreIndices = [] 
        for index in range(len(actions)):
            if actions[index] == bestScore:
                bestScoreIndices.append(index)
        bestIndex = bestScoreIndices[0]
        return bestScore, bestIndex 

    def minValue(self, gameState, depth, agentIndex, agents):
      v = +float('inf')
      legalActions = gameState.getLegalActions(agentIndex)

      if depth > self.depth or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), len(legalActions)

      actions = []
      newAgents = agents-1
      if agentIndex < newAgents:
        for action in legalActions:
          successor = gameState.generateSuccessor(agentIndex, action)
          actions.append(self.minValue(successor, depth, agentIndex+1, agents-1)[0])
          v = min(v, self.minValue(successor, depth, agentIndex+1, agents-1)[0])
        bestScore = v
        bestScoreIndices = [] 
        for index in range(len(actions)):
            if actions[index] == bestScore:
                bestScoreIndices.append(index)
        bestIndex = bestScoreIndices[0]

        return bestScore, bestIndex 
      else:
        for action in legalActions:
          successor = gameState.generateSuccessor(agentIndex, action)
          actions.append(self.maxValue(successor,depth+1,  0)[0])
          v = min(v, self.maxValue(successor, depth+1, 0)[0])
        bestScore = v
        bestScoreIndices = [] 
        for index in range(len(actions)):
            if actions[index] == bestScore:
                bestScoreIndices.append(index)
        bestIndex = bestScoreIndices[0]

        return bestScore, bestIndex 
     
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        """
        def value(state):
            if terminal: return utility
            if max: return max-value
            if min: return min-value
        def max-value(state):
            v = -inf
            for each successor:
                v = max(v, min-value(successor))
            return v
        """
        
        pacmanIndex = 0
        initialDepth = 1
        legalActions = gameState.getLegalActions(pacmanIndex) 
        legalActions.append(Directions.STOP)
    
        bestScore = self.maxValue(gameState, initialDepth,pacmanIndex)[1]
        
        return legalActions[bestScore]

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def isTerminal(self, gameState, depth):
        if gameState.isWin() or gameState.isLose():
            return True
        if depth > self.depth:                
            return True
        return False

    def minValue(self, gameState, depth, agentIndex, agents, alpha, beta):
        v = +float('inf')

        if self.isTerminal(gameState, depth) == True:
            return self.evaluationFunction(gameState), Directions.STOP
        actions = []
        agentsNum = agents - 1
        legalActions = gameState.getLegalActions(agentIndex)

        if agentIndex < agentsNum:
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                actions.append(self.minValue(successor, depth, agentIndex+1, agents-1, alpha, beta)[0])
                v = min(v, self.minValue(successor, depth, agentIndex+1, agents-1, alpha, beta)[0])
                if v < alpha:
                    break
                beta = min(v, beta)
            bestScore = v
            bestScoreIndices = [] 
            for index in range(len(actions)):
                if actions[index] == bestScore:
                    bestScoreIndices.append(index)
            bestIndex = bestScoreIndices[0]
            return bestScore, bestIndex 
        else:
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                actions.append(self.maxValue(successor, depth + 1, 0, alpha, beta)[0])
                v = min(v, self.maxValue(successor, depth + 1, 0, alpha, beta)[0])
                if v < alpha:
                    break
                beta = min(v, beta)

            bestScore = v
            bestScoreIndices = [] 
            for index in range(len(actions)):
                if actions[index] == bestScore:
                    bestScoreIndices.append(index)
            bestIndex = bestScoreIndices[0]
            return bestScore, bestIndex 

    def maxValue(self, gameState, depth, agentIndex = 0, alpha = -float('inf'), beta =float('inf')):
        v = -float('inf')
        
        if self.isTerminal(gameState, depth) == True:
            return self.evaluationFunction(gameState), Directions.STOP
        legalActions = gameState.getLegalActions()
        agents = gameState.getNumAgents()
        actions = []  
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            actions.append(self.minValue(successor, depth, 1, agents, alpha, beta )[0])
            v = max(v, self.minValue(successor, depth, 1, agents, alpha, beta )[0])
            if v > beta:
                break
            alpha = max(alpha, v)
        bestScore = v
        bestScoreIndices = [] 
        for index in range(len(actions)):
                if actions[index] == bestScore:
                    bestScoreIndices.append(index)
        bestIndex = bestScoreIndices[0]
        return bestScore, bestIndex 

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float('inf')
        beta = float('inf')

        pacmanIndex = 0
        initialDepth = 1
        legalActions = gameState.getLegalActions(pacmanIndex) 
        legalActions.append(Directions.STOP)
    
        bestScore = self.maxValue(gameState, initialDepth, pacmanIndex, alpha, beta)[1]
        
        return legalActions[bestScore]        


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
       
        def expValue(gameState, depth, agentIndex):
            v = 0

            if isTerminal(gameState, depth):
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentIndex)
            probability = 1.0 / len(legalActions)
            agents =  gameState.getNumAgents() 
            newAgents = agents - 1
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex,action)
                if agentIndex < newAgents:
                    v += probability * expValue(successor,depth,agentIndex + 1)
                else:
                    v += probability * maxValue(successor,depth - 1) 
            return v

        def maxValue(gameState, depth):
            v = -float('inf')

            if isTerminal(gameState, depth):
                return self.evaluationFunction(gameState)
 
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                if action != Directions.STOP:
                    successor = gameState.generateSuccessor(0, action)
                    v = max(v, expValue(successor, depth, 1))
            return v


        def isTerminal(gameState, depth):
            if gameState.isWin() or gameState.isLose():
                return True
            if depth == 0:
                return True
            return False

        pacman = 0
        agentIndex = 1
        legalActions = gameState.getLegalActions(pacman)

        bestScore = 0
        chosenAction = legalActions[0]
        for action in legalActions:
            if action != Directions.STOP:
                childState = gameState.generateSuccessor(pacman, action)
                score = expValue(childState,self.depth, agentIndex)
                if score >= bestScore:
                    bestScore = score
                    chosenAction = action
        return chosenAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    To make this version of the score more accurate, it now takes into account empty food positions, all the food and ghost positions, and whether or not the ghosts are scared when adding in ghost distance.
    Scared ghosts help the score, while regular ghosts add to the score like the previous ghosts did. 

    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    numCapsules = len(currentGameState.getCapsules())
    ghostPositions = currentGameState.getGhostPositions()
    foodList = foodPositions.asList()
    
    score = currentGameState.getScore()

    empty = 0
    for position in foodPositions:
        if position == False:
            empty += 1
    score += empty 

    totalFoodDist = 0
    for position in foodList:
        totalFoodDist += manhattanDistance(pacmanPosition,position)

    if totalFoodDist > 0:
        score += 1.0 / totalFoodDist

    newGhostPos=[]
    for ghost in ghostStates:
        newGhostPos.append(ghost.getPosition())

    totalGhostDistance = 0
    for position in ghostPositions:
        totalGhostDistance += manhattanDistance(pacmanPosition,position)

    totalScareTime = 0
    for state in ghostStates:
        totalScareTime += state.scaredTimer
   
    if totalScareTime <= 0:
        score += totalScareTime 
        totalGhostDistance = -1 * totalGhostDistance
 
    
    if totalGhostDistance > 0:
        score  += 1.0/totalGhostDistance

    return score 
   
    
# Abbreviation
better = betterEvaluationFunction
