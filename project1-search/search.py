# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # want last in first out. go down entire branch, then start popping values to backtrack
    frontier = util.Stack()
    startState = problem.getStartState()
    frontier.push(startState)
    expanded = [] #no priority or order needed for these nodes
    
    finalPath = util.Stack()
    finalPath.push([])

    while frontier.isEmpty() is False:

        currentNode = frontier.pop()
        path = finalPath.pop()

        if problem.isGoalState(currentNode):
            return path

        if currentNode not in expanded:
            expanded.append(currentNode)

            children = problem.getSuccessors(currentNode)
            for child, action, cost in children:
                finalPath.push(path + [action])
                frontier.push(child)
                
    
    return path



    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue() 
    startState = problem.getStartState()
    frontier.push(startState)
    expanded = [] #no priority or order needed for these nodes
    
    finalPath = util.Queue()
    finalPath.push([])

    while frontier.isEmpty() is False:

        currentNode = frontier.pop()
        path = finalPath.pop()

        if problem.isGoalState(currentNode):
            return path

        if currentNode not in expanded:
            expanded.append(currentNode)

            children = problem.getSuccessors(currentNode)
            for child, action, cost in children:
                finalPath.push(path + [action])
                frontier.push(child)
                
    
    return path
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue() 
    startState = problem.getStartState()

    frontier.push((startState), 0)
    expanded = [] #no priority or order needed for these nodes

    finalPath = util.PriorityQueue()
    finalPath.push(([]), 0)

    finalCost = util.PriorityQueue()
    finalCost.push(0,0)

    while frontier.isEmpty() is False:

        currentNode = frontier.pop()
        path = finalPath.pop()
        firstCost = finalCost.pop()
       
        if problem.isGoalState(currentNode):
            return path

        if currentNode not in expanded:
            expanded.append(currentNode)

            children = problem.getSuccessors(currentNode)
            for child, action, cost in children:
                finalCost.push(firstCost + cost, firstCost + cost)
                finalPath.push((path + [action]), firstCost + cost )
                frontier.push((child), (firstCost + cost ))
                
    
    return path
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    frontier = util.PriorityQueue() 
    startState = problem.getStartState()

    frontier.push((startState), 0)
    expanded = [] #no priority or order needed for these nodes

    finalPath = util.PriorityQueue()
    finalPath.push(([]), 0)

    finalCost = util.PriorityQueue()
    finalCost.push(0,0)

    while frontier.isEmpty() is False:

        currentNode = frontier.pop()
        path = finalPath.pop()
        firstCost = finalCost.pop()
   

        if problem.isGoalState(currentNode):
            return path

        if currentNode not in expanded:
            expanded.append(currentNode)

            children = problem.getSuccessors(currentNode)
            for child, action, cost in children:
                heuristicCost = heuristic(child, problem)

                finalCost.push(firstCost + cost, firstCost + cost + heuristicCost)
                finalPath.push((path + [action]), heuristicCost + firstCost + cost)
                frontier.push((child), heuristicCost + firstCost + cost)
                
    
    return path

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
