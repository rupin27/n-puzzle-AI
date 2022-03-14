import sys
import puzz
import pdqpq
import random


GOAL_STATE = puzz.EightPuzzleBoard("012345678")


def solve_puzzle(start_state, flavor):
    """Perform a search to find a solution to a puzzle.
    
    Args:
        start_state (EightPuzzleBoard): the start state for the search
        flavor (str): tag that indicate which type of search to run.  Can be one of the following:
            'bfs' - breadth-first search
            'ucost' - uniform-cost search
            'greedy-h1' - Greedy best-first search using a misplaced tile count heuristic
            'greedy-h2' - Greedy best-first search using a Manhattan distance heuristic
            'greedy-h3' - Greedy best-first search using a weighted Manhattan distance heuristic
            'astar-h1' - A* search using a misplaced tile count heuristic
            'astar-h2' - A* search using a Manhattan distance heuristic
            'astar-h3' - A* search using a weighted Manhattan distance heuristic
    
    Returns: 
        A dictionary containing describing the search performed, containing the following entries:
        'path' - list of 2-tuples representing the path from the start to the goal state (both 
            included).  Each entry is a (str, EightPuzzleBoard) pair indicating the move and 
            resulting successor state for each action.  Omitted if the search fails.
        'path_cost' - the total cost of the path, taking into account the costs associated with 
            each state transition.  Omitted if the search fails.
        'frontier_count' - the number of unique states added to the search frontier at any point 
            during the search.
        'expanded_count' - the number of unique states removed from the frontier and expanded 
            (successors generated)

    """
    if flavor.find('-') > -1:
        strat, heur = flavor.split('-')
    else:
        strat, heur = flavor, None

    if strat == 'bfs':
        return BreadthFirstSolver(GOAL_STATE).solve(start_state)
    elif strat == 'ucost':
        return UniformCostSolver(GOAL_STATE).solve(start_state) 
    elif strat == 'greedy':
        return GreedySolutionSolver(GOAL_STATE).solve(start_state, heur)  # delete this line!
    elif strat == 'astar':
        return AstarSolutionSolver(GOAL_STATE).solve(start_state, heur)  # delete this line!
    else:
        raise ValueError("Unknown search flavor '{}'".format(flavor))

def generate_state(num):
    state = GOAL_STATE
    path = [state]
    for i in range(num):
        successes = list(state.successors().values())
        success_state = random.choice(successes)
        state = success_state
    return state

def get_test_puzzles():
    """Return sample start states for testing the search strategies.
    
    Returns:
        A tuple containing three EightPuzzleBoard objects representing start states that have an
        optimal solution path length of 3-5, 10-15, and >=25 respectively.
    
    """ 
    return (generate_state(5), generate_state(15), generate_state(30))  # fix this line!


def print_table(flav__results, include_path = False):
    """Print out a comparison of search strategy results.

    Args:
        flav__results (dictionary): a dictionary mapping search flavor tags result statistics. See
            solve_puzzle() for detail.
        include_path (bool): indicates whether to include the actual solution paths in the table

    """
    result_tups = sorted(flav__results.items())
    c = len(result_tups)
    na = "{:>12}".format("n/a")
    rows = [  # abandon all hope ye who try to modify the table formatting code...
        "flavor  " + "".join([ "{:>12}".format(tag) for tag, _ in result_tups]),
        "--------" + ("  " + "-"*10)*c,
        "length  " + "".join([ "{:>12}".format(len(res['path'])) if 'path' in res else na 
                                for _, res in result_tups ]),
        "cost    " + "".join([ "{:>12,}".format(res['path_cost']) if 'path_cost' in res else na 
                                for _, res in result_tups ]),
        "frontier" + ("{:>12,}" * c).format(*[res['frontier_count'] for _, res in result_tups]),
        "expanded" + ("{:>12,}" * c).format(*[res['expanded_count'] for _, res in result_tups])
    ]
    if include_path:
        rows.append("path")
        longest_path = max([ len(res['path']) for _, res in result_tups if 'path' in res ])
        print("longest", longest_path)
        for i in range(longest_path):
            row = "        "
            for _, res in result_tups:
                if len(res.get('path', [])) > i:
                    move, state = res['path'][i]
                    row += " " + move[0] + " " + str(state)
                else:
                    row += " "*12
            rows.append(row)
    print("\n" + "\n".join(rows), "\n")


class PuzzleSolver:
    """Base class for 8-puzzle solver."""

    def __init__(self, goal_state):
        self.parents = {}  # state -> parent_state
        self.expanded_count = 0
        self.frontier_count = 0
        self.goal = goal_state

    def get_path(self, state):
        """Return the solution path from the start state of the search to a target.
        
        Results are obtained by retracing the path backwards through the parent tree to the start
        state for the serach at the root.
        
        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            A list of EightPuzzleBoard objects representing the path from the start state to the
            target state

        """
        path = []
        while state is not None:
            path.append(state)
            state = self.parents[state]
        path.reverse()
        return path

    def get_cost(self, state): 
        """Calculate the path cost from start state to a target state.
        
        Transition costs between states are equal to the square of the number on the tile that 
        was moved. 

        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            Integer indicating the cost of the solution path

        """
        cost = 0
        path = self.get_path(state)
        for i in range(1, len(path)):
            x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
            tile = path[i].get_tile(x, y)        
            cost += int(tile)**2
        return cost

    def get_cost_between(self, predecessor, successor):
        """ Cost function -> calculates cost between a predeccessor and its successor
        Args:
            predecessor: EightPuzzleBoard (the initial state)
            successor: EightPuzzleBoard (the state being moved to)
        Returns:
            int (weight value)
        """
        cords = predecessor.find("0") # (x ,y)
        tile_path = int(successor.get_tile(cords[0], cords[1]))
        return tile_path * tile_path

    def get_results_dict(self, state):
        """Construct the output dictionary for solve_puzzle()
        
        Args:
            state (EightPuzzleBoard): final state in the search tree
        
        Returns:
            A dictionary describing the search performed (see solve_puzzle())

        """
        results = {}
        results['frontier_count'] = self.frontier_count
        results['expanded_count'] = self.expanded_count
        if state:
            results['path_cost'] = self.get_cost(state)
            path = self.get_path(state)
            moves = ['start'] + [ path[i-1].get_move(path[i]) for i in range(1, len(path)) ]
            results['path'] = list(zip(moves, path))
        return results

    def solve(self, start_state):
        """Carry out the search for a solution path to the goal state.
        
        Args:
            start_state (EightPuzzleBoard): start state for the search 
        
        Returns:
            A dictionary describing the search from the start state to the goal state.

        """
        raise NotImplementedError('Classed inheriting from PuzzleSolver must implement solve()')

    def get_heuristic(self, state, h):
        """ 
        Heuristic function
        Args: h: "h1" | "h2" | "h3"
        Returns: int (heuristic value) | None (if h is invalid)
        """
        assert h == "h1" or h == "h2" or h == "h3", "Invalid Heuristic h"
        heuristic = 0
        if h == "h1":
        #h1 = number of misplaced tiles
            for i in range(9):
                if self.goal.find(str(i)) != state.find(str(i)):
                    heuristic += 1
        elif h == "h2":
        #h2 = the sum of the distances of the tiles from their goal positions
            for i in range (9):
                goal_cord = self.goal.find(str(i))
                state_cord = state.find(str(i))
                heuristic += abs(state_cord[0] - goal_cord[0]) + abs(state_cord[1] - goal_cord[1])
        else:
        #h3 = modified version of h2 (takes into account transition costs)
            for i in range (9):
                goal_cord = self.goal.find(str(i))
                state_cord = state.find(str(i))
                heuristic += i * i * (abs(state_cord[0] - goal_cord[0]) + abs(state_cord[1] - goal_cord[1]))
        return heuristic
        

class BreadthFirstSolver(PuzzleSolver):
    """Implementation of Breadth-First Search based on PuzzleSolver"""

    def __init__(self, goal_state):
        self.frontier = pdqpq.FifoQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node

                    # BFS checks for goal state _before_ adding to frontier
                    if succ == self.goal:
                        return self.get_results_dict(succ)
                    else:
                        self.add_to_frontier(succ)

        # if we get here, the search failed
        return self.get_results_dict(None) 

class UniformCostSolver(PuzzleSolver):
    """Implementation of Uniform-Cost Search based on PuzzleSolver"""
    def __init__(self, goal_state):
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node, n):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, n)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if node == self.goal:
                return self.get_results_dict(node)
            succs = self.expand_node(node)

            for move, succ in succs.items():
                current_cost = self.get_cost(node) + self.get_cost_between(node, succ)
                
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, current_cost)

                elif (succ in self.frontier) and (self.frontier.get(succ) > current_cost):
                    self.frontier.remove(succ)
                    self.parents[succ] = node
                    self.add_to_frontier(succ, current_cost)
        # if we get here, the search failed
        return self.get_results_dict(None) 

class GreedySolutionSolver(PuzzleSolver):
    """Implementation of Greedy-First Search based on PuzzleSolver"""
    def __init__(self, goal_state):
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node, n):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, n)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()
    
    def solve(self, start_state, h):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)
        while len(self.frontier) != 0:
            node = self.frontier.pop()
            if node == self.goal:
                return self.get_results_dict(node)
            succs = self.expand_node(node)
            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, self.get_heuristic(succ, h))
        # if we get here, the search failed
        return self.get_results_dict(None)

class AstarSolutionSolver(PuzzleSolver):
    """Implementation of A* Search based on PuzzleSolver"""
    def __init__(self, goal_state):
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node, n):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, n)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state, h):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        while len(self.frontier) !=  0:
            node = self.frontier.pop()
            if node == self.goal:
                return self.get_results_dict(node)
            succs = self.expand_node(node)
            for move, succ in succs.items():
                cost_so_far = self.get_cost(node) + self.get_cost_between(node, succ) + self.get_heuristic(succ, h) - self.get_heuristic(node, h)
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, cost_so_far)
                    
                elif (succ in self.frontier) and (self.frontier.get(succ) > cost_so_far):
                    self.frontier.remove(succ)
                    self.parents[succ] = node
                    self.add_to_frontier(succ, cost_so_far)       
        # if we get here, the search failed  
        return self.get_results_dict(None)

############################################

if __name__ == '__main__':

    # parse the command line args
    start = puzz.EightPuzzleBoard(sys.argv[1])
    if sys.argv[2] == 'all':
        flavors = ['bfs', 'ucost', 'greedy-h1', 'greedy-h2', 
                   'greedy-h3', 'astar-h1', 'astar-h2', 'astar-h3']
    else:
        flavors = sys.argv[2:]

    # run the search(es)
    results = {}
    for flav in flavors:
        print("solving puzzle {} with {}".format(start, flav))
        results[flav] = solve_puzzle(start, flav)

    print_table(results, include_path=False)  # change to True to see the paths!

