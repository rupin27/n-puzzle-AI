# n-puzzle-AI
Designed a program which uses various search techniques like BFS, DFS, Greedy Best First, and A* to solve the puzzle in Python.
Used Manhattan distance as a heuristic and data structures like HashSet and HashMap to greatly improve temporal efficiency.

# GamePlay:
The 8-puzzle consists of eight tiles on a 3x3 grid, with one open (“blank”) square. The possible configurations of tiles comprise a state space, where the goal state has the open square in the upper left, with the other tiles arranged in numeric order from left to right.
Valid moves are Up, Down, Left, and Right, which shift a tile into the open square. Depending on the position of the open square, not all of those moves may be available — in the example below, the valid moves from the start state are Up, Left, and Right.


# Running the Game:
<p>
While we will test your code by importing and directly invoking the solve_puzzle() function, you can test your solver by running solver.py from the command line or launching it from within an IDE. When run this way, you must specify at least two command line arguments1:

One or more keywords specifying which search strategies and heuristics to use, each one of the following: 
<pre>bfs, ucost, greedy-h1, greedy-h2, greedy-h3, astar-h1, astar-h2, or astar-h3.</pre>


</p>
<p>
The start state, specified as a nine digit string, with 0 denoting the blank square. The first three digits represent the top row, the next three the middle row, and the last three the bottom row. Eg: 802356174.
</p>
<br>

For example, to execute a BFS and A* (using a standard Manhattan distance heuristic):
> python3 solver.py 802356174 bfs astar-h2

