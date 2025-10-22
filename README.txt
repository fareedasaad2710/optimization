MULTI-ROBOT COVERAGE PATH PLANNING - CODE EXECUTION INSTRUCTIONS
================================================================

OVERVIEW:
This code implements a multi-robot coverage path planning framework that combines:
- DARP (Divide Areas based on Robots' Positions): generates balanced partitions
- UF-STC (Unified Formation Spanning Tree Coverage): generates continuous paths

The problem optimizes two objectives:
1. Maximize area coverage (F1)
2. Minimize workload variance (F2)

Subject to constraints:
1. Path continuity (4-neighbor connectivity)
2. Map boundary constraints
3. Obstacle avoidance

REQUIRED FILES:
- simple_robot_coverage.py (Main implementation)
- multi_robot_coverage.py (Original detailed version)

SYSTEM REQUIREMENTS:
- Python 3.6 or higher
- No additional packages required (uses only built-in libraries)

HOW TO RUN THE CODE:
====================

Method 1: Interactive Mode (Recommended)
----------------------------------------
1. Open terminal/command prompt
2. Navigate to the folder containing the code
3. Run: python3 simple_robot_coverage.py
4. Follow the prompts to enter:
   - Grid width (e.g., 3)
   - Grid height (e.g., 3) 
   - Number of robots (e.g., 2)

Method 2: Modify Configuration Variables
----------------------------------------
1. Open simple_robot_coverage.py in a text editor
2. Find the configuration section (around line 210):
   ```python
   grid_width = 3
   grid_height = 3
   num_robots = 4
   obstacles = [4]  # Cell 4 is an obstacle
   ```
3. Modify these values as needed
4. Run: python3 simple_robot_coverage.py

Method 3: Test Different Scenarios
-----------------------------------
You can test various scenarios by changing the configuration:

# Example 1: 3x3 grid, 2 robots, no obstacles
grid_width = 3
grid_height = 3
num_robots = 2
obstacles = []

# Example 2: 4x4 grid, 3 robots, with obstacles
grid_width = 4
grid_height = 4
num_robots = 3
obstacles = [5, 10]  # Multiple obstacles

# Example 3: 5x5 grid, 4 robots, no obstacles
grid_width = 5
grid_height = 5
num_robots = 4
obstacles = []

EXPECTED OUTPUT:
===============
The program will display:
1. Grid setup information
2. Cell coordinates and assignments
3. Visual grid representation
4. Robot paths
5. Evaluation results:
   - Coverage Score (higher = better)
   - Balance Score (lower = better)
   - Robot Distances
   - Constraint violations (if any)

EXAMPLE OUTPUT:
==============
Setting up a 3x3 grid with 2 robots...
Total cells: 9
Grid cells: [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]

Assignment (showing which robot covers each cell):
Cell 0: Robot 0
Cell 1: Robot 0
...

Grid Visualization:
=============
|R0|R0|R1|
=============
|R1| X |R2|
=============
|R2|R3|R3|
=============

==================================================
ROBOT COVERAGE EVALUATION RESULTS
==================================================
Coverage Score: 8 cells covered
Balance Score: 0.38 (lower = more balanced)
Robot Distances: [1.0, 2.24, 2.24, 1.0]

✓ No problems found! Solution is valid.

TROUBLESHOOTING:
===============
- If you get "command not found: python3", try "python" instead
- Make sure you're in the correct directory containing the .py files
- If you get syntax errors, check that you have Python 3.6+ installed

CODE STRUCTURE:
==============
- evaluate_robot_solution(): Main evaluation function
- count_covered_cells(): Calculates coverage (F1)
- calculate_workload_variance(): Calculates balance (F2)
- check_path_validity(): Validates constraints
- divide_cells_equally(): Assigns cells to robots
- print_grid_visualization(): Shows visual grid

For questions or issues, refer to the code comments which explain each function's purpose and how it relates to the problem formulation.

