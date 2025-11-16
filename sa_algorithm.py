"""
Simulated Annealing Algorithm for Multi-Robot Coverage Path Planning
==================================================================

WHAT IS SIMULATED ANNEALING?
- It's like finding the best way to arrange robots to cover an area
- Starts with a random solution (like throwing robots randomly)
- Tries small changes (like swapping cells between robots)
- Sometimes accepts worse solutions to avoid getting stuck
- Gradually becomes more picky about accepting worse solutions

THE FORMULA WE USE:
J = w1(1 - coverage) + w2(imbalance) + penalty

WHAT EACH PART MEANS:
- coverage: How many cells are covered (higher = better)
- imbalance: How different robot workloads are (lower = better)
- penalty: How many rules we broke (lower = better)
- w1, w2: How much we care about coverage vs balance

SA PARAMETERS:
- Temperature: Starts high (accepts bad solutions), gets lower (more picky)
- Cooling Rate: How fast temperature drops
- Iterations: How many times we try to improve
"""

import math
import random
import copy
from problem_formulation import *

class RobotCoverageSolution:
    """
    WHAT IS THIS CLASS?
    - It's like a container that holds one possible solution
    - A solution = which robot covers which cell + what path each robot takes
    - Think of it like a blueprint for robot assignments
    """
    
    def __init__(self, assignment, paths, all_cells, free_cells, obstacles, grid_width, grid_height):
        # Store all the data about this solution
        self.assignment = copy.deepcopy(assignment)  # Which robot covers which cell
        self.paths = copy.deepcopy(paths)           # What path each robot follows
        self.all_cells = all_cells                  # All cells in the grid
        self.free_cells = free_cells                # Cells robots can visit
        self.obstacles = obstacles                  # Cells robots cannot visit
        self.grid_width = grid_width               # Grid size
        self.grid_height = grid_height             # Grid size
        self.fitness = None                        # How good this solution is
        self.combined_score = None                 # Final score (lower = better)
        
    def evaluate(self):
        """
        WHAT DOES THIS DO?
        - Calculates how good this solution is
        - Uses the formula: J = w1(1 - coverage) + w2(imbalance) + penalty
        - Lower score = better solution
        """
        # Convert paths from list to dict format if needed
        # evaluate_solution expects paths as dict[int, list[int]]
        if isinstance(self.paths, list):
            paths_dict = {robot_id: path for robot_id, path in enumerate(self.paths)}
        elif isinstance(self.paths, dict):
            paths_dict = copy.deepcopy(self.paths)
        else:
            # Fallback: create empty dict
            paths_dict = {}
        
        # Ensure all paths in dict are lists
        for robot_id in paths_dict:
            if not isinstance(paths_dict[robot_id], list):
                paths_dict[robot_id] = [paths_dict[robot_id]] if paths_dict[robot_id] is not None else []
        
        # Convert all_cells to Cell objects if needed (evaluate_solution expects Cell objects)
        cells_list = self.all_cells
        if len(cells_list) > 0 and isinstance(cells_list[0], tuple):
            # Convert tuples to Cell objects
            class Cell:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            cells_list = [Cell(cell[0], cell[1]) if isinstance(cell, tuple) else cell for cell in cells_list]
        elif len(cells_list) > 0 and not (hasattr(cells_list[0], 'x') and hasattr(cells_list[0], 'y')):
            # If not tuples and not Cell objects, try to convert
            class Cell:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            cells_list = [Cell(cell[0], cell[1]) if hasattr(cell, '__getitem__') else cell for cell in cells_list]
        
        # Get basic scores (coverage, balance, problems)
        # Note: evaluate_solution expects (assignment, paths, all_cells, free_cells, obstacles, grid_width, grid_height)
        results = evaluate_solution(
            self.assignment, paths_dict, cells_list, self.free_cells, 
            self.obstacles, self.grid_width, self.grid_height
        )
        self.fitness = results
        
        # Calculate coverage ratio (0 to 1, where 1 = 100% coverage)
        coverage_ratio = results['coverage_score'] / len(self.free_cells)
        coverage_term = 1 - coverage_ratio  # Convert to minimization (0 = perfect)
        
        # Get imbalance (variance of robot workloads)
        imbalance_term = results['balance_score']
        
        # Calculate penalty for breaking rules
        penalty_term = self.calculate_penalty()
        
        # Set weights (how much we care about each thing)
        w1 = 0.7  # We care 70% about coverage
        w2 = 0.3  # We care 30% about balance
        
        # If we have perfect coverage, care more about balance
        if coverage_ratio >= 1.0:
            w1 = 0.5  # Reduce coverage weight
            w2 = 0.5  # Increase balance weight
        
        # Calculate final score (lower = better)
        self.combined_score = w1 * coverage_term + w2 * imbalance_term + penalty_term
        return self.combined_score
    
    def calculate_penalty(self):
        """
        WHAT DOES THIS DO?
        - Calculates penalty for breaking rules
        - Different penalties for different rule violations
        - Higher penalty = worse solution
        """
        violations = self.fitness['problems']
        penalty = 0
        
        # Different penalties for different rule violations
        penalty_factors = {
            'out_of_bounds': 3000,    # BIG penalty: robot goes outside grid
            'obstacle_collision': 2000, # MEDIUM penalty: robot hits obstacle  
            'path_jump': 1000          # SMALL penalty: robot jumps between non-adjacent cells
        }
        
        # Check each violation and add appropriate penalty
        for violation in violations:
            if "goes outside grid" in violation:
                penalty += penalty_factors['out_of_bounds']
            elif "hits obstacle" in violation:
                penalty += penalty_factors['obstacle_collision']
            elif "jumps from" in violation:
                penalty += penalty_factors['path_jump']
        
        return penalty
    #store best solutiojn for now
    def copy(self):
        """
        WHAT DOES THIS DO?
        - Creates an exact copy of this solution
        - Needed because we don't want to accidentally change the original
        """
        return RobotCoverageSolution(
            self.assignment, self.paths, self.all_cells, 
            self.free_cells, self.obstacles, self.grid_width, self.grid_height
        )

def generate_random_solution(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots):
    """
    WHAT DOES THIS DO?
    - Creates a random starting solution for SA algorithm
    - Like throwing robots randomly on the grid
    - Divides cells equally among robots
    """
    
    # Step 1: Randomly assign free cells to robots
    assignment = []
    obstacle_assignment = [0] * num_robots  # Obstacles assigned to no robot
    
    # Shuffle free cells for random assignment
    shuffled_free_cells = copy.deepcopy(free_cells)
    random.shuffle(shuffled_free_cells)
    
    # Assign cells to robots in round-robin fashion (like dealing cards)
    free_cell_index = 0
    for cell_idx in range(len(all_cells)):
        if cell_idx in obstacles:
            assignment.append(obstacle_assignment.copy())  # Obstacle = no robot
        else:
            robot_id = free_cell_index % num_robots  # Robot 0, 1, 2, 0, 1, 2...
            robot_assignment = [0] * num_robots
            robot_assignment[robot_id] = 1  # This cell belongs to this robot
            assignment.append(robot_assignment)
            free_cell_index += 1
    
    # Step 2: Generate simple paths for each robot
    robot_paths = []
    for robot_id in range(num_robots):
        # Find cells assigned to this robot
        robot_cells = []
        for cell_idx, assignment_row in enumerate(assignment):
            if assignment_row[robot_id] == 1:
                robot_cells.append(cell_idx)
        
        # Simple path: visit cells in order (not optimal, but valid)
        robot_paths.append(robot_cells)
    
    return RobotCoverageSolution(assignment, robot_paths, all_cells, 
                               free_cells, obstacles, grid_width, grid_height)

def generate_neighbor_solution(current_solution):
    """
    WHAT DOES THIS DO?
    - Creates a slightly different solution from the current one
    - Does this by swapping cells between two robots
    - This is how SA explores new solutions
    """
    
    # Create a copy of current solution
    neighbor = current_solution.copy()
    
    # Pick two different robots to swap cells between
    num_robots = len(neighbor.assignment[0])
    robot1 = random.randint(0, num_robots - 1)
    robot2 = random.randint(0, num_robots - 1)
    
    # Make sure they're different robots
    while robot2 == robot1:
        robot2 = random.randint(0, num_robots - 1)
    
    # Find cells assigned to each robot
    robot1_cells = []
    robot2_cells = []
    
    for cell_idx, assignment_row in enumerate(neighbor.assignment):
        if assignment_row[robot1] == 1:
            robot1_cells.append(cell_idx)
        elif assignment_row[robot2] == 1:
            robot2_cells.append(cell_idx)
    
    # Make sure both robots have at least one cell
    if len(robot1_cells) == 0 or len(robot2_cells) == 0:
        return current_solution  # Can't swap, return original
    
    # Pick random cells from each robot
    cell1 = random.choice(robot1_cells)
    cell2 = random.choice(robot2_cells)
    
    # Swap the assignments (robot1 gets cell2, robot2 gets cell1)
    neighbor.assignment[cell1][robot1] = 0
    neighbor.assignment[cell1][robot2] = 1
    neighbor.assignment[cell2][robot1] = 1
    neighbor.assignment[cell2][robot2] = 0
    
    # Update paths for affected robots
    for robot_id in [robot1, robot2]:
        robot_cells = []
        for cell_idx, assignment_row in enumerate(neighbor.assignment):
            if assignment_row[robot_id] == 1:
                robot_cells.append(cell_idx)
        neighbor.paths[robot_id] = robot_cells
    
    return neighbor

def simulated_annealing(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                       initial_temp=1000, cooling_rate=0.95, max_iterations=1000):
    """
    Simulated Annealing algorithm
    Returns: (best_solution, convergence_history)
    """
    
    print(f"Starting Simulated Annealing...")
    print(f"Parameters: T0={initial_temp}, cooling_rate={cooling_rate}, iterations={max_iterations}")
    
    # Step 1: Generate random starting solution
    current_solution = generate_random_solution(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots
    )
    current_solution.evaluate()
    
    # Keep track of the best solution found so far
    best_solution = current_solution.copy()
    best_solution.evaluate()
    
    temperature = initial_temp  # Start hot (accepts bad solutions)
    
    print(f"Initial solution: Coverage={current_solution.fitness['coverage_score']}, "
          f"Balance={current_solution.fitness['balance_score']:.3f}, "
          f"Combined={current_solution.combined_score:.3f}")
    
    # Initialize convergence tracking
    convergence_history = {
        'iteration': [],
        'best_score': [],
        'current_score': [],
        'temperature': [],
        'best_coverage': [],
        'best_balance': []
    }
    
    # Initialize best_score
    best_score = best_solution.combined_score if best_solution.combined_score is not None else float('inf')
    
    # Main SA loop - try to improve solution
    for iteration in range(max_iterations):
        # Step 2: Generate neighbor solution (slight change)
        neighbor = generate_neighbor_solution(current_solution)
        neighbor.evaluate()
        
        # Step 3: Calculate if neighbor is better or worse
        delta = neighbor.combined_score - current_solution.combined_score
        
        # Step 4: Accept or reject neighbor
        if delta < 0:  # Neighbor is better - always accept
            current_solution = neighbor
            if neighbor.combined_score < best_score:
                best_solution = neighbor.copy()
                best_score = neighbor.combined_score
        else:  # Neighbor is worse - accept with probability
            if random.random() < math.exp(-delta / temperature):
                current_solution = neighbor
        
        # Track 
        
        #MODIFIED
        convergence_history['iteration'].append(iteration)
        convergence_history['best_score'].append(best_score)
        convergence_history['current_score'].append(current_solution.combined_score)
        convergence_history['temperature'].append(temperature)
        
        if best_solution.fitness:
            convergence_history['best_coverage'].append(best_solution.fitness['coverage_score'])
            convergence_history['best_balance'].append(best_solution.fitness['balance_score'])
        
        # Step 5: Cool down temperature (become more picky)
        temperature *= cooling_rate
        
        # Print progress every iteration
        current_score = current_solution.combined_score if current_solution.combined_score is not None else 0
        best_score = best_solution.combined_score if best_solution.combined_score is not None else 0
        print(f"Iteration {iteration}: T={temperature:.2f}, "
              f"Current={current_score:.3f}, "
              f"Best={best_score:.3f}")
    
    print(f"\nSA Complete!")
    
    # Make sure best solution is evaluated
    if best_solution.fitness is None:
        best_solution.evaluate()
    
    if best_solution.fitness is not None:
        print(f"Best solution: Coverage={best_solution.fitness['coverage_score']}, "
              f"Balance={best_solution.fitness['balance_score']:.3f}, "
              f"Combined={best_solution.combined_score:.3f}")
    else:
        print("Best solution evaluation failed")
    
    return best_solution, convergence_history

def print_sa_results(solution):
    """
    WHAT DOES THIS DO?
    - Prints detailed results of the SA solution
    - Shows coverage, balance, violations, assignments, and paths
    - Helps us understand how good the solution is
    """
    print("\n" + "="*60)
    print("SIMULATED ANNEALING RESULTS")
    print("="*60)
    
    if solution.fitness is None:
        print("Solution evaluation failed!")
        return
    
    print(f"Coverage Score: {solution.fitness['coverage_score']} cells covered")
    print(f"Balance Score: {solution.fitness['balance_score']:.3f} (lower = more balanced)")
    print(f"Combined Score: {solution.combined_score:.3f}")
    print(f"Robot Distances: {solution.fitness['robot_distances']}")
    
    if solution.fitness['problems']:
        print(f"\nConstraint Violations: {len(solution.fitness['problems'])}")
        for problem in solution.fitness['problems']:
            print(f"  - {problem}")
    else:
        print("\n✓ No constraint violations!")
    
    print(f"\nRobot Assignments:")
    for i, row in enumerate(solution.assignment):
        robot_id = row.index(1) if 1 in row else -1
        print(f"Cell {i}: Robot {robot_id}")
    
    print(f"\nRobot Paths:")
    for robot_id, path in enumerate(solution.paths):
        print(f"Robot {robot_id}: {path}")
