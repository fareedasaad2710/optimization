"""
Genetic Algorithm (GA) for Multi-Robot Coverage Path Planning
=============================================================

WHAT IS GENETIC ALGORITHM?
- It's like evolution - start with random solutions (population)
- Select the best ones (survival of the fittest)
- Combine them (crossover - like parents having children)
- Add random changes (mutation - like genetic mutations)
- Keep the best ones (elitism - best solutions always survive)
- Repeat until we find a good solution

THE FORMULA WE USE (Same as SA):
J = w1(1 - coverage) + w2(imbalance) + penalty

GA PARAMETERS:
- Population Size: Number of solutions in each generation
- Generations: How many times we evolve
- Crossover Rate: Probability of combining two solutions
- Mutation Rate: Probability of random changes
- Elitism: Keep best solutions in each generation
"""

import math
import random
import copy
from problem_formulation import *
from visualization import (
    plot_convergence_history,
    plot_robot_paths,
    plot_coverage_heatmap,
    save_all_figures
)

# Add this helper function after imports
def safe_format_score(score, decimals=3):
    """
    Safely format a score value, handling None and inf
    
    Args:
        score: Score value (can be None, inf, or float)
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    if score is None:
        return "N/A"
    elif score == float('inf'):
        return "inf"
    elif score == float('-inf'):
        return "-inf"
    else:
        return f"{score:.{decimals}f}"


# Simple Cell class to match problem_formulation.py expectations
class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Cell({self.x}, {self.y})"
    
    def __eq__(self, other):
        if isinstance(other, Cell):
            return self.x == other.x and self.y == other.y
        return False
    
    def __hash__(self):
        return hash((self.x, self.y))


def convert_cells_to_objects(all_cells):
    """Convert tuple cells to Cell objects if needed"""
    if len(all_cells) == 0:
        return []
    
    # Check if first cell is already a Cell object
    if hasattr(all_cells[0], 'x') and hasattr(all_cells[0], 'y'):
        return all_cells
    
    # Convert tuples to Cell objects
    result = []
    for cell in all_cells:
        if isinstance(cell, tuple):
            result.append(Cell(cell[0], cell[1]))
        else:
            result.append(cell)
    return result

class RobotCoverageSolution:
    """
    WHAT IS THIS CLASS?
    - It's like a container that holds one possible solution
    
    IMPORTANT: A SOLUTION CONSISTS OF TWO COMPONENTS:
    1. ASSIGNMENT: assignment[cell_idx][robot_id] = which robot covers which cell
       - This is a matrix: for each cell, which robot(s) it's assigned to
       - Format: list[list[int]] where assignment[i][r] = 1 means cell i is assigned to robot r
       - ✅ SAME FORMAT as problem_formulation.py and sa_algorithm.py
       - ✅ This is our DECISION VARIABLE - we cannot change this representation!
    
    2. PATHS: paths[robot_id] = [cell1, cell2, ...] = order in which robot visits its cells
       - This is a dictionary: {robot_id: [list of cells in order]}
       - Format: dict[int, list[int]] where paths[r] = [c1, c2, ...] means robot r visits c1, then c2, etc.
       - ✅ SAME FORMAT as problem_formulation.py (expects dict)
    
    CRITICAL: Assignment and Paths must be CONSISTENT!
    - If cell X is in robot R's path, then assignment[X][R] must be 1
    - If assignment[X][R] = 1, then cell X should be in robot R's path
    - Use sync_assignment_with_paths() to ensure consistency after operations
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
        # Convert paths to dict format if needed (required by evaluate_solution)
        if isinstance(self.paths, dict):
            paths_dict = self.paths.copy()
        else:
            # Convert from list to dict format
            paths_dict = {robot_id: path for robot_id, path in enumerate(self.paths)}
        
        # Ensure all paths are lists (not ints or other types)
        for robot_id in paths_dict:
            if not isinstance(paths_dict[robot_id], list):
                paths_dict[robot_id] = [paths_dict[robot_id]] if paths_dict[robot_id] is not None else []
        
        # Convert all_cells to Cell objects if needed (required by evaluate_solution)
        cells_as_objects = convert_cells_to_objects(self.all_cells)
        
        # Get basic scores (coverage, balance, problems)
        # Note: parameter order is: assignment, paths, all_cells, free_cells, obstacles, grid_width, grid_height
        results = evaluate_solution(
            self.assignment, paths_dict, cells_as_objects, 
            self.free_cells, self.obstacles, self.grid_width, self.grid_height
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
            'out_of_bounds': 1000,    # BIG penalty: robot goes outside grid
            'obstacle_collision': 500, # MEDIUM penalty: robot hits obstacle  
            'path_jump': 100          # SMALL penalty: robot jumps between non-adjacent cells
        }
        
        # Check each violation and add appropriate penalty
        for violation in violations:
            if "goes outside grid" in violation:
                penalty += penalty_factors['out_of_bounds']
            elif "hits obstacle" in violation:
                penalty += penalty_factors['obstacle_collision']
            elif "jumps from" in violation.lower() or "jump from" in violation.lower():
                penalty += penalty_factors['path_jump']
        return penalty
    
    def sync_assignment_with_paths(self):
        """
        WHAT DOES THIS DO?
        - Ensures assignment matrix matches the paths
        - After crossover/mutation, assignment might not match paths
        - This function updates assignment to reflect what's in paths
        - IMPORTANT: A solution = assignment + paths (both must be consistent)
        """
        # Clear all assignments
        for cell_idx in range(len(self.assignment)):
            for robot_id in range(len(self.assignment[cell_idx])):
                self.assignment[cell_idx][robot_id] = 0
        
        # Set assignments based on paths
        for robot_id, path in self.paths.items():
            for cell_idx in path:
                if cell_idx < len(self.assignment):
                    self.assignment[cell_idx][robot_id] = 1
    
    def copy(self):
        """Create a deep copy of this solution"""
        import copy as copy_module
        return RobotCoverageSolution(
            copy_module.deepcopy(self.assignment),
            copy_module.deepcopy(self.paths),  # ✅ Deep copy dictionary
            self.all_cells,
            self.free_cells,
            self.obstacles,
            self.grid_width,
            self.grid_height
        )

    def get_coverage_efficiency(self):
        """Performance Index 1: Coverage Efficiency (0-1, higher=better)"""
        if self.fitness is None:
            self.evaluate()
        if len(self.free_cells) == 0:
            return 0.0
        return self.fitness['coverage_score'] / len(self.free_cells)
    
    def get_workload_balance_index(self):
        """Performance Index 2: Workload Balance (0-1, higher=better)"""
        if self.fitness is None:
            self.evaluate()
        return 1.0 / (1.0 + self.fitness['balance_score'])
    
    def get_constraint_satisfaction_rate(self):
        """Performance Index 3: Constraint Satisfaction (0-1, higher=better)"""
        if self.fitness is None:
            self.evaluate()
        total_violations = sum(self.fitness['violations'].values())
        total_moves = sum(len(path) for path in self.paths.values())
        if total_moves == 0:
            return 1.0
        return max(0.0, (total_moves - total_violations) / total_moves)
    
    def get_solution_quality_index(self):
        """Performance Index 4: Overall Quality (0-1, higher=better)"""
        coverage = self.get_coverage_efficiency()
        balance = self.get_workload_balance_index()
        constraints = self.get_constraint_satisfaction_rate()
        return 0.5 * coverage + 0.3 * balance + 0.2 * constraints
    
    def get_all_performance_metrics(self):
        """Returns dictionary of all metrics for SA vs GA comparison"""
        return {
            'coverage_efficiency': self.get_coverage_efficiency(),
            'workload_balance_index': self.get_workload_balance_index(),
            'constraint_satisfaction_rate': self.get_constraint_satisfaction_rate(),
            'solution_quality_index': self.get_solution_quality_index(),
            'combined_score': self.combined_score,
            'raw_coverage': self.fitness['coverage_score'] if self.fitness else 0,
            'raw_balance': self.fitness['balance_score'] if self.fitness else float('inf'),
            'total_violations': sum(self.fitness['violations'].values()) if self.fitness else 0,
            'cells_covered': self.fitness['coverage_score'] if self.fitness else 0,
            'total_free_cells': len(self.free_cells)
        }

def generate_random_solution(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots):
    """
    Generate a random solution for robot coverage problem
    
    Returns:
        RobotCoverageSolution with:
        - assignment: list[list[int]] - which robot covers which cell
        - paths: dict[int, list[int]] - {robot_id: [cell1, cell2, ...]}
    """
    total_cells = len(all_cells)
    
    # Step 1: Create assignment matrix
    assignment = [[0 for _ in range(num_robots)] for _ in range(total_cells)]
    
    # Randomly assign each free cell to a robot
    for cell_idx in free_cells:
        robot_id = random.randint(0, num_robots - 1)
        assignment[cell_idx][robot_id] = 1
    
    # Step 2: Generate paths for each robot as DICTIONARY
    robot_paths = {}  # ✅ Must be dictionary
    
    for robot_id in range(num_robots):
        # Get cells assigned to this robot
        robot_cells = [cell_idx for cell_idx in free_cells 
                      if assignment[cell_idx][robot_id] == 1]
        
        # IMPORTANT: Shuffle to create random path order
        # Why? Because ASSIGNMENT and PATH are two different things:
        # - ASSIGNMENT tells us WHICH cells belong to this robot (e.g., [1, 2, 3, 6, 8])
        # - PATH tells us the ORDER to visit them (e.g., [1, 8, 3, 2, 6])
        # The assignment says "Robot 0 covers cells 1,2,3,6,8"
        # The path says "Robot 0 visits them in order: 1→8→3→2→6"
        # We shuffle to create random visit orders (different solutions = different orders)
        random.shuffle(robot_cells)
        
        # Store as list in dictionary
        robot_paths[robot_id] = robot_cells  # ✅ Dictionary: {0: [cells], 1: [cells]}
    
    # Step 3: Create solution object
    solution = RobotCoverageSolution(
        assignment, 
        robot_paths,  # ✅ Pass dictionary
        all_cells, 
        free_cells, 
        obstacles, 
        grid_width, 
        grid_height
    )
    
    return solution

def initialize_population(population_size, all_cells, free_cells, obstacles, grid_width, grid_height, num_robots):
    """
    WHAT DOES THIS DO?
    - Creates a population of random solutions
    - Like creating a group of random robot arrangements
    - This is generation 0 (the starting generation)
    """
    population = []
    for _ in range(population_size):
        solution = generate_random_solution(
            all_cells, free_cells, obstacles, grid_width, grid_height, num_robots
        )
        solution.evaluate()
        population.append(solution)
    
    return population

def tournament_selection(population, tournament_size=3):
    """
    WHAT DOES THIS DO?
    - Selects a solution using tournament selection
    - Like a mini-competition: pick random solutions, best one wins
    - This is how we choose "parents" for crossover
    """
    # Pick random solutions for tournament
    tournament = random.sample(population, min(tournament_size, len(population)))
    
    # Find the best one (lowest score = best)
    winner = min(tournament, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    
    return winner

def crossover(parent1, parent2, crossover_rate=0.8):
    """
    WHAT DOES THIS DO?
    - PATH-BASED CROSSOVER: Combines robot paths from two parents
    - Each robot's path comes from either parent1 or parent2
    - This preserves path structure better than cell-by-cell crossover
    
    HOW PATH-BASED CROSSOVER WORKS:
    - For each robot, randomly choose to inherit path from parent1 or parent2
    - Example: Robot 0 from Parent1, Robot 1 from Parent2, Robot 2 from Parent1
    - This maintains path continuity and feasibility
    
    CHROMOSOME REPRESENTATION:
    - Chromosome = collection of robot paths
    - Gene = one robot's complete path
    - Crossover = swapping entire paths between parents
    """
    if random.random() > crossover_rate:
        return parent1.copy()  # No crossover, return parent1
    
    # Create child solution
    child = parent1.copy()
    
    # PATH-BASED CROSSOVER: For each robot, randomly inherit path from parent1 or parent2
    num_robots = len(parent1.paths)
    
    for robot_id in range(num_robots):
        # 50% chance to inherit this robot's path from parent2
        if random.random() < 0.5:
            # Inherit from parent2
            child.paths[robot_id] = copy.deepcopy(parent2.paths[robot_id])
            
            # Update assignment matrix to match the new path
            # First, clear this robot's assignments
            for cell_idx in range(len(child.assignment)):
                child.assignment[cell_idx][robot_id] = 0
            
            # Then, assign cells from parent2's path
            for cell_idx in parent2.paths[robot_id]:
                if cell_idx < len(child.assignment):
                    child.assignment[cell_idx][robot_id] = 1
    
    # Note: This might create conflicts where a cell is assigned to multiple robots
    # The mutation step will help fix these issues
    
    return child


def crossover_ox1_path(parent1_path, parent2_path):
    """
    Standard OX1 (Order Crossover) for paths using TWO crossover points
    
    Args:
        parent1_path: List of cells (path from parent 1)
        parent2_path: List of cells (path from parent 2)
    
    Returns:
        Child path created using OX1
    """
    if len(parent1_path) < 3 or len(parent2_path) < 3:
        # If too short, use simple one-point crossover
        if len(parent1_path) < 2:
            return parent2_path.copy()
        crossover_point = random.randint(1, len(parent1_path) - 1)
        child_path = parent1_path[:crossover_point].copy()
        used_cells = set(child_path)
        for cell in parent2_path:
            if cell not in used_cells:
                child_path.append(cell)
                if len(child_path) >= len(parent1_path):
                    break
        return child_path
    
    # Standard OX1 with TWO crossover points
    path_length = min(len(parent1_path), len(parent2_path))
    
    # Select two crossover points (point1 < point2)
    point1 = random.randint(0, path_length - 2)
    point2 = random.randint(point1 + 1, path_length - 1)
    
    # Step 1: Copy segment between point1 and point2 from parent1
    child_path = [None] * path_length
    segment = parent1_path[point1:point2]
    for i in range(point1, point2):
        child_path[i] = parent1_path[i]
    
    used_cells = set(segment)
    
    # Step 2: Fill remaining positions from parent2 in order (starting after point2, wrapping)
    # Start from position after point2 in parent2
    parent2_index = point2
    child_index = point2
    
    # Fill positions after point2
    while child_index < path_length:
        if child_path[child_index] is None:
            # Find next unused cell from parent2 (starting from parent2_index)
            found = False
            for _ in range(path_length):
                cell = parent2_path[parent2_index % path_length]
                if cell not in used_cells:
                    child_path[child_index] = cell
                    used_cells.add(cell)
                    found = True
                    break
                parent2_index += 1
            if not found:
                break
        child_index += 1
    
    # Fill positions before point1 (wrapping)
    child_index = 0
    while child_index < point1:
        if child_path[child_index] is None:
            found = False
            for _ in range(path_length):
                cell = parent2_path[parent2_index % path_length]
                if cell not in used_cells:
                    child_path[child_index] = cell
                    used_cells.add(cell)
                    found = True
                    break
                parent2_index += 1
            if not found:
                break
        child_index += 1
    
    # Fill any remaining None values with unused cells from parent1
    for i in range(path_length):
        if child_path[i] is None:
            for cell in parent1_path:
                if cell not in used_cells:
                    child_path[i] = cell
                    used_cells.add(cell)
                    break
    
    return child_path

def crossover_ox1_assignment(parent1_assignment, parent2_assignment, num_robots):
    """
    Standard OX1 (Order Crossover) for assignment matrix
    
    For each cell, we crossover which robot it's assigned to using OX1
    
    Args:
        parent1_assignment: Assignment matrix from parent 1
        parent2_assignment: Assignment matrix from parent 2
        num_robots: Number of robots
    
    Returns:
        Child assignment matrix
    """
    num_cells = len(parent1_assignment)
    child_assignment = [[0] * num_robots for _ in range(num_cells)]
    
    if num_cells < 3:
        # If too short, use simple one-point crossover
        crossover_point = num_cells // 2
        for i in range(crossover_point):
            child_assignment[i] = parent1_assignment[i].copy()
        for i in range(crossover_point, num_cells):
            child_assignment[i] = parent2_assignment[i].copy()
        return child_assignment
    
    # Standard OX1 with TWO crossover points
    point1 = random.randint(0, num_cells - 2)
    point2 = random.randint(point1 + 1, num_cells - 1)
    
    # Step 1: Copy segment between point1 and point2 from parent1
    for i in range(point1, point2):
        child_assignment[i] = parent1_assignment[i].copy()
    
    # Step 2: Fill remaining from parent2 (starting after point2, wrapping)
    parent2_index = point2
    
    # Fill positions after point2
    for i in range(point2, num_cells):
        child_assignment[i] = parent2_assignment[parent2_index % num_cells].copy()
        parent2_index += 1
    
    # Fill positions before point1 (wrapping)
    for i in range(point1):
        child_assignment[i] = parent2_assignment[parent2_index % num_cells].copy()
        parent2_index += 1
    
    return child_assignment

def crossover_order_based(parent1, parent2):
    """
    CROSSOVER with random selection: Assignment OR Path
    Uses Standard OX1 (two crossover points) for both
    
    - Random number 0 or 1:
      - 0: Crossover ASSIGNMENT using OX1
      - 1: Crossover PATH using OX1
    
    NOTE: Crossover always happens (no probability check)
    """
    child = parent1.copy()
    num_robots = len(parent1.paths)
    
    # Generate random number: 0 = crossover assignment, 1 = crossover path
    crossover_type = random.randint(0, 1)
    
    # Store crossover type in child for verbose output
    child._crossover_type = crossover_type
    
    if crossover_type == 0:
        # CROSSOVER ASSIGNMENT using OX1
        child.assignment = crossover_ox1_assignment(
            parent1.assignment, 
            parent2.assignment, 
            num_robots
        )
        
        # Update paths based on new assignment
        for robot_id in range(num_robots):
            robot_cells = []
            for cell_idx in range(len(child.assignment)):
                if child.assignment[cell_idx][robot_id] == 1:
                    robot_cells.append(cell_idx)
            # Keep order from parent1 if possible, otherwise random
            if robot_id in parent1.paths:
                # Try to preserve order from parent1
                ordered_cells = []
                for cell in parent1.paths[robot_id]:
                    if cell in robot_cells:
                        ordered_cells.append(cell)
                # Add any missing cells
                for cell in robot_cells:
                    if cell not in ordered_cells:
                        ordered_cells.append(cell)
                child.paths[robot_id] = ordered_cells
            else:
                child.paths[robot_id] = robot_cells.copy()
        
    else:
        # CROSSOVER PATH using OX1 (standard two-point OX1)
        for robot_id in range(num_robots):
            # Validate robot exists in both parents
            if robot_id not in parent1.paths or robot_id not in parent2.paths:
                continue
            
            parent1_path = parent1.paths[robot_id]
            parent2_path = parent2.paths[robot_id]
            
            # Skip if paths too short
            if len(parent1_path) < 2 or len(parent2_path) < 2:
                continue
            
            # Use standard OX1 (two crossover points)
            child_path = crossover_ox1_path(parent1_path, parent2_path)
            child.paths[robot_id] = child_path
            
            # Update assignment to match new path
            for cell_idx in range(len(child.assignment)):
                child.assignment[cell_idx][robot_id] = 0
            
            for cell_idx in child_path:
                if cell_idx < len(child.assignment):
                    child.assignment[cell_idx][robot_id] = 1
    
    # Ensure assignment and paths are fully synchronized
    child.sync_assignment_with_paths()
    
    return child


# Replace apply_crossover (around line 470):

def apply_crossover(parent1, parent2):
    """
    Apply ONE-POINT ORDER-BASED CROSSOVER (OX1)
    
    This is the only crossover method used in this implementation.
    Standard for path planning problems.
    
    NOTE: Crossover always happens (no probability check)
    """
    return crossover_order_based(parent1, parent2)
def mutate_robot_path(solution, robot_id):
    """
    Mutate a specific robot's path by swapping two cells
    
    Args:
        solution: RobotCoverageSolution to mutate
        robot_id: Which robot's path to mutate (0 to num_robots-1)
    
    Returns:
        True if mutation was applied, False otherwise
    """
    if robot_id not in solution.paths:
        return False
    
    path = solution.paths[robot_id]
    
    # Only mutate if path has more than 2 cells
    if len(path) <= 2:
        return False  # Path too short to swap
    
    # Pick two random positions (ensure they're different)
    pos1 = random.randint(0, len(path) - 1)
    pos2 = random.randint(0, len(path) - 1)
    # If same position, pick a different one
    while pos2 == pos1 and len(path) > 1:
        pos2 = random.randint(0, len(path) - 1)
    
    # Create a copy of the path to ensure mutation is detected
    new_path = path.copy()
    
    # Swap the two positions
    new_path[pos1], new_path[pos2] = new_path[pos2], new_path[pos1]
    solution.paths[robot_id] = new_path  # Assign new list object
    
    # Sync assignment with paths
    solution.sync_assignment_with_paths()
    
    return True  # Mutation succeeded

def mutate(solution, mutation_rate=0.1):
    """
    WHAT DOES THIS DO?
    - PATH-BASED MUTATION: Modifies robot paths directly
    - Mutation Type: Swap two cells within same robot's path (path reordering)
    - This mutation only changes the order of cells in a robot's path, not which cells are assigned
    """
    if random.random() > mutation_rate:
        return solution  # No mutation
    
    num_robots = len(solution.paths)
    if num_robots == 0:
        return solution  # No robots, can't mutate
    
    # MUTATION: Swap two positions in a robot's path
    # Find robots with paths of length >= 2 (need at least 2 cells to swap)
    valid_robots = [r for r in range(num_robots) if len(solution.paths[r]) >= 2]
    
    if not valid_robots:
        return solution  # No valid robots to mutate
    
    # Pick a random robot with a valid path
    robot_id = random.choice(valid_robots)
    
    # Use the new mutate_robot_path function
    mutate_robot_path(solution, robot_id)
    
    return solution  # Mutation succeeded


def mutate_path_swap(solution, mutation_rate=0.1):
    """
    ALTERNATIVE MUTATION: Simple path segment swap
    - Swaps a segment of path between two robots
    - Good for exploring different workload distributions
    """
    if random.random() > mutation_rate:
        return solution
    
    num_robots = len(solution.paths)
    if num_robots < 2:
        return solution
    
    # Select two random robots
    robot1 = random.randint(0, num_robots - 1)
    robot2 = random.randint(0, num_robots - 1)
    while robot2 == robot1:
        robot2 = random.randint(0, num_robots - 1)
    
    path1 = solution.paths[robot1]
    path2 = solution.paths[robot2]
    
    if len(path1) >= 2 and len(path2) >= 2:
        # Select random segments
        seg1_start = random.randint(0, len(path1) - 2)
        seg1_end = random.randint(seg1_start + 1, len(path1))
        
        seg2_start = random.randint(0, len(path2) - 2)
        seg2_end = random.randint(seg2_start + 1, len(path2))
        
        # Swap segments
        segment1 = path1[seg1_start:seg1_end]
        segment2 = path2[seg2_start:seg2_end]
        
        new_path1 = path1[:seg1_start] + segment2 + path1[seg1_end:]
        new_path2 = path2[:seg2_start] + segment1 + path2[seg2_end:]
        
        solution.paths[robot1] = new_path1
        solution.paths[robot2] = new_path2
        
        # Update assignments
        for cell_idx in range(len(solution.assignment)):
            solution.assignment[cell_idx][robot1] = 0
            solution.assignment[cell_idx][robot2] = 0
        
        for cell in new_path1:
            if cell < len(solution.assignment):
                solution.assignment[cell][robot1] = 1
        
        for cell in new_path2:
            if cell < len(solution.assignment):
                solution.assignment[cell][robot2] = 1
    
    return solution

def genetic_algorithm(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                      population_size=5, generations=100, 
                      verbose=True, selection_percentage=0.10, 
                      crossover_percentage=0.80, mutation_percentage=0.10):
    """
    Genetic Algorithm for robot coverage problem
    
    Args:
        verbose: If True, prints detailed logs of GA operations
    """
    
    print(f"\n{'='*70}")
    print(f"🧬 STARTING GENETIC ALGORITHM")
    print(f"{'='*70}")
    print(f"📋 Parameters:")
    print(f"   • Population Size:    {population_size}")
    print(f"   • Generations:        {generations}")
    print(f"   • Population Strategy:")
    print(f"     - Selection (Elite): {selection_percentage*100:.0f}%")
    print(f"     - Crossover:         {crossover_percentage*100:.0f}%")
    print(f"     - Mutation:          {mutation_percentage*100:.0f}%")
    print(f"{'='*70}\n")
    
    # Step 1: Initialize population (generation 0)
    print(f"🔄 STEP 1: Initializing Population (Generation 0)")
    print(f"   Creating {population_size} random solutions...")
    
    population = initialize_population(
        population_size, all_cells, free_cells, obstacles, grid_width, grid_height, num_robots
    )
    
    # Find best solution in initial population
    best_solution = min(population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    best_solution = best_solution.copy()
    best_score = best_solution.combined_score if best_solution.combined_score is not None else float('inf')
    
    # Calculate initial statistics
    initial_scores = [s.combined_score if s.combined_score is not None else float('inf') for s in population]
    initial_avg = sum(initial_scores) / len(initial_scores)
    initial_worst = max(initial_scores)
    
    print(f"   ✅ Population initialized!")
    print(f"   📊 Initial Statistics:")
    print(f"      • Best Score:     {safe_format_score(best_score)}")
    print(f"      • Average Score:  {safe_format_score(initial_avg)}")
    print(f"      • Worst Score:    {safe_format_score(initial_worst)}")
    
    if best_solution.fitness is not None:
        print(f"      • Coverage:       {best_solution.fitness['coverage_score']}/{len(free_cells)} cells")
        print(f"      • Balance:        {best_solution.fitness['balance_score']:.3f}")
    print()
    
    # Track convergence history
    convergence_history = {
        'generation': [],
        'best_score': [],
        'avg_score': [],
        'worst_score': [],
        'best_coverage': [],
        'best_balance': [],
        'crossover_count': [],
        'mutation_count': []
    }
    
    # Counters for operations
    total_crossovers = 0
    total_mutations = 0
    total_selections = 0
    
    # Main GA loop - evolve for multiple generations
    for generation in range(generations):
        
        if verbose and (generation % 10 == 0 or generation < 3):
            print(f"\n{'─'*70}")
            print(f"🔄 GENERATION {generation}")
            print(f"{'─'*70}")
        
        new_population = []
        gen_crossovers = 0
        gen_mutations = 0
        gen_selections = 0
        
        # Step 2: Create new generation with percentages
        # Configurable: Selection (Elitism), Crossover, Mutation percentages
        # Default: 10% Selection (Elitism), 80% Crossover, 10% Mutation
        
        num_selection = int(population_size * selection_percentage)
        num_crossover = int(population_size * crossover_percentage)
        num_mutation = population_size - num_selection - num_crossover  # Remaining for mutation
        
        if verbose and generation < 3:
            print(f"   🧬 Creating new population:")
            print(f"      • {num_selection} solutions via Selection (Elitism) ({selection_percentage*100:.0f}%)")
            print(f"      • {num_crossover} solutions via Crossover ({crossover_percentage*100:.0f}%)")
            print(f"      • {num_mutation} solutions via Mutation ({mutation_percentage*100:.0f}%)")
        
        import copy as copy_module
        
        # 1. SELECTION (10%): Use ELITISM - directly copy best solutions from old population
        # Sort population by score (best = lowest score)
        sorted_population = sorted(population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
        
        for i in range(num_selection):
            # Copy the i-th best solution (elitism)
            elite_solution = sorted_population[i]
            new_population.append(elite_solution.copy())
            gen_selections += 1
            if verbose and generation < 2 and i < 2:
                print(f"      Selection {i+1} (Elitism): Copied {i+1}-th best solution (score: {safe_format_score(elite_solution.combined_score)})")
        
        # 2. CROSSOVER (80%): Create offspring through crossover
        for i in range(num_crossover):
            # Select parents using tournament selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            if verbose and generation < 2 and i < 2:
                print(f"      Crossover {i+1}:")
                print(f"         • Parent 1 (score: {safe_format_score(parent1.combined_score)})")
                print(f"         • Parent 2 (score: {safe_format_score(parent2.combined_score)})")
            
            # Create child through crossover (randomly: assignment or path)
            # NOTE: Crossover always happens (no probability check) - count all attempts
            child = apply_crossover(parent1, parent2)
            
            # Always count crossover attempt (crossover always happens now)
            gen_crossovers += 1
            
            # Check if crossover actually changed something (for verbose output only)
            did_change = (child.paths != parent1.paths) or (child.assignment != parent1.assignment)
            
            if verbose and generation < 2 and i < 2:
                # Show which type of crossover was used
                crossover_type = getattr(child, '_crossover_type', None)
                if crossover_type == 0:
                    print(f"         • Random number: 0 → Crossover ASSIGNMENT using OX1 (two points)")
                elif crossover_type == 1:
                    print(f"         • Random number: 1 → Crossover PATH using OX1 (two points)")
                else:
                    # Fallback: determine by what changed
                    assignment_changed = (child.assignment != parent1.assignment)
                    path_changed = (child.paths != parent1.paths)
                    if assignment_changed and not path_changed:
                        print(f"         • ✂️  OX1 Crossover applied: ASSIGNMENT")
                    elif path_changed and not assignment_changed:
                        print(f"         • ✂️  OX1 Crossover applied: PATH")
                    else:
                        print(f"         • ✂️  OX1 Crossover applied: ASSIGNMENT & PATH")
                
                if not did_change:
                    print(f"         • ⚠️  Note: Crossover applied but result identical to Parent 1")
            
            # Evaluate child
            child.evaluate()
            
            if verbose and generation < 2 and i < 2:
                print(f"         • 📊 Child score: {safe_format_score(child.combined_score)}")
            
            # Add to new population
            new_population.append(child)
        
        # 3. MUTATION (10%): Mutate worst solutions
        # Sort population by score (worst = highest score, best = lowest score)
        sorted_population_by_worst = sorted(population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'), reverse=True)
        
        for i in range(num_mutation):
            # Pick the (i+1)-th worst solution (worst = index 0, second worst = index 1, etc.)
            worst_solution = sorted_population_by_worst[i]
            mutated = worst_solution.copy()
            
            # Save paths before mutation for comparison
            paths_before = copy_module.deepcopy(mutated.paths)
            
            # Step 1: Generate random robot ID (0 to num_robots-1)
            num_robots = len(mutated.paths)
            robot_id = random.randint(0, num_robots - 1)
            
            # Step 2: Mutate that robot's path (if path length > 2)
            did_mutate = mutate_robot_path(mutated, robot_id)
            
            if did_mutate:
                gen_mutations += 1
                if verbose and generation < 2 and i < 2:
                    print(f"      Mutation {i+1}:")
                    print(f"         • Selected: {i+1}-th worst solution (score: {safe_format_score(worst_solution.combined_score)})")
                    print(f"         • Random robot selected: Robot {robot_id}")
                    print(f"         • Robot {robot_id} path before: {paths_before.get(robot_id, [])}")
                    print(f"         • Robot {robot_id} path after:  {mutated.paths.get(robot_id, [])}")
                    print(f"         • 🧪 Mutation applied (swapped two positions in Robot {robot_id}'s path)!")
            else:
                # If mutation failed (path too short), try another robot
                valid_robots = [r for r in range(num_robots) if len(mutated.paths.get(r, [])) > 2]
                if valid_robots:
                    robot_id = random.choice(valid_robots)
                    did_mutate = mutate_robot_path(mutated, robot_id)
                    if did_mutate:
                        gen_mutations += 1
                        if verbose and generation < 2 and i < 2:
                            print(f"      Mutation {i+1}:")
                            print(f"         • Selected: {i+1}-th worst solution (score: {safe_format_score(worst_solution.combined_score)})")
                            print(f"         • Robot {robot_id} path mutated (original robot had path length <= 2)")
                            print(f"         • 🧪 Mutation applied!")
            
            # Evaluate mutated solution
            mutated.evaluate()
            
            if verbose and generation < 2 and i < 2:
                print(f"         • 📊 Mutated score: {safe_format_score(mutated.combined_score)}")
            
            # Add to new population
            new_population.append(mutated)
        
        total_crossovers += gen_crossovers
        total_mutations += gen_mutations
        total_selections += gen_selections
        
        if verbose and generation < 3:
            print(f"   📊 Generation {generation} Operations:")
            print(f"      • Selections:  {gen_selections}/{num_selection} (10% - Elitism)")
            print(f"      • Crossovers:  {gen_crossovers}/{num_crossover} (80%)")
            print(f"      • Mutations:   {gen_mutations}/{num_mutation} (10%)")
        
        # Step 4: Update population
        population = new_population
        
        # Step 5: Find best solution in current generation
        current_best = min(population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
        current_best_score = current_best.combined_score if current_best.combined_score is not None else float('inf')
        
        # Update global best if needed
        improvement = False
        if current_best_score < best_score:
            old_best = best_score
            best_solution = current_best.copy()
            best_score = current_best_score
            improvement = True
            
            if verbose and generation % 10 == 0:
                print(f"   🎉 NEW BEST SOLUTION FOUND!")
                print(f"      • Old Best: {safe_format_score(old_best)}")
                print(f"      • New Best: {safe_format_score(best_score)}")
                print(f"      • Improvement: {safe_format_score(old_best - best_score)}")
        
        # Record metrics for this generation
        scores = [sol.combined_score if sol.combined_score is not None else float('inf') 
                  for sol in population]
        convergence_history['generation'].append(generation)
        convergence_history['best_score'].append(min(scores))
        convergence_history['avg_score'].append(sum(scores) / len(scores))
        convergence_history['worst_score'].append(max(scores))
        convergence_history['crossover_count'].append(gen_crossovers)
        convergence_history['mutation_count'].append(gen_mutations)
        
        if best_solution.fitness:
            convergence_history['best_coverage'].append(best_solution.fitness['coverage_score'])
            convergence_history['best_balance'].append(best_solution.fitness['balance_score'])
        else:
            convergence_history['best_coverage'].append(0)
            convergence_history['best_balance'].append(float('inf'))
        
        # Print progress
        if generation % 10 == 0 or generation == generations - 1:
            valid_scores = [s for s in scores if s != float('inf')]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            
            status_symbol = "🎉" if improvement else "📊"
            print(f"\n   {status_symbol} Generation {generation} Summary:")
            print(f"      • Best Score:     {safe_format_score(best_score)}")
            print(f"      • Current Best:   {safe_format_score(current_best_score)}")
            print(f"      • Average:        {safe_format_score(avg_score)}")
            print(f"      • Worst:          {safe_format_score(max(scores))}")
            if best_solution.fitness:
                print(f"      • Coverage:       {best_solution.fitness['coverage_score']}/{len(free_cells)} cells")
                print(f"      • Balance:        {best_solution.fitness['balance_score']:.3f}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ GENETIC ALGORITHM COMPLETE!")
    print(f"{'='*70}")
    print(f"📊 Final Statistics:")
    print(f"   • Total Generations:      {generations}")
    print(f"   • Total Selections:        {total_selections} (10% per generation - Elitism)")
    print(f"   • Total Crossovers:        {total_crossovers} (80% per generation)")
    print(f"   • Total Mutations:         {total_mutations} (10% per generation)")
    print(f"   • Best Score Achieved:     {safe_format_score(best_score)}")
    
    if best_solution.fitness is not None:
        print(f"   • Final Coverage:         {best_solution.fitness['coverage_score']}/{len(free_cells)} cells ({best_solution.fitness['coverage_score']/len(free_cells)*100:.1f}%)")
        print(f"   • Final Balance:          {best_solution.fitness['balance_score']:.3f}")
        print(f"   • Constraint Violations:  {len(best_solution.fitness['problems'])}")
    
    print(f"{'='*70}\n")
    
    # Ensure best solution is evaluated
    if best_solution.fitness is None:
        best_solution.evaluate()
    
    # After the main GA loop ends
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. Plot convergence history
    plot_convergence_history(
        convergence_history,
        title=f"GA Convergence - Pop: {population_size}, Gen: {generations}",
        save_path="results/figures/ga_convergence.png"
    )

    # 2. Visualize best solution's robot paths
    if best_solution.fitness:
        plot_robot_paths(
            best_solution,
            grid_size=(grid_height, grid_width),
            title=f"Best Solution - Score: {safe_format_score(best_score)}",
            save_path="results/figures/ga_best_solution.png"
        )
        
        # 3. Plot coverage heatmap
        plot_coverage_heatmap(
            best_solution,
            grid_size=(grid_height, grid_width),
            title="Coverage Distribution",
            save_path="results/figures/ga_coverage_heatmap.png"
        )

    # 4. Plot GA-specific metrics
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot crossover and mutation counts
    axes[0, 0].plot(convergence_history['generation'], 
                    convergence_history['crossover_count'], 
                    label='Crossovers', marker='o')
    axes[0, 0].plot(convergence_history['generation'], 
                    convergence_history['mutation_count'], 
                    label='Mutations', marker='s')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Genetic Operations per Generation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot selection count (elite solutions copied)
    selection_count = [int(population_size * selection_percentage) for _ in convergence_history['generation']]
    axes[0, 1].plot(convergence_history['generation'], 
                    selection_count, 
                    color='purple', marker='D')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Elite Count')
    axes[0, 1].set_title(f'Elite Solutions Preserved ({selection_percentage*100:.0f}% Selection)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot coverage and balance scores
    axes[1, 0].plot(convergence_history['generation'], 
                    convergence_history['best_coverage'], 
                    label='Coverage', color='green', marker='o')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Coverage Score')
    axes[1, 0].set_title('Best Coverage Score Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(convergence_history['generation'], 
                    convergence_history['best_balance'], 
                    label='Balance', color='orange', marker='s')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Balance Score')
    axes[1, 1].set_title('Best Balance Score Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/figures/ga_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ All visualizations saved to results/figures/")

    # Return results with convergence history
    return {
        'best_solution': best_solution,
        'best_score': best_score,
        'convergence_history': convergence_history,
        'final_population': population
    }

def print_ga_results(solution):
    """
    WHAT DOES THIS DO?
    - Prints detailed results of GA solution
    - Shows coverage, balance, violations, assignments, and paths
    - Helps us understand how good the solution is
    """
    print("\n" + "="*60)
    print("GENETIC ALGORITHM RESULTS")
    print("="*60)
    
    if solution.fitness is None:
        print("Solution evaluation failed!")
        return
    
    print(f"Coverage Score: {solution.fitness['coverage_score']} cells covered")
    print(f"Balance Score: {solution.fitness['balance_score']:.3f} (lower = more balanced)")
    print(f"Combined Score: {safe_format_score(solution.combined_score)}")
    print(f"Total Distance: {solution.fitness.get('total_distance', 0):.3f}")
    print(f"Max Distance: {solution.fitness.get('max_distance', 0):.3f}")
    
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
    if isinstance(solution.paths, dict):
        for robot_id, path in solution.paths.items():
            print(f"Robot {robot_id}: {path}")
    else:
        for robot_id, path in enumerate(solution.paths):
            print(f"Robot {robot_id}: {path}")

def test_ga_parameters(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots, 
                       test_name="Parameter Sensitivity Test"):
    """
    Test GA with different parameter configurations
    REQUIRED for Milestone 4 - Parameter Analysis
    """
    results = {}
    
    print(f"\n{'='*70}")
    print(f"GA PARAMETER SENSITIVITY ANALYSIS: {test_name}")
    print(f"{'='*70}")
    
    # Test 1: Population Size Effect
    print("\n[TEST 1/4] Population Size Effect")
    print("-" * 70)
    for pop_size in [20, 50, 100]:
        print(f"  → Running with population_size={pop_size}...")
        solution, history = genetic_algorithm(
            all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
            population_size=pop_size, generations=50
        )
        results[f'pop_{pop_size}'] = {
            'solution': solution,
            'history': history,
            'metrics': solution.get_all_performance_metrics(),
            'parameter': 'population_size',
            'value': pop_size
        }
        print(f"     ✓ Best Score: {safe_format_score(solution.combined_score)}")
    
    # Test 2: Mutation Rate Effect
    print("\n[TEST 2/4] Mutation Rate Effect")
    print("-" * 70)
    # Note: mutation_rate parameter removed - mutation is controlled by mutation_percentage
    # Testing different mutation_percentage values instead
    for mut_pct in [0.05, 0.1, 0.2, 0.3]:
        print(f"  → Running with mutation_percentage={mut_pct}...")
        solution, history = genetic_algorithm(
            all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
            population_size=5, generations=50, mutation_percentage=mut_pct
        )
        results[f'mut_{mut_pct}'] = {
            'solution': solution,
            'history': history,
            'metrics': solution.get_all_performance_metrics(),
            'parameter': 'mutation_percentage',
            'value': mut_pct
        }
        print(f"     ✓ Best Score: {safe_format_score(solution.combined_score)}")
    
    # Test 3: Crossover Rate Effect (REMOVED - crossover always happens now)
    # Note: Crossover rate parameter removed - crossover always occurs for all crossover attempts
    print("\n[TEST 3/4] Crossover Percentage Effect")
    print("-" * 70)
    print("  Note: Crossover always happens (no probability check)")
    print("  Testing different crossover_percentage values instead...")
    for cross_pct in [0.6, 0.8, 0.95]:
        print(f"  → Running with crossover_percentage={cross_pct}...")
        solution, history = genetic_algorithm(
            all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
            population_size=5, generations=50, crossover_percentage=cross_pct
        )
        results[f'cross_{cross_pct}'] = {
            'solution': solution,
            'history': history,
            'metrics': solution.get_all_performance_metrics(),
            'parameter': 'crossover_percentage',
            'value': cross_pct
        }
        print(f"     ✓ Best Score: {safe_format_score(solution.combined_score)}")
    
    # Test 4: Generation Count Effect
    print("\n[TEST 4/4] Generation Count Effect")
    print("-" * 70)
    for gens in [25, 50, 100, 200]:
        print(f"  → Running with generations={gens}...")
        solution, history = genetic_algorithm(
            all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
            population_size=5, generations=gens
        )
        results[f'gen_{gens}'] = {
            'solution': solution,
            'history': history,
            'metrics': solution.get_all_performance_metrics(),
            'parameter': 'generations',
            'value': gens
        }
        print(f"     ✓ Best Score: {safe_format_score(solution.combined_score)}")
    
    print(f"\n{'='*70}")
    print("PARAMETER TESTING COMPLETE!")
    print(f"{'='*70}\n")
    
    return results

def analyze_convergence(convergence_history):
    """
    Analyze convergence behavior - REQUIRED for milestone report
    Returns metrics about how the algorithm converged
    """
    best_scores = convergence_history['best_score']
    
    analysis = {
        'total_generations': len(best_scores),
        'initial_score': best_scores[0] if best_scores else None,
        'final_score': best_scores[-1] if best_scores else None,
        'best_score_ever': min(best_scores) if best_scores else None,
        'converged_at_generation': None,
        'total_improvement': None,
        'improvement_percentage': None,
        'avg_improvement_per_gen': None,
        'stagnation_generations': 0
    }
    
    if len(best_scores) > 1:
        initial = best_scores[0]
        final = best_scores[-1]
        analysis['total_improvement'] = initial - final
        
        if initial != 0:
            analysis['improvement_percentage'] = ((initial - final) / initial) * 100
        
        analysis['avg_improvement_per_gen'] = analysis['total_improvement'] / len(best_scores)
        
        # Find convergence point (no improvement for 10 consecutive generations)
        threshold = 1e-6
        for i in range(10, len(best_scores)):
            recent_scores = best_scores[i-10:i]
            if max(recent_scores) - min(recent_scores) < threshold:
                analysis['converged_at_generation'] = i - 10
                analysis['stagnation_generations'] = len(best_scores) - (i - 10)
                break
    
    return analysis

def print_solution_summary(solution, convergence_history=None, algorithm_name="GA"):
    """
    Print comprehensive solution summary for milestone report
    """
    print("\n" + "="*70)
    print(f"{algorithm_name} SOLUTION SUMMARY")
    print("="*70)
    
    metrics = solution.get_all_performance_metrics()
    
    print(f"\n📊 PERFORMANCE METRICS (for SA vs GA Comparison):")
    print(f"  • Coverage Efficiency:          {metrics['coverage_efficiency']:.2%}")
    print(f"  • Workload Balance Index:       {metrics['workload_balance_index']:.4f}")
    print(f"  • Constraint Satisfaction:      {metrics['constraint_satisfaction_rate']:.2%}")
    print(f"  • Solution Quality Index:       {metrics['solution_quality_index']:.4f}")
    print(f"  • Combined Score (minimize):    {safe_format_score(metrics['combined_score'])}")
    
    print(f"\n📈 RAW SCORES:")
    print(f"  • Coverage:                     {metrics['cells_covered']}/{metrics['total_free_cells']} cells")
    print(f"  • Balance (variance):           {metrics['raw_balance']:.4f}")
    print(f"  • Total Violations:             {metrics['total_violations']}")
    
    if convergence_history:
        analysis = analyze_convergence(convergence_history)
        print(f"\n🔄 CONVERGENCE ANALYSIS:")
        print(f"  • Total Generations:            {analysis['total_generations']}")
        print(f"  • Initial Score:                {safe_format_score(analysis['initial_score'])}")
        print(f"  • Final Score:                  {safe_format_score(analysis['final_score'])}")
        print(f"  • Best Score Ever:              {safe_format_score(analysis['best_score_ever'])}")
        
        if analysis['converged_at_generation'] is not None:
            print(f"  • Converged at Generation:      {analysis['converged_at_generation']}")
            print(f"  • Stagnation Generations:       {analysis['stagnation_generations']}")
        else:
            print(f"  • Convergence Status:           Still improving")
        
        if analysis['improvement_percentage'] is not None:
            print(f"  • Total Improvement:            {analysis['improvement_percentage']:.2f}%")
            print(f"  • Avg Improvement/Gen:          {analysis['avg_improvement_per_gen']:.6f}")
    
    print("="*70 + "\n")

def compare_parameter_results(results):
    """
    Compare results from parameter sensitivity testing
    Generates summary for milestone report
    """
    print("\n" + "="*70)
    print("PARAMETER SENSITIVITY ANALYSIS SUMMARY")
    print("="*70)
    
    # Group by parameter type
    param_groups = {}
    for key, value in results.items():
        param_type = value['parameter']
        if param_type not in param_groups:
            param_groups[param_type] = []
        param_groups[param_type].append((value['value'], value['metrics']))
    
    # Print analysis for each parameter
    for param_name, param_data in param_groups.items():
        print(f"\n📊 {param_name.upper()} EFFECT:")
        print("-" * 70)
        print(f"{'Value':<15} {'Coverage':>12} {'Balance':>12} {'Quality':>12} {'Score':>12}")
        print("-" * 70)
        
        # Sort by parameter value
        param_data.sort(key=lambda x: x[0])
        
        for value, metrics in param_data:
            print(f"{str(value):<15} "
                  f"{metrics['coverage_efficiency']:>11.2%} "
                  f"{metrics['workload_balance_index']:>12.4f} "
                  f"{metrics['solution_quality_index']:>12.4f} "
                  f"{safe_format_score(metrics['combined_score']):>12}")
        
        # Find best value
        best_entry = min(param_data, key=lambda x: x[1]['combined_score'])
        print(f"\n  ✓ Best {param_name}: {best_entry[0]} (Score: {safe_format_score(best_entry[1]['combined_score'])})")
    
    print("\n" + "="*70 + "\n")

def analyze_parameter_effects(results, parameter_name, save_path=None):
    """
    Analyze the effect of a specific GA parameter on performance.

    Args:
        results: List of dictionaries containing results for different parameter values.
        parameter_name: Name of the parameter being analyzed.
        save_path: Optional path to save the figure.
    """
    import matplotlib.pyplot as plt

    parameter_values = [r[parameter_name] for r in results]
    scores = [r['best_score'] for r in results]
    coverage = [r['coverage_efficiency'] for r in results]
    balance = [r['workload_balance_index'] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel(parameter_name)
    ax1.set_ylabel('Best Score', color='tab:blue')
    ax1.plot(parameter_values, scores, 'o-', color='tab:blue', label='Best Score')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Performance Metrics', color='tab:green')
    ax2.plot(parameter_values, coverage, 's-', color='tab:green', label='Coverage Efficiency')
    ax2.plot(parameter_values, balance, 'd-', color='tab:orange', label='Workload Balance')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Parameter analysis plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

# Example usage
if __name__ == "__main__":
    # Configuration
    grid_width = 3
    grid_height = 3
    num_robots = 2
    obstacles = [4]  # Cell 4 is an obstacle
    
    print("Multi-Robot Coverage Path Planning with Genetic Algorithm")
    print("="*60)
    
    total_cells = grid_width * grid_height
    free_cells = [i for i in range(total_cells) if i not in obstacles]
    
    # Create grid cells
    all_cells = create_grid_cells(grid_width, grid_height)
    
    print(f"Grid: {grid_width}x{grid_height}, Robots: {num_robots}")
    print(f"Total cells: {total_cells}, Free cells: {len(free_cells)}, Obstacles: {len(obstacles)}")
    
###########################################################
    # Test 1: Population Initialization
    print("\n" + "="*60)
    print("TEST 1: Population Initialization")
    print("="*60)
    
    population = initialize_population(
        population_size=5,  # Small size for testing
        all_cells=all_cells,
        free_cells=free_cells,
        obstacles=obstacles,
        grid_width=grid_width,
        grid_height=grid_height,
        num_robots=num_robots
    )
    
    # Check results:
    print(f"✓ Population size: {len(population)} (expected: 5)")
    print(f"✓ All solutions evaluated: {all(sol.fitness is not None for sol in population)}")
    
    # Check diversity (solutions should be different)
    # What this checks: Do different solutions have Robot 0 covering different SETS of cells?
    # Example: Solution 1: Robot 0 covers {1,2,3}, Solution 2: Robot 0 covers {4,5,6} = diverse
    #          Solution 1: Robot 0 covers {1,2,3}, Solution 2: Robot 0 covers {1,2,3} = not diverse
    path_sets = [set(sol.paths[0]) for sol in population if 0 in sol.paths]
    unique_paths = len(set(tuple(sorted(p)) for p in path_sets))
    print(f"✓ Solutions are diverse: {unique_paths} unique Robot 0 paths")
    print(f"   → This means Robot 0 covers {unique_paths} different SETS of cells across {len(population)} solutions")
    print(f"   → (We ignore the ORDER of cells, only check which cells are covered)")
    
    # Print detailed information for ALL solutions
    print("\n" + "-"*60)
    print("DETAILED SOLUTION INFORMATION:")
    print("-"*60)
    
    for idx, sol in enumerate(population):
        print(f"\n🔹 SOLUTION {idx + 1}:")
        print(f"   {'='*60}")
        print(f"   COMPONENT 1: ASSIGNMENT (which robot covers which cell)")
        print(f"   {'='*60}")
        print(f"   Assignment Matrix: assignment[cell_idx][robot_id]")
        print(f"   Format: For each cell, shows which robot(s) it's assigned to")
        print(f"   ")
        # Display assignment in a readable format
        num_robots = len(sol.assignment[0]) if sol.assignment else 0
        print(f"   Cell → Robot Assignment:")
        for cell_idx in free_cells:
            assigned_robots = [r for r in range(num_robots) if sol.assignment[cell_idx][r] == 1]
            if assigned_robots:
                print(f"      Cell {cell_idx}: Assigned to Robot(s) {assigned_robots}")
        
        # Show summary
        robot_assignments = {}
        for robot_id in range(num_robots):
            cells_assigned = [cell_idx for cell_idx in free_cells if sol.assignment[cell_idx][robot_id] == 1]
            robot_assignments[robot_id] = cells_assigned
        
        print(f"\n   Summary by Robot:")
        for robot_id in range(num_robots):
            cells = robot_assignments.get(robot_id, [])
            print(f"      Robot {robot_id}: {len(cells)} cells assigned → {cells}")
        
        print(f"\n   {'='*60}")
        print(f"   COMPONENT 2: PATHS (order in which each robot visits its cells)")
        print(f"   {'='*60}")
        print(f"   ⚠️  IMPORTANT: Path ORDER may be different from assignment!")
        print(f"      - Assignment shows WHICH cells: Robot 0 covers [1, 2, 3, 6, 8]")
        print(f"      - Path shows ORDER to visit: Robot 0 path [1, 8, 3, 2, 6]")
        print(f"      - Same cells, but different order (randomly shuffled)")
        print(f"   ")
        for robot_id in sorted(sol.paths.keys()):
            path = sol.paths[robot_id]
            assigned_cells = robot_assignments.get(robot_id, [])
            print(f"      Robot {robot_id} path: {path} (length: {len(path)})")
            print(f"         → Assignment says: Robot {robot_id} covers {assigned_cells}")
            print(f"         → Path says: Robot {robot_id} visits in order: {path}")
            if set(assigned_cells) == set(path):
                print(f"         ✓ Same cells, just different order!")
            else:
                print(f"         ⚠️  WARNING: Assignment and path don't match!")
        
        # Verify assignment and paths match
        print(f"\n   {'='*60}")
        print(f"   VERIFICATION: Assignment ↔ Path Consistency")
        print(f"   {'='*60}")
        for robot_id in range(num_robots):
            assigned_cells = set(robot_assignments.get(robot_id, []))
            path_cells = set(sol.paths.get(robot_id, []))
            if assigned_cells == path_cells:
                print(f"      ✓ Robot {robot_id}: Assignment and Path match")
            else:
                missing_in_path = assigned_cells - path_cells
                extra_in_path = path_cells - assigned_cells
                if missing_in_path:
                    print(f"      ⚠ Robot {robot_id}: Cells in assignment but not in path: {missing_in_path}")
                if extra_in_path:
                    print(f"      ⚠ Robot {robot_id}: Cells in path but not in assignment: {extra_in_path}")
        
        # Show fitness evaluation details
        fitness = sol.fitness
        print(f"\n   Fitness Evaluation:")
        print(f"      Coverage Score: {fitness['coverage_score']}/{len(free_cells)} cells")
        print(f"         → This means {fitness['coverage_score']} out of {len(free_cells)} free cells are visited")
        print(f"         → Coverage ratio: {fitness['coverage_score']/len(free_cells):.2%}")
        
        # Calculate robot distances for display
        if isinstance(sol.paths, dict):
            robot_distances = {}
            for robot_id, path in sol.paths.items():
                if len(path) > 1:
                    total_dist = 0
                    for i in range(len(path) - 1):
                        cell1 = all_cells[path[i]]
                        cell2 = all_cells[path[i + 1]]
                        dist = abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])  # Manhattan distance
                        total_dist += dist
                    robot_distances[robot_id] = total_dist
                else:
                    robot_distances[robot_id] = 0
            
            print(f"      Robot Distances: {robot_distances}")
            print(f"         → Total distance: {sum(robot_distances.values()):.1f}")
            print(f"         → Max distance: {max(robot_distances.values()) if robot_distances else 0:.1f}")
            print(f"         → Balance Score (std dev): {fitness['balance_score']:.3f}")
            print(f"            → Lower balance score = more balanced workloads")
        
        print(f"      Path Jumps: {fitness['path_jumps']} (non-adjacent moves)")
        print(f"      Cell Conflicts: {fitness['cell_conflicts']} (cells assigned to multiple robots)")
        print(f"      Violations: {len(fitness['problems'])} problems")
        if fitness['problems']:
            for problem in fitness['problems'][:3]:  # Show first 3 problems
                print(f"         - {problem}")
            if len(fitness['problems']) > 3:
                print(f"         ... and {len(fitness['problems']) - 3} more")
        
        # Show how combined score is calculated
        coverage_ratio = fitness['coverage_score'] / len(free_cells)
        coverage_term = 1 - coverage_ratio  # Convert to minimization
        imbalance_term = fitness['balance_score']
        penalty_term = sol.calculate_penalty()
        w1, w2 = (0.7, 0.3) if coverage_ratio < 1.0 else (0.5, 0.5)
        
        print(f"\n   Combined Score Calculation:")
        print(f"      Coverage term: w1 × (1 - coverage) = {w1:.1f} × (1 - {coverage_ratio:.3f}) = {w1 * coverage_term:.3f}")
        print(f"      Imbalance term: w2 × balance_score = {w2:.1f} × {imbalance_term:.3f} = {w2 * imbalance_term:.3f}")
        print(f"      Penalty term: {penalty_term:.1f}")
        print(f"      ─────────────────────────────────────────────")
        print(f"      Combined Score: {safe_format_score(sol.combined_score)} (lower = better)")
        print(f"         → Formula: J = {w1:.1f}(1-coverage) + {w2:.1f}(imbalance) + penalty")
    
    print("\n" + "="*60)
    print("EXPLANATION:")
    print("="*60)
    print("Coverage 8/8 means:")
    print("  • 8 = number of free cells visited by at least one robot")
    print("  • 8 = total number of free cells in the grid")
    print("  • 8/8 = 100% coverage (all free cells are covered)")
    print("\nScore 2.750 means:")
    print("  • This is the combined fitness score (lower is better)")
    print("  • It combines: coverage (70%), balance (30%), and penalties")
    print("  • Score = 0.7×(1-coverage) + 0.3×(imbalance) + penalties")
    print("  • Even with 100% coverage, high score can come from:")
    print("    - High imbalance (robots have very different workloads)")
    print("    - Penalties (path jumps, constraint violations)")
    print("="*60)
    
    ###########################################################
    # Test 2: Tournament Selection
    print("\n" + "="*60)
    print("TEST 2: Tournament Selection")
    print("="*60)
    
    # Sort population by score to see the ranking
    sorted_pop = sorted(population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    print("\n📊 Population Ranking (by score, lower = better):")
    for idx, sol in enumerate(sorted_pop):
        print(f"   Rank {idx + 1}: Score = {safe_format_score(sol.combined_score)}, Robot 0 path = {sol.paths.get(0, [])[:5]}...")
    
    best_solution = sorted_pop[0]
    worst_solution = sorted_pop[-1]
    print(f"\n✓ Best solution: Score = {safe_format_score(best_solution.combined_score)}")
    print(f"✓ Worst solution: Score = {safe_format_score(worst_solution.combined_score)}")
    
    # Test tournament selection with different tournament sizes
    print("\n" + "-"*60)
    print("TOURNAMENT SELECTION SIMULATION:")
    print("-"*60)
    
    tournament_sizes = [2, 3, 5]
    num_selections = 10
    
    for tournament_size in tournament_sizes:
        print(f"\n🔹 Tournament Size = {tournament_size}:")
        print(f"   (Selecting {num_selections} parents using tournament of size {tournament_size})")
        
        selected_scores = []
        selected_ranks = []
        
        for i in range(num_selections):
            # Select a parent using tournament selection
            parent = tournament_selection(population, tournament_size=tournament_size)
            selected_scores.append(parent.combined_score)
            
            # Find the rank of selected solution
            rank = next((idx + 1 for idx, sol in enumerate(sorted_pop) if sol.combined_score == parent.combined_score), -1)
            selected_ranks.append(rank)
        
        # Statistics
        avg_score = sum(selected_scores) / len(selected_scores)
        avg_rank = sum(selected_ranks) / len(selected_ranks)
        best_selected = min(selected_scores)
        worst_selected = max(selected_scores)
        best_rank_selected = min(selected_ranks)
        worst_rank_selected = max(selected_ranks)
        
        print(f"   Selected Scores: {[safe_format_score(s) for s in selected_scores]}")
        print(f"   Selected Ranks:  {selected_ranks}")
        print(f"\n   Statistics:")
        print(f"      Average Score: {safe_format_score(avg_score)} (population avg: {safe_format_score(sum(s.combined_score for s in population)/len(population))})")
        print(f"      Average Rank:  {avg_rank:.1f} (out of {len(population)})")
        print(f"      Best Selected: Score {safe_format_score(best_selected)} (Rank {best_rank_selected})")
        print(f"      Worst Selected: Score {safe_format_score(worst_selected)} (Rank {worst_rank_selected})")
        print(f"      → Lower rank = better solution")
        print(f"      → Tournament size {tournament_size} tends to select rank {avg_rank:.1f} on average")
    
    # Demonstrate tournament selection step-by-step
    print("\n" + "-"*60)
    print("STEP-BY-STEP TOURNAMENT SELECTION EXAMPLE:")
    print("-"*60)
    
    print("\n🔹 Example: Tournament Size = 3")
    print("   Step 1: Randomly pick 3 solutions from population")
    
    # Do one tournament selection with detailed output
    tournament = random.sample(population, min(3, len(population)))
    print(f"   Selected for tournament:")
    for idx, sol in enumerate(tournament):
        rank = next((idx + 1 for idx, s in enumerate(sorted_pop) if s.combined_score == sol.combined_score), -1)
        print(f"      Contestant {idx + 1}: Score = {safe_format_score(sol.combined_score)} (Rank {rank}), Robot 0 path = {sol.paths.get(0, [])[:5]}...")
    
    print("\n   Step 2: Compare their scores (lower = better)")
    winner = min(tournament, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    winner_rank = next((idx + 1 for idx, sol in enumerate(sorted_pop) if sol.combined_score == winner.combined_score), -1)
    print(f"   Step 3: Winner is the one with lowest score")
    print(f"      🏆 Winner: Score = {safe_format_score(winner.combined_score)} (Rank {winner_rank})")
    print(f"      → This winner becomes a parent for crossover")
    
    # Show multiple tournaments to demonstrate selection pressure
    print("\n" + "-"*60)
    print("SELECTION PRESSURE ANALYSIS:")
    print("-"*60)
    print("Running 20 tournaments (size=3) to see selection pattern:")
    
    all_winners = []
    for _ in range(20):
        winner = tournament_selection(population, tournament_size=3)
        all_winners.append(winner.combined_score)
    
    # Count how many times each solution was selected
    winner_counts = {}
    for winner_score in all_winners:
        winner_counts[winner_score] = winner_counts.get(winner_score, 0) + 1
    
    print(f"\n   Winners distribution:")
    for score in sorted(winner_counts.keys()):
        rank = next((idx + 1 for idx, sol in enumerate(sorted_pop) if sol.combined_score == score), -1)
        count = winner_counts[score]
        print(f"      Rank {rank} (Score {safe_format_score(score)}): Selected {count}/20 times ({count*5}%)")
    
    best_count = winner_counts.get(best_solution.combined_score, 0)
    print(f"\n   ✓ Best solution (Rank 1) selected: {best_count}/20 times ({best_count*5}%)")
    print(f"   ✓ This shows tournament selection prefers better solutions")
    print(f"   ✓ But it's not deterministic - worse solutions can still be selected")
    
    print("\n" + "="*60)
    print("TOURNAMENT SELECTION CONCEPT EXPLANATION:")
    print("="*60)
    print("""
HOW TOURNAMENT SELECTION WORKS:

1. PURPOSE:
   - Selects "parents" from the population for creating new solutions
   - Like a mini-competition where the best contestant wins
   - Balances exploration (trying different solutions) vs exploitation (using good ones)

2. PROCESS:
   Step 1: Randomly pick K solutions from population (K = tournament_size)
   Step 2: Compare their fitness scores
   Step 3: Select the one with BEST score (lowest = best in our case)
   Step 4: This winner becomes a parent

3. TOURNAMENT SIZE EFFECT:
   - Small tournament (size=2): More random, more exploration
     → Gives worse solutions a chance to be selected
     → Maintains diversity
   
   - Large tournament (size=5+): More selective, more exploitation
     → Almost always picks the best solutions
     → Faster convergence but less diversity

4. WHY USE IT:
   ✓ Simple to implement
   ✓ Works well with any fitness function
   ✓ Can control selection pressure via tournament size
   ✓ Doesn't require sorting entire population
   ✓ Allows worse solutions to be selected (maintains diversity)

5. IN OUR IMPLEMENTATION:
   - tournament_size = 3 (default)
   - Picks 3 random solutions
   - Returns the one with lowest combined_score
   - Lower score = better solution in our problem
""")
    print("="*60)
    
    ###########################################################
    # Test 3: Crossover (One-Point Order-Based)
    print("\n" + "="*60)
    print("TEST 3: Crossover (One-Point Order-Based)")
    print("="*60)
    
    # Create two distinct parents
    parent1 = sorted_pop[0]  # Best solution
    parent2 = sorted_pop[1]  # Second best
    
    print("\n📊 Parent Solutions:")
    print(f"   Parent 1 (Score: {safe_format_score(parent1.combined_score)}):")
    for robot_id in sorted(parent1.paths.keys()):
        print(f"      Robot {robot_id}: {parent1.paths[robot_id]}")
    
    print(f"\n   Parent 2 (Score: {safe_format_score(parent2.combined_score)}):")
    for robot_id in sorted(parent2.paths.keys()):
        print(f"      Robot {robot_id}: {parent2.paths[robot_id]}")
    
    # Test crossover (always happens now)
    print("\n" + "-"*60)
    print("CROSSOVER SIMULATION:")
    print("-"*60)
    print("   Note: Crossover always happens (no probability check)")
    
    print(f"\n🔹 Crossover (always applied):")
    child = crossover_order_based(parent1, parent2)
    child.evaluate()
    
    print(f"   Child (Score: {safe_format_score(child.combined_score)}):")
    for robot_id in sorted(child.paths.keys()):
        print(f"      Robot {robot_id}: {child.paths[robot_id]}")
    
    # Check inheritance
    print(f"   ✓ Crossover applied: Child different from both parents")
    print(f"      Different from Parent 1: {child.paths != parent1.paths}")
    print(f"      Different from Parent 2: {child.paths != parent2.paths}")
    
    # Check if child inherits from both parents
    for robot_id in sorted(child.paths.keys()):
        child_path = child.paths[robot_id]
        p1_path = parent1.paths.get(robot_id, [])
        p2_path = parent2.paths.get(robot_id, [])
        
        from_p1 = any(cell in p1_path for cell in child_path)
        from_p2 = any(cell in p2_path for cell in child_path)
        print(f"      Robot {robot_id}: Has cells from Parent 1: {from_p1}, Parent 2: {from_p2}")
    
    # Step-by-step crossover example
    print("\n" + "-"*60)
    print("STEP-BY-STEP CROSSOVER EXAMPLE:")
    print("-"*60)
    
    print("\n🔹 Example: One-Point Order-Based Crossover")
    print("   Parent 1 Robot 0:", parent1.paths.get(0, []))
    print("   Parent 2 Robot 0:", parent2.paths.get(0, []))
    
    child_example = crossover_order_based(parent1, parent2)
    child_path = child_example.paths.get(0, [])
    p1_path = parent1.paths.get(0, [])
    p2_path = parent2.paths.get(0, [])
    
    if len(p1_path) >= 2 and len(p2_path) >= 2:
        path_length = min(len(p1_path), len(p2_path))
        crossover_point = random.randint(1, path_length - 1)
        print(f"\n   Step 1: Pick crossover point at position {crossover_point}")
        print(f"   Step 2: Copy first {crossover_point} cells from Parent 1: {p1_path[:crossover_point]}")
        print(f"   Step 3: Fill remaining with Parent 2's cells in order (avoiding duplicates)")
        print(f"   Step 4: Result: {child_path}")
        print(f"   ✓ Child has same cells as parents (just reordered)")
        print(f"   ✓ Child length: {len(child_path)} (same as Parent 1: {len(p1_path)})")
    
    print("\n" + "="*60)
    print("CROSSOVER CONCEPT EXPLANATION:")
    print("="*60)
    print("""
HOW ONE-POINT ORDER-BASED CROSSOVER WORKS:

1. PURPOSE:
   - Combines two parent solutions to create a child
   - Preserves order information (important for paths)
   - Creates new solutions by mixing parent characteristics

2. PROCESS (for each robot):
   Step 1: Pick a random crossover point (position in path)
   Step 2: Copy first segment from Parent 1 (up to crossover point)
   Step 3: Fill remaining positions with Parent 2's cells in order
          (skipping cells already used)
   Step 4: If needed, fill with Parent 1's unused cells

3. EXAMPLE:
   Parent 1: [0, 1, 2, 3, 4]
   Parent 2: [4, 3, 2, 1, 0]
   Crossover point: 2
   
   Child: [0, 1, | 4, 3, 2]
          ↑      ↑
        From P1  From P2 (in order, no duplicates)

4. WHY ORDER-BASED:
   ✓ Preserves relative order of cells
   ✓ Maintains path structure
   ✓ Good for permutation problems (like path planning)
   ✓ Prevents duplicate cells in path
""")
    print("="*60)
    
    ###########################################################
    # Test 4: Mutation (Swap Within Path)
    print("\n" + "="*60)
    print("TEST 4: Mutation (Swap Within Path)")
    print("="*60)
    
    # Use a solution for mutation testing
    test_sol = sorted_pop[0].copy()
    # Ensure solution is evaluated
    if test_sol.fitness is None or test_sol.combined_score is None:
        test_sol.evaluate()
    original_paths = {r: test_sol.paths[r].copy() for r in test_sol.paths}
    
    print("\n📊 Original Solution:")
    score_str = safe_format_score(test_sol.combined_score) if test_sol.combined_score is not None else "Not evaluated"
    print(f"   Score: {score_str}")
    for robot_id in sorted(test_sol.paths.keys()):
        print(f"   Robot {robot_id}: {test_sol.paths[robot_id]}")
    
    # Test mutation with different rates
    print("\n" + "-"*60)
    print("MUTATION SIMULATION:")
    print("-"*60)
    
    for mutation_rate in [0.0, 0.5, 1.0]:
        print(f"\n🔹 Mutation Rate = {mutation_rate}:")
        test_sol_copy = test_sol.copy()
        paths_before = {r: test_sol_copy.paths[r].copy() for r in test_sol_copy.paths}
        
        mutated_sol = mutate(test_sol_copy, mutation_rate=mutation_rate)
        paths_after = mutated_sol.paths
        
        changed = False
        for robot_id in paths_before:
            if paths_before[robot_id] != paths_after.get(robot_id, []):
                changed = True
                print(f"   Robot {robot_id} path changed:")
                print(f"      Before: {paths_before[robot_id]}")
                print(f"      After:  {paths_after[robot_id]}")
                
                # Show what was swapped
                before = paths_before[robot_id]
                after = paths_after[robot_id]
                if len(before) == len(after):
                    # Find swapped positions
                    swapped_positions = [i for i in range(len(before)) if before[i] != after[i]]
                    if len(swapped_positions) == 2:
                        print(f"      → Swapped positions {swapped_positions[0]} and {swapped_positions[1]}")
                        print(f"      → Swapped cells: {before[swapped_positions[0]]} ↔ {before[swapped_positions[1]]}")
        
        if not changed:
            print(f"   ✓ No mutation (rate={mutation_rate} didn't trigger)")
        else:
            # Verify mutation properties
            for robot_id in paths_before:
                if paths_before[robot_id] != paths_after.get(robot_id, []):
                    before_set = set(paths_before[robot_id])
                    after_set = set(paths_after.get(robot_id, []))
                    print(f"   ✓ Same cells (just reordered): {before_set == after_set}")
                    print(f"   ✓ Same length: {len(paths_before[robot_id]) == len(paths_after.get(robot_id, []))}")
    
    # Step-by-step mutation example
    print("\n" + "-"*60)
    print("STEP-BY-STEP MUTATION EXAMPLE:")
    print("-"*60)
    
    print("\n🔹 Example: Swap Mutation")
    test_sol_example = sorted_pop[0].copy()
    original_path = test_sol_example.paths[0].copy() if 0 in test_sol_example.paths else []
    
    if len(original_path) >= 2:
        print(f"   Original Robot 0 path: {original_path}")
        
        # Force mutation
        mutated_example = mutate(test_sol_example, mutation_rate=1.0)
        mutated_path = mutated_example.paths[0] if 0 in mutated_example.paths else []
        
        if mutated_path != original_path:
            # Find swapped positions
            swapped = [(i, original_path[i], mutated_path[i]) 
                      for i in range(len(original_path)) if original_path[i] != mutated_path[i]]
            
            if len(swapped) == 2:
                pos1, cell1_orig, cell1_new = swapped[0]
                pos2, cell2_orig, cell2_new = swapped[1]
                print(f"\n   Step 1: Pick Robot 0 (has {len(original_path)} cells)")
                print(f"   Step 2: Randomly pick two positions: {pos1} and {pos2}")
                print(f"   Step 3: Swap cells at these positions")
                print(f"      Position {pos1}: {cell1_orig} → {cell1_new}")
                print(f"      Position {pos2}: {cell2_orig} → {cell2_new}")
                print(f"   Step 4: Result: {mutated_path}")
                print(f"   ✓ Same cells, different order")
                print(f"   ✓ No cells added or removed")
    
    print("\n" + "="*60)
    print("MUTATION CONCEPT EXPLANATION:")
    print("="*60)
    print("""
HOW SWAP MUTATION WORKS:

1. PURPOSE:
   - Introduces small random changes to explore solution space
   - Prevents getting stuck in local optima
   - Maintains diversity in population

2. PROCESS:
   Step 1: Select worst solutions (based on mutation_percentage)
   Step 2: Find robots with paths of length >= 2
   Step 3: Pick a random robot
   Step 4: Pick two different positions in that robot's path
   Step 5: Swap the cells at these positions

3. PROPERTIES:
   ✓ Only changes order, not which cells are assigned
   ✓ Same cells before and after mutation
   ✓ Same path length
   ✓ Can improve or worsen the solution

4. MUTATION PERCENTAGE:
   - mutation_percentage = 0.1 means 10% of population will be mutated
   - Higher percentage = more exploration, more diversity
   - Lower percentage = less exploration, faster convergence
""")
    print("="*60)
    
    ###########################################################
    # Test 5: Elitism
    print("\n" + "="*60)
    print("TEST 5: Elitism")
    print("="*60)
    
    # Create old and new populations
    old_pop = sorted_pop.copy()  # Sorted by score
    print("\n📊 Old Population (Generation N):")
    for idx, sol in enumerate(old_pop):
        score_str = safe_format_score(sol.combined_score) if sol.combined_score is not None else "Not evaluated"
        print(f"   Rank {idx + 1}: Score = {score_str}")
    
    # Create new population (all with worse scores)
    new_pop = []
    old_worst_score = old_pop[-1].combined_score if old_pop[-1].combined_score is not None else 1000
    for i in range(len(old_pop)):
        sol = generate_random_solution(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots)
        sol.evaluate()
        # Make sure new solutions are worse
        sol.combined_score = old_worst_score + (i + 1) * 100
        new_pop.append(sol)
    
    new_pop_sorted = sorted(new_pop, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    print("\n📊 New Population (Generation N+1, before elitism):")
    for idx, sol in enumerate(new_pop_sorted):
        score_str = safe_format_score(sol.combined_score) if sol.combined_score is not None else "Not evaluated"
        print(f"   Rank {idx + 1}: Score = {score_str}")
    
    old_best = old_pop[0].combined_score if old_pop[0].combined_score is not None else float('inf')
    new_best = new_pop_sorted[0].combined_score if new_pop_sorted[0].combined_score is not None else float('inf')
    print(f"\n✓ Old best: {safe_format_score(old_best)}")
    print(f"✓ New best (before selection): {safe_format_score(new_best)}")
    print(f"✓ New population is worse (as expected)")
    
    # Apply selection (elitism) using selection_percentage approach
    selection_percentage = 0.10  # 10% selection
    num_selection = int(len(old_pop) * selection_percentage)
    print(f"\n   Selection percentage: {selection_percentage*100:.0f}%")
    print(f"   Number of elite solutions to copy: {num_selection}")
    
    # Create final population: copy best solutions directly (selection approach)
    final_pop = []
    # Copy top solutions from old population (elitism)
    for i in range(num_selection):
        final_pop.append(old_pop[i].copy())
        print(f"   ✓ Copied {i+1}-th best solution (score: {safe_format_score(old_pop[i].combined_score)})")
    
    # Add remaining solutions from new population
    final_pop.extend(new_pop[:len(old_pop) - num_selection])
    
    final_pop_sorted = sorted(final_pop, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    
    print("\n" + "-"*60)
    print(f"AFTER SELECTION ({selection_percentage*100:.0f}% elite copied directly):")
    print("-"*60)
    print("\n📊 Final Population:")
    for idx, sol in enumerate(final_pop_sorted):
        score_str = safe_format_score(sol.combined_score) if sol.combined_score is not None else "Not evaluated"
        print(f"   Rank {idx + 1}: Score = {score_str}")
    
    final_best_score = final_pop_sorted[0].combined_score if final_pop_sorted[0].combined_score is not None else float('inf')
    old_best_score = old_pop[0].combined_score if old_pop[0].combined_score is not None else float('inf')
    old_second_score = old_pop[1].combined_score if len(old_pop) > 1 and old_pop[1].combined_score is not None else float('inf')
    final_second_score = final_pop_sorted[1].combined_score if len(final_pop_sorted) > 1 and final_pop_sorted[1].combined_score is not None else float('inf')
    
    print(f"\n✓ Final best: {safe_format_score(final_best_score)}")
    print(f"✓ Best preserved: {final_best_score == old_best_score}")
    if num_selection >= 2:
        print(f"✓ Second best preserved: {final_second_score == old_second_score}")
    print(f"✓ Population size maintained: {len(final_pop) == len(old_pop)}")
    
    print("\n" + "="*60)
    print("SELECTION (ELITISM) CONCEPT EXPLANATION:")
    print("="*60)
    print("""
HOW SELECTION (ELITISM) WORKS:

1. PURPOSE:
   - Preserves the best solutions from previous generation
   - Ensures we never lose our best solutions
   - Prevents regression (going backwards)

2. PROCESS (using selection_percentage):
   Step 1: Sort old population by fitness (best first)
   Step 2: Calculate number of elite: num_selection = population_size * selection_percentage
   Step 3: Copy top num_selection solutions directly to new generation
   Step 4: Fill remaining slots with crossover/mutation offspring

3. EXAMPLE (selection_percentage = 10%):
   Population size = 10
   num_selection = 10 * 0.10 = 1 solution
   Old Population: [Best(0.5), Good(1.0), OK(2.0), Bad(3.0), Worst(4.0), ...]
   → Copy Best(0.5) directly to new generation
   → Create remaining 9 solutions via crossover/mutation
   
   Final: [Best(0.5), Good(1.0), New3(7.0), New4(8.0), New5(9.0)]
          ↑          ↑
        Elite      Elite

4. WHY IT'S IMPORTANT:
   ✓ Guarantees monotonic improvement (best never gets worse)
   ✓ Faster convergence
   ✓ Preserves good solutions
   ✓ Prevents losing progress
""")
    print("="*60)
    
    ###########################################################
    # Test 6: Fitness Evaluation
    print("\n" + "="*60)
    print("TEST 6: Fitness Evaluation")
    print("="*60)
    
    test_solution = sorted_pop[0]
    
    print("\n📊 Solution to Evaluate:")
    for robot_id in sorted(test_solution.paths.keys()):
        print(f"   Robot {robot_id}: {test_solution.paths[robot_id]}")
    
    # Show fitness evaluation process
    print("\n" + "-"*60)
    print("FITNESS EVALUATION PROCESS:")
    print("-"*60)
    
    fitness = test_solution.fitness
    print("\n1. COVERAGE CALCULATION:")
    covered_cells = set()
    for robot_id, path in test_solution.paths.items():
        covered_cells.update(path)
    print(f"   Covered cells: {sorted(covered_cells)}")
    print(f"   Coverage score: {fitness['coverage_score']}/{len(free_cells)}")
    print(f"   Coverage ratio: {fitness['coverage_score']/len(free_cells):.2%}")
    
    print("\n2. DISTANCE CALCULATION:")
    robot_distances = {}
    for robot_id, path in test_solution.paths.items():
        if len(path) > 1:
            total_dist = 0
            for i in range(len(path) - 1):
                cell1 = all_cells[path[i]]
                cell2 = all_cells[path[i + 1]]
                dist = abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])
                total_dist += dist
            robot_distances[robot_id] = total_dist
        else:
            robot_distances[robot_id] = 0
    
    print(f"   Robot distances: {robot_distances}")
    print(f"   Total distance: {sum(robot_distances.values())}")
    print(f"   Mean distance: {sum(robot_distances.values())/len(robot_distances):.2f}")
    
    print("\n3. BALANCE CALCULATION:")
    distances_list = list(robot_distances.values())
    mean_distance = sum(distances_list) / len(distances_list)
    variance = sum((d - mean_distance) ** 2 for d in distances_list) / len(distances_list)
    std_dev = variance ** 0.5
    print(f"   Distances: {distances_list}")
    print(f"   Mean: {mean_distance:.2f}")
    print(f"   Variance: {variance:.2f}")
    print(f"   Standard deviation (balance score): {std_dev:.3f}")
    print(f"   → Lower std dev = more balanced workloads")
    
    print("\n4. CONSTRAINT CHECKING:")
    print(f"   Path jumps: {fitness['path_jumps']}")
    print(f"   Cell conflicts: {fitness['cell_conflicts']}")
    print(f"   Total violations: {len(fitness['problems'])}")
    if fitness['problems']:
        print(f"   Violations:")
        for problem in fitness['problems'][:5]:
            print(f"      - {problem}")
    
    print("\n5. PENALTY CALCULATION:")
    penalty = test_solution.calculate_penalty()
    print(f"   Penalty: {penalty:.1f}")
    print(f"   Breakdown:")
    for violation in fitness['problems']:
        if "goes outside grid" in violation:
            print(f"      +1000 (boundary violation)")
        elif "hits obstacle" in violation:
            print(f"      +500 (obstacle collision)")
        elif "jumps from" in violation.lower() or "jump from" in violation.lower():
            print(f"      +100 (path jump)")
    
    print("\n6. COMBINED SCORE CALCULATION:")
    coverage_ratio = fitness['coverage_score'] / len(free_cells)
    coverage_term = 1 - coverage_ratio
    imbalance_term = fitness['balance_score']
    w1, w2 = (0.7, 0.3) if coverage_ratio < 1.0 else (0.5, 0.5)
    
    print(f"   Formula: J = w1(1-coverage) + w2(imbalance) + penalty")
    print(f"   Coverage term: {w1:.1f} × (1 - {coverage_ratio:.3f}) = {w1 * coverage_term:.3f}")
    print(f"   Imbalance term: {w2:.1f} × {imbalance_term:.3f} = {w2 * imbalance_term:.3f}")
    print(f"   Penalty term: {penalty:.1f}")
    print(f"   ─────────────────────────────────────────────")
    print(f"   Combined Score: {safe_format_score(test_solution.combined_score)} (lower = better)")
    
    print("\n" + "="*60)
    print("FITNESS EVALUATION CONCEPT EXPLANATION:")
    print("="*60)
    print("""
HOW FITNESS EVALUATION WORKS:

1. PURPOSE:
   - Measures how good a solution is
   - Combines multiple objectives into single score
   - Lower score = better solution

2. COMPONENTS:
   a) Coverage: How many cells are visited
      - Higher coverage = better
      - Converted to minimization: (1 - coverage_ratio)
   
   b) Balance: How evenly robots work
      - Lower variance = more balanced
      - Measured as standard deviation of distances
   
   c) Penalties: For breaking constraints
      - Boundary violations: 1000
      - Obstacle collisions: 500
      - Path jumps: 100

3. COMBINED FORMULA:
   J = w1(1-coverage) + w2(imbalance) + penalty
   
   Where:
   - w1 = 0.7 (or 0.5 if coverage is perfect)
   - w2 = 0.3 (or 0.5 if coverage is perfect)
   - Lower J = better solution

4. WHY THIS APPROACH:
   ✓ Single number to compare solutions
   ✓ Balances multiple objectives
   ✓ Penalizes constraint violations
   ✓ Guides search toward good solutions
""")
    print("="*60)
    
    ###########################################################
    # Test 7: Full GA Loop (1 Generation)
    print("\n" + "="*60)
    print("TEST 7: Full GA Loop (1 Generation)")
    print("="*60)
    
    # Use a small population for testing
    test_pop = initialize_population(5, all_cells, free_cells, obstacles, grid_width, grid_height, num_robots)
    initial_best = min(test_pop, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    initial_avg = sum(s.combined_score for s in test_pop if s.combined_score is not None) / len(test_pop)
    
    print("\n📊 Initial Population (Generation 0):")
    print(f"   Best score: {safe_format_score(initial_best.combined_score)}")
    print(f"   Average score: {safe_format_score(initial_avg)}")
    print(f"   Population size: {len(test_pop)}")
    
    print("\n" + "-"*60)
    print("RUNNING ONE GENERATION:")
    print("-"*60)
    
    # Manual GA loop for one generation
    new_pop = []
    crossovers = 0
    mutations = 0
    
    print("\n🔹 Creating new generation:")
    for i in range(len(test_pop)):
        # Selection
        parent1 = tournament_selection(test_pop, tournament_size=3)
        parent2 = tournament_selection(test_pop, tournament_size=3)
        
        if i < 2:  # Show details for first 2
            print(f"\n   Offspring {i+1}:")
            print(f"      Parent 1: Score = {safe_format_score(parent1.combined_score)}")
            print(f"      Parent 2: Score = {safe_format_score(parent2.combined_score)}")
        
        # Crossover (always happens)
        child = crossover_order_based(parent1, parent2)
        if child.paths != parent1.paths:
            crossovers += 1
            if i < 2:
                print(f"      ✓ Crossover applied")
        
        # Mutation
        paths_before = {r: child.paths[r].copy() for r in child.paths}
        child = mutate(child, mutation_rate=0.1)
        paths_after = child.paths
        if paths_before != paths_after:
            mutations += 1
            if i < 2:
                print(f"      ✓ Mutation applied")
        
        # Evaluate
        child.evaluate()
        new_pop.append(child)
        
        if i < 2:
            print(f"      Child score: {safe_format_score(child.combined_score)}")
    
    print(f"\n   Summary:")
    print(f"      Crossovers: {crossovers}/{len(test_pop)}")
    print(f"      Mutations: {mutations}/{len(test_pop)}")
    
    # Selection (Elitism) using selection_percentage approach
    selection_percentage = 0.10  # 10% selection
    num_selection = int(len(test_pop) * selection_percentage)
    print(f"\n   Selection (Elitism): {num_selection} solutions ({selection_percentage*100:.0f}%)")
    
    # Sort old population to get best solutions
    sorted_test_pop = sorted(test_pop, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    
    # Create final population: copy best solutions directly, then add offspring
    final_pop = []
    for i in range(num_selection):
        final_pop.append(sorted_test_pop[i].copy())
        print(f"      ✓ Copied {i+1}-th best solution (score: {safe_format_score(sorted_test_pop[i].combined_score)})")
    
    # Add remaining solutions from new_pop
    final_pop.extend(new_pop[:len(test_pop) - num_selection])
    
    final_best = min(final_pop, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    final_avg = sum(s.combined_score for s in final_pop if s.combined_score is not None) / len(final_pop)
    
    print("\n📊 Final Population (Generation 1):")
    print(f"   Best score: {safe_format_score(final_best.combined_score)}")
    print(f"   Average score: {safe_format_score(final_avg)}")
    print(f"   Population size: {len(final_pop)}")
    
    print(f"\n✓ Best improved or maintained: {final_best.combined_score <= initial_best.combined_score}")
    print(f"✓ Average changed: {safe_format_score(abs(final_avg - initial_avg))}")
    print(f"✓ Population size maintained: {len(final_pop) == len(test_pop)}")
    
    print("\n" + "="*60)
    print("FULL GA LOOP CONCEPT EXPLANATION:")
    print("="*60)
    print("""
HOW ONE GENERATION WORKS:

1. INITIALIZATION:
   - Start with random population
   - Evaluate all solutions

2. FOR EACH GENERATION:
   a) Selection: Pick parents using tournament selection
# Find lines 2318-2346 and replace with:

# filepath: /Users/hagarelgazzar/Desktop/Optimization_ms_3/optimization-1/GA.py
b) Crossover: Combine parents to create child
   c) Mutation: Randomly modify child
   d) Evaluation: Calculate child's fitness
   e) Add to new population
   
   f) Elitism: Replace worst solutions with best from previous generation

3. REPEAT:
   - Run for multiple generations
   - Population evolves toward better solutions
   - Best solution improves over time

4. TERMINATION:
   - After fixed number of generations, OR
   - When solution is good enough, OR
   - When no improvement for many generations
""")
    print("="*60)
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Run Genetic Algorithm
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        population_size=5, generations=50, 
        verbose=True
    )
    
    # Extract results
    best_solution = ga_results['best_solution']
    convergence_history = ga_results['convergence_history']
    
    # Print detailed results
    print_solution_summary(best_solution)
    
    # Plot convergence
    # analyze_convergence(convergence_history)