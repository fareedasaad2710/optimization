"""
Simulated Annealing for Multi-Robot Coverage Path Planning
==========================================================

This implements SA algorithm with:
- Weighted Sum: Combine F1 and F2 into single objective
- Neighbor Generation: Swapping cell assignments only
- Constraint Handling: Penalty functions for violations
- SA Parameters: Temperature=1000, cooling_rate=0.95, iterations=2000
"""

import math
import random
import copy
from simple_robot_coverage import *

class RobotCoverageSolution:
    """Represents a solution for the multi-robot coverage problem"""
    
    def __init__(self, assignment, paths, all_cells, free_cells, obstacles, grid_width, grid_height):
        self.assignment = copy.deepcopy(assignment)
        self.paths = copy.deepcopy(paths)
        self.all_cells = all_cells
        self.free_cells = free_cells
        self.obstacles = obstacles
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fitness = None
        self.combined_score = None
        
    def evaluate(self):
        """Evaluate this solution and calculate combined fitness score"""
        # Use existing evaluation function
        results = evaluate_robot_solution(
            self.all_cells, self.free_cells, self.obstacles, 
            self.assignment, self.paths, self.grid_width, self.grid_height
        )
        self.fitness = results
        
        # Calculate combined score using weighted sum
        F1 = results['coverage_score']  # Higher is better
        F2 = results['balance_score']    # Lower is better
        penalty = self.calculate_penalty()
        
        # Convert F1 to minimization: -F1 (lower is better)
        # Weighted sum: w1*(-F1) + w2*F2 + penalty
        w1 = 0.7  # Coverage weight (70%)
        w2 = 0.3  # Balance weight (30%)
        
        self.combined_score = w1 * (-F1) + w2 * F2 + penalty
        return self.combined_score
    
    def calculate_penalty(self):
        """Calculate penalty for constraint violations"""
        violations = self.fitness['problems']
        penalty = 0
        
        for violation in violations:
            if "outside grid" in violation:
                penalty += 1000  # Highest penalty for boundaries
            elif "hits obstacle" in violation:
                penalty += 500   # Middle penalty for obstacles  
            elif "jumps" in violation:
                penalty += 100   # Minimum penalty for jumping
        
        return penalty
    
    def copy(self):
        """Create a deep copy of this solution"""
        return RobotCoverageSolution(
            self.assignment, self.paths, self.all_cells, 
            self.free_cells, self.obstacles, self.grid_width, self.grid_height
        )

def generate_random_solution(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots):
    """Generate a random initial solution"""
    
    # Step 1: Randomly assign free cells to robots
    assignment = []
    obstacle_assignment = [0] * num_robots  # Obstacles assigned to no robot
    
    # Shuffle free cells for random assignment
    shuffled_free_cells = copy.deepcopy(free_cells)
    random.shuffle(shuffled_free_cells)
    
    # Assign cells to robots in round-robin fashion
    free_cell_index = 0
    for cell_idx in range(len(all_cells)):
        if cell_idx in obstacles:
            assignment.append(obstacle_assignment.copy())
        else:
            robot_id = free_cell_index % num_robots
            robot_assignment = [0] * num_robots
            robot_assignment[robot_id] = 1
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
        
        # Simple path: visit cells in order
        robot_paths.append(robot_cells)
    
    return RobotCoverageSolution(assignment, robot_paths, all_cells, 
                               free_cells, obstacles, grid_width, grid_height)

def generate_neighbor_solution(current_solution):
    """Generate a neighbor solution by swapping cell assignments"""
    
    # Create a copy of current solution
    neighbor = current_solution.copy()
    
    # Find two different robots to swap cells between
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
    
    # Swap the assignments
    neighbor.assignment[cell1][robot1] = 0
    neighbor.assignment[cell1][robot2] = 1
    neighbor.assignment[cell2][robot1] = 1
    neighbor.assignment[cell2][robot2] = 0
    
    # Regenerate paths for affected robots
    for robot_id in [robot1, robot2]:
        robot_cells = []
        for cell_idx, assignment_row in enumerate(neighbor.assignment):
            if assignment_row[robot_id] == 1:
                robot_cells.append(cell_idx)
        neighbor.paths[robot_id] = robot_cells
    
    return neighbor

def simulated_annealing(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots):
    """Main Simulated Annealing algorithm"""
    
    # SA Parameters
    initial_temp = 1000.0
    cooling_rate = 0.95
    max_iterations = 2000
    
    print(f"Starting Simulated Annealing...")
    print(f"Parameters: T0={initial_temp}, cooling_rate={cooling_rate}, iterations={max_iterations}")
    
    # Generate initial random solution
    current_solution = generate_random_solution(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots
    )
    current_solution.evaluate()
    
    best_solution = current_solution.copy()
    best_solution.evaluate()
    
    temperature = initial_temp
    
    print(f"Initial solution: Coverage={current_solution.fitness['coverage_score']}, "
          f"Balance={current_solution.fitness['balance_score']:.3f}, "
          f"Combined={current_solution.combined_score:.3f}")
    
    # Main SA loop
    for iteration in range(max_iterations):
        # Generate neighbor solution
        neighbor = generate_neighbor_solution(current_solution)
        neighbor.evaluate()
        
        # Calculate fitness difference
        delta = neighbor.combined_score - current_solution.combined_score
        
        # Accept or reject based on SA criteria
        if delta < 0:  # Neighbor is better
            current_solution = neighbor
            best_score = best_solution.combined_score if best_solution.combined_score is not None else float('inf')
            if neighbor.combined_score < best_score:
                best_solution = neighbor.copy()
        else:  # Neighbor is worse, accept with probability
            if random.random() < math.exp(-delta / temperature):
                current_solution = neighbor
        
        # Cool down temperature
        temperature *= cooling_rate
        
        # Print progress every 200 iterations
        if iteration % 200 == 0:
            current_score = current_solution.combined_score if current_solution.combined_score is not None else 0
            best_score = best_solution.combined_score if best_solution.combined_score is not None else 0
            print(f"Iteration {iteration}: T={temperature:.2f}, "
                  f"Current={current_score:.3f}, "
                  f"Best={best_score:.3f}")
    
    print(f"\nSA Complete!")
    
    # Ensure best solution is evaluated
    if best_solution.fitness is None:
        best_solution.evaluate()
    
    if best_solution.fitness is not None:
        print(f"Best solution: Coverage={best_solution.fitness['coverage_score']}, "
              f"Balance={best_solution.fitness['balance_score']:.3f}, "
              f"Combined={best_solution.combined_score:.3f}")
    else:
        print("Best solution evaluation failed")
    
    return best_solution

def print_sa_results(solution):
    """Print detailed results of SA solution"""
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

# Example usage
if __name__ == "__main__":
    # Configuration
    grid_width = 3
    grid_height = 3
    num_robots = 2
    obstacles = [4]  # Cell 4 is an obstacle
    
    print("Multi-Robot Coverage Path Planning with Simulated Annealing")
    print("="*60)
    
    total_cells = grid_width * grid_height
    free_cells = [i for i in range(total_cells) if i not in obstacles]
    
    # Create grid cells
    all_cells = []
    for y in range(grid_height):
        for x in range(grid_width):
            all_cells.append((x, y))
    
    print(f"Grid: {grid_width}x{grid_height}, Robots: {num_robots}")
    print(f"Total cells: {total_cells}, Free cells: {len(free_cells)}, Obstacles: {len(obstacles)}")
    
    # Run Simulated Annealing
    best_solution = simulated_annealing(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots
    )
    
    # Print detailed results
    print_sa_results(best_solution)
