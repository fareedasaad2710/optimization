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
        # Get basic scores (coverage, balance, problems)
        results = evaluate_solution(
            self.all_cells, self.free_cells, self.obstacles, 
            self.assignment, self.paths, self.grid_width, self.grid_height
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
            elif "jumps from" in violation:
                penalty += penalty_factors['path_jump']
        
        return penalty
    
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
    - Creates a random starting solution for GA algorithm
    - Like throwing robots randomly on the grid
    - Divides cells equally among robots
    """
    
    # Step 1: Randomly assign free cells to robots
    assignment = []
    obstacle_assignment = [0] * num_robots  # Obstacles assigned to no robot
    
    # Shuffle free cells for random assignment
    shuffled_free_cells = copy.deepcopy(free_cells)
    random.shuffle(shuffled_free_cells)
    
# 		assignment = [
#     [1, 0],  # Cell 0: Robot 0
#     [0, 1],  # Cell 1: Robot 1
#     [1, 0],  # Cell 2: Robot 0
#     [0, 1],  # Cell 3: Robot 1
#     [0, 0],  # Cell 4: Obstacle
#     [1, 0],  # Cell 5: Robot 0
#     [0, 1],  # Cell 6: Robot 1
#     [1, 0],  # Cell 7: Robot 0
#     [0, 1],  # Cell 8: Robot 1
# ]
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
# 		robot_paths = [
#     [0, 2, 5, 7],  # Robot 0's path
#     [1, 3, 6, 8],  # Robot 1's path
# ]
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
    - ONE-POINT CROSSOVER: Combines two parent solutions at a random point
    - Like cutting both parents at the same point and swapping the parts
    - With probability crossover_rate, we combine them
    - Otherwise, just return parent1
    
    HOW ONE-POINT CROSSOVER WORKS:
    - Pick a random crossover point (e.g., cell index 5)
    - Child gets: parent1's assignments before point + parent2's assignments after point
    - Example: Child = Parent1[0:5] + Parent2[5:]
    """
    if random.random() > crossover_rate:
        return parent1.copy()  # No crossover, return parent1
    
    # Create child solution
    child = parent1.copy()
    
    # ONE-POINT CROSSOVER: Pick a random crossover point
    num_cells = len(parent1.assignment)
    crossover_point = random.randint(1, num_cells - 1)  # Point between 1 and num_cells-1
    
    # Child gets parent1's assignments before crossover point
    # Child gets parent2's assignments after crossover point
    for cell_idx in range(crossover_point, num_cells):
        child.assignment[cell_idx] = copy.deepcopy(parent2.assignment[cell_idx])
    
    # Update paths for all robots (since assignments changed)
    num_robots = len(parent1.assignment[0])
    for robot_id in range(num_robots):
        robot_cells = []
        for cell_idx, assignment_row in enumerate(child.assignment):
            if assignment_row[robot_id] == 1:
                robot_cells.append(cell_idx)
        child.paths[robot_id] = robot_cells
    
    return child

def mutate(solution, mutation_rate=0.1):
    """
    WHAT DOES THIS DO?
    - Adds random changes to a solution
    - Like genetic mutations - small random changes
    - With probability mutation_rate, we swap cells between robots
    """
    if random.random() > mutation_rate:
        return solution  # No mutation
    
    # Mutation: swap cells between two random robots
    num_robots = len(solution.assignment[0])
    robot1 = random.randint(0, num_robots - 1)
    robot2 = random.randint(0, num_robots - 1)
    
    # Make sure they're different robots
    while robot2 == robot1:
        robot2 = random.randint(0, num_robots - 1)
    
    # Find cells assigned to each robot
    robot1_cells = []
    robot2_cells = []
    
    for cell_idx, assignment_row in enumerate(solution.assignment):
        if assignment_row[robot1] == 1:
            robot1_cells.append(cell_idx)
        elif assignment_row[robot2] == 1:
            robot2_cells.append(cell_idx)
    
    # Make sure both robots have at least one cell
    if len(robot1_cells) == 0 or len(robot2_cells) == 0:
        return solution  # Can't mutate, return original
    
    # Pick random cells from each robot
    cell1 = random.choice(robot1_cells)
    cell2 = random.choice(robot2_cells)
    
    # Swap the assignments
    solution.assignment[cell1][robot1] = 0
    solution.assignment[cell1][robot2] = 1
    solution.assignment[cell2][robot1] = 1
    solution.assignment[cell2][robot2] = 0
    
    # Update paths for affected robots
    for robot_id in [robot1, robot2]:
        robot_cells = []
        for cell_idx, assignment_row in enumerate(solution.assignment):
            if assignment_row[robot_id] == 1:
                robot_cells.append(cell_idx)
        solution.paths[robot_id] = robot_cells
    
    return solution

def apply_elitism(population, new_population, elitism_count=2):
    """
    WHAT DOES THIS DO?
    - Keeps the best solutions from previous generation
    - Like preserving the best genes
    - This ensures we never lose our best solutions
    """
    # Sort population by fitness (best first - lowest score is best)
    sorted_population = sorted(population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    
    # Take best solutions (elite)
    elite = [sol.copy() for sol in sorted_population[:elitism_count]]
    
    # Sort new population by fitness (worst first - highest score is worst)
    sorted_new_population = sorted(new_population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'), reverse=True)
    
    # Replace worst solutions in new population with elite
    for i, elite_solution in enumerate(elite):
        if i < len(sorted_new_population):
            sorted_new_population[i] = elite_solution
    
    return sorted_new_population

def genetic_algorithm(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                      population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.1, elitism_count=2):
    """
    WHAT IS THIS FUNCTION?
    - This is the MAIN Genetic Algorithm
    - It's like evolution - solutions get better over generations
    - Uses selection, crossover, mutation, and elitism
    
    HOW IT WORKS:
    1. Create random population
    2. For each generation:
       a. Select parents (tournament selection)
       b. Create children (crossover)
       c. Mutate children
       d. Keep best solutions (elitism)
    3. Return best solution found
    """
    
    print(f"Starting Genetic Algorithm...")
    print(f"Parameters: Population={population_size}, Generations={generations}, "
          f"Crossover={crossover_rate}, Mutation={mutation_rate}, Elitism={elitism_count}")
    
    # Step 1: Initialize population (generation 0)
    population = initialize_population(
        population_size, all_cells, free_cells, obstacles, grid_width, grid_height, num_robots
    )
    
    # Find best solution in initial population
    best_solution = min(population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    best_solution = best_solution.copy()
    best_score = best_solution.combined_score if best_solution.combined_score is not None else float('inf')
    
    if best_solution.fitness is not None:
        print(f"Initial best solution: Coverage={best_solution.fitness['coverage_score']}, "
              f"Balance={best_solution.fitness['balance_score']:.3f}, "
              f"Combined={best_score:.3f}")
    else:
        print(f"Initial best solution: Combined={best_score:.3f}")
    
    # Main GA loop - evolve for multiple generations
    for generation in range(generations):
        new_population = []
        
        # Step 2: Create new generation
        while len(new_population) < population_size:
            # Select parents using tournament selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Create child through crossover
            child = crossover(parent1, parent2, crossover_rate)
            
            # Mutate child
            child = mutate(child, mutation_rate)
            
            # Evaluate child
            child.evaluate()
            
            # Add to new population
            new_population.append(child)
        
        # Step 3: Apply elitism (keep best solutions)
        new_population = apply_elitism(population, new_population, elitism_count)
        
        # Step 4: Update population
        population = new_population
        
        # Step 5: Find best solution in current generation
        current_best = min(population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
        current_best_score = current_best.combined_score if current_best.combined_score is not None else float('inf')
        
        # Update global best if needed
        if current_best_score < best_score:
            best_solution = current_best.copy()
            best_score = current_best_score
        
        # Print progress every 10 generations
        if generation % 10 == 0 or generation == generations - 1:
            valid_scores = [s.combined_score for s in population if s.combined_score is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            print(f"Generation {generation}: Best={best_score:.3f}, "
                  f"Current Best={current_best_score:.3f}, "
                  f"Average={avg_score:.3f}")
    
    print(f"\nGA Complete!")
    
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
    
    print("Multi-Robot Coverage Path Planning with Genetic Algorithm")
    print("="*60)
    
    total_cells = grid_width * grid_height
    free_cells = [i for i in range(total_cells) if i not in obstacles]
    
    # Create grid cells
    all_cells = create_grid_cells(grid_width, grid_height)
    
    print(f"Grid: {grid_width}x{grid_height}, Robots: {num_robots}")
    print(f"Total cells: {total_cells}, Free cells: {len(free_cells)}, Obstacles: {len(obstacles)}")
    
    # Run Genetic Algorithm
    best_solution = genetic_algorithm(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.1, elitism_count=2
    )
    
    # Print detailed results
    print_ga_results(best_solution)
