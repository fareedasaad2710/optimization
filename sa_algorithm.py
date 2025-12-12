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

def generate_neighbor_solution(current_solution, verbose=False):
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
        if verbose:
            print(f"   ⚠️  Cannot generate neighbor: Robot {robot1} has {len(robot1_cells)} cells, Robot {robot2} has {len(robot2_cells)} cells")
        return current_solution  # Can't swap, return original
    
    # Pick random cells from each robot
    cell1 = random.choice(robot1_cells)
    cell2 = random.choice(robot2_cells)
    
    if verbose:
        print(f"   🔄 Generating Neighbor:")
        print(f"      • Selected Robots: Robot {robot1} (has {len(robot1_cells)} cells) ↔ Robot {robot2} (has {len(robot2_cells)} cells)")
        print(f"      • Swapping: Cell {cell1} (from Robot {robot1}) ↔ Cell {cell2} (from Robot {robot2})")
        print(f"      • Before: Robot {robot1} path length = {len(neighbor.paths[robot1])}, Robot {robot2} path length = {len(neighbor.paths[robot2])}")
    
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
    
    if verbose:
        print(f"      • After: Robot {robot1} path length = {len(neighbor.paths[robot1])}, Robot {robot2} path length = {len(neighbor.paths[robot2])}")
    
    return neighbor

def simulated_annealing(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                       initial_temp=1000, cooling_rate=0.95, max_iterations=1000):
    """
    Simulated Annealing algorithm
    Returns: (best_solution, convergence_history)
    """
    
    print(f"\n{'='*70}")
    print(f"🔥 STARTING SIMULATED ANNEALING")
    print(f"{'='*70}")
    print(f"📋 Parameters:")
    print(f"   • Initial Temperature (T0): {initial_temp}")
    print(f"   • Cooling Rate: {cooling_rate}")
    print(f"   • Max Iterations: {max_iterations}")
    print(f"   • Grid Size: {grid_width}x{grid_height}")
    print(f"   • Number of Robots: {num_robots}")
    print(f"   • Free Cells: {len(free_cells)}")
    print(f"   • Obstacles: {len(obstacles)}")
    print(f"{'='*70}\n")
    
    # Step 1: Generate random starting solution
    print(f"🔄 STEP 1: Generating Initial Random Solution...")
    current_solution = generate_random_solution(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots
    )
    current_solution.evaluate()
    
    # Keep track of the best solution found so far
    best_solution = current_solution.copy()
    best_solution.evaluate()
    
    temperature = initial_temp  # Start hot (accepts bad solutions)
    
    print(f"\n📊 Initial Solution Details:")
    print(f"   • Coverage: {current_solution.fitness['coverage_score']}/{len(free_cells)} cells")
    print(f"   • Balance Score: {current_solution.fitness['balance_score']:.3f} (lower = better)")
    print(f"   • Combined Score: {current_solution.combined_score:.3f} (lower = better)")
    print(f"   • Robot Distances: {current_solution.fitness.get('robot_distances', [])}")
    
    # Show initial assignments
    print(f"\n   📍 Initial Robot Assignments:")
    robot_cell_counts = {}
    for robot_id in range(num_robots):
        robot_cells = [i for i, row in enumerate(current_solution.assignment) if row[robot_id] == 1]
        robot_cell_counts[robot_id] = len(robot_cells)
        print(f"      Robot {robot_id}: {len(robot_cells)} cells, Path length: {len(current_solution.paths[robot_id]) if robot_id < len(current_solution.paths) else 0}")
    
    # Show violations if any
    if current_solution.fitness.get('problems'):
        print(f"\n   ⚠️  Initial Violations: {len(current_solution.fitness['problems'])}")
        for i, problem in enumerate(current_solution.fitness['problems'][:5]):  # Show first 5
            print(f"      {i+1}. {problem}")
        if len(current_solution.fitness['problems']) > 5:
            print(f"      ... and {len(current_solution.fitness['problems']) - 5} more")
    else:
        print(f"\n   ✅ No constraint violations in initial solution!")
    
    print(f"\n{'─'*70}")
    
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
    print(f"🔄 STEP 2: Starting Main SA Loop ({max_iterations} iterations)...\n")
    
    accepted_count = 0
    improved_count = 0
    worse_accepted_count = 0
    
    for iteration in range(max_iterations):
        # Step 2: Generate neighbor solution (slight change)
        verbose_iteration = (iteration < 5) or (iteration % 100 == 0) or (iteration == max_iterations - 1)
        neighbor = generate_neighbor_solution(current_solution, verbose=verbose_iteration)
        neighbor.evaluate()
        
        # Step 3: Calculate if neighbor is better or worse
        delta = neighbor.combined_score - current_solution.combined_score
        
        # Detailed tracing for first few iterations
        verbose_iteration = (iteration < 5) or (iteration % 100 == 0) or (iteration == max_iterations - 1)
        
        if verbose_iteration:
            print(f"\n{'─'*70}")
            print(f"🔄 Iteration {iteration}/{max_iterations-1}")
            print(f"   Temperature: {temperature:.2f}")
            print(f"   Current Score: {current_solution.combined_score:.3f}")
            print(f"   Neighbor Score: {neighbor.combined_score:.3f}")
            print(f"   Delta (Δ): {delta:.3f} ({'✅ Better' if delta < 0 else '❌ Worse'})")
        
        # Step 4: Accept or reject neighbor
        accepted = False
        if delta < 0:  # Neighbor is better - always accept
            current_solution = neighbor
            accepted = True
            accepted_count += 1
            improved_count += 1
            
            if verbose_iteration:
                print(f"   ✅ ACCEPTED (Better solution - always accept)")
            
            if neighbor.combined_score < best_score:
                old_best = best_score
                best_solution = neighbor.copy()
                best_score = neighbor.combined_score
                improvement = old_best - best_score
                
                if verbose_iteration:
                    print(f"   🎉 NEW BEST! Old: {old_best:.3f} → New: {best_score:.3f} (Improvement: {improvement:.3f})")
                    print(f"      Coverage: {neighbor.fitness['coverage_score']}/{len(free_cells)}")
                    print(f"      Balance: {neighbor.fitness['balance_score']:.3f}")
        else:  # Neighbor is worse - accept with probability
            acceptance_prob = math.exp(-delta / temperature) if temperature > 0 else 0
            random_val = random.random()
            
            if verbose_iteration:
                print(f"   📊 Acceptance Probability: exp(-{delta:.3f}/{temperature:.2f}) = {acceptance_prob:.4f}")
                print(f"   🎲 Random Value: {random_val:.4f}")
            
            if random_val < acceptance_prob:
                current_solution = neighbor
                accepted = True
                accepted_count += 1
                worse_accepted_count += 1
                
                if verbose_iteration:
                    print(f"   ✅ ACCEPTED (Worse solution - accepted with probability {acceptance_prob:.4f})")
            else:
                if verbose_iteration:
                    print(f"   ❌ REJECTED (Worse solution - probability too low)")
        
        # Track convergence
        convergence_history['iteration'].append(iteration)
        convergence_history['best_score'].append(best_score)
        convergence_history['current_score'].append(current_solution.combined_score)
        convergence_history['temperature'].append(temperature)
        
        if best_solution.fitness:
            convergence_history['best_coverage'].append(best_solution.fitness['coverage_score'])
            convergence_history['best_balance'].append(best_solution.fitness['balance_score'])
        
        # Step 5: Cool down temperature (become more picky)
        old_temp = temperature
        temperature *= cooling_rate
        
        if verbose_iteration:
            print(f"   🌡️  Temperature: {old_temp:.2f} → {temperature:.2f} (cooling rate: {cooling_rate})")
        
        # Print summary every 10 iterations (or last iteration)
        if (iteration + 1) % 10 == 0 or iteration == max_iterations - 1:
            acceptance_rate = (accepted_count / (iteration + 1)) * 100
            print(f"\n📊 Progress Summary (Iteration {iteration+1}/{max_iterations}):")
            print(f"   • Current Score: {current_solution.combined_score:.3f}")
            print(f"   • Best Score: {best_score:.3f}")
            print(f"   • Temperature: {temperature:.2f}")
            print(f"   • Acceptance Rate: {acceptance_rate:.1f}% ({accepted_count}/{iteration+1})")
            print(f"   • Improved Solutions: {improved_count}")
            print(f"   • Worse Accepted: {worse_accepted_count}")
    
    print(f"\n{'='*70}")
    print(f"✅ SIMULATED ANNEALING COMPLETE!")
    print(f"{'='*70}\n")
    
    # Make sure best solution is evaluated
    if best_solution.fitness is None:
        best_solution.evaluate()
    
    # Final statistics
    final_acceptance_rate = (accepted_count / max_iterations) * 100
    initial_score = convergence_history['best_score'][0] if convergence_history['best_score'] else float('inf')
    final_improvement = initial_score - best_score if initial_score != float('inf') else 0
    
    print(f"📊 FINAL STATISTICS:")
    print(f"   • Total Iterations: {max_iterations}")
    print(f"   • Solutions Accepted: {accepted_count} ({final_acceptance_rate:.1f}%)")
    print(f"   • Improved Solutions: {improved_count}")
    print(f"   • Worse Solutions Accepted: {worse_accepted_count}")
    print(f"   • Initial Score: {initial_score:.3f}")
    print(f"   • Final Best Score: {best_score:.3f}")
    print(f"   • Total Improvement: {final_improvement:.3f} ({((final_improvement/initial_score)*100):.2f}% better)")
    
    if best_solution.fitness is not None:
        print(f"\n🏆 BEST SOLUTION FOUND:")
        print(f"   • Coverage: {best_solution.fitness['coverage_score']}/{len(free_cells)} cells ({best_solution.fitness['coverage_score']/len(free_cells)*100:.1f}%)")
        print(f"   • Balance Score: {best_solution.fitness['balance_score']:.3f} (lower = better)")
        print(f"   • Combined Score: {best_solution.combined_score:.3f} (lower = better)")
        print(f"   • Robot Distances: {best_solution.fitness.get('robot_distances', [])}")
        
        # Show final assignments
        print(f"\n   📍 Final Robot Assignments:")
        for robot_id in range(num_robots):
            robot_cells = [i for i, row in enumerate(best_solution.assignment) if row[robot_id] == 1]
            path_len = len(best_solution.paths[robot_id]) if robot_id < len(best_solution.paths) else 0
            print(f"      Robot {robot_id}: {len(robot_cells)} cells, Path: {path_len} cells")
        
        # Show violations if any
        if best_solution.fitness.get('problems'):
            print(f"\n   ⚠️  Constraint Violations: {len(best_solution.fitness['problems'])}")
            for i, problem in enumerate(best_solution.fitness['problems'][:10]):  # Show first 10
                print(f"      {i+1}. {problem}")
            if len(best_solution.fitness['problems']) > 10:
                print(f"      ... and {len(best_solution.fitness['problems']) - 10} more")
        else:
            print(f"\n   ✅ No constraint violations!")
    else:
        print("\n❌ Best solution evaluation failed")
    
    print(f"\n{'='*70}\n")
    
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
