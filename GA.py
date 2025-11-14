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
                      population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1, 
                      elitism_count=2, verbose=True):
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
    print(f"   • Crossover Rate:     {crossover_rate}")
    print(f"   • Mutation Rate:      {mutation_rate}")
    print(f"   • Elitism Count:      {elitism_count}")
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
    print(f"      • Best Score:     {best_score:.3f}")
    print(f"      • Average Score:  {initial_avg:.3f}")
    print(f"      • Worst Score:    {initial_worst:.3f}")
    
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
        'mutation_count': [],
        'elite_preserved': []
    }
    
    # Counters for operations
    total_crossovers = 0
    total_mutations = 0
    
    # Main GA loop - evolve for multiple generations
    for generation in range(generations):
        
        if verbose and (generation % 10 == 0 or generation < 3):
            print(f"\n{'─'*70}")
            print(f"🔄 GENERATION {generation}")
            print(f"{'─'*70}")
        
        new_population = []
        gen_crossovers = 0
        gen_mutations = 0
        
        # Step 2: Create new generation
        if verbose and generation < 3:
            print(f"   🧬 Creating new population through selection, crossover, and mutation...")
        
        offspring_count = 0
        while len(new_population) < population_size:
            # Select parents using tournament selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            if verbose and generation < 2 and offspring_count < 2:
                print(f"      Offspring {offspring_count + 1}:")
                print(f"         • Selected Parent 1 (score: {parent1.combined_score:.3f})")
                print(f"         • Selected Parent 2 (score: {parent2.combined_score:.3f})")
            
            # Create child through crossover
            child_before_mutation = crossover(parent1, parent2, crossover_rate)
            did_crossover = (child_before_mutation.assignment != parent1.assignment)
            if did_crossover:
                gen_crossovers += 1
                if verbose and generation < 2 and offspring_count < 2:
                    print(f"         • ✂️  Crossover applied!")
            
            # Mutate child
            child = mutate(child_before_mutation, mutation_rate)
            did_mutate = (child.assignment != child_before_mutation.assignment)
            if did_mutate:
                gen_mutations += 1
                if verbose and generation < 2 and offspring_count < 2:
                    print(f"         • 🧪 Mutation applied!")
            
            # Evaluate child
            child.evaluate()
            
            if verbose and generation < 2 and offspring_count < 2:
                print(f"         • 📊 Child score: {child.combined_score:.3f}")
            
            # Add to new population
            new_population.append(child)
            offspring_count += 1
        
        total_crossovers += gen_crossovers
        total_mutations += gen_mutations
        
        if verbose and generation < 3:
            print(f"   📊 Generation {generation} Operations:")
            print(f"      • Crossovers:  {gen_crossovers}/{population_size}")
            print(f"      • Mutations:   {gen_mutations}/{population_size}")
        
        # Step 3: Apply elitism (keep best solutions)
        if verbose and generation < 3:
            print(f"   👑 Preserving top {elitism_count} elite solutions...")
        
        new_population = apply_elitism(population, new_population, elitism_count)
        
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
                print(f"      • Old Best: {old_best:.3f}")
                print(f"      • New Best: {best_score:.3f}")
                print(f"      • Improvement: {old_best - best_score:.3f}")
        
        # Record metrics for this generation
        scores = [sol.combined_score if sol.combined_score is not None else float('inf') 
                  for sol in population]
        convergence_history['generation'].append(generation)
        convergence_history['best_score'].append(min(scores))
        convergence_history['avg_score'].append(sum(scores) / len(scores))
        convergence_history['worst_score'].append(max(scores))
        convergence_history['crossover_count'].append(gen_crossovers)
        convergence_history['mutation_count'].append(gen_mutations)
        convergence_history['elite_preserved'].append(elitism_count)
        
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
            print(f"      • Best Score:     {best_score:.3f}")
            print(f"      • Current Best:   {current_best_score:.3f}")
            print(f"      • Average:        {avg_score:.3f}")
            print(f"      • Worst:          {max(scores):.3f}")
            if best_solution.fitness:
                print(f"      • Coverage:       {best_solution.fitness['coverage_score']}/{len(free_cells)} cells")
                print(f"      • Balance:        {best_solution.fitness['balance_score']:.3f}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ GENETIC ALGORITHM COMPLETE!")
    print(f"{'='*70}")
    print(f"📊 Final Statistics:")
    print(f"   • Total Generations:      {generations}")
    print(f"   • Total Crossovers:       {total_crossovers}")
    print(f"   • Total Mutations:        {total_mutations}")
    print(f"   • Best Score Achieved:    {best_score:.3f}")
    
    if best_solution.fitness is not None:
        print(f"   • Final Coverage:         {best_solution.fitness['coverage_score']}/{len(free_cells)} cells ({best_solution.fitness['coverage_score']/len(free_cells)*100:.1f}%)")
        print(f"   • Final Balance:          {best_solution.fitness['balance_score']:.3f}")
        print(f"   • Constraint Violations:  {len(best_solution.fitness['problems'])}")
    
    print(f"{'='*70}\n")
    
    # Ensure best solution is evaluated
    if best_solution.fitness is None:
        best_solution.evaluate()
    
    return best_solution, convergence_history

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
            population_size=pop_size, generations=50, crossover_rate=0.8, mutation_rate=0.1
        )
        results[f'pop_{pop_size}'] = {
            'solution': solution,
            'history': history,
            'metrics': solution.get_all_performance_metrics(),
            'parameter': 'population_size',
            'value': pop_size
        }
        print(f"     ✓ Best Score: {solution.combined_score:.4f}")
    
    # Test 2: Mutation Rate Effect
    print("\n[TEST 2/4] Mutation Rate Effect")
    print("-" * 70)
    for mut_rate in [0.05, 0.1, 0.2, 0.3]:
        print(f"  → Running with mutation_rate={mut_rate}...")
        solution, history = genetic_algorithm(
            all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
            population_size=50, generations=50, crossover_rate=0.8, mutation_rate=mut_rate
        )
        results[f'mut_{mut_rate}'] = {
            'solution': solution,
            'history': history,
            'metrics': solution.get_all_performance_metrics(),
            'parameter': 'mutation_rate',
            'value': mut_rate
        }
        print(f"     ✓ Best Score: {solution.combined_score:.4f}")
    
    # Test 3: Crossover Rate Effect
    print("\n[TEST 3/4] Crossover Rate Effect")
    print("-" * 70)
    for cross_rate in [0.6, 0.8, 0.95]:
        print(f"  → Running with crossover_rate={cross_rate}...")
        solution, history = genetic_algorithm(
            all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
            population_size=50, generations=50, crossover_rate=cross_rate, mutation_rate=0.1
        )
        results[f'cross_{cross_rate}'] = {
            'solution': solution,
            'history': history,
            'metrics': solution.get_all_performance_metrics(),
            'parameter': 'crossover_rate',
            'value': cross_rate
        }
        print(f"     ✓ Best Score: {solution.combined_score:.4f}")
    
    # Test 4: Generation Count Effect
    print("\n[TEST 4/4] Generation Count Effect")
    print("-" * 70)
    for gens in [25, 50, 100, 200]:
        print(f"  → Running with generations={gens}...")
        solution, history = genetic_algorithm(
            all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
            population_size=50, generations=gens, crossover_rate=0.8, mutation_rate=0.1
        )
        results[f'gen_{gens}'] = {
            'solution': solution,
            'history': history,
            'metrics': solution.get_all_performance_metrics(),
            'parameter': 'generations',
            'value': gens
        }
        print(f"     ✓ Best Score: {solution.combined_score:.4f}")
    
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
    print(f"  • Combined Score (minimize):    {metrics['combined_score']:.4f}")
    
    print(f"\n📈 RAW SCORES:")
    print(f"  • Coverage:                     {metrics['cells_covered']}/{metrics['total_free_cells']} cells")
    print(f"  • Balance (variance):           {metrics['raw_balance']:.4f}")
    print(f"  • Total Violations:             {metrics['total_violations']}")
    
    if convergence_history:
        analysis = analyze_convergence(convergence_history)
        print(f"\n🔄 CONVERGENCE ANALYSIS:")
        print(f"  • Total Generations:            {analysis['total_generations']}")
        print(f"  • Initial Score:                {analysis['initial_score']:.4f}")
        print(f"  • Final Score:                  {analysis['final_score']:.4f}")
        print(f"  • Best Score Ever:              {analysis['best_score_ever']:.4f}")
        
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
                  f"{metrics['combined_score']:>12.4f}")
        
        # Find best value
        best_entry = min(param_data, key=lambda x: x[1]['combined_score'])
        print(f"\n  ✓ Best {param_name}: {best_entry[0]} (Score: {best_entry[1]['combined_score']:.4f})")
    
    print("\n" + "="*70 + "\n")

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
    best_solution, convergence_history = genetic_algorithm(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.1, elitism_count=2
    )
    
    # Print detailed results
    print_ga_results(best_solution)
