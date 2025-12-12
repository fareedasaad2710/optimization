"""
Discrete Dragonfly Algorithm for Multi-Robot Coverage Path Planning
===================================================================

This implementation adapts the Discrete Dragonfly Algorithm (DDA) for solving
the multi-robot coverage path planning problem.

Based on: "An Optimized Discrete Dragonfly Algorithm Tackling the Low 
Exploitation Problem for Solving TSP" by Emambocus et al., 2022

The algorithm uses:
- Separation: Avoid crowding neighbors
- Alignment: Match neighbors' velocity
- Cohesion: Move toward neighbors
- Attraction: Move toward food (best solution)
- Distraction: Move away from enemies (worst solution)
- Local search: Hill climbing for exploitation

USAGE EXAMPLE:
    from discrete_dragonfly import DiscreteDragonflyAlgorithm
    from problem_formulation import create_grid_cells
    
    # Setup problem
    grid_width, grid_height = 5, 5
    num_robots = 2
    obstacles = [5, 10, 15]
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(len(all_cells)) if i not in obstacles]
    
    # Run algorithm
    dda = DiscreteDragonflyAlgorithm(
        population_size=15,
        max_iterations=50,
        neighborhood_size=3,
        local_search_prob=0.3
    )
    best_solution = dda.run(all_cells, free_cells, obstacles, 
                           grid_width, grid_height, num_robots)
    
    # Access results
    print(f"Coverage: {best_solution.fitness['coverage_score']}")
    print(f"Balance: {best_solution.fitness['balance_score']}")
    print(f"Paths: {best_solution.paths}")
"""

import random
import math
import copy
from typing import List, Dict, Tuple, Any
import sys
import os

# Add parent directory to path to import problem_formulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from problem_formulation import (
    evaluate_solution,
    find_neighbors,
    distance_between_points,
    create_grid_cells
)


class DragonflySolution:
    """Represents a dragonfly (solution) in the swarm"""
    
    def __init__(self, assignment, paths, all_cells, free_cells, obstacles, 
                 grid_width, grid_height):
        self.assignment = copy.deepcopy(assignment)
        self.paths = copy.deepcopy(paths)
        self.all_cells = all_cells
        self.free_cells = free_cells
        self.obstacles = obstacles
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fitness = None
        self.combined_score = None
        
        # Velocity for discrete operations (swap sequences)
        self.velocity = {}  # Will store swap operations
        
    def evaluate(self):
        """Evaluate the solution and return combined score"""
        # Convert all_cells to objects if needed
        if len(self.all_cells) > 0 and isinstance(self.all_cells[0], tuple):
            # Create Cell class locally
            class Cell:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            cells_as_objects = [Cell(x, y) for x, y in self.all_cells]
        elif len(self.all_cells) > 0 and hasattr(self.all_cells[0], 'x') and hasattr(self.all_cells[0], 'y'):
            cells_as_objects = self.all_cells
        else:
            cells_as_objects = self.all_cells
            
        results = evaluate_solution(
            self.assignment, self.paths, cells_as_objects,
            self.free_cells, self.obstacles, self.grid_width, self.grid_height
        )
        self.fitness = results
        
        # Calculate combined score (lower is better)
        # F1: Maximize coverage (minimize negative coverage)
        coverage_ratio = results['coverage_score'] / len(self.free_cells) if len(self.free_cells) > 0 else 0
        coverage_term = 1 - coverage_ratio  # Convert to minimization
        
        # F2: Minimize workload imbalance
        imbalance_term = results['balance_score']
        
        # Penalty for violations
        penalty = len(results['problems']) * 100
        
        # Weighted combination
        w1, w2 = 0.7, 0.3
        if coverage_ratio >= 1.0:
            w1, w2 = 0.5, 0.5
            
        self.combined_score = w1 * coverage_term + w2 * imbalance_term + penalty
        return self.combined_score
    
    def copy(self):
        """Create a deep copy"""
        new_solution = DragonflySolution(
            copy.deepcopy(self.assignment),
            copy.deepcopy(self.paths),
            self.all_cells,
            self.free_cells,
            self.obstacles,
            self.grid_width,
            self.grid_height
        )
        if self.fitness is not None:
            new_solution.fitness = copy.deepcopy(self.fitness)
        if self.combined_score is not None:
            new_solution.combined_score = self.combined_score
        return new_solution


def generate_random_solution(all_cells, free_cells, obstacles, grid_width, 
                            grid_height, num_robots):
    """Generate a random initial solution"""
    total_cells = len(all_cells)
    
    # Create assignment matrix
    assignment = [[0 for _ in range(num_robots)] for _ in range(total_cells)]
    
    # Randomly assign each free cell to a robot
    for cell_idx in free_cells:
        robot_id = random.randint(0, num_robots - 1)
        assignment[cell_idx][robot_id] = 1
    
    # Generate random paths for each robot
    robot_paths = {}
    for robot_id in range(num_robots):
        robot_cells = [cell_idx for cell_idx in free_cells 
                      if assignment[cell_idx][robot_id] == 1]
        random.shuffle(robot_cells)
        robot_paths[robot_id] = robot_cells
    
    return DragonflySolution(
        assignment, robot_paths, all_cells, free_cells, 
        obstacles, grid_width, grid_height
    )


def apply_swap_sequence(path, swap_sequence):
    """Apply swap sequence to a path (discrete operation)"""
    new_path = path.copy()
    for i, j in swap_sequence:
        if 0 <= i < len(new_path) and 0 <= j < len(new_path):
            new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path


def generate_swap_sequence(path1, path2, max_swaps=5):
    """Generate swap sequence to transform path1 toward path2"""
    swap_seq = []
    new_path = path1.copy()
    
    # Find differences and create swaps
    for _ in range(min(max_swaps, len(path1))):
        # Find first position where paths differ
        diff_pos = None
        for i in range(min(len(new_path), len(path2))):
            if new_path[i] != path2[i]:
                diff_pos = i
                break
        
        if diff_pos is None:
            break
            
        # Find where the correct element is in new_path
        target_val = path2[diff_pos]
        target_pos = None
        for j in range(diff_pos + 1, len(new_path)):
            if new_path[j] == target_val:
                target_pos = j
                break
        
        if target_pos is not None:
            swap_seq.append((diff_pos, target_pos))
            new_path[diff_pos], new_path[target_pos] = new_path[target_pos], new_path[diff_pos]
    
    return swap_seq


def combine_swap_sequences(seq1, seq2, weight=0.5):
    """Combine two swap sequences"""
    combined = seq1.copy()
    # Add some swaps from seq2 based on weight
    num_from_seq2 = int(len(seq2) * weight)
    combined.extend(seq2[:num_from_seq2])
    return combined


def update_dragonfly_position(dragonfly, food, enemy, neighbors, 
                              separation_weight=0.1, alignment_weight=0.1,
                              cohesion_weight=0.1, food_weight=0.2, 
                              enemy_weight=0.1):
    """Update dragonfly position using swarm behaviors"""
    new_paths = copy.deepcopy(dragonfly.paths)
    
    for robot_id in dragonfly.paths.keys():
        if robot_id not in food.paths or robot_id not in dragonfly.paths:
            continue
            
        # Separation: Move away from neighbors
        separation_seq = []
        if neighbors:
            for neighbor in neighbors:
                if robot_id in neighbor.paths:
                    swap_seq = generate_swap_sequence(
                        dragonfly.paths[robot_id], 
                        neighbor.paths[robot_id], 
                        max_swaps=2
                    )
                    # Reverse to move away
                    separation_seq.extend([(j, i) for i, j in swap_seq])
        
        # Alignment: Match neighbors' velocity
        alignment_seq = []
        if neighbors:
            for neighbor in neighbors:
                if robot_id in neighbor.paths and hasattr(neighbor, 'velocity'):
                    if robot_id in neighbor.velocity:
                        alignment_seq.extend(neighbor.velocity[robot_id])
        
        # Cohesion: Move toward neighbors
        cohesion_seq = []
        if neighbors:
            avg_path = dragonfly.paths[robot_id].copy()
            # Simple approach: move toward average
            for neighbor in neighbors:
                if robot_id in neighbor.paths:
                    swap_seq = generate_swap_sequence(
                        dragonfly.paths[robot_id],
                        neighbor.paths[robot_id],
                        max_swaps=1
                    )
                    cohesion_seq.extend(swap_seq)
        
        # Attraction: Move toward food (best solution)
        food_seq = []
        if robot_id in food.paths:
            food_seq = generate_swap_sequence(
                dragonfly.paths[robot_id],
                food.paths[robot_id],
                max_swaps=3
            )
        
        # Distraction: Move away from enemy (worst solution)
        enemy_seq = []
        if enemy and robot_id in enemy.paths:
            enemy_swap = generate_swap_sequence(
                dragonfly.paths[robot_id],
                enemy.paths[robot_id],
                max_swaps=2
            )
            # Reverse to move away
            enemy_seq = [(j, i) for i, j in enemy_swap]
        
        # Combine all behaviors
        combined_seq = []
        combined_seq.extend(separation_seq[:int(len(separation_seq) * separation_weight)])
        combined_seq.extend(alignment_seq[:int(len(alignment_seq) * alignment_weight)])
        combined_seq.extend(cohesion_seq[:int(len(cohesion_seq) * cohesion_weight)])
        combined_seq.extend(food_seq[:int(len(food_seq) * food_weight)])
        combined_seq.extend(enemy_seq[:int(len(enemy_seq) * enemy_weight)])
        
        # Apply swap sequence
        new_paths[robot_id] = apply_swap_sequence(
            dragonfly.paths[robot_id], 
            combined_seq[:5]  # Limit swaps
        )
        
        # Repair path to ensure continuity
        new_paths[robot_id] = repair_path(
            new_paths[robot_id],
            dragonfly.all_cells,
            dragonfly.free_cells,
            dragonfly.obstacles,
            dragonfly.grid_width,
            dragonfly.grid_height
        )
        
        # Store velocity
        dragonfly.velocity[robot_id] = combined_seq[:5]
    
    # Update paths
    dragonfly.paths = new_paths
    
    # Remove duplicates and sync assignment with paths
    for cell_idx in range(len(dragonfly.assignment)):
        for r in range(len(dragonfly.assignment[cell_idx])):
            dragonfly.assignment[cell_idx][r] = 0
    
    # Remove duplicate cells from paths and assign
    for robot_id, path in dragonfly.paths.items():
        # Remove duplicates while preserving order
        seen = set()
        unique_path = []
        for cell_idx in path:
            if cell_idx not in seen and cell_idx < len(dragonfly.assignment):
                unique_path.append(cell_idx)
                seen.add(cell_idx)
        dragonfly.paths[robot_id] = unique_path
        
        # Assign cells to robot
        for cell_idx in unique_path:
            dragonfly.assignment[cell_idx][robot_id] = 1


def repair_path(path, all_cells, free_cells, obstacles, grid_width, grid_height):
    """Repair path to ensure continuity by inserting intermediate cells"""
    if len(path) < 2:
        return path
    
    repaired = [path[0]]
    free_set = set(free_cells)
    obstacle_set = set(obstacles)
    
    for i in range(len(path) - 1):
        current_idx = repaired[-1]
        target_idx = path[i + 1]
        
        if current_idx == target_idx:
            continue
            
        # Get current and target coordinates
        if isinstance(all_cells[current_idx], tuple):
            cx, cy = all_cells[current_idx]
            tx, ty = all_cells[target_idx]
        else:
            cx, cy = all_cells[current_idx].x, all_cells[current_idx].y
            tx, ty = all_cells[target_idx].x, all_cells[target_idx].y
        
        # Find path using simple A* approach (Manhattan distance)
        while (cx, cy) != (tx, ty):
            # Try to move toward target
            next_x, next_y = cx, cy
            
            # Move horizontally first if needed
            if cx != tx:
                next_x = cx + (1 if tx > cx else -1)
            # Then move vertically
            elif cy != ty:
                next_y = cy + (1 if ty > cy else -1)
            
            # Find cell index for next position
            next_idx = None
            for idx, cell in enumerate(all_cells):
                if isinstance(cell, tuple):
                    cell_x, cell_y = cell
                else:
                    cell_x, cell_y = cell.x, cell.y
                    
                if (cell_x, cell_y) == (next_x, next_y):
                    if idx in free_set and idx not in obstacle_set:
                        next_idx = idx
                        break
            
            if next_idx is not None and next_idx not in repaired:
                repaired.append(next_idx)
                cx, cy = next_x, next_y
            else:
                # If can't move directly, try alternative path
                break
    
    # Ensure target is added
    if repaired[-1] != path[-1] and path[-1] not in repaired:
        repaired.append(path[-1])
    
    return repaired


def hill_climbing_local_search(solution, max_iterations=10):
    """Steepest ascent hill climbing for local exploitation"""
    current = solution.copy()
    current.evaluate()
    best_score = current.combined_score
    
    for _ in range(max_iterations):
        improved = False
        
        # Try swapping adjacent elements in paths
        for robot_id in current.paths.keys():
            path = current.paths[robot_id]
            if len(path) < 2:
                continue
                
            # Try all adjacent swaps
            for i in range(len(path) - 1):
                neighbor = current.copy()
                # Swap adjacent elements
                neighbor.paths[robot_id][i], neighbor.paths[robot_id][i+1] = \
                    neighbor.paths[robot_id][i+1], neighbor.paths[robot_id][i]
                
                # Sync assignment
                for cell_idx in range(len(neighbor.assignment)):
                    for r in range(len(neighbor.assignment[cell_idx])):
                        neighbor.assignment[cell_idx][r] = 0
                for rid, p in neighbor.paths.items():
                    for cell_idx in p:
                        if cell_idx < len(neighbor.assignment):
                            neighbor.assignment[cell_idx][rid] = 1
                
                neighbor.evaluate()
                
                if neighbor.combined_score < best_score:
                    current = neighbor
                    best_score = neighbor.combined_score
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return current


class DiscreteDragonflyAlgorithm:
    """Discrete Dragonfly Algorithm for Multi-Robot Coverage"""
    
    def __init__(self, population_size=20, max_iterations=100, 
                 neighborhood_size=5, local_search_prob=0.3):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.neighborhood_size = neighborhood_size
        self.local_search_prob = local_search_prob
        self.population = []
        self.food = None  # Best solution
        self.enemy = None  # Worst solution
        
    def initialize_population(self, all_cells, free_cells, obstacles, 
                              grid_width, grid_height, num_robots):
        """Initialize dragonfly population"""
        self.population = []
        for _ in range(self.population_size):
            solution = generate_random_solution(
                all_cells, free_cells, obstacles,
                grid_width, grid_height, num_robots
            )
            solution.evaluate()
            self.population.append(solution)
        
        # Find food (best) and enemy (worst)
        self.population.sort(key=lambda x: x.combined_score)
        self.food = self.population[0].copy()
        self.enemy = self.population[-1].copy()
    
    def get_neighbors(self, dragonfly, k=None):
        """Get k nearest neighbors of a dragonfly"""
        if k is None:
            k = self.neighborhood_size
        
        # Calculate distances (based on fitness)
        distances = []
        for other in self.population:
            if other != dragonfly:
                dist = abs(other.combined_score - dragonfly.combined_score)
                distances.append((dist, other))
        
        distances.sort(key=lambda x: x[0])
        return [d[1] for d in distances[:k]]
    
    def run(self, all_cells, free_cells, obstacles, grid_width, 
            grid_height, num_robots):
        """Run the discrete dragonfly algorithm"""
        print("Initializing population...")
        self.initialize_population(
            all_cells, free_cells, obstacles, grid_width, grid_height, num_robots
        )
        
        print(f"Initial best score: {self.food.combined_score:.4f}")
        print(f"Initial coverage: {self.food.fitness['coverage_score']}/{len(free_cells)}")
        
        for iteration in range(self.max_iterations):
            # Update each dragonfly
            for dragonfly in self.population:
                neighbors = self.get_neighbors(dragonfly)
                
                # Update position using swarm behaviors
                update_dragonfly_position(
                    dragonfly, self.food, self.enemy, neighbors
                )
                
                # Evaluate new position
                dragonfly.evaluate()
                
                # Apply local search with probability
                if random.random() < self.local_search_prob:
                    dragonfly = hill_climbing_local_search(dragonfly)
            
            # Update food and enemy
            self.population.sort(key=lambda x: x.combined_score)
            if self.population[0].combined_score < self.food.combined_score:
                self.food = self.population[0].copy()
            
            self.enemy = self.population[-1].copy()
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}: "
                      f"Best score: {self.food.combined_score:.4f}, "
                      f"Coverage: {self.food.fitness['coverage_score']}/{len(free_cells)}")
        
        print(f"\nFinal best score: {self.food.combined_score:.4f}")
        print(f"Final coverage: {self.food.fitness['coverage_score']}/{len(free_cells)}")
        print(f"Final balance score: {self.food.fitness['balance_score']:.4f}")
        
        return self.food


def main():
    """Example usage"""
    # Simple test case
    grid_width = 5
    grid_height = 5
    num_robots = 2
    obstacles = [5, 10, 15]  # Some obstacle cells
    
    # Create grid
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(len(all_cells)) if i not in obstacles]
    
    print("=" * 60)
    print("Discrete Dragonfly Algorithm for Multi-Robot Coverage")
    print("=" * 60)
    print(f"Grid: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Free cells: {len(free_cells)}")
    print(f"Obstacles: {obstacles}")
    print("=" * 60)
    
    # Run algorithm
    dda = DiscreteDragonflyAlgorithm(
        population_size=15,
        max_iterations=50,
        neighborhood_size=3,
        local_search_prob=0.3
    )
    
    best_solution = dda.run(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots
    )
    
    print("\n" + "=" * 60)
    print("BEST SOLUTION DETAILS")
    print("=" * 60)
    print(f"Coverage: {best_solution.fitness['coverage_score']}/{len(free_cells)}")
    print(f"Balance Score: {best_solution.fitness['balance_score']:.4f}")
    print(f"Combined Score: {best_solution.combined_score:.4f}")
    print(f"\nRobot Paths:")
    for robot_id, path in best_solution.paths.items():
        print(f"  Robot {robot_id}: {path}")
    print(f"\nProblems: {len(best_solution.fitness['problems'])}")
    if best_solution.fitness['problems']:
        for problem in best_solution.fitness['problems'][:5]:
            print(f"  - {problem}")


if __name__ == "__main__":
    main()

