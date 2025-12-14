
# ============================================================================
# GENETIC ALGORITHM IMPLEMENTATION - SELF-CONTAINED (FEASIBILITY-BASED)
# ============================================================================
# This file contains the COMPLETE Genetic Algorithm implementation
# No external GA.py dependency - everything is included here
# 
# IMPORTANT: Uses FEASIBILITY-BASED approach (like Dragonfly/Ant Colony)
# - NO PENALTIES - solutions are either feasible or REJECTED
# - Only accepts solutions that satisfy ALL constraints
# ============================================================================

import math
import random
import copy
import collections
from collections import deque
from typing import List, Dict, Tuple, Optional
from problem_formulation import *
from visualization import visualize_solution, plot_convergence_history, plot_best_score_only
import time
import os
from datetime import datetime


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_format_score(score, decimals=3):
    """Format score safely (handles None, inf, -inf)"""
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


# ============================================================================
# DARP PARTITIONING (from Dragonfly.py)
# ============================================================================

def darp_partition(
    grid_width: int,
    grid_height: int,
    robot_positions: List[int],
    obstacles: List[int],
) -> Dict[int, List[int]]:
    """
    Simple DARP: Distributes free cells evenly across robots.
    Assigns each cell to the nearest robot.
    """
    num_robots = len(robot_positions)
    all_cells_count = grid_width * grid_height
    free_cells = [i for i in range(all_cells_count) if i not in obstacles]
    
    # Create empty partitions
    partition = {robot_id: [] for robot_id in range(num_robots)}
    
    # Helper function to calculate Manhattan distance
    def manhattan_distance(cell1_idx, cell2_idx):
        x1, y1 = cell1_idx % grid_width, cell1_idx // grid_width
        x2, y2 = cell2_idx % grid_width, cell2_idx // grid_width
        return abs(x1 - x2) + abs(y1 - y2)
    
    # Assign each free cell to nearest robot
    for cell_idx in free_cells:
        min_dist = float('inf')
        nearest_robot = 0
        
        for robot_id, robot_pos in enumerate(robot_positions):
            dist = manhattan_distance(cell_idx, robot_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_robot = robot_id
        
        partition[nearest_robot].append(cell_idx)
    
    return partition


# ============================================================================
# SPANNING TREE PATH CONSTRUCTION (from ant3.py)
# ============================================================================

def build_spanning_tree_path(start_cell_idx, assigned_cells, all_cells, free_cells, obstacles, grid_width, grid_height):
    """
    UF-STC: Build spanning tree over assigned region and generate coverage path
    using DFS preorder traversal.
    """
    if not assigned_cells:
        return []
    
    # Convert cell indices to coordinates
    def get_cell_coords(cell_idx):
        cell = all_cells[cell_idx]
        if isinstance(cell, tuple):
            return cell
        else:
            return (cell.x, cell.y)
    
    def coord_to_index(coord):
        x, y = coord
        for idx, cell in enumerate(all_cells):
            cell_coord = get_cell_coords(idx)
            if cell_coord == (x, y):
                return idx
        return None
    
    def get_neighbor_indices(cell_idx):
        """Get 4-connected neighbor cell indices"""
        neighbors = find_neighbors(cell_idx, all_cells, grid_width, grid_height)
        neighbor_indices = []
        for neighbor_coord in neighbors:
            neighbor_idx = coord_to_index(neighbor_coord)
            if neighbor_idx is not None and neighbor_idx in free_cells and neighbor_idx not in obstacles:
                neighbor_indices.append(neighbor_idx)
        return neighbor_indices
    
    # Determine root cell
    if start_cell_idx in assigned_cells:
        root = start_cell_idx
    else:
        # Find closest assigned cell to start
        start_coord = get_cell_coords(start_cell_idx)
        min_dist = float('inf')
        root = None
        for cell_idx in assigned_cells:
            cell_coord = get_cell_coords(cell_idx)
            dist = abs(start_coord[0] - cell_coord[0]) + abs(start_coord[1] - cell_coord[1])
            if dist < min_dist:
                min_dist = dist
                root = cell_idx
        if root is None:
            root = list(assigned_cells)[0] if assigned_cells else start_cell_idx
    
    # Build spanning tree using BFS (only over assigned cells)
    q = collections.deque([root])
    parent = {root: None}
    visited = {root}
    
    while q:
        u = q.popleft()
        neighbors = get_neighbor_indices(u)
        for v in neighbors:
            if v in assigned_cells and v not in visited:
                parent[v] = u
                visited.add(v)
                q.append(v)
    
    # Build tree adjacency list
    tree_adj = collections.defaultdict(list)
    for node, p in parent.items():
        if p is not None:
            tree_adj[p].append(node)
    
    # Generate coverage path using DFS preorder traversal
    path = []
    
    def dfs(u):
        path.append(u)
        for v in tree_adj[u]:
            dfs(v)
            path.append(u)  # Return to parent
    
    if root in tree_adj or root in parent:
        dfs(root)
    
    # Remove consecutive duplicates
    if path:
        cleaned_path = [path[0]]
        for i in range(1, len(path)):
            if path[i] != path[i-1]:
                cleaned_path.append(path[i])
        path = cleaned_path
    
    # Handle disconnected cells (like ant3.py does)
    # Find continuous path to each disconnected cell using greedy navigation
    missing = set(assigned_cells) - set(parent.keys())
    if missing:
        last_cell = path[-1] if path else root
        root_coord = get_cell_coords(root)
        
        # Sort missing cells by distance from root
        missing_sorted = sorted(
            list(missing),
            key=lambda c: abs(get_cell_coords(c)[0] - root_coord[0]) + 
                          abs(get_cell_coords(c)[1] - root_coord[1])
        )
        
        for m in missing_sorted:
            # Find greedy path from last_cell to missing cell m
            current = last_cell
            target_coord = get_cell_coords(m)
            path_to_missing = []
            max_steps = grid_width + grid_height
            
            for step in range(max_steps):
                if current == m:
                    break
                
                current_coord = get_cell_coords(current)
                neighbors = get_neighbor_indices(current)
                
                if not neighbors:
                    break
                
                # Choose neighbor closest to target
                best_neighbor = None
                min_dist = float('inf')
                for nb in neighbors:
                    nb_coord = get_cell_coords(nb)
                    dist = abs(nb_coord[0] - target_coord[0]) + abs(nb_coord[1] - target_coord[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_neighbor = nb
                
                if best_neighbor is None:
                    break
                
                path_to_missing.append(best_neighbor)
                current = best_neighbor
            
            # Add path to missing cell if we reached it
            if path_to_missing:
                if path_to_missing[-1] == m or current == m:
                    path.extend(path_to_missing)
                    if path_to_missing[-1] != m:
                        path.append(m)
                    last_cell = m
            elif current == m:
                path.append(m)
                last_cell = m
    
    return path


# ============================================================================
# FEASIBILITY CHECKING FUNCTIONS (from Dragonfly.py / ant3.py)
# ============================================================================

def check_path_continuity(path: List[int], all_cells: List, grid_width: int, grid_height: int) -> Tuple[bool, List[str]]:
    """
    Check if consecutive cells in path are adjacent (4-connected).
    Returns: (is_valid, violations_list)
    """
    violations = []
    if len(path) <= 1:
        return True, violations
    
    for i in range(len(path) - 1):
        current_cell_idx = path[i]
        next_cell_idx = path[i + 1]
        
        if current_cell_idx < 0 or current_cell_idx >= len(all_cells):
            violations.append(f"Invalid cell index {current_cell_idx} at position {i}")
            continue
        if next_cell_idx < 0 or next_cell_idx >= len(all_cells):
            violations.append(f"Invalid cell index {next_cell_idx} at position {i+1}")
            continue
        
        # Get cell coordinates
        current_cell = all_cells[current_cell_idx]
        next_cell = all_cells[next_cell_idx]
        
        if isinstance(current_cell, tuple):
            current_x, current_y = current_cell
        else:
            current_x, current_y = current_cell.x, current_cell.y
        
        if isinstance(next_cell, tuple):
            next_x, next_y = next_cell
        else:
            next_x, next_y = next_cell.x, next_cell.y
        
        # Check if cells are adjacent (Manhattan distance = 1)
        manhattan_dist = abs(current_x - next_x) + abs(current_y - next_y)
        if manhattan_dist > 1:
            violations.append(
                f"Path jump: cell {current_cell_idx} to {next_cell_idx} not adjacent"
            )
    
    return len(violations) == 0, violations


def check_boundary_constraint(path: List[int], all_cells: List, grid_width: int, grid_height: int) -> Tuple[bool, List[str]]:
    """
    Check if all cells in path are within grid boundaries.
    Returns: (is_valid, violations_list)
    """
    violations = []
    for i, cell_idx in enumerate(path):
        if cell_idx < 0 or cell_idx >= len(all_cells):
            violations.append(f"Cell index {cell_idx} out of bounds at position {i}")
            continue
        
        cell = all_cells[cell_idx]
        if isinstance(cell, tuple):
            x, y = cell
        else:
            x, y = cell.x, cell.y
        
        if not (0 <= x < grid_width and 0 <= y < grid_height):
            violations.append(f"Cell {cell_idx} at ({x}, {y}) outside grid")
    
    return len(violations) == 0, violations


def check_obstacle_avoidance(path: List[int], obstacles: List[int]) -> Tuple[bool, List[str]]:
    """
    Check if path contains any obstacle cells.
    Returns: (is_valid, violations_list)
    """
    violations = []
    obstacles_set = set(obstacles)
    for i, cell_idx in enumerate(path):
        if cell_idx in obstacles_set:
            violations.append(f"Robot enters obstacle at cell {cell_idx}")
    return len(violations) == 0, violations


def is_solution_feasible(solution, all_cells, obstacles, grid_width, grid_height) -> Tuple[bool, List[str]]:
    """
    Check if solution satisfies ALL constraints (like Dragonfly/Ant Colony).
    Returns: (is_feasible, violations_list)
    
    A solution is FEASIBLE if:
    1. All paths are within boundaries
    2. All paths avoid obstacles
    3. All paths have continuity (adjacent cells only)
    """
    all_violations = []
    
    if not isinstance(solution.paths, dict):
        return False, [f"Paths must be a dictionary, got {type(solution.paths)}"]
    
    num_robots = len(solution.paths)
    
    for robot_id in range(num_robots):
        path = solution.paths.get(robot_id, [])
        if not isinstance(path, list):
            all_violations.append(f"Robot {robot_id}: path must be a list")
            continue
        
        # Check boundary constraint
        boundary_valid, boundary_violations = check_boundary_constraint(
            path, all_cells, grid_width, grid_height
        )
        if not boundary_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in boundary_violations])
        
        # Check obstacle avoidance
        obstacle_valid, obstacle_violations = check_obstacle_avoidance(path, obstacles)
        if not obstacle_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in obstacle_violations])
        
        # Check path continuity
        continuity_valid, continuity_violations = check_path_continuity(
            path, all_cells, grid_width, grid_height
        )
        if not continuity_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in continuity_violations])
    
    is_feasible = len(all_violations) == 0
    return is_feasible, all_violations


# ============================================================================
# ROBOT COVERAGE SOLUTION CLASS
# ============================================================================

class RobotCoverageSolution:
    """Represents one solution (assignment + paths) for multi-robot coverage"""
    
    def __init__(self, assignment, paths, all_cells, free_cells, obstacles, grid_width, grid_height):
        self.assignment = copy.deepcopy(assignment)  # Which robot covers which cell
        self.paths = copy.deepcopy(paths)           # What path each robot follows
        self.all_cells = all_cells
        self.free_cells = free_cells
        self.obstacles = obstacles
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fitness = None
        self.combined_score = None
        
    def evaluate(self):
        """
        Calculate fitness score: J = w1(1 - coverage) + w2(imbalance)
        
        ⚠️  IMPORTANT: NO PENALTIES - infeasible solutions are REJECTED before evaluation
        """
        # Convert paths to dict format if needed
        if isinstance(self.paths, dict):
            paths_dict = self.paths.copy()
        else:
            paths_dict = {robot_id: path for robot_id, path in enumerate(self.paths)}
        
        # Ensure all paths are lists
        for robot_id in paths_dict:
            if not isinstance(paths_dict[robot_id], list):
                paths_dict[robot_id] = [paths_dict[robot_id]] if paths_dict[robot_id] is not None else []
        
        # Convert all_cells to Cell objects if needed
        cells_as_objects = convert_cells_to_objects(self.all_cells)
        
        # Get basic scores from problem_formulation
        results = evaluate_solution(
            self.assignment, paths_dict, cells_as_objects, 
            self.free_cells, self.obstacles, self.grid_width, self.grid_height
        )
        self.fitness = results
        
        # Calculate coverage ratio
        coverage_ratio = results['coverage_score'] / len(self.free_cells)
        coverage_term = 1 - coverage_ratio  # Convert to minimization (0 = perfect)
        
        # Get imbalance
        imbalance_term = results['balance_score']
        
        # ✅ NO PENALTY TERM - only feasible solutions reach this point
        
        # Set weights
        w1 = 0.7  # 70% coverage
        w2 = 0.3  # 30% balance
        
        # If perfect coverage, care more about balance
        if coverage_ratio >= 1.0:
            w1 = 0.5
            w2 = 0.5
        
        # Calculate final score (lower = better) - NO PENALTIES
        self.combined_score = w1 * coverage_term + w2 * imbalance_term
        return self.combined_score
    
    def sync_assignment_with_paths(self):
        """Ensure assignment matrix matches the paths"""
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
        new_solution = RobotCoverageSolution(
            copy_module.deepcopy(self.assignment),
            copy_module.deepcopy(self.paths),
            self.all_cells,
            self.free_cells,
            self.obstacles,
            self.grid_width,
            self.grid_height
        )
        if self.fitness is not None:
            new_solution.fitness = copy_module.deepcopy(self.fitness)
        if self.combined_score is not None:
            new_solution.combined_score = self.combined_score
        return new_solution

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


# ============================================================================
# POPULATION INITIALIZATION
# ============================================================================

def generate_random_solution(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots, max_attempts=100):
    """
    Generate a random FEASIBLE solution using DARP + UF-STC (like ant3.py/Dragonfly).
    
    ⚠️  CRITICAL: Uses DARP partitioning + spanning tree path construction
    """
    for attempt in range(max_attempts):
        total_cells = len(all_cells)
        
        # STEP 1: DARP PARTITIONING
        # Generate random robot starting positions
        robot_positions = random.sample(free_cells, num_robots)
        
        # Apply DARP partitioning (assigns cells to nearest robot)
        partition = darp_partition(
            grid_width, grid_height,
            robot_positions, obstacles
        )
        
        # Create assignment matrix from partition
        assignment = [[0 for _ in range(num_robots)] for _ in range(total_cells)]
        
        for robot_id, assigned_cells in partition.items():
            for cell_idx in assigned_cells:
                if cell_idx < total_cells:
                    assignment[cell_idx][robot_id] = 1
        
        # STEP 2: UF-STC PATH CONSTRUCTION
        # Build spanning tree paths for each robot
        robot_paths = {}
        
        for robot_id in range(num_robots):
            robot_cells = partition.get(robot_id, [])
            start_pos = robot_positions[robot_id]
            
            if len(robot_cells) > 0:
                # ✅ Build spanning tree path (ensures continuity)
                path = build_spanning_tree_path(
                    start_pos, robot_cells, all_cells, 
                    free_cells, obstacles, grid_width, grid_height
                )
                robot_paths[robot_id] = path if path else robot_cells
            else:
                robot_paths[robot_id] = []
        
        solution = RobotCoverageSolution(
            assignment, 
            robot_paths,
            all_cells, 
            free_cells, 
            obstacles, 
            grid_width, 
            grid_height
        )
        
        # ✅ CHECK FEASIBILITY - only return if feasible
        is_feasible, violations = is_solution_feasible(
            solution, all_cells, obstacles, grid_width, grid_height
        )
        
        if is_feasible:
            return solution  # ✅ Found feasible solution
        
        # ❌ Infeasible - try again
    
    # If max attempts reached, raise error
    raise RuntimeError(
        f"❌ Failed to generate feasible solution after {max_attempts} attempts. "
        "Grid constraints may be too strict."
    )


def initialize_population(population_size, all_cells, free_cells, obstacles, grid_width, grid_height, num_robots):
    """
    Initialize population with ONLY FEASIBLE solutions.
    
    ⚠️  Each solution is verified to satisfy ALL constraints
    """
    population = []
    attempts = 0
    max_total_attempts = population_size * 200  # Allow many attempts
    
    while len(population) < population_size and attempts < max_total_attempts:
        attempts += 1
        try:
            solution = generate_random_solution(
                all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                max_attempts=50
            )
            solution.evaluate()
            population.append(solution)
        except RuntimeError:
            # Failed to generate feasible solution, try again
            continue
    
    if len(population) < population_size:
        raise RuntimeError(
            f"❌ Only generated {len(population)}/{population_size} feasible solutions after {attempts} attempts"
        )
    
    return population


# ============================================================================
# SELECTION
# ============================================================================

def tournament_selection(population, tournament_size=3):
    """Select best solution from random tournament"""
    tournament = random.sample(population, min(tournament_size, len(population)))
    winner = min(tournament, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
    return winner


# ============================================================================
# CROSSOVER OPERATORS
# ============================================================================

def crossover_ox1_path(parent1_path, parent2_path, verbose=False):
    """Order Crossover (OX1) for paths"""
    if len(parent1_path) == 0 and len(parent2_path) == 0:
        return [] if not verbose else ([], {})
    
    if len(parent1_path) == 0:
        result = parent2_path.copy()
        return result if not verbose else (result, {'note': 'Parent1 empty, using parent2'})
    if len(parent2_path) == 0:
        result = parent1_path.copy()
        return result if not verbose else (result, {'note': 'Parent2 empty, using parent1'})
    
    # Get ALL unique cells from both parents
    all_unique_cells = list(set(parent1_path) | set(parent2_path))
    
    if len(parent1_path) < 2:
        result = []
        used = set()
        for cell in parent2_path:
            if cell not in used:
                result.append(cell)
                used.add(cell)
        for cell in parent1_path:
            if cell not in used:
                result.append(cell)
        return result if not verbose else (result, {'note': 'Parent1 too short'})
    
    debug_info = {}
    
    # Standard OX1 with two crossover points
    point1 = random.randint(0, max(0, len(parent1_path) - 2))
    point2 = random.randint(point1 + 1, len(parent1_path) - 1)
    
    debug_info['point1'] = point1
    debug_info['point2'] = point2
    
    # Copy segment from parent1
    segment = parent1_path[point1:point2]
    child_path = segment.copy()
    used_cells = set(segment)
    
    # Fill remaining from parent2 in order
    parent2_index = point2 % len(parent2_path) if len(parent2_path) > 0 else 0
    
    cells_from_parent2 = []
    
    while len(child_path) < len(all_unique_cells):
        attempts = 0
        found = False
        
        while attempts < len(parent2_path) and not found:
            cell = parent2_path[parent2_index % len(parent2_path)]
            if cell not in used_cells:
                child_path.append(cell)
                cells_from_parent2.append(cell)
                used_cells.add(cell)
                found = True
            parent2_index = (parent2_index + 1) % len(parent2_path)
            attempts += 1
        
        if not found:
            for cell in parent1_path:
                if cell not in used_cells and len(child_path) < len(all_unique_cells):
                    child_path.append(cell)
                    used_cells.add(cell)
                    break
            else:
                break
    
    # Ensure ALL cells from union are included
    missing_cells = set(all_unique_cells) - set(child_path)
    if missing_cells:
        child_path.extend(list(missing_cells))
        debug_info['added_missing_cells'] = list(missing_cells)
    
    debug_info['cells_from_parent2'] = cells_from_parent2.copy()
    debug_info['segment'] = segment.copy()
    
    if verbose:
        return child_path, debug_info
    return child_path


def crossover_order_based(parent1, parent2, verbose=False, free_cells=None):
    """Order-based crossover for robot solutions"""
    child = parent1.copy()
    num_robots = len(parent1.paths)
    
    # Random: 0 = crossover assignment, 1 = crossover path
    crossover_type = random.randint(0, 1)
    
    child._crossover_type = crossover_type
    child._crossover_debug = {}
    
    if crossover_type == 0:
        # ASSIGNMENT CROSSOVER
        # Simplified version - just swap some assignments
        for cell_idx in free_cells if free_cells else range(len(parent1.assignment)):
            if random.random() < 0.5:
                child.assignment[cell_idx] = parent2.assignment[cell_idx].copy()
        
        # Update paths based on new assignment
        for robot_id in range(num_robots):
            robot_cells = []
            for cell_idx in range(len(child.assignment)):
                if child.assignment[cell_idx][robot_id] == 1:
                    robot_cells.append(cell_idx)
            child.paths[robot_id] = robot_cells
    else:
        # PATH CROSSOVER
        for robot_id in range(num_robots):
            if robot_id not in parent1.paths or robot_id not in parent2.paths:
                continue
            
            parent1_path = parent1.paths[robot_id]
            parent2_path = parent2.paths[robot_id]
            
            if len(parent1_path) == 0 or len(parent2_path) == 0:
                continue
            
            # Use OX1 crossover
            child_path = crossover_ox1_path(parent1_path, parent2_path, verbose=False)
            child.paths[robot_id] = child_path
            
            # Update assignment
            for cell_idx in range(len(child.assignment)):
                child.assignment[cell_idx][robot_id] = 0
            for cell_idx in child.paths[robot_id]:
                if cell_idx < len(child.assignment):
                    child.assignment[cell_idx][robot_id] = 1
    
    # Sync assignment and paths
    child.sync_assignment_with_paths()
    child.evaluate()
    
    return child


def apply_crossover(parent1, parent2, verbose=False, free_cells=None, all_cells=None, obstacles=None, grid_width=None, grid_height=None, max_attempts=20):
    """
    Apply crossover operator - only returns FEASIBLE offspring.
    
    ⚠️  Retries until feasible child is created or max attempts reached
    """
    for attempt in range(max_attempts):
        child = crossover_order_based(parent1, parent2, verbose=verbose, free_cells=free_cells)
        
        # ✅ CHECK FEASIBILITY
        if all_cells is not None and obstacles is not None:
            is_feasible, violations = is_solution_feasible(
                child, all_cells, obstacles, grid_width, grid_height
            )
            
            if is_feasible:
                return child  # ✅ Found feasible child
        else:
            # No feasibility check params provided, return child
            return child
    
    # If max attempts reached, return parent1 (safest fallback)
    return parent1.copy()


# ============================================================================
# MUTATION OPERATORS
# ============================================================================

def mutate_robot_path(solution, robot_id):
    """Mutate a robot's path by swapping two positions"""
    if robot_id not in solution.paths:
        return False
    
    path = solution.paths[robot_id]
    
    if len(path) <= 2:
        return False
    
    # Pick two random positions
    pos1 = random.randint(0, len(path) - 1)
    pos2 = random.randint(0, len(path) - 1)
    while pos2 == pos1 and len(path) > 1:
        pos2 = random.randint(0, len(path) - 1)
    
    # Swap
    new_path = path.copy()
    new_path[pos1], new_path[pos2] = new_path[pos2], new_path[pos1]
    solution.paths[robot_id] = new_path
    
    # Sync assignment
    solution.sync_assignment_with_paths()
    
    return True


def mutate(solution, mutation_rate=0.1, all_cells=None, obstacles=None, grid_width=None, grid_height=None, max_attempts=20):
    """
    Mutate solution - only returns FEASIBLE mutated solution.
    
    ⚠️  Retries until feasible mutation or returns original
    """
    if random.random() > mutation_rate:
        return solution
    
    num_robots = len(solution.paths)
    if num_robots == 0:
        return solution
    
    # Find robots with paths of length >= 2
    valid_robots = [r for r in range(num_robots) if len(solution.paths[r]) >= 2]
    
    if not valid_robots:
        return solution
    
    # Try to create feasible mutation
    for attempt in range(max_attempts):
        mutated = solution.copy()
        robot_id = random.choice(valid_robots)
        did_mutate = mutate_robot_path(mutated, robot_id)
        
        if did_mutate:
            # ✅ CHECK FEASIBILITY
            if all_cells is not None and obstacles is not None:
                is_feasible, violations = is_solution_feasible(
                    mutated, all_cells, obstacles, grid_width, grid_height
                )
                
                if is_feasible:
                    return mutated  # ✅ Found feasible mutation
            else:
                return mutated
    
    # If no feasible mutation found, return original
    return solution


# ============================================================================
# MAIN GENETIC ALGORITHM
# ============================================================================

def genetic_algorithm(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                      population_size=50, generations=100, 
                      verbose=True, selection_percentage=0.10, 
                      crossover_percentage=0.80, mutation_percentage=0.10):
    """
    Main Genetic Algorithm for Multi-Robot Coverage Path Planning
    
    Parameters:
    - all_cells: list of (x,y) tuples
    - free_cells: list of cell indices that are free
    - obstacles: list of cell indices that are obstacles
    - grid_width, grid_height: grid dimensions
    - num_robots: number of robots
    - population_size: size of population
    - generations: number of generations
    - selection_percentage: percentage of elite solutions to preserve (default 10%)
    - crossover_percentage: percentage of offspring from crossover (default 80%)
    - mutation_percentage: percentage of offspring from mutation (default 10%)
    """
    
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
    
    # Step 1: Initialize population
    print(f"🔄 STEP 1: Initializing Population (Generation 0)")
    print(f"   Creating {population_size} random solutions...")
    
    population = initialize_population(
        population_size, all_cells, free_cells, obstacles, grid_width, grid_height, num_robots
    )
    
    # Evaluate all solutions
    for solution in population:
        solution.evaluate()
    
    # Find best solution
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
    
    # Track convergence
    convergence_history = {
        'generation': [],
        'best_score': [],
        'avg_score': [],
        'worst_score': [],
        'best_coverage': [],
        'best_balance': []
    }
    
    # Main GA loop
    for generation in range(generations):
        
        if verbose and (generation % 10 == 0 or generation < 3):
            print(f"\n{'─'*70}")
            print(f"🔄 GENERATION {generation}")
            print(f"{'─'*70}")
        
        new_population = []
        
        num_selection = int(population_size * selection_percentage)
        num_crossover = int(population_size * crossover_percentage)
        num_mutation = population_size - num_selection - num_crossover
        
        if verbose and generation < 3:
            print(f"\n   🧬 Creating new population:")
            print(f"      • {num_selection} solutions via Selection (Elitism) ({selection_percentage*100:.0f}%)")
            print(f"      • {num_crossover} solutions via Crossover ({crossover_percentage*100:.0f}%)")
            print(f"      • {num_mutation} solutions via Mutation ({mutation_percentage*100:.0f}%)")
        
        # 1. SELECTION (Elitism) - copy best solutions
        sorted_population = sorted(population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'))
        
        for i in range(num_selection):
            elite_solution = sorted_population[i]
            elite_copy = elite_solution.copy()
            new_population.append(elite_copy)
        
        # 2. CROSSOVER - create offspring (only feasible)
        for i in range(num_crossover):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = apply_crossover(
                parent1, parent2, 
                verbose=False, 
                free_cells=free_cells,
                all_cells=all_cells,
                obstacles=obstacles,
                grid_width=grid_width,
                grid_height=grid_height
            )
            child.evaluate()
            new_population.append(child)
        
        # 3. MUTATION - mutate worst solutions (only feasible)
        sorted_population_by_worst = sorted(population, key=lambda x: x.combined_score if x.combined_score is not None else float('inf'), reverse=True)
        
        for i in range(num_mutation):
            worst_solution = sorted_population_by_worst[i]
            mutated = mutate(
                worst_solution, 
                mutation_rate=1.0,  # Force mutation
                all_cells=all_cells,
                obstacles=obstacles,
                grid_width=grid_width,
                grid_height=grid_height
            )
            mutated.evaluate()
            new_population.append(mutated)
        
        # Update population
        population = new_population
        
        # Find best solution in current generation
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
        
        # Record metrics
        scores = [sol.combined_score if sol.combined_score is not None else float('inf') 
                  for sol in population]
        convergence_history['generation'].append(generation)
        convergence_history['best_score'].append(min(scores))
        convergence_history['avg_score'].append(sum(scores) / len(scores))
        convergence_history['worst_score'].append(max(scores))
        
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
            if best_solution.fitness:
                print(f"      • Coverage:       {best_solution.fitness['coverage_score']}/{len(free_cells)} cells")
                print(f"      • Balance:        {best_solution.fitness['balance_score']:.3f}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ GENETIC ALGORITHM COMPLETE!")
    print(f"{'='*70}")
    print(f"📊 Final Statistics:")
    print(f"   • Total Generations:      {generations}")
    print(f"   • Best Score Achieved:     {safe_format_score(best_score)}")
    
    if best_solution.fitness is not None:
        print(f"   • Final Coverage:         {best_solution.fitness['coverage_score']}/{len(free_cells)} cells ({best_solution.fitness['coverage_score']/len(free_cells)*100:.1f}%)")
        print(f"   • Final Balance:          {best_solution.fitness['balance_score']:.3f}")
        print(f"   • Constraint Violations:  {len(best_solution.fitness['problems'])}")
    
    print(f"{'='*70}\n")
    
    # Ensure best solution is evaluated
    if best_solution.fitness is None:
        best_solution.evaluate()
    
    # Return results
    return {
        'best_solution': best_solution,
        'best_score': best_score,
        'convergence_history': convergence_history,
        'final_population': population
    }


# ============================================================================
# CONVERGENCE HISTORY NORMALIZATION
# ============================================================================

def normalize_ga_convergence_history(convergence_history):
    """
    Normalize GA convergence history to 0-1 range for fair comparison with ACO.
    
    GA uses: w1 * (1 - coverage) + w2 * imbalance + penalty (raw values, can be 1000+)
    ACO uses: normalized values (0-1 range)
    
    This function normalizes GA scores to match ACO's scale while preserving "lower = better".
    """
    if not convergence_history or 'best_score' not in convergence_history:
        return convergence_history
    
    best_scores = convergence_history['best_score']
    if not best_scores:
        return convergence_history
    
    # Find min and max for normalization
    min_score = min(best_scores)
    max_score = max(best_scores)
    score_range = max_score - min_score
    
    normalized_history = convergence_history.copy()
    
    if score_range > 0:
        # Normalize to 0.3-0.7 range (similar to ACO)
        normalized_history['best_score'] = [
            0.3 + 0.4 * ((score - min_score) / score_range)
            for score in best_scores
        ]
        
        if 'avg_score' in convergence_history:
            avg_scores = convergence_history['avg_score']
            if score_range > 0:
                normalized_history['avg_score'] = [
                    0.3 + 0.4 * ((score - min_score) / score_range)
                    for score in avg_scores
                ]
        
        if 'worst_score' in convergence_history:
            worst_scores = convergence_history['worst_score']
            if score_range > 0:
                normalized_history['worst_score'] = [
                    0.3 + 0.4 * ((score - min_score) / score_range)
                    for score in worst_scores
                ]
    else:
        # All scores are the same
        normalized_history['best_score'] = [0.5] * len(best_scores)
    
    return normalized_history


# ============================================================================
# CASE STUDIES
# ============================================================================

def run_ga_case_study_1():
    """Run GA on Case Study 1: Small Grid (4x4, 2 Robots)"""
    
    print("\n" + "="*80)
    print("RUNNING GA ONLY - CASE STUDY 1: SMALL GRID (4x4, 2 Robots)")
    print("="*80)
    print("⚠️  FEASIBILITY-BASED: No penalties - infeasible solutions are REJECTED")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 4, 4
    num_robots = 2
    obstacles = [5, 10]  # 2 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # GA parameters
    ga_params = {
        'population_size': 30,
        'generations': 50,
        'selection_percentage': 0.10,
        'crossover_percentage': 0.80,
        'mutation_percentage': 0.10
    }
    
    print(f"\nGA Parameters: {ga_params}")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/ga_only/case1_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run GA
    print("\n" + "-"*80)
    print("Running Genetic Algorithm (FEASIBILITY-BASED)...")
    print("-"*80)
    
    start_time = time.time()
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\n⏱️  GA Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    # Generate visualizations with timestamp
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        best_solution = ga_results['best_solution']
        
        # Visualize solution
        solution_path = f"{results_dir}/ga_solution_{timestamp}.png"
        visualize_solution(
            best_solution,
            title=f"GA Best Solution (Case 1) - {timestamp}",
            save_path=solution_path
        )
        print(f"✅ GA solution saved: {solution_path}")
        
        # Normalize and plot convergence
        normalized_history = normalize_ga_convergence_history(ga_results['convergence_history'])
        
        convergence_path = f"{results_dir}/ga_convergence_{timestamp}.png"
        plot_convergence_history(
            normalized_history,
            title=f"GA Convergence (Case 1) - {timestamp}",
            save_path=convergence_path
        )
        print(f"✅ GA convergence saved: {convergence_path}")
        
        best_score_path = f"{results_dir}/ga_best_score_{timestamp}.png"
        plot_best_score_only(
            normalized_history,
            title=f"GA Best Score (Case 1) - {timestamp}",
            save_path=best_score_path
        )
        print(f"✅ GA best score saved: {best_score_path}")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print results summary
    print("\n" + "="*80)
    print("📊 GA RESULTS SUMMARY")
    print("="*80)
    
    if best_solution:
        print(f"Best Combined Score: {best_solution.combined_score:.4f}")
        if best_solution.fitness:
            print(f"Coverage: {best_solution.fitness.get('coverage_score', 'N/A')}/{len(free_cells)} cells")
            print(f"Balance: {best_solution.fitness.get('balance_score', 'N/A'):.4f}")
        if hasattr(best_solution, 'get_coverage_efficiency'):
            print(f"Coverage Efficiency: {best_solution.get_coverage_efficiency():.2f}%")
        if hasattr(best_solution, 'get_workload_balance_index'):
            print(f"Workload Balance Index: {best_solution.get_workload_balance_index():.4f}")
    
    print(f"\n✅ GA run completed!")
    print(f"Results saved to: {results_dir}/")
    
    return ga_results


def run_ga_case_study_2():
    """Run GA on Case Study 2: Medium Grid (6x6, 3 Robots)"""
    
    print("\n" + "="*80)
    print("RUNNING GA ONLY - CASE STUDY 2: MEDIUM GRID (6x6, 3 Robots)")
    print("="*80)
    print("⚠️  FEASIBILITY-BASED: No penalties - infeasible solutions are REJECTED")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 6, 6
    num_robots = 3
    obstacles = [1, 7, 13, 19, 25, 31]  # 6 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # GA parameters
    ga_params = {
        'population_size': 50,
        'generations': 100,
        'selection_percentage': 0.10,
        'crossover_percentage': 0.80,
        'mutation_percentage': 0.10
    }
    
    print(f"\nGA Parameters: {ga_params}")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/ga_only/case2_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run GA
    print("\n" + "-"*80)
    print("Running Genetic Algorithm (FEASIBILITY-BASED)...")
    print("-"*80)
    
    start_time = time.time()
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\n⏱️  GA Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    # Generate visualizations with timestamp
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        best_solution = ga_results['best_solution']
        
        solution_path = f"{results_dir}/ga_solution_{timestamp}.png"
        visualize_solution(
            best_solution,
            title=f"GA Best Solution (Case 2) - {timestamp}",
            save_path=solution_path
        )
        print(f"✅ GA solution saved: {solution_path}")
        
        normalized_history = normalize_ga_convergence_history(ga_results['convergence_history'])
        
        convergence_path = f"{results_dir}/ga_convergence_{timestamp}.png"
        plot_convergence_history(
            normalized_history,
            title=f"GA Convergence (Case 2) - {timestamp}",
            save_path=convergence_path
        )
        print(f"✅ GA convergence saved: {convergence_path}")
        
        best_score_path = f"{results_dir}/ga_best_score_{timestamp}.png"
        plot_best_score_only(
            normalized_history,
            title=f"GA Best Score (Case 2) - {timestamp}",
            save_path=best_score_path
        )
        print(f"✅ GA best score saved: {best_score_path}")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print results summary
    print("\n" + "="*80)
    print("📊 GA RESULTS SUMMARY")
    print("="*80)
    
    if best_solution:
        print(f"Best Combined Score: {best_solution.combined_score:.4f}")
        if best_solution.fitness:
            print(f"Coverage: {best_solution.fitness.get('coverage_score', 'N/A')}/{len(free_cells)} cells")
            print(f"Balance: {best_solution.fitness.get('balance_score', 'N/A'):.4f}")
        if hasattr(best_solution, 'get_coverage_efficiency'):
            print(f"Coverage Efficiency: {best_solution.get_coverage_efficiency():.2f}%")
        if hasattr(best_solution, 'get_workload_balance_index'):
            print(f"Workload Balance Index: {best_solution.get_workload_balance_index():.4f}")
    
    print(f"\n✅ GA run completed!")
    print(f"Results saved to: {results_dir}/")
    
    return ga_results


def run_ga_case_study_3():
    """Run GA on Case Study 3: Large Grid (10x10, 5 Robots)"""
    
    print("\n" + "="*80)
    print("RUNNING GA ONLY - CASE STUDY 3: LARGE GRID (10x10, 5 Robots)")
    print("="*80)
    print("⚠️  FEASIBILITY-BASED: No penalties - infeasible solutions are REJECTED")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 10, 10
    num_robots = 5
    obstacles = [11, 12, 13, 22, 32, 42, 52, 62, 67, 68, 69, 77, 88]  # 13 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # GA parameters
    ga_params = {
        'population_size': 100,
        'generations': 200,
        'selection_percentage': 0.10,
        'crossover_percentage': 0.80,
        'mutation_percentage': 0.10
    }
    
    print(f"\nGA Parameters: {ga_params}")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/ga_only/case3_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run GA
    print("\n" + "-"*80)
    print("Running Genetic Algorithm (FEASIBILITY-BASED)...")
    print("-"*80)
    
    start_time = time.time()
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\n⏱️  GA Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    # Generate visualizations with timestamp
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        best_solution = ga_results['best_solution']
        
        solution_path = f"{results_dir}/ga_solution_{timestamp}.png"
        visualize_solution(
            best_solution,
            title=f"GA Best Solution (Case 3) - {timestamp}",
            save_path=solution_path
        )
        print(f"✅ GA solution saved: {solution_path}")
        
        normalized_history = normalize_ga_convergence_history(ga_results['convergence_history'])
        
        convergence_path = f"{results_dir}/ga_convergence_{timestamp}.png"
        plot_convergence_history(
            normalized_history,
            title=f"GA Convergence (Case 3) - {timestamp}",
            save_path=convergence_path
        )
        print(f"✅ GA convergence saved: {convergence_path}")
        
        best_score_path = f"{results_dir}/ga_best_score_{timestamp}.png"
        plot_best_score_only(
            normalized_history,
            title=f"GA Best Score (Case 3) - {timestamp}",
            save_path=best_score_path
        )
        print(f"✅ GA best score saved: {best_score_path}")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print results summary
    print("\n" + "="*80)
    print("📊 GA RESULTS SUMMARY")
    print("="*80)
    
    if best_solution:
        print(f"Best Combined Score: {best_solution.combined_score:.4f}")
        if best_solution.fitness:
            print(f"Coverage: {best_solution.fitness.get('coverage_score', 'N/A')}/{len(free_cells)} cells")
            print(f"Balance: {best_solution.fitness.get('balance_score', 'N/A'):.4f}")
        if hasattr(best_solution, 'get_coverage_efficiency'):
            print(f"Coverage Efficiency: {best_solution.get_coverage_efficiency():.2f}%")
        if hasattr(best_solution, 'get_workload_balance_index'):
            print(f"Workload Balance Index: {best_solution.get_workload_balance_index():.4f}")
    
    print(f"\n✅ GA run completed!")
    print(f"Results saved to: {results_dir}/")
    
    return ga_results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if case study number is provided
    if len(sys.argv) > 1:
        case_num = int(sys.argv[1])
    else:
        # Default to case study 1
        case_num = 1
    
    print("\n" + "="*80)
    print("🧬 GA ONLY RUNNER")
    print("="*80)
    print("\nThis script runs ONLY the Genetic Algorithm (GA)")
    print("No Simulated Annealing (SA) will be executed")
    print("="*80)
    
    if case_num == 1:
        run_ga_case_study_1()
    elif case_num == 2:
        run_ga_case_study_2()
    elif case_num == 3:
        run_ga_case_study_3()
    else:
        print(f"\n❌ Error: Case study {case_num} not available")
        print("Available case studies: 1, 2, 3")
        print("\nUsage:")
        print("  python3 run_ga_only.py          # Run case study 1 (default)")
        print("  python3 run_ga_only.py 1       # Run case study 1")
        print("  python3 run_ga_only.py 2       # Run case study 2")
        print("  python3 run_ga_only.py 3       # Run case study 3")
