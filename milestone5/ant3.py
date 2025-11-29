"""
Ant Colony Optimization for Multi-Robot Coverage Path Planning
==============================================================

This implementation combines:
- DARP (Divide Areas based on Robot Positions): ACO-based deterministic partitioning
- UF-STC (Uniform Coverage Spanning Tree Construction): Spanning tree + DFS/BFS traversal

Problem Formulation:
- Decision Variables: ai,r (assignment), πr (paths)
- Objective F1: Maximize area coverage
- Objective F2: Minimize workload imbalance
- Constraints: Path continuity, boundary, obstacle avoidance

Based on research paper approach with full problem formulation implementation.
"""

import copy
import math
import random
import sys
import os
import collections
from typing import List, Dict, Set, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem_formulation import (
    evaluate_solution,
    find_neighbors,
    calculate_robot_distances,
    distance_between_points,
    create_grid_cells
)


class ACOSolution:
    """
    Solution representation for ACO algorithm
    
    Decision Variables:
    - assignment[cell_idx][robot_id] = ai,r (0 or 1)
    - paths[robot_id] = πr = [ir,1, ir,2, ..., ir,nr]
    """
    def __init__(self, assignment, paths, all_cells, free_cells, obstacles, grid_width, grid_height):
        self.assignment = copy.deepcopy(assignment)  # ai,r: assignment matrix
        self.paths = copy.deepcopy(paths)  # πr: paths dict {robot_id: [cell1, cell2, ...]}
        self.all_cells = all_cells
        self.free_cells = free_cells
        self.obstacles = obstacles
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Objective values
        self.F1 = None  # Coverage objective
        self.F2 = None  # Workload imbalance objective
        self.Lr = {}  # Path lengths per robot: {robot_id: Lr}
        self.fitness = None  # Combined fitness for ACO
        
    def copy(self):
        """Create a deep copy of this solution"""
        new_solution = ACOSolution(
            copy.deepcopy(self.assignment),
            copy.deepcopy(self.paths),
            self.all_cells,
            self.free_cells,
            self.obstacles,
            self.grid_width,
            self.grid_height
        )
        if self.F1 is not None:
            new_solution.F1 = self.F1
        if self.F2 is not None:
            new_solution.F2 = self.F2
        if self.Lr:
            new_solution.Lr = copy.deepcopy(self.Lr)
        if self.fitness is not None:
            new_solution.fitness = self.fitness
        return new_solution
    
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


def create_empty_solution(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots):
    """Create initial empty solution"""
    total_cells = len(all_cells)
    assignment = [[0 for _ in range(num_robots)] for _ in range(total_cells)]
    paths = {robot_id: [] for robot_id in range(num_robots)}
    
    solution = ACOSolution(
        assignment,
        paths,
        all_cells,
        free_cells,
        obstacles,
        grid_width,
        grid_height
    )
    return solution


def check_path_continuity(path, all_cells, grid_width, grid_height):
    """
    Constraint: Path continuity - (ir,k, ir,k+1) ∈ Egrid
    Check if consecutive cells in path are adjacent (4-connected)
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
        # Allow same cell (distance 0) - robot can stay in place
        manhattan_dist = abs(current_x - next_x) + abs(current_y - next_y)
        if manhattan_dist > 1:  # Changed from != 1 to > 1 to allow distance 0
            violations.append(
                f"Path jump: cell {current_cell_idx} ({current_x},{current_y}) to "
                f"{next_cell_idx} ({next_x},{next_y}) are not adjacent"
            )
    
    return len(violations) == 0, violations


def check_boundary_constraint(path, all_cells, grid_width, grid_height):
    """
    Constraint: Boundary constraint - ir,k ∈ C
    Check if all cells in path are within grid boundaries
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
            violations.append(f"Cell {cell_idx} at ({x}, {y}) is outside grid boundaries")
    
    return len(violations) == 0, violations


def check_obstacle_avoidance(path, obstacles):
    """
    Constraint: Obstacle avoidance - ir,k ∉ O
    Check if path contains any obstacle cells
    """
    violations = []
    for i, cell_idx in enumerate(path):
        if cell_idx in obstacles:
            violations.append(f"Robot enters obstacle at cell {cell_idx} (position {i} in path)")
    return len(violations) == 0, violations


def is_solution_feasible(solution: ACOSolution) -> Tuple[bool, List[str]]:
    """
    Check all constraints for a solution
    Returns: (is_feasible, violations_list)
    """
    all_violations = []
    
    if not isinstance(solution.paths, dict):
        return False, [f"Paths must be a dictionary, got {type(solution.paths)}"]
    
    num_robots = len(solution.paths)
    
    for robot_id in range(num_robots):
        path = solution.paths.get(robot_id, [])
        if not isinstance(path, list):
            all_violations.append(f"Robot {robot_id}: path must be a list, got {type(path)}")
            continue
        
        # Check boundary constraint
        boundary_valid, boundary_violations = check_boundary_constraint(
            path, solution.all_cells, solution.grid_width, solution.grid_height
        )
        if not boundary_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in boundary_violations])
        
        # Check obstacle avoidance
        obstacle_valid, obstacle_violations = check_obstacle_avoidance(path, solution.obstacles)
        if not obstacle_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in obstacle_violations])
        
        # Check path continuity
        continuity_valid, continuity_violations = check_path_continuity(
            path, solution.all_cells, solution.grid_width, solution.grid_height
        )
        if not continuity_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in continuity_violations])
    
    is_feasible = len(all_violations) == 0
    return is_feasible, all_violations


def build_spanning_tree_path(start_cell_idx, assigned_cells, all_cells, free_cells, obstacles, grid_width, grid_height):
    """
    UF-STC: Build spanning tree over assigned region and generate coverage path
    
    Steps:
    1. Build spanning tree using BFS over assigned cells
    2. Generate coverage path using DFS preorder traversal
    3. Return path πr = [ir,1, ir,2, ..., ir,nr]
    """
    if not assigned_cells:
        return []
    
    # Convert cell indices to coordinates for easier neighbor checking
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
    
    # Validate start cell
    if start_cell_idx < 0 or start_cell_idx >= len(all_cells):
        # Invalid start, use first assigned cell
        if assigned_cells:
            root = next(iter(assigned_cells))
        else:
            return []
    elif start_cell_idx not in free_cells or start_cell_idx in obstacles:
        # Start is obstacle or invalid, find closest assigned cell
        if assigned_cells:
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
                return []
        else:
            return []
    # Find root: use start if assigned, otherwise closest assigned cell
    elif start_cell_idx in assigned_cells:
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
            return []
    
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
            path.append(u)  # Return to parent (coverage may revisit)
    
    if root in tree_adj or root in parent:
        dfs(root)
    
    # Remove consecutive duplicates (robot staying in same cell)
    if path:
        cleaned_path = [path[0]]
        for i in range(1, len(path)):
            if path[i] != path[i-1]:
                cleaned_path.append(path[i])
        path = cleaned_path
    
    # Handle disconnected cells (cells not reached by BFS)
    # Instead of just appending, we need to connect them properly
    missing = set(assigned_cells) - set(parent.keys())
    if missing:
        # For disconnected cells, find path from last cell in path to each missing cell
        # This ensures path continuity
        last_cell = path[-1] if path else root
        root_coord = get_cell_coords(root)
        missing_sorted = sorted(
            list(missing),
            key=lambda c: abs(get_cell_coords(c)[0] - root_coord[0]) + 
                          abs(get_cell_coords(c)[1] - root_coord[1])
        )
        
        for m in missing_sorted:
            # Find path from last cell to missing cell
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
            
            # Append path to missing cell, then the missing cell itself
            if path_to_missing:
                # Only add if path actually reaches target
                if path_to_missing[-1] == m or current == m:
                    path.extend(path_to_missing)
                    if path_to_missing[-1] != m:
                        path.append(m)
                    last_cell = m
                else:
                    # Can't reach, skip this disconnected cell
                    pass
            elif current == m:
                # Already at target
                path.append(m)
                last_cell = m
    
    # Ensure path starts from robot's start position
    # Handle empty path case
    if not path:
        # Empty path - just add start if it's assigned
        if start_cell_idx in assigned_cells:
            path = [start_cell_idx]
        else:
            # Start not assigned, return empty path
            return []
    elif path[0] != start_cell_idx:
        # Check if start is adjacent to first cell in path
        first_cell_neighbors = get_neighbor_indices(path[0])
        if start_cell_idx in first_cell_neighbors:
            # Start is adjacent, safe to prepend
            path = [start_cell_idx] + path
        else:
            # Start is not adjacent, need to find path from start to first cell
            # Use simple greedy path: move towards first cell step by step
            current = start_cell_idx
            path_to_first = []
            first_cell_coord = get_cell_coords(path[0])
            
            max_steps = grid_width + grid_height  # Prevent infinite loop
            steps = 0
            while current != path[0] and steps < max_steps:
                steps += 1
                current_coord = get_cell_coords(current)
                # Find neighbor that moves closer to first cell
                neighbors = get_neighbor_indices(current)
                if not neighbors:
                    break
                
                # Choose neighbor that minimizes distance to first cell
                best_neighbor = None
                min_dist = float('inf')
                for nb in neighbors:
                    nb_coord = get_cell_coords(nb)
                    dist_to_first = abs(nb_coord[0] - first_cell_coord[0]) + abs(nb_coord[1] - first_cell_coord[1])
                    if dist_to_first < min_dist:
                        min_dist = dist_to_first
                        best_neighbor = nb
                
                if best_neighbor is None:
                    break
                
                path_to_first.append(best_neighbor)
                current = best_neighbor
                
                # If we reached first cell, stop
                if current == path[0]:
                    break
            
            # Prepend path from start to first cell
            if path_to_first and path_to_first[-1] == path[0]:
                # Successfully found path to first cell
                path = [start_cell_idx] + path_to_first + path
            elif start_cell_idx == path[0]:
                # Already starts at start cell, no need to prepend
                pass
            else:
                # Can't find valid path, try to find any path or skip prepending
                # For now, skip prepending if we can't find valid path
                # This might cause issues but better than invalid path
                pass
    
    return path


def calculate_path_length(path, all_cells):
    """
    Calculate path length Lr = Σ d(ir,k, ir,k+1)
    where d is Manhattan distance between consecutive cells
    """
    if len(path) <= 1:
        return 0.0
    
    total_length = 0.0
    for k in range(len(path) - 1):
        cell1_idx = path[k]
        cell2_idx = path[k + 1]
        
        cell1 = all_cells[cell1_idx]
        cell2 = all_cells[cell2_idx]
        
        if isinstance(cell1, tuple):
            x1, y1 = cell1
        else:
            x1, y1 = cell1.x, cell1.y
        
        if isinstance(cell2, tuple):
            x2, y2 = cell2
        else:
            x2, y2 = cell2.x, cell2.y
        
        # Manhattan distance
        d = abs(x2 - x1) + abs(y2 - y1)
        total_length += d
    
    return total_length


def evaluate_objectives(solution: ACOSolution):
    """
    Evaluate both objectives:
    
    Objective F1: Maximize area coverage
    F1 = Σ(i=1 to N) Σ(r=1 to M) ai,r
    
    Objective F2: Minimize workload imbalance
    F2 = Σ(r=1 to M) |Lr - L̄|
    where Lr = Σ(k=1 to nr-1) d(ir,k, ir,k+1)
    and L̄ = (1/M) Σ(r=1 to M) Lr
    """
    # Calculate F1: Coverage
    N = len(solution.all_cells)
    M = len(solution.paths)
    F1 = 0
    
    for cell_idx in solution.free_cells:
        for robot_id in range(M):
            if cell_idx < len(solution.assignment) and solution.assignment[cell_idx][robot_id] == 1:
                F1 += 1
                break  # Count each cell only once
    
    # Calculate F2: Workload imbalance
    # First, calculate Lr for each robot
    Lr_dict = {}
    for robot_id, path in solution.paths.items():
        Lr = calculate_path_length(path, solution.all_cells)
        Lr_dict[robot_id] = Lr
    
    solution.Lr = Lr_dict
    
    # Calculate average path length L̄
    if M > 0 and Lr_dict:
        L_bar = sum(Lr_dict.values()) / M
    else:
        L_bar = 0.0
    
    # Calculate F2: Sum of absolute deviations
    F2 = 0.0
    for robot_id in range(M):
        Lr = Lr_dict.get(robot_id, 0.0)
        F2 += abs(Lr - L_bar)
    
    solution.F1 = F1
    solution.F2 = F2
    
    return F1, F2


class AntColonyOptimization:
    """
    Ant Colony Optimization for Multi-Robot Coverage Path Planning
    
    Implements:
    - DARP: ACO-based partitioning (assign cells to robots)
    - UF-STC: Spanning tree path generation
    - Multi-objective optimization: F1 (coverage) and F2 (workload imbalance)
    """
    
    def __init__(self, all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                 num_ants=20, initial_pheromone=1.0, rho=0.5, alpha=1.0, beta=1.0, iterations=50,
                 robot_starts=None, seed=None, gamma=1.0):
        """
        Initialize ACO algorithm
        
        Parameters:
        - all_cells: List of all cells in grid
        - free_cells: List of free (non-obstacle) cell indices
        - obstacles: Set of obstacle cell indices
        - grid_width, grid_height: Grid dimensions
        - num_robots: Number of robots
        - num_ants: Number of ants per iteration
        - initial_pheromone: Initial pheromone level
        - rho: Pheromone evaporation rate
        - alpha: Pheromone importance
        - beta: Heuristic importance
        - iterations: Number of iterations
        - robot_starts: List of starting cell indices for each robot (optional)
        - seed: Random seed (optional)
        - gamma: Multi-objective weighting parameter for workload balance (γ)
        """
        self.all_cells = all_cells
        self.free_cells = free_cells
        self.obstacles = set(obstacles)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_robots = num_robots
        self.num_ants = num_ants
        self.initial_pheromone = initial_pheromone
        self.rho = rho  # Evaporation rate
        self.alpha = alpha  # Pheromone weight
        self.beta = beta  # Heuristic weight
        self.gamma = gamma  # Multi-objective workload balance weight
        self.iterations = iterations
        
        if seed is not None:
            random.seed(seed)
        
        # Set robot starting positions
        if robot_starts is None:
            # Default: distribute robots evenly across free cells
            self.robot_starts = [free_cells[i % len(free_cells)] for i in range(num_robots)]
        else:
            self.robot_starts = robot_starts
        
        # Initialize pheromone matrix: τ[r][i] for robot r and cell i
        self.pheromone = {}
        for robot_id in range(num_robots):
            self.pheromone[robot_id] = {}
            for cell_idx in free_cells:
                self.pheromone[robot_id][cell_idx] = initial_pheromone
        
        # Initialize heuristic matrix: η[r][i] based on distance from robot start
        self.heuristic = {}
        for robot_id in range(num_robots):
            self.heuristic[robot_id] = {}
            start_cell_idx = self.robot_starts[robot_id]
            for cell_idx in free_cells:
                # Heuristic: inverse distance (closer = more desirable)
                eta = self._calculate_distance_heuristic(start_cell_idx, cell_idx)
                self.heuristic[robot_id][cell_idx] = eta
        
        # Best solution tracking
        self.best_solution = None
        self.best_F1 = 0
        self.best_F2 = float('inf')
        self.history = []
    
    def _get_cell_coords(self, cell_idx):
        """Get (x, y) coordinates from cell index"""
        cell = self.all_cells[cell_idx]
        if isinstance(cell, tuple):
            return cell
        else:
            return (cell.x, cell.y)
    
    def _calculate_distance_heuristic(self, from_cell_idx, to_cell_idx):
        """
        Calculate basic distance heuristic (for initialization)
        Returns: 1.0 / (1.0 + distance) - closer cells have higher heuristic
        """
        from_coord = self._get_cell_coords(from_cell_idx)
        to_coord = self._get_cell_coords(to_cell_idx)
        
        # Manhattan distance
        dist = abs(to_coord[0] - from_coord[0]) + abs(to_coord[1] - from_coord[1])
        if dist == 0:
            dist = 0.1  # Avoid division by zero
        
        return 1.0 / (1.0 + dist)
    
    def _calculate_multi_objective_heuristic(self, from_cell_idx, to_cell_idx, robot_id, 
                                            current_Lr, L_bar):
        """
        Multi-objective desirability: η_ij = 1 / (dist(i, j) + γ|L_r - L̄|)
        
        Parameters:
        - from_cell_idx: Starting cell
        - to_cell_idx: Target cell
        - robot_id: Robot identifier
        - current_Lr: Current path length of robot r (estimated)
        - L_bar: Average path length across all robots (estimated)
        
        Returns:
        - Multi-objective desirability value
        """
        from_coord = self._get_cell_coords(from_cell_idx)
        to_coord = self._get_cell_coords(to_cell_idx)
        
        # Manhattan distance
        dist = abs(to_coord[0] - from_coord[0]) + abs(to_coord[1] - from_coord[1])
        if dist == 0:
            dist = 0.1  # Avoid division by zero
        
        # Workload imbalance penalty
        imbalance_penalty = abs(current_Lr - L_bar)
        
        # Multi-objective desirability formula
        eta = 1.0 / (dist + self.gamma * imbalance_penalty)
        
        return eta
    
    def construct_solution(self, ant_id: int) -> Optional[ACOSolution]:
        """
        DARP Phase: Construct solution for one ant
        
        Process:
        1. Partition cells among robots using ACO probability
        2. Build spanning tree paths (UF-STC) for each robot
        3. Evaluate objectives F1 and F2
        4. Return feasible solution
        """
        solution = create_empty_solution(
            self.all_cells, self.free_cells, self.obstacles,
            self.grid_width, self.grid_height, self.num_robots
        )
        
        # Step 1: DARP Partitioning - Assign each free cell to a robot
        # Using ACO probability: P(r|i) ∝ (τ[r][i]^α * η[r][i]^β)
        # Multi-objective desirability: η_ij = 1 / (dist(i, j) + γ|L_r - L̄|)
        assignments_per_robot = {robot_id: set() for robot_id in range(self.num_robots)}
        
        # Track estimated path lengths during assignment (for multi-objective heuristic)
        estimated_path_lengths = {robot_id: 0 for robot_id in range(self.num_robots)}
        
        for cell_idx in self.free_cells:
            # Calculate current average path length
            avg_length = (sum(estimated_path_lengths.values()) / self.num_robots 
                         if self.num_robots > 0 else 0)
            
            # Calculate probability for each robot to get this cell
            weights = []
            for robot_id in range(self.num_robots):
                tau = self.pheromone[robot_id].get(cell_idx, self.initial_pheromone)
                
                # Use multi-objective heuristic with workload balance
                current_Lr = estimated_path_lengths[robot_id]
                eta = self._calculate_multi_objective_heuristic(
                    self.robot_starts[robot_id],  # from robot start
                    cell_idx,                      # to candidate cell
                    robot_id,
                    current_Lr,
                    avg_length
                )
                
                # ACO probability formula
                weight = (tau ** self.alpha) * (eta ** self.beta)
                weights.append(weight)
            
            # Normalize to probabilities
            total_weight = sum(weights)
            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
            else:
                probabilities = [1.0 / self.num_robots] * self.num_robots
            
            # Select robot probabilistically
            selected_robot = random.choices(range(self.num_robots), weights=probabilities, k=1)[0]
            assignments_per_robot[selected_robot].add(cell_idx)
            
            # Update estimated path length (rough estimate: number of assigned cells)
            estimated_path_lengths[selected_robot] = len(assignments_per_robot[selected_robot])
        
        # Step 2: UF-STC Path Generation - Build spanning tree paths for each robot
        for robot_id in range(self.num_robots):
            assigned_cells = assignments_per_robot[robot_id]
            start_cell_idx = self.robot_starts[robot_id]
            
            # Build spanning tree path
            path = build_spanning_tree_path(
                start_cell_idx, assigned_cells, self.all_cells,
                self.free_cells, self.obstacles,
                self.grid_width, self.grid_height
            )
            
            solution.paths[robot_id] = path
        
        # Step 3: Sync assignment matrix with paths
        solution.sync_assignment_with_paths()
        
        # Step 4: Check feasibility
        is_feasible, violations = is_solution_feasible(solution)
        if not is_feasible:
            # Debug: Print violations for first few failed solutions
            if ant_id < 3:  # Only print for first 3 ants to avoid spam
                print(f"\n⚠️  Ant {ant_id} solution rejected. Violations:")
                for v in violations[:5]:  # Show first 5 violations
                    print(f"   - {v}")
                if len(violations) > 5:
                    print(f"   ... and {len(violations) - 5} more")
                # Print path info for debugging
                for robot_id, path in solution.paths.items():
                    print(f"   Robot {robot_id}: path length={len(path)}, path={path[:10]}..." if len(path) > 10 else f"   Robot {robot_id}: path={path}")
            return None
        
        # Step 5: Evaluate objectives
        F1, F2 = evaluate_objectives(solution)
        
        return solution
    
    def update_pheromone(self, ant_solutions: List[ACOSolution]):
        """
        Update pheromone trails based on ant solutions
        
        Process:
        1. Evaporate: τ[r][i] = (1 - ρ) * τ[r][i]
        2. Deposit: τ[r][i] += Δτ based on solution quality
        """
        # Evaporation
        for robot_id in range(self.num_robots):
            for cell_idx in self.free_cells:
                self.pheromone[robot_id][cell_idx] *= (1.0 - self.rho)
        
        # Deposit pheromone based on solution quality
        for solution in ant_solutions:
            if solution is None:
                continue
            
            # Calculate solution quality (higher F1, lower F2 = better)
            # Normalize F1: coverage ratio (0 to 1)
            coverage_ratio = solution.F1 / len(self.free_cells) if len(self.free_cells) > 0 else 0
            
            # Normalize F2: inverse imbalance (lower F2 = higher reward)
            # Use max F2 from all solutions to normalize
            max_F2 = max((s.F2 for s in ant_solutions if s is not None), default=1.0)
            if max_F2 > 0:
                imbalance_ratio = 1.0 - (solution.F2 / max_F2)
            else:
                imbalance_ratio = 1.0
            
            # Combined quality score
            quality = coverage_ratio * 0.7 + imbalance_ratio * 0.3
            
            # Deposit pheromone on used edges
            for robot_id, path in solution.paths.items():
                for cell_idx in path:
                    if cell_idx in self.free_cells:
                        # Deposit proportional to quality
                        deposit = quality * self.initial_pheromone
                        self.pheromone[robot_id][cell_idx] += deposit
    
    def run(self, verbose=False):
        """
        Main ACO algorithm loop
        
        Returns:
        - best_solution: Best solution found
        - history: List of (iteration, best_F1, best_F2) tuples
        """
        print(f"\n{'='*70}")
        print(f"🐜 ANT COLONY OPTIMIZATION - DARP + UF-STC")
        print(f"{'='*70}")
        print(f"Parameters:")
        print(f"  • Number of ants: {self.num_ants}")
        print(f"  • Initial pheromone: {self.initial_pheromone}")
        print(f"  • Pheromone decay (ρ): {self.rho}")
        print(f"  • Alpha (α): {self.alpha}")
        print(f"  • Beta (β): {self.beta}")
        print(f"  • Gamma (γ): {self.gamma} (workload balance weight)")
        print(f"  • Iterations: {self.iterations}")
        print(f"  • Grid: {self.grid_width}x{self.grid_height}")
        print(f"  • Robots: {self.num_robots}")
        print(f"  • Free cells: {len(self.free_cells)}")
        print(f"  • Robot starts: {self.robot_starts}")
        print(f"{'='*70}\n")
        
        for iteration in range(self.iterations):
            # Construct solutions for all ants
            ant_solutions = []
            feasible_count = 0
            
            for ant_id in range(self.num_ants):
                solution = self.construct_solution(ant_id)
                if solution is not None:
                    ant_solutions.append(solution)
                    feasible_count += 1
                    
                    # Update best solution
                    if (solution.F1 > self.best_F1 or 
                        (solution.F1 == self.best_F1 and solution.F2 < self.best_F2)):
                        self.best_solution = solution.copy()
                        self.best_F1 = solution.F1
                        self.best_F2 = solution.F2
            
            # Update pheromone trails
            if ant_solutions:
                self.update_pheromone(ant_solutions)
            
            # Record history
            self.history.append({
                'iteration': iteration + 1,
                'best_F1': self.best_F1,
                'best_F2': self.best_F2,
                'feasible_solutions': feasible_count,
                'total_ants': self.num_ants
            })
            
            # Print progress
            if verbose and (iteration % max(1, self.iterations // 10) == 0 or iteration == self.iterations - 1):
                print(f"Iteration {iteration + 1}/{self.iterations}: "
                      f"F1={self.best_F1}, F2={self.best_F2:.2f}, "
                      f"Feasible={feasible_count}/{self.num_ants}")
        
        print(f"\n{'='*70}")
        print(f"✅ OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        if self.best_solution:
            print(f"Best Solution:")
            print(f"  • F1 (Coverage): {self.best_F1}/{len(self.free_cells)} cells")
            print(f"  • F2 (Workload Imbalance): {self.best_F2:.2f}")
            print(f"  • Robot Path Lengths: {self.best_solution.Lr}")
        print(f"{'='*70}\n")
        
        return self.best_solution, self.history


if __name__ == "__main__":
    # Example usage
    grid_width = 5
    grid_height = 5
    num_robots = 2
    obstacles = [12]  # Cell 12 is obstacle
    
    total_cells = grid_width * grid_height
    free_cells = [i for i in range(total_cells) if i not in obstacles]
    all_cells = create_grid_cells(grid_width, grid_height)
    
    # Initialize ACO
    aco = AntColonyOptimization(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        num_ants=20, initial_pheromone=1.0, rho=0.5, alpha=1.0, beta=1.0, iterations=50,
        seed=42, gamma=1.0  # Multi-objective workload balance weight
    )
    
    # Run optimization
    best_solution, history = aco.run(verbose=True)
    
    # Print results
    if best_solution:
        print("\nBest Solution Details:")
        print(f"Assignment matrix: {len(best_solution.assignment)} cells x {len(best_solution.assignment[0])} robots")
        print(f"Paths:")
        for robot_id, path in best_solution.paths.items():
            print(f"  Robot {robot_id}: {len(path)} cells, Length={best_solution.Lr.get(robot_id, 0):.2f}")
            print(f"    Path: {path[:10]}..." if len(path) > 10 else f"    Path: {path}")

