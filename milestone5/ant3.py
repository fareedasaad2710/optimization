
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
        self.combined_score = None  # Combined score: w1*(1-coverage) + w2*imbalance + penalty
        #keeps a copy of solution
    def copy(self):
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
        if self.combined_score is not None:
            new_solution.combined_score = self.combined_score
        return new_solution
    
    def calculate_combined_score(self):

        if self.F1 is None or self.F2 is None:
            self.combined_score = float('inf')
            return self.combined_score
        
        # Calculate coverage ratio (0 to 1, where 1 = 100% coverage)
        max_possible_coverage = len(self.free_cells) if self.free_cells else 1
        if max_possible_coverage == 0:
            self.combined_score = float('inf')
            return self.combined_score
        
        coverage_ratio = self.F1 / max_possible_coverage
        coverage_term = 1 - coverage_ratio  # Convert to minimization (0 = perfect)
        
        # Get imbalance (F2 is already the workload imbalance)
        imbalance_term = self.F2
        
        # Calculate penalty for constraint violations
        penalty_term = calculate_penalty(self)
        
        # Set weights (same as GA)
        w1 = 0.7  # We care 70% about coverage
        w2 = 0.3  # We care 30% about balance
        
        # If we have perfect coverage, care more about balance
        if coverage_ratio >= 1.0:
            w1 = 0.5  # Reduce coverage weight
            w2 = 0.5  # Increase balance weight
        
        # Calculate final score (lower = better)
        self.combined_score = w1 * coverage_term + w2 * imbalance_term + penalty_term
        return self.combined_score
    
    #Ensure assignment matrix matches the paths
    def sync_assignment_with_paths(self):
        # Clear all assignments
        for cell_idx in range(len(self.assignment)):
            for robot_id in range(len(self.assignment[cell_idx])):
                self.assignment[cell_idx][robot_id] = 0
        
        # Set assignments based on paths
        for robot_id, path in self.paths.items():
            for cell_idx in path:
                if cell_idx < len(self.assignment):
                    self.assignment[cell_idx][robot_id] = 1

#Create initial empty solution
def create_empty_solution(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots):
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

#    Check if consecutive cells in path are adjacent (4-connected)
def check_path_continuity(path, all_cells, grid_width, grid_height):

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

#    Check if all cells in path are within grid boundaries

def check_boundary_constraint(path, all_cells, grid_width, grid_height):

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

#    Check if path contains any obstacle cells

def check_obstacle_avoidance(path, obstacles):

    violations = []
    for i, cell_idx in enumerate(path):
        if cell_idx in obstacles:
            violations.append(f"Robot enters obstacle at cell {cell_idx} (position {i} in path)")
    return len(violations) == 0, violations

#    Check all constraints for a solution
    #Returns: (is_feasible, violations_list)
def is_solution_feasible(solution: ACOSolution) -> Tuple[bool, List[str]]:

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

#    UF-STC: Build spanning tree over assigned region and generate coverage path
# using DFS preorder traversal
def build_spanning_tree_path(start_cell_idx, assigned_cells, all_cells, free_cells, obstacles, grid_width, grid_height):

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
    #Get 4-connected neighbor cell indices
    def get_neighbor_indices(cell_idx):
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

#    Calculate path length Lr = Σ d(ir,k, ir,k+1)
   # where d is Manhattan distance between consecutive cells
def calculate_path_length(path, all_cells):

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

#    Evaluate both objectives:

def evaluate_objectives(solution: ACOSolution):
    
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
    
    # Calculate combined score (same formula as GA)
    solution.calculate_combined_score()
    
    return F1, F2


def calculate_penalty(solution: ACOSolution):

    is_feasible, violations = is_solution_feasible(solution)
    
    if is_feasible:
        return 0.0
    
    penalty = 0.0
    
    # Different penalties for different rule violations (same as GA)
    penalty_factors = {
        'out_of_bounds': 1000,    # BIG penalty: robot goes outside grid
        'obstacle_collision': 500, # MEDIUM penalty: robot hits obstacle  
        'path_jump': 100          # SMALL penalty: robot jumps between non-adjacent cells
    }
    
    # Check each violation and add appropriate penalty
    for violation in violations:
        if "outside grid" in violation.lower() or "out of bounds" in violation.lower():
            penalty += penalty_factors['out_of_bounds']
        elif "obstacle" in violation.lower():
            penalty += penalty_factors['obstacle_collision']
        elif "jump" in violation.lower() or "not adjacent" in violation.lower():
            penalty += penalty_factors['path_jump']
    
    return penalty


	  #   Ant Colony Optimization for Multi-Robot Coverage Path Planning
    
    # Implements:
    # - DARP: ACO-based partitioning (assign cells to robots)
    # - UF-STC: Spanning tree path generation
    # - Multi-objective optimization: F1 (coverage) and F2 (workload imbalance)

        # Initialize ACO algorithm
        
        # Parameters:
        # - all_cells: List of all cells in grid
        # - free_cells: List of free (non-obstacle) cell indices
        # - obstacles: Set of obstacle cell indices
        # - grid_width, grid_height: Grid dimensions
        # - num_robots: Number of robots
        # - num_ants: Number of ants per iteration
        # - initial_pheromone: Initial pheromone level
        # - rho: Pheromone evaporation rate
        # - alpha: Pheromone importance
        # - beta: Heuristic importance
        # - iterations: Number of iterations
        # - robot_starts: List of starting cell indices for each robot (optional)
        # - seed: Random seed (optional)
        # - gamma: Multi-objective weighting parameter for workload balance (γ)
class AntColonyOptimization:

    def __init__(self, all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                 num_ants=20, initial_pheromone=1.0, rho=0.5, alpha=1.0, beta=1.0, iterations=50,
                 robot_starts=None, seed=None, gamma=1.0):

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
        self.best_combined_score = float('inf')  # Best combined score (lower = better)
        self.history = []
    
		#Get (x, y) coordinates from cell index
    def _get_cell_coords(self, cell_idx):
        cell = self.all_cells[cell_idx]
        if isinstance(cell, tuple):
            return cell
        else:
            return (cell.x, cell.y)
    #        Returns: 1.0 / (1.0 + distance) - closer cells have higher heuristic

    def _calculate_distance_heuristic(self, from_cell_idx, to_cell_idx):

        from_coord = self._get_cell_coords(from_cell_idx)
        to_coord = self._get_cell_coords(to_cell_idx)
        
        # Manhattan distance
        dist = abs(to_coord[0] - from_coord[0]) + abs(to_coord[1] - from_coord[1])
        if dist == 0:
            dist = 0.1  # Avoid division by zero
        
        return 1.0 / (1.0 + dist)
   #        Multi-objective desirability: η_ij = 1 / (dist(i, j) + γ|L_r - L̄|)
#desirability 
    def _calculate_multi_objective_heuristic(self, from_cell_idx, to_cell_idx, robot_id, 
                                            current_Lr, L_bar):

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
    
		#builds one solution (one ant's attempt) to assign cells to robots and create paths.
    def construct_solution(self, ant_id: int) -> Optional[ACOSolution]:

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
        
        # Calculate minimum cells per robot to ensure all robots are utilized
        min_cells_per_robot = max(1, len(self.free_cells) // (self.num_robots * 2))  # At least 1, or 1/2 of average
        
        # Track which robots need minimum assignment
        robots_needing_cells = set(range(self.num_robots))
        
        # Shuffle free cells for random processing order
        shuffled_cells = list(self.free_cells)
        random.shuffle(shuffled_cells)
        
        for cell_idx in shuffled_cells:
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
                
                # Boost probability for underutilized robots
                boost_factor = 1.0
                num_assigned = len(assignments_per_robot[robot_id])
                
                # If robot hasn't reached minimum assignment, boost its probability
                if num_assigned < min_cells_per_robot and robot_id in robots_needing_cells:
                    # Strong boost: multiply weight by factor that decreases as we approach minimum
                    boost_factor = 1.0 + (min_cells_per_robot - num_assigned) * 2.0
                
                # If robot has very few assignments compared to average, boost it
                if num_assigned > 0:
                    avg_assigned = sum(len(assignments_per_robot[r]) for r in range(self.num_robots)) / self.num_robots
                    if num_assigned < avg_assigned * 0.5:  # Less than 50% of average
                        boost_factor *= 1.5
                
                # ACO probability formula with boost
                weight = (tau ** self.alpha) * (eta ** self.beta) * boost_factor
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
            
            # Remove robot from needing-cells set if it reached minimum
            if len(assignments_per_robot[selected_robot]) >= min_cells_per_robot:
                robots_needing_cells.discard(selected_robot)
        
        # Post-processing: Ensure all robots have at least some assignments
        # If any robot has zero assignments, redistribute some cells
        for robot_id in range(self.num_robots):
            if len(assignments_per_robot[robot_id]) == 0:
                # Find robot with most assignments
                max_robot = max(range(self.num_robots), 
                              key=lambda r: len(assignments_per_robot[r]))
                
                # Transfer some cells from max_robot to empty robot
                if len(assignments_per_robot[max_robot]) > min_cells_per_robot:
                    cells_to_transfer = list(assignments_per_robot[max_robot])[:min_cells_per_robot]
                    for cell in cells_to_transfer:
                        assignments_per_robot[max_robot].remove(cell)
                        assignments_per_robot[robot_id].add(cell)
                        estimated_path_lengths[robot_id] = len(assignments_per_robot[robot_id])
                        estimated_path_lengths[max_robot] = len(assignments_per_robot[max_robot])
        
        # Step 2: Check connectivity and reassign unreachable cells
        # Helper function to check if a cell is reachable from a start position
        def is_reachable_from(start_idx, target_idx, free_cells_set, obstacles_set):
            if start_idx == target_idx:
                return True
            
            queue = collections.deque([start_idx])
            visited = {start_idx}
            
            while queue:
                current = queue.popleft()
                if current == target_idx:
                    return True
                
                neighbors = find_neighbors(current, self.all_cells, self.grid_width, self.grid_height)
                for nb_coord in neighbors:
                    # Convert to index
                    nb_idx = None
                    for idx, cell in enumerate(self.all_cells):
                        cell_coord = self._get_cell_coords(idx)
                        if isinstance(nb_coord, tuple) and cell_coord == nb_coord:
                            nb_idx = idx
                            break
                    
                    if (nb_idx is not None and 
                        nb_idx in free_cells_set and 
                        nb_idx not in obstacles_set and
                        nb_idx not in visited):
                        visited.add(nb_idx)
                        queue.append(nb_idx)
            
            return False
        
        # Reassign unreachable cells to robots that can reach them
        free_cells_set = set(self.free_cells)
        obstacles_set = set(self.obstacles)
        unreachable_cells = {}  # {cell_idx: [list of robots that can't reach it]}
        
        for robot_id in range(self.num_robots):
            assigned_cells = assignments_per_robot[robot_id]
            start_cell_idx = self.robot_starts[robot_id]
            
            # Check which assigned cells are unreachable
            for cell_idx in list(assigned_cells):
                if not is_reachable_from(start_cell_idx, cell_idx, free_cells_set, obstacles_set):
                    # This cell is unreachable by this robot
                    if cell_idx not in unreachable_cells:
                        unreachable_cells[cell_idx] = []
                    unreachable_cells[cell_idx].append(robot_id)
                    assignments_per_robot[robot_id].remove(cell_idx)
        
        # Reassign unreachable cells to robots that CAN reach them
        for cell_idx, unreachable_robots in unreachable_cells.items():
            best_robot = None
            min_distance = float('inf')
            
            # Find the closest robot that CAN reach this cell
            for robot_id in range(self.num_robots):
                if robot_id not in unreachable_robots:
                    start_cell_idx = self.robot_starts[robot_id]
                    if is_reachable_from(start_cell_idx, cell_idx, free_cells_set, obstacles_set):
                        # Calculate distance
                        start_coord = self._get_cell_coords(start_cell_idx)
                        cell_coord = self._get_cell_coords(cell_idx)
                        dist = abs(start_coord[0] - cell_coord[0]) + abs(start_coord[1] - cell_coord[1])
                        
                        if dist < min_distance:
                            min_distance = dist
                            best_robot = robot_id
            
            # Reassign to best robot (or keep with original if all robots can't reach)
            if best_robot is not None:
                assignments_per_robot[best_robot].add(cell_idx)
            else:
                # If no robot can reach it, assign to closest robot anyway
                # (might be an isolated cell - will handle in path generation)
                closest_robot = min(range(self.num_robots), 
                                  key=lambda r: abs(self._get_cell_coords(self.robot_starts[r])[0] - 
                                                   self._get_cell_coords(cell_idx)[0]) +
                                               abs(self._get_cell_coords(self.robot_starts[r])[1] - 
                                                   self._get_cell_coords(cell_idx)[1]))
                assignments_per_robot[closest_robot].add(cell_idx)
        
        # Step 3: UF-STC Path Generation - Build spanning tree paths for each robot
        # Set assignments in solution
        for robot_id in range(self.num_robots):
            for cell_idx in assignments_per_robot[robot_id]:
                if cell_idx < len(solution.assignment):
                    solution.assignment[cell_idx][robot_id] = 1
        
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
            
            # Ensure all assigned cells are in the path (add disconnected cells at the end if needed)
            path_cells_set = set(path)
            missing_in_path = assigned_cells - path_cells_set
            
            if missing_in_path:
                # Try to add missing cells to path
                # Find the last cell in current path
                last_cell = path[-1] if path else start_cell_idx
                
                for missing_cell in missing_in_path:
                    # Try to find a path from last_cell to missing_cell
                    # Use simple greedy approach
                    current = last_cell
                    path_to_missing = []
                    max_attempts = self.grid_width + self.grid_height
                    
                    for attempt in range(max_attempts):
                        if current == missing_cell:
                            break
                        
                        # Get neighbors
                        neighbors = find_neighbors(current, self.all_cells, self.grid_width, self.grid_height)
                        # Convert to indices
                        neighbor_indices = []
                        for nb_coord in neighbors:
                            for idx, cell in enumerate(self.all_cells):
                                cell_coord = self._get_cell_coords(idx)
                                if isinstance(nb_coord, tuple) and cell_coord == nb_coord:
                                    neighbor_indices.append(idx)
                                    break
                        
                        # Filter to free cells
                        valid_neighbors = [idx for idx in neighbor_indices 
                                         if idx in self.free_cells and idx not in self.obstacles]
                        
                        if not valid_neighbors:
                            break
                        
                        # Choose neighbor closest to missing cell
                        missing_coord = self._get_cell_coords(missing_cell)
                        best_neighbor = None
                        min_dist = float('inf')
                        
                        for nb_idx in valid_neighbors:
                            nb_coord = self._get_cell_coords(nb_idx)
                            dist = abs(nb_coord[0] - missing_coord[0]) + abs(nb_coord[1] - missing_coord[1])
                            if dist < min_dist:
                                min_dist = dist
                                best_neighbor = nb_idx
                        
                        if best_neighbor is None:
                            break
                        
                        path_to_missing.append(best_neighbor)
                        current = best_neighbor
                    
                    # If we reached the missing cell, add the path
                    if current == missing_cell and path_to_missing:
                        path.extend(path_to_missing)
                        if path_to_missing[-1] != missing_cell:
                            path.append(missing_cell)
                        last_cell = missing_cell
                    elif current == missing_cell:
                        # Already at target
                        path.append(missing_cell)
                        last_cell = missing_cell
                    else:
                        # Couldn't find a path - try alternative: use BFS to find path through all free cells
                        # Use BFS to find shortest path (can use any free cell, not just assigned)
                        queue = collections.deque([(last_cell, [])])
                        visited_bfs = {last_cell}
                        found_path = False
                        
                        while queue and not found_path:
                            current_bfs, path_bfs = queue.popleft()
                            
                            if current_bfs == missing_cell:
                                # Found path!
                                if path_bfs:
                                    path.extend(path_bfs)
                                    if path_bfs[-1] != missing_cell:
                                        path.append(missing_cell)
                                    last_cell = missing_cell
                                    found_path = True
                                break
                            
                            # Get all neighbors (including non-assigned free cells)
                            neighbors_bfs = find_neighbors(current_bfs, self.all_cells, self.grid_width, self.grid_height)
                            for nb_coord in neighbors_bfs:
                                # Convert to index
                                nb_idx = None
                                for idx, cell in enumerate(self.all_cells):
                                    cell_coord = self._get_cell_coords(idx)
                                    if isinstance(nb_coord, tuple) and cell_coord == nb_coord:
                                        nb_idx = idx
                                        break
                                
                                if (nb_idx is not None and 
                                    nb_idx in self.free_cells and 
                                    nb_idx not in self.obstacles and
                                    nb_idx not in visited_bfs):
                                    visited_bfs.add(nb_idx)
                                    new_path = path_bfs + [nb_idx]
                                    queue.append((nb_idx, new_path))
                        
                        # If still couldn't find path, verify reachability and try one more time
                        if not found_path:
                            # Since we checked reachability before, this cell SHOULD be reachable
                            # Try one more time with a longer BFS limit
                            queue2 = collections.deque([(last_cell, [])])
                            visited_bfs2 = {last_cell}
                            max_bfs_depth = (self.grid_width + self.grid_height) * 2  # Longer search
                            depth = 0
                            
                            while queue2 and depth < max_bfs_depth and not found_path:
                                depth += 1
                                current_bfs2, path_bfs2 = queue2.popleft()
                                
                                if current_bfs2 == missing_cell:
                                    if path_bfs2:
                                        path.extend(path_bfs2)
                                        if path_bfs2[-1] != missing_cell:
                                            path.append(missing_cell)
                                        last_cell = missing_cell
                                        found_path = True
                                    break
                                
                                neighbors_bfs2 = find_neighbors(current_bfs2, self.all_cells, self.grid_width, self.grid_height)
                                for nb_coord in neighbors_bfs2:
                                    nb_idx = None
                                    for idx, cell in enumerate(self.all_cells):
                                        cell_coord = self._get_cell_coords(idx)
                                        if isinstance(nb_coord, tuple) and cell_coord == nb_coord:
                                            nb_idx = idx
                                            break
                                    
                                    if (nb_idx is not None and 
                                        nb_idx in self.free_cells and 
                                        nb_idx not in self.obstacles and
                                        nb_idx not in visited_bfs2):
                                        visited_bfs2.add(nb_idx)
                                        new_path2 = path_bfs2 + [nb_idx]
                                        queue2.append((nb_idx, new_path2))
                            
                            # If STILL couldn't find path, the cell might be truly isolated
                            # In this case, we've already reassigned it, so skip it
                            # (It will be handled by the robot it was reassigned to)
                            if not found_path:
                                # Remove from this robot's assignments since it can't be reached
                                assignments_per_robot[robot_id].discard(missing_cell)
                                if missing_cell < len(solution.assignment):
                                    solution.assignment[missing_cell][robot_id] = 0
        
        # Step 4: Final pass - ensure all assigned cells are in paths
        # Check for any cells that are still assigned but not in any path
        all_path_cells = set()
        for path in solution.paths.values():
            all_path_cells.update(path)
        
        # Find cells assigned but not in any path
        for cell_idx in self.free_cells:
            assigned_robots = []
            for robot_id in range(self.num_robots):
                if cell_idx < len(solution.assignment) and solution.assignment[cell_idx][robot_id] == 1:
                    assigned_robots.append(robot_id)
            
            if assigned_robots and cell_idx not in all_path_cells:
                # This cell is assigned but not in any path - add it to the first assigned robot's path
                # Use BFS to find path from robot's last position to this cell
                robot_id = assigned_robots[0]
                robot_path = solution.paths[robot_id]
                start_from = robot_path[-1] if robot_path else self.robot_starts[robot_id]
                
                # Use BFS to find path
                queue_final = collections.deque([(start_from, [])])
                visited_final = {start_from}
                found_final = False
                
                while queue_final and not found_final:
                    current_final, path_final = queue_final.popleft()
                    
                    if current_final == cell_idx:
                        if path_final:
                            robot_path.extend(path_final)
                            if path_final[-1] != cell_idx:
                                robot_path.append(cell_idx)
                        else:
                            robot_path.append(cell_idx)
                        solution.paths[robot_id] = robot_path
                        found_final = True
                        break
                    
                    neighbors_final = find_neighbors(current_final, self.all_cells, self.grid_width, self.grid_height)
                    for nb_coord in neighbors_final:
                        nb_idx = None
                        for idx, cell in enumerate(self.all_cells):
                            cell_coord = self._get_cell_coords(idx)
                            if isinstance(nb_coord, tuple) and cell_coord == nb_coord:
                                nb_idx = idx
                                break
                        
                        if (nb_idx is not None and 
                            nb_idx in self.free_cells and 
                            nb_idx not in self.obstacles and
                            nb_idx not in visited_final):
                            visited_final.add(nb_idx)
                            queue_final.append((nb_idx, path_final + [nb_idx]))
                
                # If still couldn't find, try adding directly (ensures visualization)
                if not found_final:
                    robot_path.append(cell_idx)
                    solution.paths[robot_id] = robot_path
        
        # Step 5: Sync assignment matrix with final paths
        # Clear and rebuild based on actual paths
        for cell_idx in range(len(solution.assignment)):
            for robot_id in range(len(solution.assignment[cell_idx])):
                solution.assignment[cell_idx][robot_id] = 0
        
        for robot_id, path in solution.paths.items():
            for cell_idx in path:
                if cell_idx < len(solution.assignment):
                    solution.assignment[cell_idx][robot_id] = 1
        
        # Step 6: Check feasibility
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
    
		    #     1. Evaporate: τ[r][i] = (1 - ρ) * τ[r][i]
        # 2. Deposit: τ[r][i] += z_i (Ant Quantity Model)
        #    where z_i = desirability (η) - our multi-objective heuristic
    def update_pheromone(self, ant_solutions: List[ACOSolution]):
 
        # Evaporation
        for robot_id in range(self.num_robots):
            for cell_idx in self.free_cells:
                self.pheromone[robot_id][cell_idx] *= (1.0 - self.rho)
        
        # Deposit pheromone using Ant Quantity Model (each ant deposits)
        if not ant_solutions:
            return
        
        # Calculate average path length for all solutions (for desirability calculation)
        all_Lr_values = []
        for solution in ant_solutions:
            if solution is not None and solution.Lr:
                all_Lr_values.extend(solution.Lr.values())
        
        avg_L_bar = sum(all_Lr_values) / len(all_Lr_values) if all_Lr_values else 0.0
        
        # Each ant deposits pheromone on its path
        for solution in ant_solutions:
            if solution is None:
                continue
            
            # For each robot in this solution
            for robot_id, path in solution.paths.items():
                if not path:
                    continue
                
                # Get robot's current path length (for desirability calculation)
                current_Lr = solution.Lr.get(robot_id, 0.0)
                robot_start = self.robot_starts[robot_id]
                
                # Deposit on each cell in the path
                for cell_idx in path:
                    if cell_idx not in self.free_cells:
                        continue
                    
                    # Calculate desirability (z_i = η) using multi-objective heuristic
                    eta = self._calculate_multi_objective_heuristic(
                        robot_start,      # from robot start
                        cell_idx,         # to this cell
                        robot_id,
                        current_Lr,       # current path length
                        avg_L_bar         # average path length
                    )
                    
                    # Ant Quantity Model: deposit = z_i = η (desirability)
                    # In knapsack: z_i = value/weight, in our problem: z_i = η
                    deposit = eta
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
                    
                    # Update best solution using combined_score (same as GA)
                    # Lower combined_score = better solution
                    if solution.combined_score is not None and solution.combined_score < self.best_combined_score:
                        self.best_solution = solution.copy()
                        self.best_F1 = solution.F1
                        self.best_F2 = solution.F2
                        self.best_combined_score = solution.combined_score
            
            # Update pheromone trails
            if ant_solutions:
                self.update_pheromone(ant_solutions)
            
            # Record history
            self.history.append({
                'iteration': iteration + 1,
                'best_F1': self.best_F1,
                'best_F2': self.best_F2,
                'best_combined_score': self.best_combined_score,
                'feasible_solutions': feasible_count,
                'total_ants': self.num_ants
            })
            
            # Print progress
            if verbose and (iteration % max(1, self.iterations // 10) == 0 or iteration == self.iterations - 1):
                print(f"Iteration {iteration + 1}/{self.iterations}: "
                      f"F1={self.best_F1}, F2={self.best_F2:.2f}, "
                      f"Combined Score={self.best_combined_score:.4f}, "
                      f"Feasible={feasible_count}/{self.num_ants}")
        
        print(f"\n{'='*70}")
        print(f"✅ OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        if self.best_solution:
            print(f"Best Solution:")
            print(f"  • F1 (Coverage): {self.best_F1}/{len(self.free_cells)} cells")
            print(f"  • F2 (Workload Imbalance): {self.best_F2:.2f}")
            print(f"  • Combined Score: {self.best_combined_score:.4f} (lower = better)")
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

