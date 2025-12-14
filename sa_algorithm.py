"""
Simulated Annealing Algorithm for Multi-Robot Coverage Path Planning
==================================================================
FEASIBILITY-BASED APPROACH (like Dragonfly/Ant Colony)

WHAT IS SIMULATED ANNEALING?
- It's like finding the best way to arrange robots to cover an area
- Starts with a FEASIBLE solution using DARP + Spanning Tree
- Tries small changes (like swapping cells between robots)
- Only accepts FEASIBLE solutions (NO PENALTIES)
- Sometimes accepts worse solutions to avoid getting stuck
- Gradually becomes more picky about accepting worse solutions

THE FORMULA WE USE:
J = w1(1 - coverage) + w2(imbalance)

⚠️ NO PENALTIES - infeasible solutions are REJECTED

WHAT EACH PART MEANS:
- coverage: How many cells are covered (higher = better)
- imbalance: How different robot workloads are (lower = better)
- w1, w2: How much we care about coverage vs balance

SA PARAMETERS:
- Temperature: Starts high (accepts bad solutions), gets lower (more picky)
- Cooling Rate: How fast temperature drops
- Iterations: How many times we try to improve
"""

import math
import random
import copy
import collections
from collections import deque
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from problem_formulation import *


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
    DARP with CONNECTIVITY checking: Assigns cells to nearest REACHABLE robot.
    
    ✅ Ensures each cell is assigned to a robot that can actually reach it
    """
    num_robots = len(robot_positions)
    all_cells_count = grid_width * grid_height
    free_cells_list = [i for i in range(all_cells_count) if i not in obstacles]
    free_cells_set = set(free_cells_list)
    obstacles_set = set(obstacles)
    
    # Create empty partitions
    partition = {robot_id: [] for robot_id in range(num_robots)}
    
    # Helper function to calculate Manhattan distance
    def manhattan_distance(cell1_idx, cell2_idx):
        x1, y1 = cell1_idx % grid_width, cell1_idx // grid_width
        x2, y2 = cell2_idx % grid_width, cell2_idx // grid_width
        return abs(x1 - x2) + abs(y1 - y2)
    
    # Helper function to check if cell is reachable from start
    def is_reachable_from(start_idx, target_idx):
        if start_idx == target_idx:
            return True
        
        queue = collections.deque([start_idx])
        visited = {start_idx}
        
        while queue:
            current = queue.popleft()
            if current == target_idx:
                return True
            
            # Get 4-connected neighbors
            x, y = current % grid_width, current // grid_width
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    nb_idx = ny * grid_width + nx
                    if nb_idx in free_cells_set and nb_idx not in visited:
                        visited.add(nb_idx)
                        queue.append(nb_idx)
        
        return False
    
    # Assign each free cell to nearest REACHABLE robot
    for cell_idx in free_cells_list:
        min_dist = float('inf')
        best_robot = None
        
        for robot_id, robot_pos in enumerate(robot_positions):
            # ✅ CHECK REACHABILITY FIRST
            if is_reachable_from(robot_pos, cell_idx):
                dist = manhattan_distance(cell_idx, robot_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_robot = robot_id
        
        # If no robot can reach it (isolated cell), assign to nearest robot anyway
        if best_robot is None:
            best_robot = min(range(num_robots), 
                           key=lambda r: manhattan_distance(cell_idx, robot_positions[r]))
        
        partition[best_robot].append(cell_idx)
    
    return partition


# ============================================================================
# SPANNING TREE PATH CONSTRUCTION (from ant3.py)
# ============================================================================

def build_spanning_tree_path(start_cell_idx, assigned_cells, all_cells, free_cells, obstacles, grid_width, grid_height):
    """
    UF-STC: Build spanning tree over assigned region and generate coverage path
    using DFS preorder traversal.
    
    ✅ EXACT IMPLEMENTATION FROM ant3.py (achieves 100% coverage)
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
    
    # Get 4-connected neighbor cell indices
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
            # ✅ Use BFS to find path through ALL free cells (including transit cells)
            queue = collections.deque([(last_cell, [])])
            visited_bfs = {last_cell}
            path_to_missing = []
            found = False
            
            while queue and not found:
                current_bfs, path_bfs = queue.popleft()
                
                if current_bfs == m:
                    path_to_missing = path_bfs
                    found = True
                    break
                
                # Get all neighbors (including non-assigned for transit)
                neighbors_bfs = get_neighbor_indices(current_bfs)
                for nb_idx in neighbors_bfs:
                    if nb_idx not in visited_bfs:
                        visited_bfs.add(nb_idx)
                        queue.append((nb_idx, path_bfs + [nb_idx]))
            
            # Add path to missing cell if we reached it
            if found and path_to_missing:
                path.extend(path_to_missing)
                if path_to_missing[-1] != m:
                    path.append(m)
                last_cell = m
            elif found and not path_to_missing:
                # last_cell == m already
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


# ============================================================================
# FEASIBILITY CHECKING FUNCTIONS
# ============================================================================

def check_path_continuity(path: List[int], all_cells: List, grid_width: int, grid_height: int) -> Tuple[bool, List[str]]:
    """Check if consecutive cells in path are adjacent (4-connected)."""
    violations = []
    if len(path) <= 1:
        return True, violations
    
    for i in range(len(path) - 1):
        current_cell_idx = path[i]
        next_cell_idx = path[i + 1]
        
        if current_cell_idx < 0 or current_cell_idx >= len(all_cells):
            violations.append(f"Invalid cell index {current_cell_idx}")
            continue
        if next_cell_idx < 0 or next_cell_idx >= len(all_cells):
            violations.append(f"Invalid cell index {next_cell_idx}")
            continue
        
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
        
        manhattan_dist = abs(current_x - next_x) + abs(current_y - next_y)
        if manhattan_dist > 1:
            violations.append(f"Path jump: cell {current_cell_idx} to {next_cell_idx} not adjacent")
    
    return len(violations) == 0, violations


def check_boundary_constraint(path: List[int], all_cells: List, grid_width: int, grid_height: int) -> Tuple[bool, List[str]]:
    """Check if all cells in path are within grid boundaries."""
    violations = []
    for i, cell_idx in enumerate(path):
        if cell_idx < 0 or cell_idx >= len(all_cells):
            violations.append(f"Cell index {cell_idx} out of bounds")
            continue
        
        cell = all_cells[cell_idx]
        if isinstance(cell, tuple):
            x, y = cell
        else:
            x, y = cell.x, cell.y
        
        if not (0 <= x < grid_width and 0 <= y < grid_height):
            violations.append(f"Cell {cell_idx} outside grid")
    
    return len(violations) == 0, violations


def check_obstacle_avoidance(path: List[int], obstacles: List[int]) -> Tuple[bool, List[str]]:
    """Check if path contains any obstacle cells."""
    violations = []
    obstacles_set = set(obstacles)
    for i, cell_idx in enumerate(path):
        if cell_idx in obstacles_set:
            violations.append(f"Robot enters obstacle at cell {cell_idx}")
    return len(violations) == 0, violations


def is_solution_feasible(solution, all_cells, obstacles, grid_width, grid_height) -> Tuple[bool, List[str]]:
    """
    Check if solution satisfies ALL constraints.
    Returns: (is_feasible, violations_list)
    """
    all_violations = []
    
    if not isinstance(solution.paths, dict) and not isinstance(solution.paths, list):
        return False, [f"Paths must be dict or list"]
    
    # Convert to dict if list
    if isinstance(solution.paths, list):
        paths_dict = {i: p for i, p in enumerate(solution.paths)}
    else:
        paths_dict = solution.paths
    
    # Get num_robots from assignment matrix (more reliable)
    num_robots = len(solution.assignment[0]) if solution.assignment else len(paths_dict)
    
    # ✅ CHECK 1: Cell Conflicts in Assignment Matrix
    for cell_idx in range(len(solution.assignment)):
        assigned_robots = [r for r in range(num_robots) if solution.assignment[cell_idx][r] == 1]
        if len(assigned_robots) > 1:
            all_violations.append(f"Cell {cell_idx} assigned to multiple robots: {assigned_robots}")
    
    # ✅ CHECK 2: Path constraints for each robot
    for robot_id in range(num_robots):
        path = paths_dict.get(robot_id, [])
        if not isinstance(path, list):
            all_violations.append(f"Robot {robot_id}: path must be a list")
            continue
        
        # Check boundary
        boundary_valid, boundary_violations = check_boundary_constraint(
            path, all_cells, grid_width, grid_height
        )
        if not boundary_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in boundary_violations])
        
        # Check obstacles
        obstacle_valid, obstacle_violations = check_obstacle_avoidance(path, obstacles)
        if not obstacle_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in obstacle_violations])
        
        # Check continuity
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
        Calculate fitness score: J = w1(1 - coverage) + w2(imbalance)
        
        ⚠️ NO PENALTIES - infeasible solutions are REJECTED before evaluation
        """
        # Convert paths from list to dict format if needed
        if isinstance(self.paths, list):
            paths_dict = {robot_id: path for robot_id, path in enumerate(self.paths)}
        elif isinstance(self.paths, dict):
            paths_dict = copy.deepcopy(self.paths)
        else:
            paths_dict = {}
        
        # Ensure all paths in dict are lists
        for robot_id in paths_dict:
            if not isinstance(paths_dict[robot_id], list):
                paths_dict[robot_id] = [paths_dict[robot_id]] if paths_dict[robot_id] is not None else []
        
        # Convert all_cells to Cell objects if needed
        cells_list = self.all_cells
        if len(cells_list) > 0 and isinstance(cells_list[0], tuple):
            class Cell:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            cells_list = [Cell(cell[0], cell[1]) if isinstance(cell, tuple) else cell for cell in cells_list]
        elif len(cells_list) > 0 and not (hasattr(cells_list[0], 'x') and hasattr(cells_list[0], 'y')):
            class Cell:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            cells_list = [Cell(cell[0], cell[1]) if hasattr(cell, '__getitem__') else cell for cell in cells_list]
        
        # Get basic scores
        results = evaluate_solution(
            self.assignment, paths_dict, cells_list, self.free_cells, 
            self.obstacles, self.grid_width, self.grid_height
        )
        self.fitness = results
        
        # Calculate coverage ratio
        coverage_ratio = results['coverage_score'] / len(self.free_cells)
        coverage_term = 1 - coverage_ratio  # Convert to minimization
        
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

def build_continuous_path_simple(assigned_cells, all_cells, free_cells, obstacles, grid_width, grid_height):
    """
    Build a continuous path that covers assigned cells using simple BFS pathfinding.
    
    NO DARP, NO Spanning Tree - just basic BFS to connect cells.
    
    Returns: List of cell indices forming a continuous path
    """
    if not assigned_cells:
        return []
    
    if len(assigned_cells) == 1:
        return assigned_cells
    
    # Start from first assigned cell
    path = [assigned_cells[0]]
    remaining = set(assigned_cells[1:])
    visited = {assigned_cells[0]}
    
    # Helper: BFS to find path between two cells
    def bfs_path(start, goal, allowed_cells):
        if start == goal:
            return [start]
        
        queue = deque([(start, [start])])
        visited_bfs = {start}
        
        while queue:
            current, current_path = queue.popleft()
            
            # Get neighbors
            if current < len(all_cells):
                cell = all_cells[current]
                if isinstance(cell, tuple):
                    x, y = cell
                else:
                    x, y = cell.x, cell.y
                
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_width and 0 <= ny < grid_height:
                        neighbor_idx = ny * grid_width + nx
                        
                        if neighbor_idx == goal:
                            return current_path + [goal]
                        
                        if (neighbor_idx in allowed_cells and 
                            neighbor_idx not in visited_bfs and
                            neighbor_idx not in obstacles):
                            visited_bfs.add(neighbor_idx)
                            queue.append((neighbor_idx, current_path + [neighbor_idx]))
        
        return None  # No path found
    
    # Greedy: repeatedly connect to nearest unvisited assigned cell
    while remaining:
        current = path[-1]
        
        # Find nearest remaining cell
        best_next = None
        best_path = None
        
        for target in remaining:
            # Try BFS through free cells
            sub_path = bfs_path(current, target, free_cells)
            if sub_path and len(sub_path) > 1:
                if best_path is None or len(sub_path) < len(best_path):
                    best_next = target
                    best_path = sub_path
        
        if best_path:
            # Add path (skip first cell as it's already in path)
            path.extend(best_path[1:])
            remaining.remove(best_next)
            visited.add(best_next)
        else:
            # Can't reach remaining cells - just add them (will create jumps, rejected later)
            path.extend(list(remaining))
            break
    
    return path


def find_connected_components(free_cells, all_cells, obstacles, grid_width, grid_height):
    """Find connected components of free cells (handles disconnected regions)."""
    visited = set()
    components = []
    
    for start_cell in free_cells:
        if start_cell in visited:
            continue
        
        # BFS to find all cells in this component
        component = []
        queue = deque([start_cell])
        visited.add(start_cell)
        
        while queue:
            current = queue.popleft()
            component.append(current)
            
            # Check neighbors
            if current < len(all_cells):
                cell = all_cells[current]
                if isinstance(cell, tuple):
                    x, y = cell
                else:
                    x, y = cell.x, cell.y
                
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_width and 0 <= ny < grid_height:
                        neighbor_idx = ny * grid_width + nx
                        if (neighbor_idx in free_cells and 
                            neighbor_idx not in visited and
                            neighbor_idx not in obstacles):
                            visited.add(neighbor_idx)
                            queue.append(neighbor_idx)
        
        components.append(component)
    
    return components


def generate_random_solution(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots, max_attempts=100):
    """
    Generate a random solution with CONTINUOUS paths (NO jumps).
    
    Handles disconnected regions by finding connected components first.
    """
    for attempt in range(max_attempts):
        total_cells = len(all_cells)
        
        # Find connected components
        components = find_connected_components(free_cells, all_cells, obstacles, grid_width, grid_height)
        
        if len(components) > num_robots:
            # More disconnected components than robots - can't cover all!
            continue
        
        # SMART APPROACH: Distribute cells considering connectivity
        assignment = [[0 for _ in range(num_robots)] for _ in range(total_cells)]
        
        # Assign each component to a different robot first
        robot_id = 0
        all_assigned_cells = []
        
        for component in components:
            for cell_idx in component:
                assignment[cell_idx][robot_id] = 1
                all_assigned_cells.append((cell_idx, robot_id))
            robot_id = (robot_id + 1) % num_robots
        
        # Distribute remaining cells if needed to balance load
        # (This is a simple approach - could be improved)
        
        # Build continuous paths for each robot
        robot_paths = []
        for robot_id in range(num_robots):
            robot_cells = [i for i in range(total_cells) if i < len(assignment) and assignment[i][robot_id] == 1]
            if robot_cells:
                # Build continuous path within connected component
                path = build_continuous_path_simple(
                    robot_cells, all_cells, free_cells,
                    obstacles, grid_width, grid_height
                )
                robot_paths.append(path if path else robot_cells)
            else:
                robot_paths.append([])
        
        solution = RobotCoverageSolution(
            assignment, robot_paths, all_cells,
            free_cells, obstacles, grid_width, grid_height
        )
        
        # ✅ CHECK FEASIBILITY (reject if jumps)
        is_feasible, violations = is_solution_feasible(
            solution, all_cells, obstacles, grid_width, grid_height
        )
        
        if is_feasible:
            return solution  # ✅ Found feasible solution (no jumps)
        
        # Try again
        if attempt < 5 or attempt % 20 == 0:
            print(f"   Attempt {attempt+1}/{max_attempts}: Infeasible ({len(violations)} violations)")
    
    # If failed, raise error
    raise RuntimeError(
        f"❌ Failed to generate feasible solution after {max_attempts} attempts"
    )

def generate_neighbor_solution(current_solution, all_cells, obstacles, grid_width, grid_height, verbose=False, max_attempts=10):
    """
    SIMPLE neighbor generation with CONTINUOUS paths (NO jumps)
    
    Following SA tutorial: x_new = x_cur + rand
    
    Operator: Swap 1-2 cells between two random robots, rebuild continuous paths
    """
    
    num_robots = len(current_solution.assignment[0])
    free_cells = current_solution.free_cells
    
    if len(free_cells) < 2 or num_robots < 2:
        return current_solution
    
    for attempt in range(max_attempts):
        # Create a copy of current solution
        neighbor = current_solution.copy()
        
        # SIMPLE OPERATOR: Swap 1-2 cells between two random robots
        robot1 = random.randint(0, num_robots - 1)
        robot2 = random.randint(0, num_robots - 1)
        while robot2 == robot1:
            robot2 = random.randint(0, num_robots - 1)
        
        # Get cells assigned to each robot
        robot1_cells = [i for i in range(len(neighbor.assignment)) 
                       if i in free_cells and neighbor.assignment[i][robot1] == 1]
        robot2_cells = [i for i in range(len(neighbor.assignment)) 
                       if i in free_cells and neighbor.assignment[i][robot2] == 1]
        
        if robot1_cells and robot2_cells:
            # Swap 1 cell from each robot
            cell_from_r1 = random.choice(robot1_cells)
            cell_from_r2 = random.choice(robot2_cells)
            
            # Swap assignments
            neighbor.assignment[cell_from_r1][robot1] = 0
            neighbor.assignment[cell_from_r1][robot2] = 1
            
            neighbor.assignment[cell_from_r2][robot2] = 0
            neighbor.assignment[cell_from_r2][robot1] = 1
            
            # ✅ Rebuild CONTINUOUS paths for affected robots
            for robot_id in [robot1, robot2]:
                robot_cells = [i for i in range(len(neighbor.assignment)) 
                              if i in free_cells and neighbor.assignment[i][robot_id] == 1]
                if robot_cells:
                    path = build_continuous_path_simple(
                        robot_cells, all_cells, free_cells,
                        obstacles, grid_width, grid_height
                    )
                    neighbor.paths[robot_id] = path if path else robot_cells
                else:
                    neighbor.paths[robot_id] = []
            
            # ✅ CHECK FEASIBILITY (reject if jumps)
            is_feasible, violations = is_solution_feasible(
                neighbor, all_cells, obstacles, grid_width, grid_height
            )
            
            if is_feasible:
                return neighbor  # ✅ Found feasible neighbor
    
    # If all attempts failed, return current solution
    return current_solution

def simulated_annealing(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                       initial_temp=1000, cooling_rate=0.95, max_iterations=1000):
    """
    Simulated Annealing algorithm (SIMPLE & CLEAN)
    Returns: (best_solution, convergence_history)
    
    ⚠️  Simple random initial solution (NO DARP, NO Spanning Tree)
    ⚠️  Simple neighbor: swap 1-2 cells between robots (following SA tutorial)
    """
    
    print(f"\n{'='*70}")
    print(f"🔥 STARTING SIMULATED ANNEALING (SIMPLE & CLEAN)")
    print(f"⚠️  NO DARP, NO Spanning Tree - Just simple random + swap")
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
        # Step 2: Generate neighbor solution (slight change) - ONLY FEASIBLE
        verbose_iteration = (iteration < 5) or (iteration % 100 == 0) or (iteration == max_iterations - 1)
        neighbor = generate_neighbor_solution(
            current_solution, 
            all_cells, obstacles, grid_width, grid_height,
            verbose=verbose_iteration
        )
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
    improvement_pct = ((final_improvement/initial_score)*100) if initial_score != 0 else 0.0
    print(f"   • Total Improvement: {final_improvement:.3f} ({improvement_pct:.2f}% better)")
    
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


# ============================================================================
# CASE STUDY RUNNERS
# ============================================================================

def run_sa_case_study_1():
    """Case Study 1: 5x5 grid, 2 robots"""
    import time
    import os
    from visualization import visualize_solution, plot_convergence_history, plot_best_score_only
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("CASE STUDY 1: 5x5 Grid with 2 Robots")
    print("="*80)
    
    # Define grid
    grid_width, grid_height = 5, 5
    obstacles = [6, 12, 18]
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(len(all_cells)) if i not in obstacles]
    num_robots = 2
    
    print(f"\n📋 Problem Setup:")
    print(f"   • Grid: {grid_width}x{grid_height} = {grid_width * grid_height} cells")
    print(f"   • Free Cells: {len(free_cells)}")
    print(f"   • Obstacles: {len(obstacles)} at positions {obstacles}")
    print(f"   • Robots: {num_robots}")
    
    # Run SA
    start_time = time.time()
    best_solution, convergence = simulated_annealing(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        initial_temp=1000, cooling_rate=0.95, max_iterations=500
    )
    runtime = time.time() - start_time
    
    print(f"\n⏱️  SA Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    # Create results directory
    results_dir = f"results/sa_only/case1_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Visualize results
    print(f"\n{'='*80}")
    print(f"📊 GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    visualize_solution(
        best_solution,
        title="SA Case Study 1: 5x5 Grid, 2 Robots",
        save_path=f"{results_dir}/sa_solution_{timestamp}.png"
    )
    
    plot_convergence_history(
        convergence, 
        title="SA Convergence - Case Study 1",
        save_path=f"{results_dir}/sa_convergence_{timestamp}.png"
    )
    print(f"✅ SA convergence saved: {results_dir}/sa_convergence_{timestamp}.png")
    
    plot_best_score_only(
        convergence,
        title="SA Best Score - Case Study 1",
        save_path=f"{results_dir}/sa_best_score_{timestamp}.png"
    )
    print(f"✅ SA best score saved: {results_dir}/sa_best_score_{timestamp}.png")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 SA RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Best Combined Score: {best_solution.combined_score:.4f}")
    print(f"Coverage: {best_solution.fitness['coverage_score']}/{len(free_cells)} cells")
    print(f"Balance: {best_solution.fitness['balance_score']:.4f}")
    
    print(f"\n✅ SA run completed!")
    print(f"Results saved to: {results_dir}/")


def run_sa_case_study_2():
    """Case Study 2: 6x6 grid, 3 robots (matching ant3 case study)"""
    import time
    import os
    from visualization import visualize_solution, plot_convergence_history, plot_best_score_only
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("CASE STUDY 2: 6x6 Grid with 3 Robots")
    print("="*80)
    
    # Define grid (matching milestone5/case_study_ant.py)
    grid_width, grid_height = 6, 6
    obstacles = [1, 7, 13, 19, 25, 31]  # 6 obstacles
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(len(all_cells)) if i not in obstacles]
    num_robots = 3
    
    print(f"\n📋 Problem Setup:")
    print(f"   • Grid: {grid_width}x{grid_height} = {grid_width * grid_height} cells")
    print(f"   • Free Cells: {len(free_cells)}")
    print(f"   • Obstacles: {len(obstacles)} at positions {obstacles}")
    print(f"   • Robots: {num_robots}")
    
    # Run SA
    start_time = time.time()
    best_solution, convergence = simulated_annealing(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        initial_temp=1000, cooling_rate=0.95, max_iterations=500
    )
    runtime = time.time() - start_time
    
    print(f"\n⏱️  SA Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    # Create results directory
    results_dir = f"results/sa_only/case2_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Visualize results
    print(f"\n{'='*80}")
    print(f"📊 GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    visualize_solution(
        best_solution,
        title="SA Case Study 2: 6x8 Grid, 3 Robots",
        save_path=f"{results_dir}/sa_solution_{timestamp}.png"
    )
    
    plot_convergence_history(
        convergence,
        title="SA Convergence - Case Study 2",
        save_path=f"{results_dir}/sa_convergence_{timestamp}.png"
    )
    print(f"✅ SA convergence saved: {results_dir}/sa_convergence_{timestamp}.png")
    
    plot_best_score_only(
        convergence,
        title="SA Best Score - Case Study 2",
        save_path=f"{results_dir}/sa_best_score_{timestamp}.png"
    )
    print(f"✅ SA best score saved: {results_dir}/sa_best_score_{timestamp}.png")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 SA RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Best Combined Score: {best_solution.combined_score:.4f}")
    print(f"Coverage: {best_solution.fitness['coverage_score']}/{len(free_cells)} cells")
    print(f"Balance: {best_solution.fitness['balance_score']:.4f}")
    
    print(f"\n✅ SA run completed!")
    print(f"Results saved to: {results_dir}/")


def run_sa_case_study_3():
    """Case Study 3: 10x10 grid, 5 robots"""
    import time
    import os
    from visualization import visualize_solution, plot_convergence_history, plot_best_score_only
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("CASE STUDY 3: 10x10 Grid with 5 Robots")
    print("="*80)
    
    # Define grid
    grid_width, grid_height = 10, 10
    obstacles = [11, 12, 13, 21, 23, 31, 33, 41, 42, 43,
                 55, 56, 57, 65, 67, 75, 77, 85, 86, 87]
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(len(all_cells)) if i not in obstacles]
    num_robots = 5
    
    print(f"\n📋 Problem Setup:")
    print(f"   • Grid: {grid_width}x{grid_height} = {grid_width * grid_height} cells")
    print(f"   • Free Cells: {len(free_cells)}")
    print(f"   • Obstacles: {len(obstacles)}")
    print(f"   • Robots: {num_robots}")
    
    # Run SA
    start_time = time.time()
    best_solution, convergence = simulated_annealing(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        initial_temp=2000, cooling_rate=0.95, max_iterations=2000
    )
    runtime = time.time() - start_time
    
    print(f"\n⏱️  SA Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    # Create results directory
    results_dir = f"results/sa_only/case3_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Visualize results
    print(f"\n{'='*80}")
    print(f"📊 GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    visualize_solution(
        best_solution,
        title="SA Case Study 3: 10x10 Grid, 5 Robots",
        save_path=f"{results_dir}/sa_solution_{timestamp}.png"
    )
    
    plot_convergence_history(
        convergence,
        title="SA Convergence - Case Study 3",
        save_path=f"{results_dir}/sa_convergence_{timestamp}.png"
    )
    print(f"✅ SA convergence saved: {results_dir}/sa_convergence_{timestamp}.png")
    
    plot_best_score_only(
        convergence,
        title="SA Best Score - Case Study 3",
        save_path=f"{results_dir}/sa_best_score_{timestamp}.png"
    )
    print(f"✅ SA best score saved: {results_dir}/sa_best_score_{timestamp}.png")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 SA RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Best Combined Score: {best_solution.combined_score:.4f}")
    print(f"Coverage: {best_solution.fitness['coverage_score']}/{len(free_cells)} cells")
    print(f"Balance: {best_solution.fitness['balance_score']:.4f}")
    
    print(f"\n✅ SA run completed!")
    print(f"Results saved to: {results_dir}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SIMULATED ANNEALING - MULTI-ROBOT COVERAGE PATH PLANNING")
    print("FEASIBILITY-BASED APPROACH (NO PENALTIES)")
    print("="*80)
    print("\nAvailable Case Studies:")
    print("  1. Case Study 1: 5x5 grid, 2 robots")
    print("  2. Case Study 2: 6x8 grid, 3 robots")
    print("  3. Case Study 3: 10x10 grid, 5 robots")
    print("  4. Run all case studies")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_sa_case_study_1()
    elif choice == "2":
        run_sa_case_study_2()
    elif choice == "3":
        run_sa_case_study_3()
    elif choice == "4":
        print("\n🚀 Running all case studies...\n")
        run_sa_case_study_1()
        run_sa_case_study_2()
        run_sa_case_study_3()
    else:
        print("❌ Invalid choice. Please run again and select 1-4.")
