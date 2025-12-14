"""
Multi-Robot Coverage Path Planning - Problem Formulation
=======================================================

This module contains the core problem definition, evaluation functions,
and constraint handling for the multi-robot coverage path planning problem.

PROBLEM OVERVIEW:
- Multiple robots need to cover an area (grid) cooperatively
- Each robot gets assigned cells and follows a path to visit them
- Goal: Cover as many cells as possible while keeping robot workloads balanced

DECISION VARIABLES:
- assignment[i][r]: Binary (0/1) - is cell i assigned to robot r?
- paths[r]: List of cell indices - what order does robot r visit its cells?

OBJECTIVE FUNCTIONS:
- F1: Maximize coverage (cover as many free cells as possible)
- F2: Minimize workload imbalance (all robots should work similar amounts)

CONSTRAINTS:
- Path continuity: Robots can only move to adjacent cells (no jumping)
- Boundary: Robots must stay within the grid
- Obstacle avoidance: Robots cannot enter obstacle cells
"""

import math
from typing import List, Dict, Any

def distance_between_points(point1, point2):
    """
    Calculate Euclidean distance between two 2D points
    Used for: Computing robot path lengths (F2 objective)
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def find_neighbors(cell_index, all_cells, grid_width, grid_height):
    """
    Find 4-connected neighbors of a cell (path continuity constraint)
    Used for: Checking if robot moves are valid (no jumping between non-adjacent cells)
    """
    x, y = all_cells[cell_index]
    neighbors = []
    
    # Check all 4 directions: right, left, down, up
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbor_pos = (x + dx, y + dy)
        if 0 <= x + dx < grid_width and 0 <= y + dy < grid_height:
            neighbors.append(neighbor_pos)
    
    return neighbors

def count_covered_cells(assignment, free_cells):
    """
    Count how many free cells are covered by at least one robot (F1 objective)
    Used for: Measuring coverage - higher is better
    """
    covered = 0
    for cell in free_cells:
        for robot in range(len(assignment[0])):
            if assignment[cell][robot] == 1:
                covered += 1
                break
    return covered


def calculate_robot_distances(paths, all_cells):
    """
    Calculate total distance for each robot's path
    
    Args:
        paths: dict[int, list[int]] - {robot_id: [cell1, cell2, ...]}
        all_cells: list of Cell objects
    
    Returns:
        dict[int, float] - {robot_id: distance}
    """
    distances = {}
    
    # ✅ Iterate over dictionary
    for robot_id, path in paths.items():  # Not paths.values()!
        
        # ✅ Check if path is a list
        if not isinstance(path, list):
            print(f"ERROR: path for robot {robot_id} is {type(path)}, not list!")
            distances[robot_id] = 0
            continue
        
        if len(path) == 0:
            distances[robot_id] = 0
            continue
        
        total_distance = 0
        for i in range(len(path) - 1):
            cell1 = all_cells[path[i]]
            cell2 = all_cells[path[i + 1]]
            distance = abs(cell1.x - cell2.x) + abs(cell1.y - cell2.y)
            total_distance += distance
        
        distances[robot_id] = total_distance
    
    return distances

def calculate_workload_variance(distances):
    """
    Calculate variance of robot workloads (F2 objective)
    Used for: Measuring workload imbalance - lower variance = more balanced robots
    """
    if len(distances) == 0:
        return 0
    
    average = sum(distances) / len(distances)
    variance = sum((d - average) ** 2 for d in distances) / len(distances)
    return variance

def check_path_validity(paths, all_cells, obstacles, grid_width, grid_height):
    """
    Check all constraints: path continuity, boundaries, obstacle avoidance
    Used for: Validating solutions and applying penalties in SA algorithm
    """
    problems = []
    
    for robot_id, path in enumerate(paths):
        for i, cell_index in enumerate(path):
            # Constraint 1: Boundary constraint
            if cell_index < 0 or cell_index >= len(all_cells):
                problems.append(f"Robot {robot_id} goes outside grid at step {i}")
                continue
                
            # Constraint 2: Obstacle avoidance
            if cell_index in obstacles:
                problems.append(f"Robot {robot_id} hits obstacle at cell {cell_index}")
                continue
            
            # Constraint 3: Path continuity
            if i < len(path) - 1:
                next_cell_index = path[i + 1]
                current_cell = all_cells[cell_index]
                next_cell = all_cells[next_cell_index]
                
                neighbors = find_neighbors(cell_index, all_cells, grid_width, grid_height)
                if next_cell not in neighbors:
                    problems.append(f"Robot {robot_id} jumps from {current_cell} to {next_cell}")
    
    return problems

def evaluate_solution(assignment, paths, all_cells, free_cells, obstacles, grid_width, grid_height):
    """
    Evaluate a robot coverage solution
    
    Args:
        assignment: list[list[int]] - assignment[cell][robot] = 0 or 1
        paths: dict[int, list[int]] - {robot_id: [cell1, cell2, ...]}
        all_cells: list of Cell objects
        free_cells: list of free cell indices
        obstacles: list of obstacle cell indices
        grid_width: grid width
        grid_height: grid height
    
    Returns:
        dict with fitness metrics:
        - coverage_score: number of free cells covered
        - total_distance: sum of all robot distances
        - max_distance: maximum distance among robots
        - balance_score: standard deviation of distances (lower is better)
        - path_jumps: number of non-adjacent cell transitions
        - cell_conflicts: number of cells assigned to multiple robots
        - problems: list of constraint violations
    """
    
    # Debug: Validate paths structure
    if not isinstance(paths, dict):
        raise TypeError(f"paths must be dict, got {type(paths)}")
    
    for robot_id, path in paths.items():
        if not isinstance(path, list):
            raise TypeError(f"paths[{robot_id}] must be list, got {type(path)}")
    
    # Initialize results
    results = {
        'coverage_score': 0,
        'total_distance': 0,
        'max_distance': 0,
        'balance_score': 0,
        'path_jumps': 0,
        'cell_conflicts': 0,
        'problems': []
    }
    
    num_robots = len(paths)
    
    # 1. Calculate Coverage
    covered_cells = set()
    for robot_id, path in paths.items():
        covered_cells.update(path)
    
    results['coverage_score'] = len(covered_cells)
    
    # Check for uncovered cells
    uncovered = set(free_cells) - covered_cells
    if uncovered:
        results['problems'].append(f"Uncovered cells: {sorted(uncovered)}")
    
    # 2. Calculate Robot Distances
    robot_distances = calculate_robot_distances(paths, all_cells)
    
    results['total_distance'] = sum(robot_distances.values())
    results['max_distance'] = max(robot_distances.values()) if robot_distances else 0
    
    # 3. Calculate Balance Score (standard deviation of distances)
    if len(robot_distances) > 1:
        distances_list = list(robot_distances.values())
        mean_distance = sum(distances_list) / len(distances_list)
        variance = sum((d - mean_distance) ** 2 for d in distances_list) / len(distances_list)
        results['balance_score'] = math.sqrt(variance)
    else:
        results['balance_score'] = 0
    
    # 4. Check for Path Jumps (non-adjacent cells)
    path_jumps = 0
    for robot_id, path in paths.items():
        for i in range(len(path) - 1):
            cell1 = all_cells[path[i]]
            cell2 = all_cells[path[i + 1]]
            
            # Check if cells are adjacent (Manhattan distance = 1)
            if abs(cell1.x - cell2.x) + abs(cell1.y - cell2.y) != 1:
                path_jumps += 1
                results['problems'].append(
                    f"Robot {robot_id}: Jump from cell {path[i]} to {path[i+1]}"
                )
    
    results['path_jumps'] = path_jumps
    
    # 5. Check for Cell Conflicts (multiple robots assigned to same cell)
    # Only flag if cell is assigned to MULTIPLE DIFFERENT robots (not same robot visiting multiple times)
    cell_to_robots = {}  # cell_idx -> set of robot_ids
    for robot_id, path in paths.items():
        for cell_idx in path:
            if cell_idx not in cell_to_robots:
                cell_to_robots[cell_idx] = set()
            cell_to_robots[cell_idx].add(robot_id)
    
    conflicts = 0
    for cell_idx, robots_set in cell_to_robots.items():
        if len(robots_set) > 1:  # Multiple different robots
            conflicts += 1
            results['problems'].append(
                f"Cell {cell_idx} assigned to multiple robots: {sorted(robots_set)}"
            )
    
    results['cell_conflicts'] = conflicts
    
    # 6. Validate assignment matrix matches paths
    for cell_idx in range(len(assignment)):
        assigned_robots = [r for r in range(num_robots) if assignment[cell_idx][r] == 1]
        
        # Check if cell is in paths
        in_paths = []
        for robot_id, path in paths.items():
            if cell_idx in path:
                in_paths.append(robot_id)
        
        # Warn if mismatch (but don't fail)
        if set(assigned_robots) != set(in_paths):
            # This is just a warning, not counted as a problem
            pass
    
    return results

def create_grid_cells(grid_width, grid_height):
    """
    Create list of cell coordinates for the grid
    Used for: Setting up the workspace for robot coverage
    """
    all_cells = []
    for y in range(grid_height):
        for x in range(grid_width):
            all_cells.append((x, y))
    return all_cells

def print_grid_visualization(all_cells, obstacles, assignment, grid_width, grid_height):
    """
    Print visual representation of the grid with obstacles and robot assignments
    Used for: Debugging and visualizing solutions
    """
    print("\nGrid Visualization:")
    print("=" * (grid_width * 4 + 1))
    
    for y in range(grid_height):
        row_str = "|"
        for x in range(grid_width):
            cell_idx = y * grid_width + x
            if cell_idx in obstacles:
                row_str += " X |"  # X = obstacle
            else:
                robot_id = assignment[cell_idx].index(1) if 1 in assignment[cell_idx] else -1
                if robot_id >= 0:
                    row_str += f"R{robot_id}|"
                else:
                    row_str += "  |"
        print(row_str)
        print("=" * (grid_width * 4 + 1))
    
    print("Legend: X = Obstacle, R0, R1, R2, R3 = Robot assignments")
