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
    Calculate total distance traveled by each robot
    Used for: Computing workload imbalance (F2 objective)
    """
    distances = []
    for path in paths:
        total_distance = 0
        for i in range(len(path) - 1):
            current_cell = all_cells[path[i]]
            next_cell = all_cells[path[i + 1]]
            total_distance += distance_between_points(current_cell, next_cell)
        distances.append(total_distance)
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

def evaluate_solution(all_cells, free_cells, obstacles, assignment, paths, grid_width, grid_height):
    """
    Main evaluation function for multi-robot coverage path planning
    Used for: Computing fitness scores for SA algorithm
    
    Returns:
    - coverage_score: Number of cells covered (F1 objective) - higher is better
    - balance_score: Workload variance (F2 objective) - lower is better  
    - robot_distances: Distance traveled by each robot
    - problems: List of constraint violations (for penalties)
    """
    
    # Objective F1: Coverage (maximize)
    covered_count = count_covered_cells(assignment, free_cells)
    coverage_score = covered_count
    
    # Objective F2: Workload balance (minimize variance)
    robot_distances = calculate_robot_distances(paths, all_cells)
    balance_score = calculate_workload_variance(robot_distances)
    
    # Constraint validation (for penalty functions)
    problems = check_path_validity(paths, all_cells, obstacles, grid_width, grid_height)
    
    return {
        'coverage_score': coverage_score,
        'balance_score': balance_score,
        'robot_distances': robot_distances,
        'problems': problems
    }

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
