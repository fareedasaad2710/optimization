"""
Multi-Robot Coverage Path Planning Framework
============================================

This module implements a multi-robot coverage path planning framework that combines:
- DARP (Divide Areas based on Robots' Positions): generates balanced, connected partitions
- UF-STC (Unified Formation Spanning Tree Coverage): generates continuous traversal paths

The problem optimizes two objectives:
1. Maximize area coverage (minimize negative coverage)
2. Minimize workload imbalance (variance of path lengths)

Subject to constraints:
1. Path continuity (4-neighbor connectivity)
2. Map boundary constraints
3. Obstacle avoidance
"""

import math
from typing import List, Dict, Any


def euclidean(a: tuple, b: tuple) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        a: First point as (x, y) tuple
        b: Second point as (x, y) tuple
    
    Returns:
        Euclidean distance between the two points
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_index_map(cell_coords: List[tuple]) -> Dict[tuple, int]:
    """
    Build a mapping from cell coordinates to their index in the cell_coords list.
    
    This is used for efficient neighbor lookups and coordinate-to-index conversions.
    
    Args:
        cell_coords: List of (x, y) coordinate tuples for all cells
    
    Returns:
        Dictionary mapping (x, y) coordinates to their index
    """
    return {tuple(cell_coords[i]): i for i in range(len(cell_coords))}


def get_4neighbors_idx(idx: int, cell_coords: List[tuple], index_map: dict) -> List[int]:
    """
    Get the indices of 4-connected neighbors of a cell.
    
    Uses 4-neighbor connectivity (up, down, left, right) to find adjacent cells.
    This enforces the path continuity constraint.
    
    Args:
        idx: Index of the current cell
        cell_coords: List of all cell coordinates
        index_map: Mapping from coordinates to indices
    
    Returns:
        List of neighbor cell indices
    """
    x, y = cell_coords[idx]
    neighs = []
    
    # Check all 4 directions: right, left, down, up
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = (x + dx, y + dy)
        if nb in index_map:
            neighs.append(index_map[nb])
    
    return neighs


def evaluate_solution(cell_coords: List[tuple], 
                     free_cells: List[int], 
                     obstacle_cells: List[int], 
                     assignment: List[List[int]], 
                     paths: List[List[int]], 
                     battery_limits: List[float] = None) -> Dict[str, Any]:
    """
    Evaluate DARP + UF-STC objectives and constraints.
    
    This function implements the multi-objective optimization evaluation:
    - Objective F1: Maximize coverage (minimize negative coverage)
    - Objective F2: Minimize workload variance
    - Constraints: path continuity, boundaries, obstacle avoidance
    
    Args:
        cell_coords: List of (x, y) coordinates for all cells
        free_cells: List of indices of free (non-obstacle) cells
        obstacle_cells: List of indices of obstacle cells
        assignment: 2D list where assignment[i][r] = 1 if cell i assigned to robot r
        paths: List of paths, where each path is a list of cell indices
        battery_limits: Optional battery capacity limits (not used in current implementation)
    
    Returns:
        Dictionary containing:
        - F1: Coverage objective value (negative number of covered cells)
        - F2: Workload variance objective value
        - covered_cells: Number of cells covered by at least one robot
        - per_robot_distance: List of total distances for each robot
        - violations: List of constraint violations
    """
    N = len(cell_coords)  # Total number of cells
    M = len(paths)        # Number of robots
    index_map = build_index_map(cell_coords)
    free_set = set(free_cells)
    obstacle_set = set(obstacle_cells)
    violations = []
    
    # ==========================================
    # OBJECTIVE 1: Coverage (F1)
    # ==========================================
    # Maximize area coverage by minimizing negative coverage
    # A cell is covered if assigned to at least one robot
    cover_count = 0
    for i in free_cells:
        if any(assignment[i][r] == 1 for r in range(M)):
            cover_count += 1
    
    F1 = -float(cover_count)  # Negative because we want to minimize this
    
    # ==========================================
    # OBJECTIVE 2: Workload Variance (F2)
    # ==========================================
    # Minimize workload imbalance by minimizing variance of path lengths
    per_robot_distance = []
    for r in range(M):
        path = paths[r]
        # Calculate total distance for robot r by summing Euclidean distances
        # between consecutive cells in its path
        dist = sum(euclidean(cell_coords[a], cell_coords[b]) 
                  for a, b in zip(path[:-1], path[1:]))
        per_robot_distance.append(dist)
    
    # Calculate variance of distances across all robots
    mean_d = sum(per_robot_distance) / M if M > 0 else 0
    F2 = sum((d - mean_d) ** 2 for d in per_robot_distance) / M if M > 0 else 0
    
    # ==========================================
    # CONSTRAINT CHECKS
    # ==========================================
    
    # Check constraints for each robot's path
    for r in range(M):
        path = paths[r]
        
        # Constraint 1: Boundary constraint
        # Robots cannot leave the environment grid
        for idx in path:
            if idx not in range(N):
                violations.append(f"Robot {r} out of map at cell {idx}")
        
        # Constraint 2: Obstacle avoidance
        # Robots cannot enter obstacle cells
        for idx in path:
            if idx in obstacle_set:
                violations.append(f"Robot {r} enters obstacle {idx}")
        
        # Constraint 3: Path continuity
        # Robots can only move between adjacent cells (4-neighbor connectivity)
        for a, b in zip(path[:-1], path[1:]):
            if b not in get_4neighbors_idx(a, cell_coords, index_map):
                violations.append(f"Robot {r} jumps from {a} to {b}")
    
    # ==========================================
    # RETURN RESULTS
    # ==========================================
    result = {
        'F1': F1,                           # Coverage objective (negative coverage)
        'F2': F2,                           # Workload variance objective
        'covered_cells': cover_count,        # Number of cells covered
        'per_robot_distance': per_robot_distance,  # Distance traveled by each robot
        'violations': violations             # List of constraint violations
    }
    
    return result


# ==========================================
# EXAMPLE TEST CASE
# ==========================================
if __name__ == "__main__":
    # Create a simple 3x3 grid
    coords = [(x, y) for y in range(3) for x in range(3)]
    print("Cell coordinates:", coords)
    
    # All cells are free (no obstacles)
    free = list(range(9))
    obstacles = []
    
    # Setup: 2 robots, 9 cells
    M = 2  # Number of robots
    N = len(coords)  # Number of cells
    
    # Assignment matrix: first 5 cells to robot 0, last 4 cells to robot 1
    assignment = [[0] * M for _ in range(N)]
    for i in range(N):
        if i < 5:
            assignment[i][0] = 1  # Assign to robot 0
        else:
            assignment[i][1] = 1  # Assign to robot 1
    
    print("Assignment matrix:")
    for i, row in enumerate(assignment):
        print(f"Cell {i}: {row}")
    
    # Define paths for each robot
    path0 = [0, 1, 2, 5, 4]  # Robot 0's path
    path1 = [3, 6, 7, 8]     # Robot 1's path
    
    print(f"\nRobot 0 path: {path0}")
    print(f"Robot 1 path: {path1}")
    
    # Evaluate the solution
    result = evaluate_solution(coords, free, obstacles, assignment, [path0, path1])
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Coverage Objective (F1): {result['F1']}")
    print(f"Workload Variance (F2): {result['F2']}")
    print(f"Covered cells: {result['covered_cells']}")
    print(f"Per-robot distances: {result['per_robot_distance']}")
    
    if result['violations']:
        print(f"Constraint violations: {result['violations']}")
    else:
        print("No constraint violations!")
