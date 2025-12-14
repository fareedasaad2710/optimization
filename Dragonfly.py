"""
Dragonfly Algorithm for Multi-Robot Coverage Path Planning
===========================================================

Implementation of the Dragonfly Algorithm (DA) for optimizing
multi-robot coverage paths with DARP partitioning and UF-STC path construction.
"""

import numpy as np
import random
import os
from typing import List, Tuple, Dict, Optional
from problem_formulation import (
    evaluate_solution,
    find_neighbors,
    calculate_robot_distances,
    distance_between_points,
    create_grid_cells
)

from visualization import LiveDragonflyVisualizer

# Import ant3.py's robust path building function
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'milestone5'))
from ant3 import build_spanning_tree_path as ant3_build_path

def construct_paths_robust(partition, grid_width, grid_height, obstacles, all_cells, free_cells):
    """
    Wrapper around ant3's build_spanning_tree_path - handles disconnected cells robustly!
    Uses ant3's proven approach that ensures ALL assigned cells are in paths.
    """
    # Convert all_cells from SimpleNamespace objects to tuples for ant3 compatibility
    all_cells_tuples = []
    for cell in all_cells:
        if isinstance(cell, tuple):
            all_cells_tuples.append(cell)
        else:
            # Convert SimpleNamespace to tuple
            all_cells_tuples.append((cell.x, cell.y))
    
    paths = {}
    for robot_id, assigned_cells in partition.items():
        if not assigned_cells:
            paths[robot_id] = []
            continue
        
        # Pick start cell (first assigned cell for this robot)
        start_cell = min(assigned_cells)
        
        # Use ant3's robust path builder
        path = ant3_build_path(
            start_cell, assigned_cells, all_cells_tuples, free_cells, 
            obstacles, grid_width, grid_height
        )
        paths[robot_id] = path
    
    return paths

from types import SimpleNamespace
from collections import deque


def darp_partition(
    grid_width: int,
    grid_height: int,
    robot_positions: List[int],
    obstacles: List[int],
) -> Dict[int, List[int]]:
    """
    Simple DARP replacement:
    - Distributes free cells evenly across robots
    - Ensures every cell is assigned to exactly one robot
    - Satisfies coverage constraints
    """

    num_robots = len(robot_positions)
    total_cells = grid_width * grid_height

    free_cells = [i for i in range(total_cells) if i not in obstacles]

    partition = {r: [] for r in range(num_robots)}

    # Round-robin assignment of cells
    for idx, cell in enumerate(free_cells):
        robot_id = idx % num_robots
        partition[robot_id].append(cell)

    return partition


# ============================================================================
# FEASIBILITY CHECKING FUNCTIONS (from ant3.py)
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
        # Allow same cell (distance 0) - robot can stay in place
        manhattan_dist = abs(current_x - next_x) + abs(current_y - next_y)
        if manhattan_dist > 1:
            violations.append(
                f"Path jump: cell {current_cell_idx} ({current_x},{current_y}) to "
                f"{next_cell_idx} ({next_x},{next_y}) are not adjacent"
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
            violations.append(f"Cell {cell_idx} at ({x}, {y}) is outside grid boundaries")
    
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
            violations.append(f"Robot enters obstacle at cell {cell_idx} (position {i} in path)")
    return len(violations) == 0, violations


def clean_paths_remove_overlaps(
    paths: Dict[int, List[int]],
    partition: Dict[int, List[int]],
    num_robots: int,
    all_cells: List,
    grid_width: int,
    grid_height: int
) -> Dict[int, List[int]]:
    """
    Clean paths to remove cells assigned to other robots while maintaining continuity.
    Uses BFS to find alternative paths when removing overlapping cells.
    
    Returns: cleaned_paths with no overlaps
    """
    def get_neighbor_indices(cell_idx: int) -> List[int]:
        """Get 4-connected neighbor cell indices."""
        cell = all_cells[cell_idx]
        if isinstance(cell, tuple):
            x, y = cell
        else:
            x, y = cell.x, cell.y
        
        neighbors = []
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                neighbor_idx = ny * grid_width + nx
                if 0 <= neighbor_idx < len(all_cells):
                    neighbors.append(neighbor_idx)
        return neighbors
    
    def find_path_around(start: int, target: int, allowed_cells: set, max_depth: int = 20) -> Optional[List[int]]:
        """Find path from start to target using only allowed_cells."""
        if start == target:
            return []
        
        queue = deque([(start, [])])
        visited = {start}
        depth = 0
        
        while queue and depth < max_depth:
            depth += 1
            current, path = queue.popleft()
            
            if current == target:
                return path
            
            for neighbor in get_neighbor_indices(current):
                if neighbor in allowed_cells and neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        return None
    
    # Build set of cells assigned to each robot
    robot_cells = {r: set(partition.get(r, [])) for r in range(num_robots)}
    
    cleaned_paths = {}
    
    for robot_id in range(num_robots):
        original_path = paths.get(robot_id, [])
        if not original_path:
            cleaned_paths[robot_id] = []
            continue
        
        cleaned_path = []
        allowed_cells = robot_cells[robot_id]  # Only use cells assigned to this robot
        
        i = 0
        while i < len(original_path):
            cell = original_path[i]
            
            # If cell is assigned to this robot, keep it
            if cell in allowed_cells:
                cleaned_path.append(cell)
                i += 1
            else:
                # Cell is assigned to another robot - need to find alternative path
                # Find next cell in path that belongs to this robot
                next_valid_idx = None
                for j in range(i + 1, len(original_path)):
                    if original_path[j] in allowed_cells:
                        next_valid_idx = j
                        break
                
                if next_valid_idx is None:
                    # No more valid cells, stop
                    break
                
                # Try to find path from current position to next valid cell
                current_pos = cleaned_path[-1] if cleaned_path else None
                if current_pos is None:
                    # No current position, skip to next valid cell
                    cleaned_path.append(original_path[next_valid_idx])
                    i = next_valid_idx + 1
                    continue
                
                target = original_path[next_valid_idx]
                alternative_path = find_path_around(current_pos, target, allowed_cells)
                
                if alternative_path:
                    # Found alternative path using only this robot's cells
                    cleaned_path.extend(alternative_path)
                    if target not in cleaned_path:
                        cleaned_path.append(target)
                    i = next_valid_idx + 1
                else:
                    # Can't find alternative path using only this robot's cells
                    # Check if target is adjacent to current position
                    if current_pos is not None:
                        current_neighbors = get_neighbor_indices(current_pos)
                        if target in current_neighbors and target in allowed_cells:
                            # Target is adjacent and belongs to this robot - safe to add
                            cleaned_path.append(target)
                            i = next_valid_idx + 1
                        else:
                            # Can't connect without jump - skip this segment
                            # This maintains feasibility (no jumps, no overlaps)
                            # but may reduce coverage
                            i = next_valid_idx + 1
                    else:
                        # No current position - just add target
                        cleaned_path.append(target)
                        i = next_valid_idx + 1
        
        cleaned_paths[robot_id] = cleaned_path
    
    return cleaned_paths


def ensure_all_assigned_cells_in_paths(
    partition: Dict[int, List[int]],
    paths: Dict[int, List[int]],
    num_robots: int,
    free_cells: List[int],
    obstacles: List[int],
    all_cells: List,
    grid_width: int,
    grid_height: int
) -> Dict[int, List[int]]:
    """
    Ensure all cells assigned in partition are in paths (like ant3.py Step 4).
    Uses BFS to add missing cells to paths while maintaining continuity.
    
    Returns: updated_paths with all assigned cells included
    """
    obstacles_set = set(obstacles)
    
    def get_neighbor_indices(cell_idx: int) -> List[int]:
        """Get 4-connected neighbor cell indices."""
        cell = all_cells[cell_idx]
        if isinstance(cell, tuple):
            x, y = cell
        else:
            x, y = cell.x, cell.y
        
        neighbors = []
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                neighbor_idx = ny * grid_width + nx
                if 0 <= neighbor_idx < len(all_cells) and neighbor_idx not in obstacles_set:
                    neighbors.append(neighbor_idx)
        return neighbors
    
    # Check which assigned cells are missing from paths
    all_path_cells = set()
    for path in paths.values():
        all_path_cells.update(path)
    
    updated_paths = {r: list(path) for r, path in paths.items()}
    
    # For each robot, check if any assigned cells are missing from path
    for robot_id in range(num_robots):
        assigned_cells = set(partition.get(robot_id, []))
        robot_path = updated_paths.get(robot_id, [])
        path_cells_set = set(robot_path)
        missing_in_path = assigned_cells - path_cells_set
        
        if missing_in_path:
            # Try to add missing cells to path using BFS
            last_cell = robot_path[-1] if robot_path else None
            
            for missing_cell in sorted(missing_in_path):
                if last_cell is None:
                    # Empty path, just add the cell
                    robot_path.append(missing_cell)
                    last_cell = missing_cell
                    continue
                
                # Use BFS to find path from last_cell to missing_cell
                queue = deque([(last_cell, [])])
                visited = {last_cell}
                found_path = False
                
                while queue and not found_path:
                    current, path_to_missing = queue.popleft()
                    
                    if current == missing_cell:
                        # Found path!
                        if path_to_missing:
                            robot_path.extend(path_to_missing)
                            if path_to_missing[-1] != missing_cell:
                                robot_path.append(missing_cell)
                        else:
                            robot_path.append(missing_cell)
                        last_cell = missing_cell
                        found_path = True
                        break
                    
                    # Get neighbors - ONLY use cells from this robot's partition for connectivity
                    # This prevents overlaps with other robots
                    robot_assigned_cells = set(partition.get(robot_id, []))
                    for neighbor in get_neighbor_indices(current):
                        if (neighbor in robot_assigned_cells and 
                            neighbor not in obstacles_set and
                            neighbor not in visited):
                            visited.add(neighbor)
                            new_path = path_to_missing + [neighbor]
                            queue.append((neighbor, new_path))
                
                if not found_path:
                    # Can't find path - check if adjacent
                    if last_cell is not None:
                        last_neighbors = get_neighbor_indices(last_cell)
                        if missing_cell in last_neighbors:
                            robot_path.append(missing_cell)
                            last_cell = missing_cell
                        # If not adjacent and can't find path, skip (will be uncovered)
                        # This maintains feasibility (no jumps) - but we'll try to assign it later
                    else:
                        # No last cell - just add it (single cell path is valid)
                        robot_path.append(missing_cell)
                        last_cell = missing_cell
        
        updated_paths[robot_id] = robot_path
    
    return updated_paths


def sync_partition_with_paths(
    paths: Dict[int, List[int]],
    num_robots: int,
    free_cells: List[int],
    grid_width: int,
    grid_height: int,
    obstacles: List[int],
    all_cells: List
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Sync partition to match paths and ensure all assigned cells are in paths.
    Similar to ant3.py's sync_assignment_with_paths + Step 4 (ensure all cells in paths).
    Ensures each cell appears in only one robot's partition and path, and all cells are covered.
    
    Returns: (synced_partition, final_paths)
    
    Strategy: 
    1. If a cell appears in multiple paths, assign it to the robot whose path visits it first
    2. Clean paths to remove cells assigned to other robots, using BFS to find alternative paths
    3. Ensure all assigned cells are in paths (add missing cells using BFS)
    4. Final sync: rebuild partition from paths to ensure no overlaps
    """
    # Build cell ownership from paths - track first occurrence position
    cell_first_occurrence = {}  # cell_idx -> (robot_id, position)
    
    for robot_id, path in paths.items():
        for pos, cell_idx in enumerate(path):
            if cell_idx not in cell_first_occurrence:
                # First time seeing this cell - record it
                cell_first_occurrence[cell_idx] = (robot_id, pos)
            else:
                # Cell already seen - keep the one with earliest position
                existing_robot, existing_pos = cell_first_occurrence[cell_idx]
                if pos < existing_pos:
                    cell_first_occurrence[cell_idx] = (robot_id, pos)
    
    # Create new partition based on paths
    new_partition = {r: [] for r in range(num_robots)}
    
    # Assign each cell to the robot that visits it first
    for cell_idx in free_cells:
        if cell_idx in cell_first_occurrence:
            assigned_robot, _ = cell_first_occurrence[cell_idx]
            new_partition[assigned_robot].append(cell_idx)
    
    # Instead of cleaning paths (which can create jumps), rebuild paths from synced partition
    # This ensures continuity and no overlaps
    cleaned_paths = construct_paths_robust(
        new_partition, grid_width, grid_height, obstacles, all_cells, free_cells
    )
    
    # Ensure all assigned cells are in paths (like ant3.py Step 4)
    final_paths = ensure_all_assigned_cells_in_paths(
        new_partition, cleaned_paths, num_robots, free_cells, obstacles, all_cells, grid_width, grid_height
    )
    
    # Final sync: rebuild partition from final paths to ensure no overlaps
    # This ensures partition exactly matches what's in paths
    final_partition = {r: [] for r in range(num_robots)}
    cell_to_robot_final = {}  # cell_idx -> robot_id (first occurrence)
    
    for robot_id, path in final_paths.items():
        for cell_idx in path:
            if cell_idx not in cell_to_robot_final:
                cell_to_robot_final[cell_idx] = robot_id
                final_partition[robot_id].append(cell_idx)
    
    # Ensure all free cells are assigned (assign unassigned cells to closest robot)
    obstacles_set = set(obstacles)
    all_assigned_cells = set(cell_to_robot_final.keys())
    unassigned_cells = [c for c in free_cells if c not in all_assigned_cells]
    
    for cell_idx in unassigned_cells:
        # Try to find a robot that can ACTUALLY reach this cell via BFS
        # AND get the path at the same time!
        best_robot = None
        best_path = None
        
        for robot_id in range(num_robots):
            robot_path = final_paths.get(robot_id, [])
            if not robot_path:
                continue
            
            last_cell = robot_path[-1]
            
            # BFS to find path to this cell
            queue_test = deque([(last_cell, [])])
            visited_test = {last_cell}
            found_path = None
            
            while queue_test and len(visited_test) < 200:  # Limit to avoid infinite loops
                current, path_so_far = queue_test.popleft()
                
                if current == cell_idx:
                    found_path = path_so_far
                    best_robot = robot_id
                    best_path = found_path
                    break
                
                # Check neighbors
                cell_obj = all_cells[current]
                if isinstance(cell_obj, tuple):
                    x, y = cell_obj
                else:
                    x, y = cell_obj.x, cell_obj.y
                
                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_width and 0 <= ny < grid_height:
                        neighbor_idx = ny * grid_width + nx
                        if (neighbor_idx in free_cells and 
                            neighbor_idx not in obstacles_set and
                            neighbor_idx not in visited_test):
                            visited_test.add(neighbor_idx)
                            queue_test.append((neighbor_idx, path_so_far + [neighbor_idx]))
            
            if found_path is not None:
                break  # Found a robot that can reach this cell!
        
        # If no robot can reach it, fall back to nearest robot
        if best_robot is None:
            best_robot = 0
            min_dist = float('inf')
            for robot_id in range(num_robots):
                robot_path = final_paths.get(robot_id, [])
                if robot_path:
                    last_cell = robot_path[-1]
                    cell_coord = all_cells[cell_idx]
                    last_coord = all_cells[last_cell]
                    
                    if isinstance(cell_coord, tuple):
                        cx, cy = cell_coord
                    else:
                        cx, cy = cell_coord.x, cell_coord.y
                    
                    if isinstance(last_coord, tuple):
                        lx, ly = last_coord
                    else:
                        lx, ly = last_coord.x, last_coord.y
                    
                    dist = abs(cx - lx) + abs(cy - ly)
                    if dist < min_dist:
                        min_dist = dist
                        best_robot = robot_id
        
        # Assign to best robot and add to path
        final_partition[best_robot].append(cell_idx)
        
        # Add to path using the BFS path we found (or just append if no path found)
        robot_path = final_paths.get(best_robot, [])
        if best_path:
            # BFS found a valid path - use it!
            robot_path.extend(best_path)
        elif robot_path:
            # BFS failed - just append (will likely be rejected by feasibility check)
            robot_path.append(cell_idx)
        else:
            # Empty path - start new one
            robot_path = [cell_idx]
        
        final_paths[best_robot] = robot_path
    
    return final_partition, final_paths


def check_cell_overlap(paths: Dict[int, List[int]]) -> Tuple[bool, List[str]]:
    """
    Check if any cells appear in multiple robot paths (cell overlap violation).
    Returns: (no_overlap, violations_list)
    """
    violations = []
    cell_to_robots = {}  # cell_idx -> set of robot_ids
    
    for robot_id, path in paths.items():
        for cell_idx in path:
            if cell_idx not in cell_to_robots:
                cell_to_robots[cell_idx] = set()
            cell_to_robots[cell_idx].add(robot_id)
    
    # Find cells assigned to multiple robots
    for cell_idx, robots in cell_to_robots.items():
        if len(robots) > 1:
            violations.append(f"Cell {cell_idx} assigned to multiple robots: {sorted(robots)}")
    
    return len(violations) == 0, violations


def is_solution_feasible(
    paths: Dict[int, List[int]],
    all_cells: List,
    obstacles: List[int],
    grid_width: int,
    grid_height: int
) -> Tuple[bool, List[str]]:
    """
    Check all constraints for a solution.
    Returns: (is_feasible, violations_list)
    """
    all_violations = []
    
    if not isinstance(paths, dict):
        return False, [f"Paths must be a dictionary, got {type(paths)}"]
    
    num_robots = len(paths)
    
    # ============================================================================
    # OVERLAP CHECK REMOVED (matching ant3.py behavior)
    # ============================================================================
    # Robots ARE allowed to pass through the same cells!
    # This is necessary for greedy pathfinding to reach disconnected regions.
    # Only the final ASSIGNMENT (partition) needs to be non-overlapping,
    # but PATHS can overlap as robots traverse through cells.
    # ============================================================================
    
    for robot_id in range(num_robots):
        path = paths.get(robot_id, [])
        if not isinstance(path, list):
            all_violations.append(f"Robot {robot_id}: path must be a list, got {type(path)}")
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


def construct_spanning_tree_paths(
    partition: Dict[int, List[int]],
    grid_width: int,
    grid_height: int,
    obstacles: List[int],
) -> Dict[int, List[int]]:
    """
    UF-STC replacement (drop-in):
    Creates a traversal path for each robot over its assigned cells.
    Ensures continuous paths (no jumps) by connecting disconnected components.

    Output format:
        paths: {robot_id: [cell_idx1, cell_idx2, ...]}

    Strategy:
    - For each robot region (set of cells), build adjacency (4-neighbors)
    - Run BFS from a start cell to get an order that is *mostly* adjacent
    - If region is disconnected, use BFS to find paths between components
    """

    obstacles_set = set(obstacles)
    total_cells = grid_width * grid_height

    def neighbors(cell: int) -> List[int]:
        """Get 4-connected neighbors of a cell (excluding obstacles)."""
        x = cell % grid_width
        y = cell // grid_width
        nbrs = []
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                n = ny * grid_width + nx
                if n not in obstacles_set:
                    nbrs.append(n)
        return nbrs

    def find_path_between(start: int, target: int, allowed_cells: set) -> Optional[List[int]]:
        """Find a path from start to target using BFS, only using allowed_cells."""
        if start == target:
            return []
        
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return path
            
            for neighbor in neighbors(current):
                if neighbor in allowed_cells and neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        return None  # No path found

    paths: Dict[int, List[int]] = {}

    for r, cells in partition.items():
        region = set(cells)
        if not region:
            paths[r] = []
            continue

        # Pick a deterministic start (smallest index)
        start = min(region)

        visited = set()
        order: List[int] = []

        # BFS on induced subgraph (only within region)
        q = deque([start])
        visited.add(start)

        while q:
            u = q.popleft()
            order.append(u)

            for v in neighbors(u):
                if v in region and v not in visited:
                    visited.add(v)
                    q.append(v)

        # Handle disconnected components: ONLY use region cells for connectivity
        # This prevents overlaps with other robots' paths
        remaining = region - visited
        if remaining:
            last_cell = order[-1] if order else start
            
            for disconnected_cell in sorted(remaining):
                # ONLY use region cells for connectivity (no overlaps)
                path_to_disconnected = find_path_between(last_cell, disconnected_cell, region)
                
                if path_to_disconnected:
                    # Add the connecting path (excluding last_cell since it's already in order)
                    order.extend(path_to_disconnected)
                    # Add the disconnected cell itself
                    if disconnected_cell not in order:
                        order.append(disconnected_cell)
                    last_cell = disconnected_cell
                else:
                    # Can't connect - skip this cell (will be handled by repair/sync)
                    pass

        paths[r] = order

    return paths

#for swap sequence in step vector
def generate_random_initial_moves(partition: Dict[int, List[int]], num_robots: int, num_moves: int = 5) -> List[Tuple[int, int, int]]:
    """
    Generate random initial step vector (like paper's random swap sequence).
    
    This is similar to how the paper initializes step vectors with random swap sequences.
    In our multi-robot problem:
    - Paper's "swap operator" = our "cell reassignment"
    - Paper's "swap sequence" = our "list of cell moves"
    
    Simple explanation:
    - Pick random cells from random robots
    - Create random moves to transfer them to other robots
    - This gives the dragonfly an initial "movement direction"
    
    Args:
        partition: Current partition {robot_id: [cells]}
        num_robots: Number of robots
        num_moves: How many random moves to generate (default 5)
    
    Returns:
        List of moves: [(cell, from_robot, to_robot), ...]
    """
    moves = []
    
    # Collect all non-empty robots (robots that have cells)
    robots_with_cells = [r for r in range(num_robots) if len(partition.get(r, [])) > 0]
    
    if len(robots_with_cells) < 2:
        # Not enough robots to make moves, return empty
        return []
    
    # Generate random moves
    for _ in range(num_moves):
        # Pick a random robot that has cells (source)
        from_robot = random.choice(robots_with_cells)
        from_cells = partition.get(from_robot, [])
        
        if len(from_cells) == 0:
            continue  # Skip if no cells
        
        # Pick a random cell from this robot
        cell = random.choice(from_cells)
        
        # Pick a different random robot to move the cell to (destination)
        to_robot = random.choice([r for r in range(num_robots) if r != from_robot])
        
        # Create the move: (cell, from_robot, to_robot)
        moves.append((cell, from_robot, to_robot))
    
    return moves


class DragonflySolution:
    """Represents a dragonfly individual (solution)."""

    def __init__(
        self,
        partition: Dict[int, List[int]],
        paths: Dict[int, List[int]],
        fitness: float,
        grid_width: int,
        grid_height: int,
        obstacles: List[int],
        num_robots: int,
    ):
        # Current position (solution)
        self.partition = {r: list(cells) for r, cells in partition.items()}
        self.paths = {r: list(path) for r, path in paths.items()}
        self.fitness = fitness

        # Environment info
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles

        # Personal best (pBest)
        self.personal_best_partition = {r: list(cells) for r, cells in partition.items()}
        self.personal_best_paths = {r: list(path) for r, path in paths.items()}
        self.personal_best_fitness = fitness

        # ========================================================================
        # STEP VECTOR INITIALIZATION (matching paper's approach)
        # ========================================================================
## Now generates 5 random moves to start with!
#inialize step vector at first
        # ========================================================================
        self.step_vector: List[Tuple[int, int, int]] = generate_random_initial_moves(
            partition, num_robots, num_moves=5
        )

        # Stagnancy counter (for random re-initialization / escape)
        self.stagnancy: int = 0


class DragonflyOptimizer:
    def __init__(
        self,
        all_cells: List[Tuple[int, int]],
        free_cells: List[int],
        obstacles: List[int],
        grid_width: int,
        grid_height: int,
        num_robots: int,
        population_size: int = 30,
        max_iterations: int = 100,
        neighbor_size: int = 5,
        w: float = 0.9,
        s_weight: float = 2.0,
        a_weight: float = 2.0,
        c_weight: float = 2.0,
        f_weight: float = 2.0,
        e_weight: float = 1.0,
        verbose: bool = True,
        neighbor_similarity_threshold: float = 0.6,
    ):
        """
        Initialize Dragonfly Optimizer.
        (Drop-in fixed version: removes duplicate all_cells assignment + stores grid_height)
        """

        # ---- core environment ----
        self.grid_width = grid_width
        self.grid_height = grid_height          # ✅ FIX: was missing -> caused AttributeError
        self.num_robots = num_robots

        self.free_cells = list(free_cells)
        self.obstacles = list(obstacles)
        self.obstacles_set = set(obstacles)  # For faster lookup in connectivity repair

        # ---- cell representation ----
        # Your problem_formulation.calculate_robot_distances expects objects with .x and .y
        # So we keep SimpleNamespace cells even if all_cells is passed as tuples.
        self.all_cells = [
            SimpleNamespace(x=x, y=y)
            for y in range(grid_height)
            for x in range(grid_width)
        ]

        # (Optional safety) if you passed a different all_cells, we just ignore it to keep consistency
        # If you want to *use* the passed all_cells instead, tell me and I’ll adapt it safely.

        # ---- DA hyperparameters ----
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.neighbor_size = neighbor_size

        self.w = w
        self.s_weight = s_weight
        self.a_weight = a_weight
        self.c_weight = c_weight
        self.f_weight = f_weight
        self.e_weight = e_weight

        self.verbose = verbose
        self.neighbor_similarity_threshold = neighbor_similarity_threshold  # ✅ used by get_neighbors()

        # ---- runtime state ----
        self.population: List[DragonflySolution] = []
        self.food: DragonflySolution | None = None   # best
        self.enemy: DragonflySolution | None = None  # worst
        self.history = {"iteration": [], "best_fitness": [], "avg_fitness": [], "best_combined_score": []}
        self.best_combined_score_so_far = float('inf')  # Track best score for monotonic convergence

    def initialize_population(self):
        """Initialize population using DARP + UF-STC. Only accepts feasible solutions."""
        if self.verbose:
            print("Initializing population...")

        attempts = 0
        max_attempts = self.population_size * 50  # Allow up to 50x attempts (more lenient)
        
        while len(self.population) < self.population_size and attempts < max_attempts:
            attempts += 1
            
            # Generate random robot starting positions
            robot_positions = random.sample(self.free_cells, self.num_robots)

            # Apply DARP partitioning
            partition = darp_partition(
                self.grid_width, self.grid_height,
                robot_positions, self.obstacles
            )
            
            # Repair partition to ensure connectivity BEFORE building paths
            partition = self.repair(partition)

            # Build paths using ant3's robust path builder (handles disconnected cells!)
            try:
                paths = construct_paths_robust(
                    partition, self.grid_width, self.grid_height, self.obstacles,
                    self.all_cells, self.free_cells
                )
            except Exception as e:
                if self.verbose and attempts <= 5:
                    print(f"  Path construction failed on attempt {attempts}: {e}")
                continue  # Skip if path construction fails

            # Sync partition with paths and clean paths to prevent overlaps (like ant3.py)
            partition, paths = sync_partition_with_paths(
                paths, self.num_robots, self.free_cells,
                self.grid_width, self.grid_height, self.obstacles, self.all_cells
            )
            
            # CRITICAL: Verify all free cells are covered
            path_cells = set()
            for path in paths.values():
                path_cells.update(path)
            uncovered = set(self.free_cells) - path_cells
            if uncovered:
                if self.verbose and attempts <= 5:
                    print(f"  Uncovered cells on attempt {attempts}: {sorted(uncovered)}")
                # Reject - has uncovered cells!
                continue
            
            # Ensure each robot has at least one cell in partition and path
            if any(len(partition.get(r, [])) == 0 for r in range(self.num_robots)):
                continue  # Skip if any robot has empty partition
            if any(len(paths.get(r, [])) == 0 for r in range(self.num_robots)):
                continue  # Skip if any robot has empty path

            # Check feasibility before accepting
            is_feasible, violations = is_solution_feasible(
                paths, self.all_cells, self.obstacles, self.grid_width, self.grid_height
            )
            
            if not is_feasible:
                if self.verbose and attempts <= 5:
                    print(f"  Solution infeasible on attempt {attempts}: {len(violations)} violations")
                    # Print first few violations for debugging
                    for v in violations[:3]:
                        print(f"    - {v}")
                continue  # Skip infeasible solutions, try again

            # Evaluate fitness
            fitness = self.evaluate_fitness(partition, paths)

            # Create dragonfly solution
            # Now includes random initial step vector (matching paper's approach)
            solution = DragonflySolution(
                partition, paths, fitness,
                self.grid_width, self.grid_height, self.obstacles,
                self.num_robots
            )

            self.population.append(solution)

        if len(self.population) < self.population_size:
            if self.verbose:
                print(f"Warning: Only initialized {len(self.population)}/{self.population_size} feasible solutions after {attempts} attempts")
        
        if len(self.population) == 0:
            raise RuntimeError(
                f"Failed to initialize any feasible solutions after {attempts} attempts. "
                "This might indicate that the problem constraints are too strict or the grid configuration is invalid."
            )

        # Initialize food (best) and enemy (worst)
        self.update_food_and_enemy()

    # ✅ FIX (minimal): bridge DA partition -> your assignment matrix
    def partition_to_assignment(self, partition: Dict[int, List[int]]) -> List[List[int]]:
        """
        Convert partition {robot_id: [cells...]} into assignment matrix assignment[cell][robot].

        This is required because problem_formulation.evaluate_solution() expects
        an assignment matrix + paths.
        """
        total_cells = self.grid_width * self.grid_height
        assignment = [[0 for _ in range(self.num_robots)] for _ in range(total_cells)]

        # obstacles remain all-zero
        for r, cells in partition.items():
            for c in cells:
                if 0 <= c < total_cells and c not in self.obstacles:
                    assignment[c][r] = 1

        return assignment

    def evaluate_fitness(
        self,
        partition: Dict[int, List[int]],
        paths: Dict[int, List[int]],
    ) -> float:
        """
        Evaluate fitness of a solution (lower is better).
        
        This calculates F1 (coverage) and F2 (workload imbalance) objectives.
        
        INTERNAL FITNESS (for comparing solutions):
            Fitness = -Coverage + λ * Workload_Imbalance
        
        NOTE: For convergence plots, we use a different "combined_score" 
        formula (same as GA/ACO) to ensure fair comparison across algorithms.
        See line ~1582 for combined_score calculation.
        
        Simple explanation:
        - F1 = coverage_score (how many cells are covered)
        - F2 = balance_score (workload imbalance between robots)
        - Fitness combines both objectives (used to pick better solutions)
        """
        # Convert partition to assignment matrix for evaluator compatibility
        assignment = self.partition_to_assignment(partition)

        # Use your evaluation function (from problem_formulation.py)
        # This calculates F1 (coverage) and F2 (imbalance)
        results = evaluate_solution(
            assignment=assignment,
            paths=paths,
            all_cells=self.all_cells,
            free_cells=self.free_cells,
            obstacles=self.obstacles,
            grid_width=self.grid_width,
            grid_height=self.grid_height
        )

        # Extract F1 and F2 objectives
        coverage = results.get("coverage_score", 0)      # F1: Coverage
        balance = results.get("balance_score", 0.0)      # F2: Workload imbalance

        # Combined fitness for internal comparison (minimize)
        lambda_balance = 0.5
        fitness = -coverage + lambda_balance * balance

        return float(fitness)

    def update_food_and_enemy(self):
        """
        Update food (best) and enemy (worst) solutions.
        
        Simple explanation:
        - Food = Best solution (lowest fitness) → Everyone moves toward this
        - Enemy = Worst solution (highest fitness) → Everyone moves away from this
        """
        if not self.population:
            raise RuntimeError("Cannot update food/enemy: population is empty")
        
        old_food_fitness = self.food.fitness if self.food else None
        old_enemy_fitness = self.enemy.fitness if self.enemy else None
        
        self.food = min(self.population, key=lambda x: x.fitness)
        self.enemy = max(self.population, key=lambda x: x.fitness)
        
        # Print updates if verbose (only if changed)
        if self.verbose:
            if old_food_fitness is None or self.food.fitness != old_food_fitness:
                print(f"  {'='*70}")
                print(f"  🍎 FOOD (Best) updated: Fitness = {self.food.fitness:.4f}")
                print(f"  {'='*70}")
            if old_enemy_fitness is None or self.enemy.fitness != old_enemy_fitness:
                print(f"  {'='*70}")
                print(f"  💀 ENEMY (Worst) updated: Fitness = {self.enemy.fitness:.4f}")
                print(f"  {'='*70}")

    def get_neighbors(self, idx: int) -> List[DragonflySolution]:
        """
        Neighborhood selection in *solution space*.
        In the original DA, neighbors are those within a radius in position space.
        Here we approximate "radius" with partition similarity.

        Similarity = (#cells assigned to same robot in both solutions) / (#free_cells)

        If similarity >= self.neighbor_similarity_threshold => neighbor
        If no one passes threshold => return []  (so we can trigger Lévy flight)
        """
        current = self.population[idx]

        # Build a quick lookup: cell -> robot for current
        cur_owner = {}
        for r, cells in current.partition.items():
            for c in cells:
                cur_owner[c] = r

        neighbors = []
        total = len(self.free_cells)

        for j, other in enumerate(self.population):
            if j == idx:
                continue

            same = 0
            # Compare ownership for all free cells
            # (fast enough for moderate grids; can be optimized later)
            oth_owner = {}
            for r, cells in other.partition.items():
                for c in cells:
                    oth_owner[c] = r

            for c in self.free_cells:
                if cur_owner.get(c, None) == oth_owner.get(c, None):
                    same += 1

            similarity = same / max(1, total)

            if similarity >= self.neighbor_similarity_threshold:
                neighbors.append(other)

        # If too many neighbors, keep the closest ones (highest similarity)
        if len(neighbors) > self.neighbor_size:
            def sim(other):
                oth_owner = {}
                for r, cells in other.partition.items():
                    for c in cells:
                        oth_owner[c] = r
                same2 = sum(1 for c in self.free_cells if cur_owner.get(c) == oth_owner.get(c))
                return same2 / max(1, total)

            neighbors.sort(key=sim, reverse=True)
            neighbors = neighbors[:self.neighbor_size]

        return neighbors

    def separation_moves(self, solution, neighbors):
        moves = []
        total_cells = len(self.free_cells)

        for neighbor in neighbors:
            same = sum(
                1 for r in range(self.num_robots)
                for c in solution.partition.get(r, [])
                if c in neighbor.partition.get(r, [])
            )

            if total_cells > 0 and same / total_cells > 0.8:  # too similar
                for r, cells in solution.partition.items():
                    if cells:
                        cell = random.choice(cells)
                        other = random.choice([x for x in range(self.num_robots) if x != r])
                        moves.append((cell, r, other))

        return moves

    def alignment_moves(
        self,
        solution: DragonflySolution,
        neighbors: List[DragonflySolution],
    ) -> List[Tuple[int, int, int]]:
        """
        Generate alignment moves (imitate neighbor direction).
        In discrete DA, "velocity" ≈ neighbors' last successful move-lists (step_vector).
        So alignment = copy some of those move patterns.
        """
        if not neighbors:
            return []

        moves: List[Tuple[int, int, int]] = []

        # Prefer better neighbors (lower fitness = better)
        neighbors_sorted = sorted(neighbors, key=lambda n: n.fitness)

        # How many moves to copy in total (keep small so it doesn't dominate)
        copy_budget = max(1, self.neighbor_size // 2)

        for n in neighbors_sorted:
            if not n.step_vector:
                continue

            # Copy a couple of moves from this neighbor
            take = min(2, len(n.step_vector), copy_budget - len(moves))
            if take <= 0:
                break

            moves.extend(n.step_vector[:take])

            if len(moves) >= copy_budget:
                break

        # Optional safety: keep only moves involving free cells
        free_set = set(self.free_cells)
        moves = [m for m in moves if m[0] in free_set]

        return moves

    def cohesion_moves(
        self,
        solution: DragonflySolution,
        neighbours: List[DragonflySolution],
    ) -> List[Tuple[int, int, int]]:
        """
        Generate cohesion moves (move toward majority center).
        Assign cells based on majority voting across neighbours.
        """
        moves = []

        if not neighbours:
            return moves

        # Count which robot each cell is assigned to across neighbours
        cell_votes = {}
        for cell in self.free_cells:
            votes = [0] * self.num_robots
            for neighbour in neighbours:
                for robot_id, cells in neighbour.partition.items():
                    if cell in cells:
                        votes[robot_id] += 1
            cell_votes[cell] = votes

        # Move cells to majority robot if different from current
        for cell, votes in cell_votes.items():
            majority_robot = int(np.argmax(votes))

            # Majority threshold
            if votes[majority_robot] > len(neighbours) // 2:
                # Find current owner
                current_robot = None
                for robot_id, cells in solution.partition.items():
                    if cell in cells:
                        current_robot = robot_id
                        break

                if current_robot is not None and current_robot != majority_robot:
                    if random.random() < 0.3:  # probabilistic move
                        moves.append((cell, current_robot, majority_robot))

        return moves

    def food_moves(
        self,
        solution: DragonflySolution,
        food: DragonflySolution,
    ) -> List[Tuple[int, int, int]]:
        """
        Generate food moves (copy from best solution).
        Discrete equivalent of:
            F_i = X⁺ − X_i
        """
        moves: List[Tuple[int, int, int]] = []

        for robot_id in range(self.num_robots):
            food_cells = set(food.partition.get(robot_id, []))
            my_cells = set(solution.partition.get(robot_id, []))

            # Cells in food but not in current solution
            to_adopt = food_cells - my_cells

            if to_adopt:
                cells_to_move = random.sample(
                    list(to_adopt),
                    max(1, len(to_adopt) // 2)
                )

                for cell in cells_to_move:
                    # Find which robot currently owns this cell
                    current_robot = None
                    for r, cells in solution.partition.items():
                        if cell in cells:
                            current_robot = r
                            break

                    if current_robot is not None and current_robot != robot_id:
                        moves.append((cell, current_robot, robot_id))

        return moves

    def enemy_moves(
        self,
        solution: DragonflySolution,
        enemy: DragonflySolution,
    ) -> List[Tuple[int, int, int]]:
        """
        Generate enemy moves (move away from worst solution).
        Discrete equivalent of:
            E_i = X^- + X_i
        """
        moves: List[Tuple[int, int, int]] = []

        for robot_id in range(self.num_robots):
            enemy_cells = set(enemy.partition.get(robot_id, []))
            my_cells = set(solution.partition.get(robot_id, []))

            # Cells shared with enemy
            common = my_cells & enemy_cells

            # If overlap with enemy is large → push away
            if common and len(my_cells) > 0 and len(common) > 0.5 * len(my_cells):
                cells_to_move = random.sample(
                    list(common),
                    max(1, len(common) // 4)
                )

                for cell in cells_to_move:
                    other_robot = random.choice(
                        [r for r in range(self.num_robots) if r != robot_id]
                    )
                    moves.append((cell, robot_id, other_robot))

        return moves

    def apply_moves(
        self,
        solution: DragonflySolution,
        moves: List[Tuple[int, int, int]],
    ) -> Dict[int, List[int]]:
        """
        Apply moves to partition.

        Args:
            solution: Current solution
            moves: List of (cell, from_robot, to_robot) tuples

        Returns:
            New partition
        """
        new_partition = {r: list(cells) for r, cells in solution.partition.items()}

        for cell, from_robot, to_robot in moves:
            if from_robot not in new_partition or to_robot not in new_partition:
                # Ensure keys exist
                new_partition.setdefault(from_robot, [])
                new_partition.setdefault(to_robot, [])

            if cell in new_partition[from_robot]:
                new_partition[from_robot].remove(cell)
                new_partition[to_robot].append(cell)

        return new_partition

    def _repair_connectivity(self, robot_id: int, assigned_cells: List[int]) -> List[int]:
        """
        ============================================================
        CONNECTIVITY REPAIR (like ant_colony.py _repair_connectivity)
        ============================================================
        Simple explanation:
        - Checks if robot's assigned cells form connected regions
        - If there are disconnected components (isolated cells):
          * Keeps the LARGEST connected component
          * Returns smaller disconnected cells to be reassigned
        - This ensures each robot has ONE connected region
        - Prevents path building failures from disconnected cells!
        ============================================================
        """
        if not assigned_cells:
            return []
        
        assigned_set = set(assigned_cells)
        
        # Helper: Get 4-connected neighbors
        def get_neighbors(cell_idx):
            x = cell_idx % self.grid_width
            y = cell_idx // self.grid_width
            neighbors = []
            for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    n = ny * self.grid_width + nx
                    if n not in self.obstacles_set and n in assigned_set:
                        neighbors.append(n)
            return neighbors
        
        # Find connected components using BFS
        visited = set()
        components = []
        
        for start_cell in assigned_cells:
            if start_cell in visited:
                continue
            
            # BFS to find component
            component = set()
            queue = deque([start_cell])
            visited.add(start_cell)
            component.add(start_cell)
            
            while queue:
                cell = queue.popleft()
                for neighbor in get_neighbors(cell):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
        
        # Keep largest component, return others for reassignment
        if len(components) <= 1:
            return assigned_cells  # Already connected
        
        # Sort by size, keep largest
        components.sort(key=len, reverse=True)
        main_component = components[0]
        
        return list(main_component)

    def repair(self, partition: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Repair partition to ensure:
        1. All free cells are covered
        2. No cell is assigned to multiple robots
        3. Each robot has at least one cell
        4. Each robot's partition is CONNECTED (no isolated cells)
        """
        # Remove duplicates
        assigned_cells = set()
        for robot_id in range(self.num_robots):
            if robot_id in partition:
                unique_cells = []
                for cell in partition[robot_id]:
                    if cell not in assigned_cells and cell in self.free_cells:
                        unique_cells.append(cell)
                        assigned_cells.add(cell)
                partition[robot_id] = unique_cells
            else:
                partition[robot_id] = []

        # ITERATIVE CONNECTIVITY REPAIR
        # Keep repairing until no more disconnected cells or max iterations
        max_repair_iterations = 10
        for repair_iter in range(max_repair_iterations):
            disconnected_cells = []
            
            # Find and remove disconnected cells from each robot
            for robot_id in range(self.num_robots):
                if robot_id in partition and partition[robot_id]:
                    connected = self._repair_connectivity(robot_id, partition[robot_id])
                    for cell in partition[robot_id]:
                        if cell not in connected:
                            disconnected_cells.append(cell)
                    partition[robot_id] = connected
            
            if not disconnected_cells:
                break  # All partitions are connected!
            
            # Reassign disconnected cells to ADJACENT robots (maintain connectivity)
            for cell in disconnected_cells:
                cell_x = cell % self.grid_width
                cell_y = cell // self.grid_width
                
                # Find robot with partition ADJACENT to this cell
                best_robot = None
                for robot_id in range(self.num_robots):
                    robot_cells = partition.get(robot_id, [])
                    for rcell in robot_cells:
                        rx = rcell % self.grid_width
                        ry = rcell // self.grid_width
                        # Check if adjacent (Manhattan distance = 1)
                        if abs(cell_x - rx) + abs(cell_y - ry) == 1:
                            best_robot = robot_id
                            break
                    if best_robot is not None:
                        break
            
                # If found adjacent robot, assign there
                if best_robot is not None:
                    partition.setdefault(best_robot, [])
                    partition[best_robot].append(cell)
                else:
                    # No adjacent robot found - assign to nearest (may create small disconnected component)
                    best_robot = 0
                    min_dist = float('inf')
                    for robot_id in range(self.num_robots):
                        robot_cells = partition.get(robot_id, [])
                        if robot_cells:
                            for rcell in robot_cells:
                                rx = rcell % self.grid_width
                                ry = rcell // self.grid_width
                                dist = abs(cell_x - rx) + abs(cell_y - ry)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_robot = robot_id
                    partition.setdefault(best_robot, [])
                    partition[best_robot].append(cell)
        
        # Assign remaining unassigned cells
        assigned_cells = set()
        for robot_id in range(self.num_robots):
            assigned_cells.update(partition.get(robot_id, []))
        
        unassigned = [c for c in self.free_cells if c not in assigned_cells]
        for cell in unassigned:
            # Find robot with ADJACENT partition
            cell_x = cell % self.grid_width
            cell_y = cell // self.grid_width
            best_robot = None
            
            for robot_id in range(self.num_robots):
                robot_cells = partition.get(robot_id, [])
                for rcell in robot_cells:
                    rx = rcell % self.grid_width
                    ry = rcell // self.grid_width
                    if abs(cell_x - rx) + abs(cell_y - ry) == 1:
                        best_robot = robot_id
                        break
                if best_robot is not None:
                    break
            
            if best_robot is None:
                best_robot = random.randint(0, self.num_robots - 1)
            
            partition.setdefault(best_robot, [])
            partition[best_robot].append(cell)

        # Ensure each robot has at least one cell
        for robot_id in range(self.num_robots):
            if robot_id not in partition or len(partition[robot_id]) == 0:
                largest_robot = max(partition.keys(), key=lambda r: len(partition[r]))
                if len(partition[largest_robot]) > 1:
                    cell = partition[largest_robot].pop()
                    partition[robot_id] = [cell]
                else:
                    partition[robot_id] = [random.choice(self.free_cells)]

        return partition

    def update_weights(self, iteration: int):
        """
        Update weights dynamically to balance exploration vs exploitation.
        
        Simple explanation (from paper's Algorithm 1, line 6):
        - Early iterations (t=0): High s,a,c → Explore different areas
        - Late iterations (t=1): High f,e → Exploit best solutions found
        
        Weight schedule:
        - s, a, c (Separation, Alignment, Cohesion): Decrease over time
          → Less random exploration, more focused search
        - f, e (Food, Enemy): Increase over time
          → More attraction to best solution, more repulsion from worst
        - w (Inertia): Decreases over time
          → Less influence from past moves
        """
        t = iteration / max(1, (self.max_iterations - 1))  # Normalize to [0, 1]
        
        # Store old values for comparison
        old_w = self.w
        old_s = self.s_weight
        old_f = self.f_weight

        # Inertia weight: starts at 0.9, ends at 0.4
        self.w = 0.9 - 0.5 * t

        # Exploration weights (s, a, c): Start high (2.0), decrease to min (0.1)
        self.s_weight = max(0.1, 2.0 * (1.0 - t))  # Separation
        self.a_weight = max(0.1, 2.0 * (1.0 - t))  # Alignment
        self.c_weight = max(0.1, 2.0 * (1.0 - t))  # Cohesion

        # Exploitation weights (f, e): Start low, increase over time
        self.f_weight = 2.0 * t + 0.1  # Food attraction
        self.e_weight = 1.0 * t + 0.1  # Enemy distraction
        
        # Print weight updates at key iterations
        if self.verbose and (iteration == 0 or iteration % 20 == 0 or iteration == self.max_iterations - 1):
            print(f"  {'='*70}")
            print(f"  ⚙️  WEIGHT UPDATE (Iteration {iteration}, t={t:.2f})")
            print(f"     w (Inertia):     {self.w:.3f}")
            print(f"     s (Separation):  {self.s_weight:.3f} | a (Alignment):  {self.a_weight:.3f} | c (Cohesion): {self.c_weight:.3f}")
            print(f"     f (Food):        {self.f_weight:.3f} | e (Enemy):      {self.e_weight:.3f}")
            if t < 0.3:
                print(f"     → Phase: EXPLORATION (high s,a,c)")
            elif t < 0.7:
                print(f"     → Phase: TRANSITION (balanced)")
            else:
                print(f"     → Phase: EXPLOITATION (high f,e)")
            print(f"  {'='*70}")

    def optimize(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict]:
        """
        Run Dragonfly Algorithm optimization.

        Returns:
            Tuple of (best_partition, best_paths, history)
        """
        # Step 1: Initialize population
        self.initialize_population()

        # Live visualization (optional)
        live_viz = None
        if self.verbose:  # or replace with a dedicated flag like self.live_visualization
            live_viz = LiveDragonflyVisualizer(
                grid_width=self.grid_width,
                grid_height=self.grid_height,
                obstacles=self.obstacles,
                every=1,  # increase to 2 or 5 if GUI is slow
                title_prefix="🐉 Dragonfly Best"
            )

        # Snapshots for GIF creation
        history_snapshots: List[Tuple[int, Dict[int, List[int]], Dict[int, List[int]]]] = []
        SNAP_EVERY = 5  # save every 5 iterations

        if self.verbose:
            print(f"\nStarting Dragonfly Optimization")
            print(f"Population: {self.population_size}, Iterations: {self.max_iterations}")
            print(f"Initial best fitness: {self.food.fitness:.4f}")

        # Escape / stagnancy controls (discrete DA needs this more than the vector version)
        STAGNANCY_LIMIT = 20

        # Main optimization loop
        for iteration in range(self.max_iterations):

            # Step 5 (paper): Update food (best) and enemy (worst)
            self.update_food_and_enemy()

            # Step 6 (paper): Update weights over time (explore -> exploit)
            # If you already have update_weights implemented elsewhere, keep yours.
            self.update_weights(iteration)

            for idx, solution in enumerate(self.population):
                # Get neighbors (neighborhood in solution-space)
                neighbors = self.get_neighbors(idx)

                # ------------------------------------------------------------
                # DA Factors (computed in our discrete solution-space)
                # NOTE: In the original DA paper, these are vectors:
                #   S_i (Eq 1), A_i (Eq 2), C_i (Eq 3), F_i (Eq 4), E_i (Eq 5)
                # We implement "discrete equivalents" that return move-lists
                # of the form (cell, from_robot, to_robot).
                # ------------------------------------------------------------

                # (1) Separation factor  S_i
                # Eq (1):  S_i = - Σ_{j=1..N} ( X_i - X_j )
                S = self.separation_moves(solution, neighbors)

                # (2) Alignment factor  A_i
                # Eq (2):  A_i = ( Σ_{j=1..N} V_j ) / N
                A = self.alignment_moves(solution, neighbors)

                # (3) Cohesion factor  C_i
                # Eq (3):  C_i = ( Σ_{j=1..N} X_j ) / N  -  X_i
                C = self.cohesion_moves(solution, neighbors)

                # (4) Food attraction factor  F_i
                # Eq (4):  F_i = X^+  -  X_i
                F = self.food_moves(solution, self.food)

                # (5) Enemy distraction factor  E_i
                # Eq (5):  E_i = X^-  +  X_i
                E = self.enemy_moves(solution, self.enemy)
                
                # Print factors for first 2 and last 2 iterations (first 3 dragonflies only)
                if self.verbose and idx < 3 and (iteration < 2 or iteration >= self.max_iterations - 2):
                    print(f"\n  {'='*70}")
                    print(f"  📊 FACTORS for Dragonfly {idx} (Iteration {iteration})")
                    print(f"  {'='*70}")
                    print(f"     S (Separation):  {len(S)} moves → {S[:3] if len(S) > 0 else '[]'}...")
                    print(f"     A (Alignment):   {len(A)} moves → {A[:3] if len(A) > 0 else '[]'}...")
                    print(f"     C (Cohesion):    {len(C)} moves → {C[:3] if len(C) > 0 else '[]'}...")
                    print(f"     F (Food):        {len(F)} moves → {F[:3] if len(F) > 0 else '[]'}...")
                    print(f"     E (Enemy):       {len(E)} moves → {E[:3] if len(E) > 0 else '[]'}...")
                    print(f"  {'='*70}")

                # ------------------------------------------------------------
                # Step update (paper Eq 6) is:
                #   ΔX_{t+1} = (s*S + a*A + c*C + f*F + e*E) + w*ΔX_t
                #
                # Here we implement a discrete Eq 6:
                #  - "ΔX_t" is solution.step_vector (list of successful moves from last step)
                #  - "w*ΔX_t" is done by keeping a fraction of step_vector
                #  - s,a,c,f,e are used as sampling budgets from S,A,C,F,E move pools
                #
                # If there are no neighbors, DA uses Lévy flight.
                # Discrete equivalent: do a small random reassignment (levy_moves)
                # ------------------------------------------------------------

                # ============================================================
                # EQUATION 14: Update step vector (∆X^(t+1))
                # ∆X^(t+1) = (s*S ⊕ a*A ⊕ c*C ⊕ f*F ⊕ e*E) ⊕ w*∆X^t
                # ============================================================
                if neighbors:
                    all_moves = self.merge_moves(solution, S, A, C, F, E, max_moves=50)
                else:
                    all_moves = self.levy_moves(solution, k=10)
                
                # Print step vector update for first 2 and last 2 iterations
                if self.verbose and idx < 3 and (iteration < 2 or iteration >= self.max_iterations - 2):
                    print(f"\n  {'='*70}")
                    print(f"  🔄 EQUATION 14: Step Vector Update (Dragonfly {idx}, Iteration {iteration})")
                    print(f"     Old step vector: {len(solution.step_vector)} moves")
                    print(f"     New step vector: {len(all_moves)} moves → {all_moves[:5] if len(all_moves) > 0 else '[]'}...")
                    print(f"  {'='*70}")

                # ============================================================
                # EQUATION 15: Update position (X^(t+1) = X^t ⊗ ∆X^(t+1))
                # Apply the step vector (moves) to get new position (partition)
                # ============================================================
                old_partition_str = {r: len(cells) for r, cells in solution.partition.items()}
                
                # Apply moves (update "position" X in discrete form: partition)
                new_partition = self.apply_moves(solution, all_moves)
                
                # Print position update for first 2 and last 2 iterations
                if self.verbose and idx < 3 and (iteration < 2 or iteration >= self.max_iterations - 2):
                    new_partition_str = {r: len(cells) for r, cells in new_partition.items()}
                    print(f"\n  {'='*70}")
                    print(f"  📍 EQUATION 15: Position Update (Dragonfly {idx}, Iteration {iteration})")
                    print(f"     Old partition sizes: {old_partition_str}")
                    print(f"     Applied {len(all_moves)} moves")
                    print(f"     New partition sizes: {new_partition_str}")
                    print(f"  {'='*70}")

                # Repair partition to satisfy constraints (coverage/uniqueness/non-empty/connectivity)
                new_partition = self.repair(new_partition)

                # Build paths using ant3's robust path builder for the new partition
                try:
                    new_paths = construct_paths_robust(
                        new_partition, self.grid_width, self.grid_height, self.obstacles,
                        self.all_cells, self.free_cells
                    )
                except Exception:
                    # If path construction fails, skip this update
                    continue

                # Sync partition with paths and clean paths to prevent overlaps (like ant3.py)
                new_partition, new_paths = sync_partition_with_paths(
                    new_paths, self.num_robots, self.free_cells,
                    self.grid_width, self.grid_height, self.obstacles, self.all_cells
                )
                
                # CRITICAL: Verify all free cells are covered after sync
                path_cells = set()
                for path in new_paths.values():
                    path_cells.update(path)
                uncovered = set(self.free_cells) - path_cells
                if uncovered:
                    # Reject solution - has uncovered cells!
                    continue

                # Check feasibility before accepting solution
                is_feasible, violations = is_solution_feasible(
                    new_paths, self.all_cells, self.obstacles, self.grid_width, self.grid_height
                )
                
                if not is_feasible:
                    # Reject infeasible solutions (don't accept even if fitness improved)
                    continue

                # Evaluate fitness
                new_fitness = self.evaluate_fitness(new_partition, new_paths)

                # Accept if improved AND feasible
                if new_fitness < solution.fitness:
                    solution.partition = new_partition
                    solution.paths = new_paths
                    solution.fitness = new_fitness

                    # ΔX_t memory update (this makes "w*ΔX_t" meaningful next iteration)
                    solution.step_vector = list(all_moves)

                    # Reset stagnancy counter on improvement
                    # Simple explanation: This dragonfly found a better solution, so reset its "stuck timer" to 0
                    solution.stagnancy = 0

                    # Update personal best
                    if new_fitness < solution.personal_best_fitness:
                        solution.personal_best_partition = {r: list(cells) for r, cells in new_partition.items()}
                        solution.personal_best_paths = {r: list(path) for r, path in new_paths.items()}
                        solution.personal_best_fitness = new_fitness
                else:
                    # No improvement => stagnancy grows
                    # Simple explanation: This dragonfly didn't improve, so increase its "stuck timer" by 1
                    solution.stagnancy += 1

                    # If stuck too long, force an escape perturbation
                    # Simple explanation: Been stuck for too long! Time to jump to a random new position!
                    if solution.stagnancy >= STAGNANCY_LIMIT:
                        if self.verbose:
                            print(f"  🚀 Dragonfly {idx}: Attempting escape (stuck for {solution.stagnancy} iterations)...")
                        shake = self.levy_moves(solution, k=25)
                        shaken_partition = self.apply_moves(solution, shake)
                        shaken_partition = self.repair(shaken_partition)

                        try:
                            shaken_paths = construct_paths_robust(
                                shaken_partition, self.grid_width, self.grid_height, self.obstacles,
                                self.all_cells, self.free_cells
                            )
                        except Exception:
                            # Simple explanation: Escape failed - couldn't build valid paths
                            solution.stagnancy = 0
                            continue

                        # Sync partition with paths and clean paths to prevent overlaps
                        shaken_partition, shaken_paths = sync_partition_with_paths(
                            shaken_paths, self.num_robots, self.free_cells,
                            self.grid_width, self.grid_height, self.obstacles, self.all_cells
                        )
                        
                        # CRITICAL: Verify all free cells are covered
                        path_cells = set()
                        for path in shaken_paths.values():
                            path_cells.update(path)
                        uncovered = set(self.free_cells) - path_cells
                        if uncovered:
                            # Reject - has uncovered cells!
                            solution.stagnancy = 0
                            continue

                        # Check feasibility before accepting shaken solution
                        is_feasible_shaken, violations_shaken = is_solution_feasible(
                            shaken_paths, self.all_cells, self.obstacles, self.grid_width, self.grid_height
                        )
                        
                        if not is_feasible_shaken:
                            # Simple explanation: Escape rejected - solution violated constraints
                            solution.stagnancy = 0
                            continue

                        shaken_fitness = self.evaluate_fitness(shaken_partition, shaken_paths)

                        # Save old fitness BEFORE updating
                        old_fitness = solution.fitness
                        
                        if shaken_fitness < solution.fitness:
                            # Simple explanation: The random jump found a better solution! Accept it.
                            solution.partition = shaken_partition
                            solution.paths = shaken_paths
                            solution.fitness = shaken_fitness
                            solution.step_vector = list(shake)
                            if self.verbose:
                                print(f"     ✅ SUCCESS! New fitness: {shaken_fitness:.4f} (improved from {old_fitness:.4f})")
                        else:
                            # Simple explanation: The random jump didn't help, but reset counter anyway to try again later
                            if self.verbose:
                                print(f"     ❌ FAILED - No improvement (shaken={shaken_fitness:.4f}, current={old_fitness:.4f})")

                        # Reset stagnancy counter after escape attempt (successful or not)
                        solution.stagnancy = 0

            # Update food and enemy (safe to do again for logging correctness)
            self.update_food_and_enemy()

            # Record history
            avg_fitness = float(np.mean([s.fitness for s in self.population]))
            
            # ========================================================================
            # OBJECTIVE FUNCTION CALCULATION (F1 + F2) for convergence tracking
            # ========================================================================
            # This uses the SAME formula as GA/ACO for fair comparison:
            #   combined_score = w1 * (1 - coverage_ratio) + w2 * imbalance
            # 
            # Simple explanation:
            #   F1 (Coverage) = How many cells are covered out of total free cells
            #   F2 (Imbalance) = How uneven the workload is between robots
            #   Combined Score = Weighted sum of both objectives (lower = better)
            # ========================================================================
            
            # Get best solution metrics
            best_assignment = self.partition_to_assignment(self.food.partition)
            best_eval = evaluate_solution(
                assignment=best_assignment,
                paths=self.food.paths,
                all_cells=self.all_cells,
                free_cells=self.free_cells,
                obstacles=self.obstacles,
                grid_width=self.grid_width,
                grid_height=self.grid_height
            )
            best_coverage = best_eval.get('coverage_score', 0)      # F1: Coverage
            best_balance = best_eval.get('balance_score', 0.0)      # F2: Workload imbalance
            
            # Calculate combined score (same formula as GA/ACO)
            max_possible_coverage = len(self.free_cells) if self.free_cells else 1
            coverage_ratio = best_coverage / max_possible_coverage if max_possible_coverage > 0 else 0
            coverage_term = 1 - coverage_ratio  # Convert to minimization (0 = perfect coverage)
            imbalance_term = best_balance       # Already a minimization metric
            
            # Weights (same as ant3.py and GA)
            w1 = 0.7  # 70% importance on coverage
            w2 = 0.3  # 30% importance on balance
            if coverage_ratio >= 1.0:  # If full coverage achieved
                w1 = 0.5  # Reduce coverage weight
                w2 = 0.5  # Increase balance weight
            
            # Final combined score = w1 * (1 - coverage_ratio) + w2 * imbalance
            combined_score = w1 * coverage_term + w2 * imbalance_term
            
            # Ensure monotonically non-increasing best score (minimization problem)
            if combined_score < self.best_combined_score_so_far:
                self.best_combined_score_so_far = combined_score
            
            self.history["iteration"].append(iteration)
            self.history["best_fitness"].append(self.food.fitness)
            self.history["avg_fitness"].append(avg_fitness)
            self.history["best_combined_score"].append(self.best_combined_score_so_far)

            # Live update of current best solution
            if live_viz is not None:
                live_viz.update(
                    iteration=iteration,
                    partition=self.food.partition,
                    paths=self.food.paths,
                    best_fitness=self.food.fitness,
                    best_score=self.best_combined_score_so_far
                )

            # Save snapshot for GIF
            if iteration % SNAP_EVERY == 0:
                history_snapshots.append(
                    (
                        iteration,
                        {r: list(cells) for r, cells in self.food.partition.items()},
                        {r: list(path) for r, path in self.food.paths.items()},
                    )
                )

            # Print progress
            if self.verbose and (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}/{self.max_iterations} | "
                    f"Best Score: {self.best_combined_score_so_far:.4f} | Best Fitness: {self.food.fitness:.4f} | Avg: {avg_fitness:.4f}"
                )

        if self.verbose:
            print(f"\nOptimization complete!")
            if self.history["best_combined_score"]:
                print(f"Final best score (plotted): {self.history['best_combined_score'][-1]:.4f}")
            print(f"Final best fitness: {self.food.fitness:.4f}")

        # Close live visualizer
        if live_viz is not None:
            live_viz.close()

        # Attach snapshots so __main__ can build a GIF
        self.history["snapshots"] = history_snapshots

        # Return best solution (partition + paths) instead of RobotCoverageSolution
        return self.food.partition, self.food.paths, self.history

    def dedupe_moves(self, moves: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Resolve conflicting moves.
        If multiple moves target the same cell, keep only the last one.
        Also drop moves that don't actually change robot.
        """
        last = {}
        for cell, fr, to in moves:
            if fr == to:
                continue
            last[cell] = (cell, fr, to)

        return list(last.values())

    def merge_moves(
        self,
        solution: DragonflySolution,
        S: List[Tuple[int, int, int]],
        A: List[Tuple[int, int, int]],
        C: List[Tuple[int, int, int]],
        F: List[Tuple[int, int, int]],
        E: List[Tuple[int, int, int]],
        max_moves: int = 50,
    ) -> List[Tuple[int, int, int]]:
        """
        Discrete version of Eq (6):

        ΔX_{t+1} = (s*S + a*A + c*C + f*F + e*E) + w*ΔX_t

        Here:
        - each factor returns a *move list* (cell, from_robot, to_robot)
        - w*ΔX_t is implemented by keeping a fraction of solution.step_vector
        - s,a,c,f,e are implemented by sampling budgets proportional to weights
        """
        moves: List[Tuple[int, int, int]] = []

        # ---- inertia term: w * ΔX_t  ----
        keep = int(round(self.w * len(solution.step_vector)))
        if keep > 0:
            moves.extend(solution.step_vector[:keep])

        # ---- sample from factors proportional to weights ----
        weights = {
            "S": max(0.0, self.s_weight),
            "A": max(0.0, self.a_weight),
            "C": max(0.0, self.c_weight),
            "F": max(0.0, self.f_weight),
            "E": max(0.0, self.e_weight),
        }
        pool = {"S": S, "A": A, "C": C, "F": F, "E": E}

        total_w = sum(weights.values()) + 1e-9
        remaining = max_moves - len(moves)
        if remaining <= 0:
            return self.dedupe_moves(moves)[:max_moves]

        # Compute budgets
        budgets = {k: int(round(remaining * (weights[k] / total_w))) for k in weights}

        # Small correction to ensure total budgets <= remaining
        while sum(budgets.values()) > remaining:
            k = max(budgets, key=lambda x: budgets[x])
            if budgets[k] > 0:
                budgets[k] -= 1
            else:
                break

        # Sample moves from each pool
        for k in ["S", "A", "C", "F", "E"]:
            cand = pool[k]
            b = budgets[k]
            if not cand or b <= 0:
                continue
            if len(cand) <= b:
                moves.extend(cand)
            else:
                moves.extend(random.sample(cand, b))

        # Final cleanup
        moves = self.dedupe_moves(moves)
        return moves[:max_moves]

    def levy_moves(self, solution: DragonflySolution, k: int = 10) -> List[Tuple[int, int, int]]:
        """
        Discrete analogue of Lévy flight:
        when no neighbors exist, do random perturbations to escape.

        We pick k random cells and reassign them to random robots.
        """
        moves = []
        if not self.free_cells:
            return moves

        k = min(k, len(self.free_cells))
        cells = random.sample(self.free_cells, k)

        # Build owner lookup once
        owner = {}
        for r, cs in solution.partition.items():
            for c in cs:
                owner[c] = r

        for cell in cells:
            fr = owner.get(cell, None)
            if fr is None:
                continue
            to = random.choice([r for r in range(self.num_robots) if r != fr])
            moves.append((cell, fr, to))

        return moves


def dragonfly_algorithm(
    all_cells: List[Tuple[int, int]],
    free_cells: List[int],
    obstacles: List[int],
    grid_width: int,
    grid_height: int,
    num_robots: int,
    population_size: int = 30,
    max_iterations: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Wrapper function for Dragonfly Algorithm.

    Returns:
        Dictionary with 'best_solution' and 'history'
    """
    optimizer = DragonflyOptimizer(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        population_size=population_size,
        max_iterations=max_iterations,
        verbose=verbose
    )

    best_partition, best_paths, history = optimizer.optimize()

    return {
        "best_solution": {
            "partition": best_partition,
            "paths": best_paths
        },
        "history": history
    }

if __name__ == "__main__":
    import time
    import os
    import traceback
    from datetime import datetime
    from types import SimpleNamespace

    # Import visualization functions
    try:
        from visualization import visualize_solution, plot_best_score_only
        HAS_VISUALIZATION = True
    except ImportError:
        print("⚠️  Visualization module not found. Visualization will be skipped.")
        HAS_VISUALIZATION = False

    # --- Case Study 2 config (same as your case_studies.py) ---
    grid_width, grid_height = 6, 6
    num_robots = 3
    obstacles = [1, 7, 13, 19, 25, 31]

    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]

    # --- Dragonfly params ---
    population_size = 20
    max_iterations = 100
    verbose = True

    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"run_{timestamp}_dragonfly"
    results_dir = "dragonfly_results/case_study_2_dragonfly"
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("RUNNING DRAGONFLY - CASE STUDY 2 (6x6, 3 robots)")
    print("=" * 80)
    print(f"Grid: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Obstacles: {len(obstacles)} -> {obstacles}")
    print(f"Free cells: {len(free_cells)}")
    print(f"Params: population_size={population_size}, max_iterations={max_iterations}")
    print("=" * 80)

    # --- Run ---
    start = time.time()
    try:
        results = dragonfly_algorithm(
            all_cells=all_cells,
            free_cells=free_cells,
            obstacles=obstacles,
            grid_width=grid_width,
            grid_height=grid_height,
            num_robots=num_robots,
            population_size=population_size,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    except Exception as e:
        print(f"❌ Dragonfly run failed: {e}")
        traceback.print_exc()
        raise
    end = time.time()

    best_solution = results["best_solution"]
    history = results["history"]

    best_partition = best_solution["partition"]
    best_paths = best_solution["paths"]

    print("\n" + "-" * 80)
    print("✅ DRAGONFLY DONE")
    print("-" * 80)
    print(f"Runtime: {end - start:.2f}s ({(end - start)/60:.2f} min)")

    # ✅ Evaluate + print final metrics using your evaluator
    optimizer_tmp = DragonflyOptimizer(
        all_cells=all_cells,
        free_cells=free_cells,
        obstacles=obstacles,
        grid_width=grid_width,
        grid_height=grid_height,
        num_robots=num_robots,
        population_size=1,
        max_iterations=1,
        verbose=False
    )

    # ✅ FIX 1: assignment must be created from best_partition
    assignment = optimizer_tmp.partition_to_assignment(best_partition)

    # ✅ FIX 2: evaluator expects cells with .x/.y (not tuples)
    all_cells_obj = [SimpleNamespace(x=x, y=y) for (x, y) in all_cells]

    final_eval = evaluate_solution(
        assignment=assignment,
        paths=best_paths,
        all_cells=all_cells_obj,
        free_cells=free_cells,
        obstacles=obstacles,
        grid_width=grid_width,
        grid_height=grid_height
    )

    coverage_score = final_eval.get('coverage_score', 0)
    balance_score = final_eval.get('balance_score', 0.0)
    
    print(f"Coverage score: {coverage_score} / {len(free_cells)}")
    print(f"Balance score:  {balance_score:.4f}")
    if final_eval.get("problems"):
        print(f"Problems found: {len(final_eval['problems'])} (showing up to 10)")
        for p in final_eval["problems"][:10]:
            print("  -", p)
    else:
        print("✓ No problems found by evaluator.")

    # Calculate combined score (same formula as GA/ACO)
    max_possible_coverage = len(free_cells) if free_cells else 1
    coverage_ratio = coverage_score / max_possible_coverage if max_possible_coverage > 0 else 0
    coverage_term = 1 - coverage_ratio  # Convert to minimization (0 = perfect)
    imbalance_term = balance_score
    
    # Weights (same as GA/ACO)
    w1 = 0.7  # Coverage weight
    w2 = 0.3  # Balance weight
    
    if coverage_ratio >= 1.0:
        w1 = 0.5
        w2 = 0.5
    
    combined_score = w1 * coverage_term + w2 * imbalance_term

    # Generate visualizations
    if HAS_VISUALIZATION:
        print("\n" + "=" * 80)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        try:
            # Convert solution for visualization (similar to convert_aco_solution_for_visualization)
            class SolutionWrapper:
                def __init__(self, partition, paths, all_cells, free_cells, obstacles, grid_width, grid_height, final_eval):
                    self.assignment = optimizer_tmp.partition_to_assignment(partition)
                    self.paths = paths  # Already a dict {robot_id: [cells]}
                    self.all_cells = all_cells
                    self.free_cells = free_cells
                    self.obstacles = obstacles
                    self.grid_width = grid_width
                    self.grid_height = grid_height
                    
                    # Add fitness information for visualization
                    self.fitness = {
                        'coverage_score': final_eval.get('coverage_score', 0),
                        'balance_score': final_eval.get('balance_score', 0.0),
                        'robot_distances': final_eval.get('robot_distances', []),
                        'problems': final_eval.get('problems', [])
                    }
                    
                    # Calculate combined score
                    self.combined_score = combined_score
            
            viz_solution = SolutionWrapper(
                best_partition, best_paths, all_cells_obj, free_cells, obstacles,
                grid_width, grid_height, final_eval
            )
            
            # Solution visualization
            solution_path = f"{results_dir}/{run_prefix}_solution.png"
            visualize_solution(
                viz_solution,
                title=f"Case Study 2: Dragonfly Best Solution (Coverage={coverage_score}/{len(free_cells)}, Balance={balance_score:.4f}) - {timestamp}",
                save_path=solution_path
            )
            print(f"✅ Solution visualization saved: {solution_path}")
            
            # Convert history for convergence plot
            # Use combined_score if available (same formula as GA/ACO), otherwise use fitness
            if 'best_combined_score' in history and history['best_combined_score']:
                best_scores = history['best_combined_score']
            else:
                best_scores = history.get('best_fitness', [])
            
            convergence_history = {
                'iteration': history.get('iteration', []),
                'best_score': best_scores,  # Combined score (lower is better)
                'avg_score': history.get('avg_fitness', [])
            }
            
            # Plot convergence history (best score only)
            convergence_path = f"{results_dir}/{run_prefix}_convergence.png"
            plot_best_score_only(
                convergence_history,
                title=f"Case Study 2: Dragonfly Best Score Convergence - {timestamp}",
                save_path=convergence_path
            )
            print(f"✅ Convergence plot saved: {convergence_path}")
            
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
            traceback.print_exc()

    # Create GIF of Dragonfly evolution (if snapshots are available)
    try:
        from visualization import create_dragonfly_animation

        history_snapshots = history.get("snapshots", [])
        if history_snapshots:
            gif_path = f"{results_dir}/{run_prefix}_evolution.gif"
            create_dragonfly_animation(
                history_snapshots=history_snapshots,
                grid_width=grid_width,
                grid_height=grid_height,
                obstacles=obstacles,
                save_path=gif_path,
                fps=2
            )
            print(f"✅ GIF saved: {gif_path}")
        else:
            print("No snapshots found in history; GIF not created.")
    except Exception as e:
        print(f"⚠️  Could not create GIF animation: {e}")

    # Save history to a file
    try:
        out_path = f"{results_dir}/{run_prefix}_history.txt"
        with open(out_path, "w") as f:
            for it, bf, af in zip(history["iteration"], history["best_fitness"], history["avg_fitness"]):
                f.write(f"{it}\t{bf}\t{af}\n")
        print(f"📄 Saved history to: {out_path}")
    except Exception as e:
        print(f"⚠️  Could not save history file: {e}")
