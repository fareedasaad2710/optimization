import copy
import math
import random
import sys
import os
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

    #assignemnt --> 2d list ,where assign[cell_index][robot_id] = 0 or 1
		#paths:dict{robot_id: [cell1, cell2, ...]}
    def __init__(self, assignment, paths, all_cells, free_cells, obstacles, grid_width, grid_height):
        self.assignment = copy.deepcopy(assignment)
        self.paths = copy.deepcopy(paths)
        self.all_cells = all_cells
        self.free_cells = free_cells
        self.obstacles = obstacles
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fitness = None
        self.combined_score = None
    
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
        if self.fitness is not None:
            new_solution.fitness = copy.deepcopy(self.fitness)
        if self.combined_score is not None:
            new_solution.combined_score = self.combined_score
        return new_solution
#check assignemnet and path are same    
    def sync_assignment_with_paths(self):
        """Ensure assignment matrix matches the paths"""
        for cell_idx in range(len(self.assignment)):
            for robot_id in range(len(self.assignment[cell_idx])):
                self.assignment[cell_idx][robot_id] = 0
        
        for robot_id, path in self.paths.items():
            for cell_idx in path:
                if cell_idx < len(self.assignment):
                    self.assignment[cell_idx][robot_id] = 1

#create initial empty solution
def create_empty_solution(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots):

    total_cells = len(all_cells)
		#set all assignment with zeros
    assignment = [[0 for _ in range(num_robots)] for _ in range(total_cells)]
		#set all paths with empty list
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

# check first constarinbts
#return is valid(sol),violations list
def check_path_continuity(path, all_cells, grid_width, grid_height):

    violations = []
    #check if path empty,has one single cell 
    if len(path) <= 1:
        return True, violations
 #iterate through cells   
    for i in range(len(path) - 1):
        current_cell_idx = path[i]
        next_cell_idx = path[i + 1]
        #check if cell index is valid >0,within no.of cells
        if current_cell_idx < 0 or current_cell_idx >= len(all_cells):
            violations.append(f"Invalid cell index {current_cell_idx} at position {i}")
            continue
        #check if next cell index is valid >0,within no.of cells
        if next_cell_idx < 0 or next_cell_idx >= len(all_cells):
            violations.append(f"Invalid cell index {next_cell_idx} at position {i+1}")
            continue
        #get current cell ,next one
        current_cell = all_cells[current_cell_idx]
        next_cell = all_cells[next_cell_idx]
        #find neighbors of current cell,return four neighb(up,down,left,right)
        neighbors = find_neighbors(current_cell_idx, all_cells, grid_width, grid_height)
        #get coordinates of current cell
        current_coord = all_cells[current_cell_idx]
#check is cell tuplr ex: (0, 0), (5, 3) or object Cell(x=5, y=3)
        if isinstance(current_coord, tuple):
            current_x, current_y = current_coord
        else:
            current_x, current_y = current_coord.x, current_coord.y
        
        next_coord = all_cells[next_cell_idx]
        if isinstance(next_coord, tuple):
            next_x, next_y = next_coord
        else:
            next_x, next_y = next_coord.x, next_coord.y
        #convert list of tuples coming from find_neighbours,
				#to sets of tuples {(0, 0), (1, 0), (0, 1), (1, 1)}
        neighbor_coords = set(neighbors)
        #check if current,next not neighbours
        if (next_x, next_y) not in neighbor_coords:
            violations.append(
                f"Path jump: cell {current_cell_idx} ({current_x},{current_y}) to {next_cell_idx} ({next_x},{next_y}) are not adjacent"
            )
    
    return len(violations) == 0, violations

#check all cellss in path inside grid,return is valid,violations
def check_boundary_constraint(path, all_cells, grid_width, grid_height):
    violations = []
		#loop each cell in path
    for i, cell_idx in enumerate(path):
			#check if cell index (cell_index_ is neg or too largw)
        if cell_idx < 0 or cell_idx >= len(all_cells):
            violations.append(f"Cell index {cell_idx} out of bounds at position {i}")
            continue
        #get cell data 
        cell = all_cells[cell_idx]
				#get cell coordinates
        x = cell.x if hasattr(cell, 'x') else cell[0]
        y = cell.y if hasattr(cell, 'y') else cell[1]
        #check i cell coord is in grid or not
        if not (0 <= x < grid_width and 0 <= y < grid_height):
            violations.append(f"Cell {cell_idx} at ({x}, {y}) is outside grid boundaries")
    
    return len(violations) == 0, violations
##check that path has no obstacles (Cells) return is valid,violat.
def check_obstacle_avoidance(path, obstacles):
    violations = []
    for i, cell_idx in enumerate(path):
			#check if curr cell is an obstacle
        if cell_idx in obstacles:
            violations.append(f"Robot enters obstacle at cell {cell_idx} (position {i} in path)")   
    return len(violations) == 0, violations

#takes the sol. check all constraints,return is valid,violations
def is_solution_feasible(solution: ACOSolution) -> Tuple[bool, List[str]]:

    all_violations = []
    #check is sol.paths is dict
    if not isinstance(solution.paths, dict):
        return False, [f"Paths must be a dictionary, got {type(solution.paths)}"]
    #keep track of no of robots
    num_robots = len(solution.paths)
    #loop through eac robot
    for robot_id in range(num_robots):
			#get path of this robot
        path = solution.paths.get(robot_id, [])
        if not isinstance(path, list):
            all_violations.append(f"Robot {robot_id}: path must be a list, got {type(path)}")
            continue
        #check boundary
        boundary_valid, boundary_violations = check_boundary_constraint(
            path, solution.all_cells, solution.grid_width, solution.grid_height
        )
        if not boundary_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in boundary_violations])
        #check obstacles
        obstacle_valid, obstacle_violations = check_obstacle_avoidance(path, solution.obstacles)
        if not obstacle_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in obstacle_violations])
        #path valid
        continuity_valid, continuity_violations = check_path_continuity(
            path, solution.all_cells, solution.grid_width, solution.grid_height
        )
        if not continuity_valid:
            all_violations.extend([f"Robot {robot_id}: {v}" for v in continuity_violations])
    
    is_feasible = len(all_violations) == 0
    return is_feasible, all_violations


def convert_cells_to_objects(all_cells):
    """Convert tuple cells to Cell objects if needed"""
    if len(all_cells) == 0:
        return []
    
    if hasattr(all_cells[0], 'x') and hasattr(all_cells[0], 'y'):
        return all_cells
    
    class Cell:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    result = []
    for cell in all_cells:
        if isinstance(cell, tuple):
            result.append(Cell(cell[0], cell[1]))
        else:
            result.append(cell)
    return result

#num ants = 2
#initial_pheromone: Initial pheromone level (1.0)
#p=0.5
#alpgha:1,beta=1
#it no=50
class AntColonyOptimization:
#constructor
    def __init__(self, all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                 num_ants=2, initial_pheromone=1.0, rho=0.5, alpha=1.0, beta=1.0, iterations=50):
        self.all_cells = all_cells
        self.free_cells = free_cells
        self.obstacles = obstacles
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_robots = num_robots
        self.num_ants = num_ants
        self.initial_pheromone = initial_pheromone
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        
        # Assign starting position to each robot
        # Each robot starts at a different free cell
        self.robot_starts = [free_cells[i % len(free_cells)] for i in range(num_robots)]
       
        num_cells = len(all_cells)
        num_free = len(free_cells)
        #initialize phermone dict
        self.pheromone = {}
        for robot_id in range(num_robots):
            self.pheromone[robot_id] = {}
            for cell_idx in free_cells:
                self.pheromone[robot_id][cell_idx] = initial_pheromone
        #create dict for robot heurist 
        self.heuristic = {}
        #calculate desirablitity from st. point of robot to every other free cell
        for robot_id in range(num_robots):
            self.heuristic[robot_id] = {}
            start_cell = self.robot_starts[robot_id]
            for cell_idx in free_cells:
                # Calculate heuristic based on distance from robot start
                self.heuristic[robot_id][cell_idx] = self._calculate_distance_heuristic(start_cell, cell_idx)
        self.best_solution = None
        self.best_score = float('inf')
   #Convert (x, y) coordinate to cell index 
    def _coordinate_to_index(self, coord):
        if isinstance(coord, (int,)):
            return coord
        x, y = coord[0], coord[1]
        for idx, cell in enumerate(self.all_cells):
            cell_coord = self._get_cell_coords(idx)
            if cell_coord == (x, y):
                return idx
        return None
    
    def _get_cell_coords(self, cell_idx):
        """Get (x, y) coordinates from cell index"""
        cell = self.all_cells[cell_idx]
        if isinstance(cell, tuple):
            return cell
        else:
            return (cell.x, cell.y)
    
    def _calculate_distance_heuristic(self, from_cell: int, to_cell: int) -> float:
        """
        Calculate heuristic desirability based on distance between two cells
        Returns: 1.0 / (1.0 + distance) - closer cells have higher heuristic
        """
        from_coord = self._get_cell_coords(from_cell)
        to_coord = self._get_cell_coords(to_cell)
        dist = distance_between_points(from_coord, to_coord)
        if dist == 0:
            dist = 0.1  # Avoid division by zero
        return 1.0 / (1.0 + dist)  # Closer = higher heuristic
    #create first empty sol.
    def get_empty_solution(self):
        return create_empty_solution(
            self.all_cells, self.free_cells, self.obstacles,
            self.grid_width, self.grid_height, self.num_robots
        )
    #check sol. feasibile
    def check_feasibility(self, solution: ACOSolution) -> Tuple[bool, List[str]]:
        return is_solution_feasible(solution)
    #construct sol. for 1 ant
    def construct_solution(self, ant_id: int) -> Optional[ACOSolution]:
        solution = self.get_empty_solution()
        covered_cells = set()
        
        # Initialize each robot at its starting position
        for robot_id in range(self.num_robots):
            start_cell = self.robot_starts[robot_id]
            solution.paths[robot_id] = [start_cell]
            covered_cells.add(start_cell)
        
        # Build paths iteratively - each robot takes turns adding cells
        max_iterations = len(self.free_cells) * 2  # Prevent infinite loops
        iteration = 0
        
        while len(covered_cells) < len(self.free_cells) and iteration < max_iterations:
            iteration += 1
            progress_made = False
            
            for robot_id in range(self.num_robots):
                if len(covered_cells) >= len(self.free_cells):
                    break
                #For each robot, gets the current cell (last in its path)
                current_cell = solution.paths[robot_id][-1]
                
                # Get neighboring cells that are free and not obstacles
                neighbor_indices = find_neighbors(current_cell, self.all_cells, 
                                                 self.grid_width, self.grid_height)
                
                # Convert neighbor coordinates to cell indices
                available_indices = []
                for neighbor_coord in neighbor_indices:
                    neighbor_idx = self._coordinate_to_index(neighbor_coord)
										# filters to free cells.
                    if neighbor_idx is not None and neighbor_idx in self.free_cells:
                        available_indices.append(neighbor_idx)
                
                # Prioritize uncovered cells
                uncovered = [idx for idx in available_indices if idx not in covered_cells]
                
                if uncovered:
                    next_cell = self.select_next_cell(robot_id, current_cell, 
                                                     uncovered, covered_cells)
                elif available_indices:
                    # If all neighbors are covered, allow revisiting (for path continuity)
                    next_cell = self.select_next_cell(robot_id, current_cell, 
                                                     available_indices, covered_cells)
                else:
                    continue  # No available moves for this robot
                #append the next cell(selected cell to path),marks as covere
                if next_cell is not None:
                    solution.paths[robot_id].append(next_cell)
                    covered_cells.add(next_cell)
                    progress_made = True
            if not progress_made:
                break  # No robot can make progress      
        # Sync assignment matrix with paths
        solution.sync_assignment_with_paths()
        
        # Check feasibility
        is_feasible, violations = self.check_feasibility(solution)        
        if not is_feasible:
            return None
        
        return solution
				#calculate probabilities 
    def select_next_cell(self, robot_id: int, current_cell: int, 
                        candidate_cells: List[int], covered_cells: Set[int]) -> Optional[int]:

        if not candidate_cells:
            return None     
        # Calculate probabilities for each candidate
        probabilities = []
        for cell_idx in candidate_cells:
            # Get pheromone value
            tau = self.pheromone[robot_id].get(cell_idx, self.initial_pheromone) 
            # Calculate heuristic (distance-based, prefer uncovered)
            if cell_idx in covered_cells:
                eta = 0.01  # Low desirability for covered cells
            else:
                # Use distance from current position to next cell
                eta = self._calculate_distance_heuristic(current_cell, cell_idx)
            
            # ACO probability: τ^α * η^β
            prob = (tau ** self.alpha) * (eta ** self.beta)
            probabilities.append(prob)
        
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            # If all probabilities are 0, choose randomly
            return random.choice(candidate_cells)
        
        probabilities = [p / total for p in probabilities]
        
        # Select based on probabilities
        return random.choices(candidate_cells, weights=probabilities, k=1)[0]
    
    def update_pheromone(self, ant_solutions: List[ACOSolution]):

        for robot_id in range(self.num_robots):
            for cell_idx in self.free_cells:
                self.pheromone[robot_id][cell_idx] *= (1.0 - self.rho)
    
    def run(self, verbose=False):
        """
        Main ACO loop
        This is a placeholder - will be implemented in next steps
        """
        print(f"\n{'='*70}")
        print(f"🐜 ANT COLONY OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Parameters:")
        print(f"  • Number of ants: {self.num_ants}")
        print(f"  • Initial pheromone: {self.initial_pheromone}")
        print(f"  • Pheromone decay (ρ): {self.rho}")
        print(f"  • Alpha (α): {self.alpha}")
        print(f"  • Beta (β): {self.beta}")
        print(f"  • Iterations: {self.iterations}")
        print(f"  • Grid: {self.grid_width}x{self.grid_height}")
        print(f"  • Robots: {self.num_robots}")
        print(f"  • Free cells: {len(self.free_cells)}")
        print(f"{'='*70}\n")
        
        for iteration in range(self.iterations):
            if verbose and (iteration % 10 == 0 or iteration == self.iterations - 1):
                print(f"Iteration {iteration + 1}/{self.iterations}")
        
        return self.best_solution


if __name__ == "__main__":
    grid_width = 5
    grid_height = 5
    num_robots = 2
    obstacles = [12]
    
    total_cells = grid_width * grid_height
    free_cells = [i for i in range(total_cells) if i not in obstacles]
    all_cells = create_grid_cells(grid_width, grid_height)
    
    print("="*70)
    print("ANT COLONY OPTIMIZATION - PROBLEM REPRESENTATION TEST")
    print("="*70)
    
    aco = AntColonyOptimization(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        num_ants=2, initial_pheromone=1.0, rho=0.5, alpha=1.0, beta=1.0, iterations=50
    )
    
    print("\n1. Testing Empty Solution:")
    print("-" * 70)
    empty_solution = aco.get_empty_solution()
    print(f"  Assignment: {len(empty_solution.assignment)} cells x {len(empty_solution.assignment[0])} robots")
    print(f"  Paths: {empty_solution.paths}")
    
    is_feasible, violations = aco.check_feasibility(empty_solution)
    print(f"  Feasibility: {is_feasible}")
    if violations:
        print(f"  Violations: {violations}")
    else:
        print(f"  ✓ No violations (empty solution is always feasible)")
    
    print("\n2. Testing Invalid Solution (Path Jump):")
    print("-" * 70)
    invalid_solution = empty_solution.copy()
    invalid_solution.paths[0] = [0, 10]
    is_feasible, violations = aco.check_feasibility(invalid_solution)
    print(f"  Path: {invalid_solution.paths[0]}")
    print(f"  Feasibility: {is_feasible}")
    if violations:
        print(f"  ✗ Violations found:")
        for v in violations:
            print(f"    - {v}")
    
    print("\n3. Testing Invalid Solution (Obstacle):")
    print("-" * 70)
    invalid_solution2 = empty_solution.copy()
    invalid_solution2.paths[0] = [11, 12]
    is_feasible, violations = aco.check_feasibility(invalid_solution2)
    print(f"  Path: {invalid_solution2.paths[0]} (cell 12 is an obstacle)")
    print(f"  Feasibility: {is_feasible}")
    if violations:
        print(f"  ✗ Violations found:")
        for v in violations:
            print(f"    - {v}")
    
    print("\n4. Testing Valid Solution (Adjacent Path):")
    print("-" * 70)
    valid_solution = empty_solution.copy()
    valid_solution.paths[0] = [0, 1, 2]
    valid_solution.sync_assignment_with_paths()
    is_feasible, violations = aco.check_feasibility(valid_solution)
    print(f"  Path: {valid_solution.paths[0]}")
    print(f"  Feasibility: {is_feasible}")
    if violations:
        print(f"  ✗ Violations found:")
        for v in violations:
            print(f"    - {v}")
    else:
        print(f"  ✓ No violations (valid adjacent path)")
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print("✓ Problem representation implemented (ACOSolution class)")
    print("✓ Empty solution creation works")
    print("✓ Feasibility checking implemented:")
    print("  - Path continuity constraint")
    print("  - Boundary constraint")
    print("  - Obstacle avoidance constraint")
    print("✓ Solution rejection mechanism ready (returns False for invalid solutions)")
    print("\nNext steps:")
    print("  - Implement ant solution construction")
    print("  - Implement pheromone update")
    print("  - Implement main ACO loop")
    print("="*70)

