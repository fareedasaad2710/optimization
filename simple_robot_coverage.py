"""

DECISION VARIABLES:
- assignment[i][r]: Binary variable (0 or 1) - assigns cell i to robot r
- paths[r]: List of cell indices representing robot r's path sequence

OBJECTIVE FUNCTIONS:
- F1: Maximize coverage (cover as many cells as possible)
- F2: Minimize workload variance (balance robot workloads)

CONSTRAINTS:
- Path continuity: Robots can only move to adjacent cells (4-neighbor connectivity)
- Boundary constraint: Robots must stay within the grid
- Obstacle avoidance: Robots cannot enter obstacle cells

The problem: Multiple robots need to cover an area while:
1. Covering as many cells as possible
2. Having balanced workloads
3. Following valid paths (no jumping, no obstacles)
"""

import math
#eucliden distance
def distance_between_points(point1, point2):

    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

#1-avoid boundaries,move bewteen adjacent cells
def find_neighbors(cell_index, all_cells, grid_width, grid_height):

		#gets the (x, y) coordinates for the cell at cell_index
    x, y = all_cells[cell_index]
		#createw empty array for neibours
    neighbors = []
    
    #  right 
    if x + 1 < grid_width:
        neighbors.append((x+1, y))
    #  left   
    if x - 1 >= 0:
        neighbors.append((x-1, y))
    #  down 
    if y + 1 < grid_height:
        neighbors.append((x, y+1))
    #  up 
    if y - 1 >= 0:
        neighbors.append((x, y-1))
    
    return neighbors

#count each robot covered kam cell
def count_covered_cells(assignment, notCovered_cells):

    covered = 0
    for cell in notCovered_cells:

        for robot in range(len(assignment[0])):
            if assignment[cell][robot] == 1:
                covered += 1
								# no need to search other robots if cell is already occuoied
                break  
    return covered

#calculate kol robot meshy ad eh
def calculate_robot_distances(paths, all_cells):

    distances = []
    for path in paths:
        total_distance = 0
        for i in range(len(path) - 1):
					#calculate distance between lo; cell el robot ra7 menha li cell el b3dha
            current_cell = all_cells[path[i]]
            next_cell = all_cells[path[i + 1]]
            total_distance += distance_between_points(current_cell, next_cell)
        distances.append(total_distance)
    return distances

#varience 2olail = balanced
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!refence from paper no.??
def calculate_workload_variance(distances):

    if len(distances) == 0:
        return 0  
    # Calculate average distance
    average = sum(distances) / len(distances)   
    # Calculate variance
    variance = sum((d - average) ** 2 for d in distances) / len(distances)
    return variance

#checlk kol el constraints
def check_path_validity(paths, all_cells, obstacles, grid_width, grid_height):

    problems = []
    
    for robot_id, path in enumerate(paths):
        for i, cell_index in enumerate(path):
            # Check if cell is within grid bounds
            if cell_index < 0 or cell_index >= len(all_cells):
                problems.append(f"Robot {robot_id} goes outside grid at step {i}")
                continue
                
            # Check if cell is an obstacle
            if cell_index in obstacles:
                problems.append(f"Robot {robot_id} hits obstacle at cell {cell_index}")
                continue
            
            # Check if next move is valid (only to neighbors)
            if i < len(path) - 1:
                next_cell_index = path[i + 1]
                current_cell = all_cells[cell_index]
                next_cell = all_cells[next_cell_index]
                
                # Check if next cell is a neighbor
                neighbors = find_neighbors(cell_index, all_cells, grid_width, grid_height)
                if next_cell not in neighbors:
                    problems.append(f"Robot {robot_id} jumps from {current_cell} to {next_cell}")
    
    return problems
#mbd2yn dih el evaluation bt3na mesh 3rfeen b3d keda eh
def evaluate_robot_solution(all_cells, notCovered_cells, obstacles, assignment, paths, grid_width, grid_height):

    covered_count = count_covered_cells(assignment, notCovered_cells)
    coverage_score = covered_count  # Higher is better
    # 2. Calculate workload balance
    robot_distances = calculate_robot_distances(paths, all_cells)
    balance_score = calculate_workload_variance(robot_distances)  # Lower is better 
    # 3. Check for problems

    problems = check_path_validity(paths, all_cells, obstacles, grid_width, grid_height)
    
    return {
        'coverage_score': coverage_score,
        'balance_score': balance_score,
        'robot_distances': robot_distances,
        'problems': problems
    }

def print_results(results):
    print(f"Coverage Score: {results['coverage_score']} cells covered")
    print(f"Balance Score: {results['balance_score']:.2f} (lower = more balanced)")
    print(f"Robot Distances: {results['robot_distances']}")
    if results['problems']:
        print("\nPROBLEMS FOUND:")
        for problem in results['problems']:
            print(f"  - {problem}")
    else:
        print("\n✓ No problems found! Solution is valid.")

def divide_cells_equally(total_cells, num_robots):
    cells_per_robot = total_cells // num_robots
    remainder = total_cells % num_robots
    
    assignments = []
    start_idx = 0
    
    for robot in range(num_robots):
        # Some robots get one extra cell if there's a remainder
        robot_cell_count = cells_per_robot + (1 if robot < remainder else 0)
        end_idx = start_idx + robot_cell_count
        
        # Create assignment for this robot
        robot_assignment = [0] * num_robots
        robot_assignment[robot] = 1
        
        # Assign cells to this robot
        for cell_idx in range(start_idx, end_idx):
            if cell_idx < total_cells:
                assignments.append(robot_assignment.copy())
        
        start_idx = end_idx
    
    return assignments

def generate_simple_path(cell_indices):
    if not cell_indices:
        return []
    return cell_indices

def print_grid_visualization(all_cells, obstacles, assignment, grid_width, grid_height):
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

# Example usage
if __name__ == "__main__":
    # Configuration - Change these values to test different scenarios
    grid_width = 3
    grid_height = 3
    num_robots = 4  # Change this to test with different numbers of robots
    
    print(f"Setting up a {grid_width}x{grid_height} grid with {num_robots} robots...")
    
    total_cells = grid_width * grid_height
    print(f"Total cells: {total_cells}")

    all_cells = []
    for y in range(grid_height):
        for x in range(grid_width):
            all_cells.append((x, y))
    print(f"Grid cells: {all_cells}")
    
    # Define obstacles - cells that robots cannot enter
    obstacles = [4]  # Cell 4 (middle center) is an obstacle
    free_cells = [i for i in range(total_cells) if i not in obstacles]
    
    print(f"Obstacles: {obstacles}")
    print(f"Free cells: {free_cells}")
    
    # Automatically divide FREE cells equally among robots
    assignment = divide_cells_equally(len(free_cells), num_robots)
    
    # Create full assignment matrix (including obstacles)
    full_assignment = []
    obstacle_assignment = [0] * num_robots  # Obstacles assigned to no robot
    
    free_cell_idx = 0
    for cell_idx in range(total_cells):
        if cell_idx in obstacles:
            full_assignment.append(obstacle_assignment.copy())
        else:
            full_assignment.append(assignment[free_cell_idx])
            free_cell_idx += 1
    
    assignment = full_assignment
    
    print(f"\nAssignment (showing which robot covers each cell):")
    for i, row in enumerate(assignment):
        robot_id = row.index(1) if 1 in row else -1
        print(f"Cell {i}: Robot {robot_id}")
    
    # Print visual grid
    print_grid_visualization(all_cells, obstacles, assignment, grid_width, grid_height)
    
    # Generate simple paths for each robot
    robot_paths = []
    for robot_id in range(num_robots):
        # Find cells assigned to this robot
        robot_cells = []
        for cell_idx, assignment_row in enumerate(assignment):
            if assignment_row[robot_id] == 1:
                robot_cells.append(cell_idx)
        
        # Generate a simple path for this robot
        robot_path = generate_simple_path(robot_cells)
        robot_paths.append(robot_path)
        print(f"Robot {robot_id} path: {robot_path}")
    
    # Evaluate the solution
    results = evaluate_robot_solution(
        all_cells, free_cells, obstacles, assignment, robot_paths, 
        grid_width, grid_height
    )
    
    # Print results
    print_results(results)
