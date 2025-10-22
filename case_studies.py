"""
Case Studies for Multi-Robot Coverage Path Planning
==================================================

This module defines different test cases of varying sizes and complexity
to validate the SA algorithm implementation.

Case Studies:
1. Small Grid (3x3) - 2 robots, 1 obstacle
2. Medium Grid (4x4) - 3 robots, 2 obstacles  
3. Large Grid (5x5) - 4 robots, 3 obstacles
4. Complex Grid (6x6) - 5 robots, 5 obstacles
5. Benchmark Cases - Known optimal solutions for comparison
"""

from problem_formulation import *
from sa_algorithm import *

def case_study_1_small_grid():
    """Case Study 1: Small 3x3 grid with 2 robots and 1 obstacle"""
    print("=" * 60)
    print("CASE STUDY 1: Small Grid (3x3)")
    print("=" * 60)
    
    grid_width = 3
    grid_height = 3
    num_robots = 2
    obstacles = [4]  # Middle cell
    
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(len(all_cells)) if i not in obstacles]
    
    print(f"Grid: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Total cells: {len(all_cells)}")
    print(f"Free cells: {len(free_cells)}")
    print(f"Obstacles: {obstacles}")
    
    # Run SA algorithm
    solution = simulated_annealing(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        initial_temp=1000, cooling_rate=0.95, max_iterations=50
    )
    
    print_sa_results(solution)
    return solution

def case_study_2_medium_grid():
    """Case Study 2: Medium 4x4 grid with 3 robots and 2 obstacles"""
    print("=" * 60)
    print("CASE STUDY 2: Medium Grid (4x4)")
    print("=" * 60)
    
    grid_width = 4
    grid_height = 4
    num_robots = 3
    obstacles = [5, 10]  # Two obstacles
    
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(len(all_cells)) if i not in obstacles]
    
    print(f"Grid: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Total cells: {len(all_cells)}")
    print(f"Free cells: {len(free_cells)}")
    print(f"Obstacles: {obstacles}")
    
    # Run SA algorithm
    solution = simulated_annealing(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        initial_temp=1000, cooling_rate=0.95, max_iterations=50
    )
    
    print_sa_results(solution)
    return solution

def case_study_3_large_grid():
    """Case Study 3: Large 5x5 grid with 4 robots and 3 obstacles"""
    print("=" * 60)
    print("CASE STUDY 3: Large Grid (5x5)")
    print("=" * 60)
    
    grid_width = 5
    grid_height = 5
    num_robots = 4
    obstacles = [6, 12, 18]  # Three obstacles
    
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(len(all_cells)) if i not in obstacles]
    
    print(f"Grid: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Total cells: {len(all_cells)}")
    print(f"Free cells: {len(free_cells)}")
    print(f"Obstacles: {obstacles}")
    
    # Run SA algorithm
    solution = simulated_annealing(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        initial_temp=1000, cooling_rate=0.95, max_iterations=50
    )
    
    print_sa_results(solution)
    return solution

def case_study_4_complex_grid():
    """Case Study 4: Complex 6x6 grid with 5 robots and 5 obstacles"""
    print("=" * 60)
    print("CASE STUDY 4: Complex Grid (6x6)")
    print("=" * 60)
    
    grid_width = 6
    grid_height = 6
    num_robots = 5
    obstacles = [7, 14, 21, 28, 35]  # Five obstacles
    
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(len(all_cells)) if i not in obstacles]
    
    print(f"Grid: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Total cells: {len(all_cells)}")
    print(f"Free cells: {len(free_cells)}")
    print(f"Obstacles: {obstacles}")
    
    # Run SA algorithm
    solution = simulated_annealing(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        initial_temp=1000, cooling_rate=0.95, max_iterations=50
    )
    
    print_sa_results(solution)
    return solution

def benchmark_case_optimal():
    """Benchmark Case: Known optimal solution for validation"""
    print("=" * 60)
    print("BENCHMARK CASE: Known Optimal Solution")
    print("=" * 60)
    
    grid_width = 3
    grid_height = 3
    num_robots = 2
    obstacles = []
    
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(len(all_cells)) if i not in obstacles]
    
    # Create optimal solution manually
    assignment = []
    for i in range(9):
        if i < 4:
            assignment.append([1, 0])  # First 4 cells to robot 0
        else:
            assignment.append([0, 1])  # Last 5 cells to robot 1
    
    # Optimal paths
    paths = [
        [0, 1, 2, 3],  # Robot 0: top row
        [4, 5, 6, 7, 8]  # Robot 1: bottom rows
    ]
    
    # Create solution
    solution = RobotCoverageSolution(
        assignment, paths, all_cells, free_cells, obstacles, grid_width, grid_height
    )
    solution.evaluate()
    
    print(f"Grid: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Obstacles: None")
    print(f"Expected: Perfect coverage and balance")
    
    print_sa_results(solution)
    return solution

def run_all_case_studies():
    """Run all case studies and return results"""
    print("MULTI-ROBOT COVERAGE PATH PLANNING - CASE STUDIES")
    print("=" * 80)
    
    results = {}
    
    # Run all case studies
    results['case_1'] = case_study_1_small_grid()
    results['case_2'] = case_study_2_medium_grid()
    results['case_3'] = case_study_3_large_grid()
    results['case_4'] = case_study_4_complex_grid()
    results['benchmark'] = benchmark_case_optimal()
    
    # Summary
    print("\n" + "=" * 80)
    print("CASE STUDIES SUMMARY")
    print("=" * 80)
    
    for case_name, solution in results.items():
        print(f"{case_name.upper()}:")
        print(f"  Coverage: {solution.fitness['coverage_score']}")
        print(f"  Balance: {solution.fitness['balance_score']:.3f}")
        print(f"  Combined: {solution.combined_score:.3f}")
        print(f"  Violations: {len(solution.fitness['problems'])}")
        print()
    
    return results

def performance_analysis(results):
    """Analyze performance across different case studies"""
    print("PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    case_names = list(results.keys())
    coverage_scores = [results[name].fitness['coverage_score'] for name in case_names]
    balance_scores = [results[name].fitness['balance_score'] for name in case_names]
    combined_scores = [results[name].combined_score for name in case_names]
    
    print(f"Average Coverage: {sum(coverage_scores)/len(coverage_scores):.2f}")
    print(f"Average Balance: {sum(balance_scores)/len(balance_scores):.3f}")
    print(f"Average Combined: {sum(combined_scores)/len(combined_scores):.3f}")
    
    # Find best solution
    best_case = min(case_names, key=lambda x: results[x].combined_score)
    print(f"Best Solution: {best_case} (Combined Score: {results[best_case].combined_score:.3f})")

if __name__ == "__main__":
    # Run all case studies
    results = run_all_case_studies()
    
    # Performance analysis
    performance_analysis(results)
