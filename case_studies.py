"""
Case Studies for Multi-Robot Coverage Path Planning
===================================================

This module contains different case studies to test and validate
the GA and SA algorithms on various problem configurations.
"""

from problem_formulation import *
from GA import genetic_algorithm
from sa_algorithm import simulated_annealing
from algorithm_comparison import compare_sa_vs_ga, generate_comparison_report
from visualization import visualize_solution, plot_convergence_history, save_all_figures
import os


def case_study_1_small_grid():
    """
    Case Study 1: Small Grid - Basic Functionality Test
    Grid: 4x4
    Robots: 2
    Obstacles: Few scattered obstacles
    Purpose: Validate basic algorithm functionality
    """
    print("\n" + "="*80)
    print("CASE STUDY 1: SMALL GRID (4x4, 2 Robots)")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 4, 4
    num_robots = 2
    obstacles = [5, 10]  # 2 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print(f"\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Total Cells: {grid_width * grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # GA Parameters
    ga_params = {
        'population_size': 30,
        'generations': 50,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'elitism_count': 2
    }
    
    # SA Parameters
    sa_params = {
        'initial_temp': 1000,
        'cooling_rate': 0.95,
        'max_iterations': 50
    }
    
    print(f"\nGA Parameters: {ga_params}")
    print(f"SA Parameters: {sa_params}")
    
    # Create output directory
    os.makedirs("results/case_study_1", exist_ok=True)
    
    # Run GA
    print("\n" + "-"*80)
    print("Running Genetic Algorithm...")
    print("-"*80)
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    
    # Run SA
    print("\n" + "-"*80)
    print("Running Simulated Annealing...")
    print("-"*80)
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
        verbose=True
    )
    
    # Generate comparison report
    print("\n" + "-"*80)
    print("Generating Comparison Report...")
    print("-"*80)
    comparison = generate_comparison_report(
        (sa_solution, sa_history),
        ga_results,
        case_study_name="Case Study 1: Small Grid"
    )
    
    # Save visualizations
    ga_solution = ga_results['best_solution']
    visualize_solution(ga_solution, 
                      title="Case Study 1: GA Solution",
                      save_path="results/case_study_1/ga_solution.png")
    visualize_solution(sa_solution,
                      title="Case Study 1: SA Solution", 
                      save_path="results/case_study_1/sa_solution.png")
    
    print("\n✅ Case Study 1 completed!")
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def case_study_2_medium_grid():
    """
    Case Study 2: Medium Grid - Moderate Complexity
    Grid: 6x6
    Robots: 3
    Obstacles: Moderate obstacle density
    Purpose: Test scalability and performance
    """
    print("\n" + "="*80)
    print("CASE STUDY 2: MEDIUM GRID (6x6, 3 Robots)")
    print("="*80)
    
    grid_width, grid_height = 6, 6
    num_robots = 3
    obstacles = [7, 8, 13, 14, 19, 20]  # 2x3 obstacle block
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print(f"\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # Algorithm parameters
    ga_params = {
        'population_size': 50,
        'generations': 100,
        'crossover_rate': 0.8,
        'mutation_rate': 0.12,
        'elitism_count': 3
    }
    
    sa_params = {
        'initial_temp': 1500,
        'cooling_rate': 0.95,
        'max_iterations': 100
    }
    
    print(f"\nGA Parameters: {ga_params}")
    print(f"SA Parameters: {sa_params}")
    
    os.makedirs("results/case_study_2", exist_ok=True)
    
    # Run algorithms
    print("\n[1/2] Running GA...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    
    print("\n[2/2] Running SA...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
        verbose=True
    )
    
    # Comparison
    comparison = generate_comparison_report(
        (sa_solution, sa_history),
        ga_results,
        case_study_name="Case Study 2: Medium Grid"
    )
    
    # Visualizations
    ga_solution = ga_results['best_solution']
    visualize_solution(ga_solution,
                      title="Case Study 2: GA Solution",
                      save_path="results/case_study_2/ga_solution.png")
    visualize_solution(sa_solution,
                      title="Case Study 2: SA Solution",
                      save_path="results/case_study_2/sa_solution.png")
    
    print("\n✅ Case Study 2 completed!")
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def case_study_3_large_grid():
    """
    Case Study 3: Large Grid - High Complexity
    Grid: 10x10
    Robots: 5
    Obstacles: Complex obstacle pattern
    Purpose: Test algorithm performance on large-scale problems
    """
    print("\n" + "="*80)
    print("CASE STUDY 3: LARGE GRID (10x10, 5 Robots)")
    print("="*80)
    
    grid_width, grid_height = 10, 10
    num_robots = 5
    # Create L-shaped obstacle pattern
    obstacles = list(range(30, 40)) + list(range(40, 70, 10))  # Horizontal + vertical wall
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print(f"\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # Algorithm parameters - larger for complex problem
    ga_params = {
        'population_size': 100,
        'generations': 200,
        'crossover_rate': 0.8,
        'mutation_rate': 0.15,
        'elitism_count': 5
    }
    
    sa_params = {
        'initial_temp': 2000,
        'cooling_rate': 0.97,
        'max_iterations': 200
    }
    
    print(f"\nGA Parameters: {ga_params}")
    print(f"SA Parameters: {sa_params}")
    
    os.makedirs("results/case_study_3", exist_ok=True)
    
    # Run algorithms
    print("\n[1/2] Running GA...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    
    print("\n[2/2] Running SA...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
        verbose=True
    )
    
    # Comparison
    comparison = generate_comparison_report(
        (sa_solution, sa_history),
        ga_results,
        case_study_name="Case Study 3: Large Grid"
    )
    
    # Visualizations
    ga_solution = ga_results['best_solution']
    visualize_solution(ga_solution,
                      title="Case Study 3: GA Solution",
                      save_path="results/case_study_3/ga_solution.png")
    visualize_solution(sa_solution,
                      title="Case Study 3: SA Solution",
                      save_path="results/case_study_3/sa_solution.png")
    
    print("\n✅ Case Study 3 completed!")
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def case_study_4_many_robots():
    """
    Case Study 4: Many Robots - Coordination Challenge
    Grid: 8x8
    Robots: 6
    Obstacles: Moderate
    Purpose: Test workload balancing with many robots
    """
    print("\n" + "="*80)
    print("CASE STUDY 4: MANY ROBOTS (8x8, 6 Robots)")
    print("="*80)
    
    grid_width, grid_height = 8, 8
    num_robots = 6
    obstacles = [18, 19, 20, 21, 26, 27, 28, 29]  # 2x4 central obstacle
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print(f"\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # Algorithm parameters
    ga_params = {
        'population_size': 80,
        'generations': 150,
        'crossover_rate': 0.85,
        'mutation_rate': 0.15,
        'elitism_count': 4
    }
    
    sa_params = {
        'initial_temp': 1800,
        'cooling_rate': 0.96,
        'max_iterations': 150
    }
    
    print(f"\nGA Parameters: {ga_params}")
    print(f"SA Parameters: {sa_params}")
    
    os.makedirs("results/case_study_4", exist_ok=True)
    
    # Run algorithms
    print("\n[1/2] Running GA...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    
    print("\n[2/2] Running SA...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
        verbose=True
    )
    
    # Comparison
    comparison = generate_comparison_report(
        (sa_solution, sa_history),
        ga_results,
        case_study_name="Case Study 4: Many Robots"
    )
    
    # Visualizations
    ga_solution = ga_results['best_solution']
    visualize_solution(ga_solution,
                      title="Case Study 4: GA Solution",
                      save_path="results/case_study_4/ga_solution.png")
    visualize_solution(sa_solution,
                      title="Case Study 4: SA Solution",
                      save_path="results/case_study_4/sa_solution.png")
    
    print("\n✅ Case Study 4 completed!")
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def run_all_case_studies():
    """
    Run all case studies sequentially
    """
    print("\n" + "="*80)
    print("RUNNING ALL CASE STUDIES")
    print("="*80)
    
    results = {}
    
    # Run each case study
    try:
        results['case_1'] = case_study_1_small_grid()
    except Exception as e:
        print(f"❌ Case Study 1 failed: {str(e)}")
    
    try:
        results['case_2'] = case_study_2_medium_grid()
    except Exception as e:
        print(f"❌ Case Study 2 failed: {str(e)}")
    
    try:
        results['case_3'] = case_study_3_large_grid()
    except Exception as e:
        print(f"❌ Case Study 3 failed: {str(e)}")
    
    try:
        results['case_4'] = case_study_4_many_robots()
    except Exception as e:
        print(f"❌ Case Study 4 failed: {str(e)}")
    
    # Generate overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY - ALL CASE STUDIES")
    print("="*80)
    
    for case_name, case_results in results.items():
        if 'comparison' in case_results:
            comp = case_results['comparison']
            print(f"\n{case_name.upper()}:")
            print(f"  Winner: {comp['overall_winner']}")
            print(f"  SA Score: {comp['sa_score']:.4f}")
            print(f"  GA Score: {comp['ga_score']:.4f}")
    
    return results


def run_single_case_study(case_number):
    """
    Run a specific case study
    
    Args:
        case_number: 1, 2, 3, or 4
    """
    case_functions = {
        1: case_study_1_small_grid,
        2: case_study_2_medium_grid,
        3: case_study_3_large_grid,
        4: case_study_4_many_robots
    }
    
    if case_number in case_functions:
        return case_functions[case_number]()
    else:
        print(f"Invalid case study number: {case_number}")
        print("Available case studies: 1, 2, 3, 4")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific case study
        case_num = int(sys.argv[1])
        run_single_case_study(case_num)
    else:
        # Run all case studies
        run_all_case_studies()
