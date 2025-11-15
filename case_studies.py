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
import traceback


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
    
    print("\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Total Cells: {grid_width * grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # Algorithm parameters
    ga_params = {
        'population_size': 30,
        'generations': 50,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'elitism_count': 2
    }
    
    sa_params = {
        'initial_temp': 1000,
        'cooling_rate': 0.95,
        'max_iterations': 50
    }
    
    print(f"\nGA Parameters: {ga_params}")
    print(f"SA Parameters: {sa_params}")
    
    # Create results directory
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
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        # Visualize GA solution
        visualize_solution(
            ga_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 1: GA Best Solution",
            save_path="results/case_study_1/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        # Visualize SA solution
        visualize_solution(
            sa_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 1: SA Best Solution",
            save_path="results/case_study_1/sa_solution.png"
        )
        print("✅ SA solution visualization saved")
        
        # Plot convergence histories
        plot_convergence_history(
            ga_results['convergence_history'],
            title="Case Study 1: GA Convergence",
            save_path="results/case_study_1/ga_convergence.png"
        )
        print("✅ GA convergence plot saved")
        
        plot_convergence_history(
            sa_history,
            title="Case Study 1: SA Convergence",
            save_path="results/case_study_1/sa_convergence.png"
        )
        print("✅ SA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        traceback.print_exc()
    
    # Generate comparison report
    print("\n" + "-"*80)
    print("Generating Comparison Report...")
    print("-"*80)
    
    try:
        comparison = generate_comparison_report(
            sa_solution, sa_history,
            ga_results['best_solution'], ga_results['convergence_history'],
            grid_width, grid_height,
            case_study_name="Case Study 1: Small Grid"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 1 completed!")
    print(f"Results saved to: results/case_study_1/")
    
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def case_study_2_medium_grid():
    """
    Case Study 2: Medium Grid
    Grid: 6x6
    Robots: 3
    Obstacles: Moderate number of obstacles
    Purpose: Test scalability and load balancing
    """
    print("\n" + "="*80)
    print("CASE STUDY 2: MEDIUM GRID (6x6, 3 Robots)")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 6, 6
    num_robots = 3
    obstacles = [1, 7, 13, 19, 25, 31]  # 6 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
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
    
    # Run GA
    print("\n[1/2] Running GA...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    
    # Run SA
    print("\n[2/2] Running SA...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
        verbose=True
    )
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        visualize_solution(
            ga_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 2: GA Best Solution",
            save_path="results/case_study_2/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        visualize_solution(
            sa_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 2: SA Best Solution",
            save_path="results/case_study_2/sa_solution.png"
        )
        print("✅ SA solution visualization saved")
        
        plot_convergence_history(
            ga_results['convergence_history'],
            title="Case Study 2: GA Convergence",
            save_path="results/case_study_2/ga_convergence.png"
        )
        print("✅ GA convergence plot saved")
        
        plot_convergence_history(
            sa_history,
            title="Case Study 2: SA Convergence",
            save_path="results/case_study_2/sa_convergence.png"
        )
        print("✅ SA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        traceback.print_exc()
    
    # Generate comparison report
    try:
        comparison = generate_comparison_report(
            sa_solution, sa_history,
            ga_results['best_solution'], ga_results['convergence_history'],
            grid_width, grid_height,
            case_study_name="Case Study 2: Medium Grid"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 2 completed!")
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def case_study_3_large_grid():
    """
    Case Study 3: Large Grid
    Grid: 10x10
    Robots: 5
    Obstacles: Complex environment
    Purpose: Test algorithm performance on larger problems
    """
    print("\n" + "="*80)
    print("CASE STUDY 3: LARGE GRID (10x10, 5 Robots)")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 10, 10
    num_robots = 5
    obstacles = [11, 12, 13, 22, 32, 42, 52, 62, 67, 68, 69, 77, 88]  # 13 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # Algorithm parameters
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
    
    # Run GA
    print("\n[1/2] Running GA...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    
    # Run SA
    print("\n[2/2] Running SA...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
        verbose=True
    )
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        visualize_solution(
            ga_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 3: GA Best Solution",
            save_path="results/case_study_3/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        visualize_solution(
            sa_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 3: SA Best Solution",
            save_path="results/case_study_3/sa_solution.png"
        )
        print("✅ SA solution visualization saved")
        
        plot_convergence_history(
            ga_results['convergence_history'],
            title="Case Study 3: GA Convergence",
            save_path="results/case_study_3/ga_convergence.png"
        )
        print("✅ GA convergence plot saved")
        
        plot_convergence_history(
            sa_history,
            title="Case Study 3: SA Convergence",
            save_path="results/case_study_3/sa_convergence.png"
        )
        print("✅ SA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        traceback.print_exc()
    
    # Generate comparison report
    try:
        comparison = generate_comparison_report(
            sa_solution, sa_history,
            ga_results['best_solution'], ga_results['convergence_history'],
            grid_width, grid_height,
            case_study_name="Case Study 3: Large Grid"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 3 completed!")
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def case_study_4_many_robots():
    """
    Case Study 4: Many Robots
    Grid: 8x8
    Robots: 6
    Obstacles: Moderate obstacles
    Purpose: Test coordination with many robots
    """
    print("\n" + "="*80)
    print("CASE STUDY 4: MANY ROBOTS (8x8, 6 Robots)")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 8, 8
    num_robots = 6
    obstacles = [9, 10, 17, 18, 26, 35, 44, 53]  # 8 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
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
    
    # Run GA
    print("\n[1/2] Running GA...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    
    # Run SA
    print("\n[2/2] Running SA...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
        verbose=True
    )
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        visualize_solution(
            ga_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 4: GA Best Solution",
            save_path="results/case_study_4/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        visualize_solution(
            sa_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 4: SA Best Solution",
            save_path="results/case_study_4/sa_solution.png"
        )
        print("✅ SA solution visualization saved")
        
        plot_convergence_history(
            ga_results['convergence_history'],
            title="Case Study 4: GA Convergence",
            save_path="results/case_study_4/ga_convergence.png"
        )
        print("✅ GA convergence plot saved")
        
        plot_convergence_history(
            sa_history,
            title="Case Study 4: SA Convergence",
            save_path="results/case_study_4/sa_convergence.png"
        )
        print("✅ SA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        traceback.print_exc()
    
    # Generate comparison report
    try:
        comparison = generate_comparison_report(
            sa_solution, sa_history,
            ga_results['best_solution'], ga_results['convergence_history'],
            grid_width, grid_height,
            case_study_name="Case Study 4: Many Robots"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 4 completed!")
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def run_single_case_study(case_number):
    """Run a single case study by number"""
    case_studies = {
        1: case_study_1_small_grid,
        2: case_study_2_medium_grid,
        3: case_study_3_large_grid,
        4: case_study_4_many_robots
    }
    
    if case_number not in case_studies:
        print(f"❌ Error: Case study {case_number} does not exist.")
        print(f"Available case studies: {list(case_studies.keys())}")
        return None
    
    return case_studies[case_number]()


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
import traceback


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
    
    print("\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Total Cells: {grid_width * grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # Algorithm parameters
    ga_params = {
        'population_size': 30,
        'generations': 50,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'elitism_count': 2
    }
    
    sa_params = {
        'initial_temp': 1000,
        'cooling_rate': 0.95,
        'max_iterations': 50
    }
    
    print(f"\nGA Parameters: {ga_params}")
    print(f"SA Parameters: {sa_params}")
    
    # Create results directory
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
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        # Visualize GA solution
        visualize_solution(
            ga_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 1: GA Best Solution",
            save_path="results/case_study_1/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        # Visualize SA solution
        visualize_solution(
            sa_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 1: SA Best Solution",
            save_path="results/case_study_1/sa_solution.png"
        )
        print("✅ SA solution visualization saved")
        
        # Plot convergence histories
        plot_convergence_history(
            ga_results['convergence_history'],
            title="Case Study 1: GA Convergence",
            save_path="results/case_study_1/ga_convergence.png"
        )
        print("✅ GA convergence plot saved")
        
        plot_convergence_history(
            sa_history,
            title="Case Study 1: SA Convergence",
            save_path="results/case_study_1/sa_convergence.png"
        )
        print("✅ SA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        traceback.print_exc()
    
    # Generate comparison report
    print("\n" + "-"*80)
    print("Generating Comparison Report...")
    print("-"*80)
    
    try:
        comparison = generate_comparison_report(
            sa_solution, sa_history,
            ga_results['best_solution'], ga_results['convergence_history'],
            grid_width, grid_height,
            case_study_name="Case Study 1: Small Grid"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 1 completed!")
    print(f"Results saved to: results/case_study_1/")
    
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def case_study_2_medium_grid():
    """
    Case Study 2: Medium Grid
    Grid: 6x6
    Robots: 3
    Obstacles: Moderate number of obstacles
    Purpose: Test scalability and load balancing
    """
    print("\n" + "="*80)
    print("CASE STUDY 2: MEDIUM GRID (6x6, 3 Robots)")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 6, 6
    num_robots = 3
    obstacles = [1, 7, 13, 19, 25, 31]  # 6 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
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
    
    # Run GA
    print("\n[1/2] Running GA...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    
    # Run SA
    print("\n[2/2] Running SA...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
        verbose=True
    )
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        visualize_solution(
            ga_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 2: GA Best Solution",
            save_path="results/case_study_2/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        visualize_solution(
            sa_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 2: SA Best Solution",
            save_path="results/case_study_2/sa_solution.png"
        )
        print("✅ SA solution visualization saved")
        
        plot_convergence_history(
            ga_results['convergence_history'],
            title="Case Study 2: GA Convergence",
            save_path="results/case_study_2/ga_convergence.png"
        )
        print("✅ GA convergence plot saved")
        
        plot_convergence_history(
            sa_history,
            title="Case Study 2: SA Convergence",
            save_path="results/case_study_2/sa_convergence.png"
        )
        print("✅ SA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        traceback.print_exc()
    
    # Generate comparison report
    try:
        comparison = generate_comparison_report(
            sa_solution, sa_history,
            ga_results['best_solution'], ga_results['convergence_history'],
            grid_width, grid_height,
            case_study_name="Case Study 2: Medium Grid"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 2 completed!")
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def case_study_3_large_grid():
    """
    Case Study 3: Large Grid
    Grid: 10x10
    Robots: 5
    Obstacles: Complex environment
    Purpose: Test algorithm performance on larger problems
    """
    print("\n" + "="*80)
    print("CASE STUDY 3: LARGE GRID (10x10, 5 Robots)")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 10, 10
    num_robots = 5
    obstacles = [11, 12, 13, 22, 32, 42, 52, 62, 67, 68, 69, 77, 88]  # 13 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # Algorithm parameters
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
    
    # Run GA
    print("\n[1/2] Running GA...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    
    # Run SA
    print("\n[2/2] Running SA...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
        verbose=True
    )
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        visualize_solution(
            ga_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 3: GA Best Solution",
            save_path="results/case_study_3/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        visualize_solution(
            sa_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 3: SA Best Solution",
            save_path="results/case_study_3/sa_solution.png"
        )
        print("✅ SA solution visualization saved")
        
        plot_convergence_history(
            ga_results['convergence_history'],
            title="Case Study 3: GA Convergence",
            save_path="results/case_study_3/ga_convergence.png"
        )
        print("✅ GA convergence plot saved")
        
        plot_convergence_history(
            sa_history,
            title="Case Study 3: SA Convergence",
            save_path="results/case_study_3/sa_convergence.png"
        )
        print("✅ SA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        traceback.print_exc()
    
    # Generate comparison report
    try:
        comparison = generate_comparison_report(
            sa_solution, sa_history,
            ga_results['best_solution'], ga_results['convergence_history'],
            grid_width, grid_height,
            case_study_name="Case Study 3: Large Grid"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 3 completed!")
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def case_study_4_many_robots():
    """
    Case Study 4: Many Robots
    Grid: 8x8
    Robots: 6
    Obstacles: Moderate obstacles
    Purpose: Test coordination with many robots
    """
    print("\n" + "="*80)
    print("CASE STUDY 4: MANY ROBOTS (8x8, 6 Robots)")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 8, 8
    num_robots = 6
    obstacles = [9, 10, 17, 18, 26, 35, 44, 53]  # 8 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
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
    
    # Run GA
    print("\n[1/2] Running GA...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    
    # Run SA
    print("\n[2/2] Running SA...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
        verbose=True
    )
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        visualize_solution(
            ga_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 4: GA Best Solution",
            save_path="results/case_study_4/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        visualize_solution(
            sa_solution,
            grid_width=grid_width,
            grid_height=grid_height,
            title="Case Study 4: SA Best Solution",
            save_path="results/case_study_4/sa_solution.png"
        )
        print("✅ SA solution visualization saved")
        
        plot_convergence_history(
            ga_results['convergence_history'],
            title="Case Study 4: GA Convergence",
            save_path="results/case_study_4/ga_convergence.png"
        )
        print("✅ GA convergence plot saved")
        
        plot_convergence_history(
            sa_history,
            title="Case Study 4: SA Convergence",
            save_path="results/case_study_4/sa_convergence.png"
        )
        print("✅ SA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        traceback.print_exc()
    
    # Generate comparison report
    try:
        comparison = generate_comparison_report(
            sa_solution, sa_history,
            ga_results['best_solution'], ga_results['convergence_history'],
            grid_width, grid_height,
            case_study_name="Case Study 4: Many Robots"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 4 completed!")
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison
    }


def run_single_case_study(case_number):
    """Run a single case study by number"""
    case_studies = {
        1: case_study_1_small_grid,
        2: case_study_2_medium_grid,
        3: case_study_3_large_grid,
        4: case_study_4_many_robots
    }
    
    if case_number not in case_studies:
        print(f"❌ Error: Case study {case_number} does not exist.")
        print(f"Available case studies: {list(case_studies.keys())}")
        return None
    
    return case_studies[case_number]()


def run_all_case_studies():
    """Run all case studies sequentially"""
    print("\n" + "="*80)
    print("RUNNING ALL CASE STUDIES")
    print("="*80)
    
    results = {}
    
    for i in range(1, 5):
        try:
            print(f"\n{'='*80}")
            print(f"STARTING CASE STUDY {i}")
            print(f"{'='*80}")
            results[f'case_{i}'] = run_single_case_study(i)
            
            # Print individual case study results
            if results[f'case_{i}'] is not None:
                print(f"\n{'─'*80}")
                print(f"📊 CASE STUDY {i} RESULTS SUMMARY")
                print(f"{'─'*80}")
                
                case_result = results[f'case_{i}']
                ga_result = case_result['ga']
                sa_result = case_result['sa']
                
                ga_solution = ga_result['best_solution']
                sa_solution = sa_result[0]
                
                print(f"\n🧬 Genetic Algorithm:")
                print(f"   • Final Score:        {ga_solution.combined_score:.3f}")
                print(f"   • Coverage:           {ga_solution.get_coverage_efficiency():.2f}%")
                print(f"   • Balance Index:      {ga_solution.get_workload_balance_index():.4f}")
                
                print(f"\n🔥 Simulated Annealing:")
                print(f"   • Final Score:        {sa_solution.combined_score:.3f}")
                print(f"   • Coverage:           {sa_solution.get_coverage_efficiency():.2f}%")
                print(f"   • Balance Index:      {sa_solution.get_workload_balance_index():.4f}")
                
                # Winner determination
                if ga_solution.combined_score < sa_solution.combined_score:
                    winner = "GA"
                    improvement = ((sa_solution.combined_score - ga_solution.combined_score) / sa_solution.combined_score) * 100
                else:
                    winner = "SA"
                    improvement = ((ga_solution.combined_score - sa_solution.combined_score) / ga_solution.combined_score) * 100
                
                print(f"\n🏆 Winner: {winner}")
                print(f"   • Better by: {improvement:.2f}%")
                
                # Check if images were created
                case_dir = f"results/case_study_{i}"
                if os.path.exists(case_dir):
                    files = [f for f in os.listdir(case_dir) if f.endswith('.png')]
                    if files:
                        print(f"\n📊 Generated visualizations:")
                        for f in sorted(files):
                            print(f"   • {case_dir}/{f}")
                    else:
                        print(f"\n⚠️ No visualizations generated in {case_dir}/")
                else:
                    print(f"\n⚠️ Directory {case_dir}/ not found")
                
                print(f"{'─'*80}")
                
        except Exception as e:
            print(f"❌ Case Study {i} failed: {e}")
            traceback.print_exc()
            results[f'case_{i}'] = None
    
    # Overall comparison across all case studies
    print("\n" + "="*80)
    print("OVERALL SUMMARY - ALL CASE STUDIES")
    print("="*80)
    
    successful = sum(1 for v in results.values() if v is not None)
    print(f"\n✅ Case studies completed: {successful}/4")
    
    if successful > 0:
        print("\n" + "="*80)
        print("📊 ALGORITHM COMPARISON ACROSS ALL CASE STUDIES")
        print("="*80)
        
        ga_wins = 0
        sa_wins = 0
        ga_scores = []
        sa_scores = []
        ga_coverage = []
        sa_coverage = []
        ga_balance = []
        sa_balance = []
        
        for i in range(1, 5):
            case_key = f'case_{i}'
            if results[case_key] is not None:
                ga_result = results[case_key]['ga']
                sa_result = results[case_key]['sa']
                
                ga_solution = ga_result['best_solution']
                sa_solution = sa_result[0]
                
                ga_scores.append(ga_solution.combined_score)
                sa_scores.append(sa_solution.combined_score)
                ga_coverage.append(ga_solution.get_coverage_efficiency())
                sa_coverage.append(sa_solution.get_coverage_efficiency())
                ga_balance.append(ga_solution.get_workload_balance_index())
                sa_balance.append(sa_solution.get_workload_balance_index())
                
                if ga_solution.combined_score < sa_solution.combined_score:
                    ga_wins += 1
                else:
                    sa_wins += 1
        
        # Print comparison table
        print(f"\n{'Metric':<25} {'GA':<20} {'SA':<20} {'Winner':<10}")
        print("─" * 75)
        
        # Average scores
        avg_ga_score = sum(ga_scores) / len(ga_scores) if ga_scores else 0
        avg_sa_score = sum(sa_scores) / len(sa_scores) if sa_scores else 0
        score_winner = "GA" if avg_ga_score < avg_sa_score else "SA"
        print(f"{'Average Score':<25} {avg_ga_score:<20.3f} {avg_sa_score:<20.3f} {score_winner:<10}")
        
        # Average coverage
        avg_ga_cov = sum(ga_coverage) / len(ga_coverage) if ga_coverage else 0
        avg_sa_cov = sum(sa_coverage) / len(sa_coverage) if sa_coverage else 0
        cov_winner = "GA" if avg_ga_cov > avg_sa_cov else "SA"
        print(f"{'Average Coverage (%)':<25} {avg_ga_cov:<20.2f} {avg_sa_cov:<20.2f} {cov_winner:<10}")
        
        # Average balance
        avg_ga_bal = sum(ga_balance) / len(ga_balance) if ga_balance else 0
        avg_sa_bal = sum(sa_balance) / len(sa_balance) if sa_balance else 0
        bal_winner = "GA" if avg_ga_bal > avg_sa_bal else "SA"
        print(f"{'Average Balance Index':<25} {avg_ga_bal:<20.4f} {avg_sa_bal:<20.4f} {bal_winner:<10}")
        
        # Win count
        print(f"{'Case Study Wins':<25} {ga_wins:<20} {sa_wins:<20} {'-':<10}")
        
        print("\n" + "="*80)
        print("🏆 FINAL VERDICT")
        print("="*80)
        
        if ga_wins > sa_wins:
            print(f"✅ Genetic Algorithm wins overall!")
            print(f"   • Won {ga_wins}/{successful} case studies")
            print(f"   • Average score: {avg_ga_score:.3f}")
        elif sa_wins > ga_wins:
            print(f"✅ Simulated Annealing wins overall!")
            print(f"   • Won {sa_wins}/{successful} case studies")
            print(f"   • Average score: {avg_sa_score:.3f}")
        else:
            print(f"🤝 It's a tie!")
            print(f"   • Both algorithms won {ga_wins} case studies each")
        
        print("\n" + "="*80)
        print("📈 DETAILED CASE-BY-CASE BREAKDOWN")
        print("="*80)
        
        for i in range(1, 5):
            case_key = f'case_{i}'
            if results[case_key] is not None:
                ga_solution = results[case_key]['ga']['best_solution']
                sa_solution = results[case_key]['sa'][0]
                
                winner = "GA" if ga_solution.combined_score < sa_solution.combined_score else "SA"
                ga_score = ga_solution.combined_score
                sa_score = sa_solution.combined_score
                
                print(f"\nCase Study {i}:")
                print(f"   • GA Score: {ga_score:.3f}")
                print(f"   • SA Score: {sa_score:.3f}")
                print(f"   • Winner: {winner} {'✅' if winner == 'GA' else '🔥'}")
                
                if winner == "GA":
                    improvement = ((sa_score - ga_score) / sa_score) * 100
                    print(f"   • GA better by: {improvement:.2f}%")
                else:
                    improvement = ((ga_score - sa_score) / ga_score) * 100
                    print(f"   • SA better by: {improvement:.2f}%")
        
        print("\n" + "="*80)
    
    print(f"\n📁 Results saved to: results/case_study_*/")
    
    return results