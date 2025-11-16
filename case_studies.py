"""
Case Studies for Multi-Robot Coverage Path Planning
===================================================

This module contains different case studies to test and validate
the GA and SA algorithms on various problem configurations.
"""

from problem_formulation import *
from GA import genetic_algorithm
from sa_algorithm import simulated_annealing
from algorithm_comparison import compare_sa_vs_ga, generate_comparison_report, run_algorithm_multiple_times
from visualization import visualize_solution, plot_convergence_history, save_all_figures
import os
import traceback
import time


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
    
    # Run GA with runtime measurement
    print("\n" + "-"*80)
    print("Running Genetic Algorithm...")
    print("-"*80)
    
    ga_start_time = time.time()
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    ga_end_time = time.time()
    ga_runtime = ga_end_time - ga_start_time
    print(f"\n⏱️  GA Runtime: {ga_runtime:.2f} seconds ({ga_runtime/60:.2f} minutes)")
    
    # Run SA with runtime measurement
    print("\n" + "-"*80)
    print("Running Simulated Annealing...")
    print("-"*80)
    
    sa_start_time = time.time()
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
    )
    sa_end_time = time.time()
    sa_runtime = sa_end_time - sa_start_time
    print(f"\n⏱️  SA Runtime: {sa_runtime:.2f} seconds ({sa_runtime/60:.2f} minutes)")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        # Visualize GA solution
        visualize_solution(
            ga_solution,
            title="Case Study 1: GA Best Solution",
            save_path="results/case_study_1/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        # Visualize SA solution
        visualize_solution(
            sa_solution,
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
        # Add runtime to results for comparison
        ga_results['runtime'] = ga_runtime
        if isinstance(sa_history, dict):
            sa_history['runtime'] = sa_runtime
        else:
            # If sa_history is not a dict, wrap it
            sa_history_dict = {'convergence_history': sa_history, 'runtime': sa_runtime}
            sa_history = sa_history_dict
        
        comparison = generate_comparison_report(
            (sa_solution, sa_history),
            ga_results,
            case_study_name="Case Study 1: Small Grid"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 1 completed!")
    print(f"Results saved to: results/case_study_1/")
    print(f"\n⏱️  Runtime Summary:")
    print(f"   • GA: {ga_runtime:.2f} seconds ({ga_runtime/60:.2f} minutes)")
    print(f"   • SA: {sa_runtime:.2f} seconds ({sa_runtime/60:.2f} minutes)")
    if ga_runtime > 0 and sa_runtime > 0:
        if ga_runtime > sa_runtime:
            speedup = ga_runtime / sa_runtime
            faster = "SA"
        else:
            speedup = sa_runtime / ga_runtime
            faster = "GA"
        print(f"   • Speedup: {speedup:.2f}x ({faster} is faster)")
    
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison,
        'ga_runtime': ga_runtime,
        'sa_runtime': sa_runtime
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
    }
    
    sa_params = {
        'initial_temp': 1500,
        'cooling_rate': 0.95,
        'max_iterations': 100
    }
    
    print(f"\nGA Parameters: {ga_params}")
    print(f"SA Parameters: {sa_params}")
    
    os.makedirs("results/case_study_2", exist_ok=True)
    
    # Run GA with runtime measurement
    print("\n[1/2] Running GA...")
    ga_start_time = time.time()
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    ga_end_time = time.time()
    ga_runtime = ga_end_time - ga_start_time
    print(f"\n⏱️  GA Runtime: {ga_runtime:.2f} seconds ({ga_runtime/60:.2f} minutes)")
    
    # Run SA with runtime measurement
    print("\n[2/2] Running SA...")
    sa_start_time = time.time()
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
    )
    sa_end_time = time.time()
    sa_runtime = sa_end_time - sa_start_time
    print(f"\n⏱️  SA Runtime: {sa_runtime:.2f} seconds ({sa_runtime/60:.2f} minutes)")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        visualize_solution(
            ga_solution,
            title="Case Study 2: GA Best Solution",
            save_path="results/case_study_2/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        visualize_solution(
            sa_solution,
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
        # Add runtime to results for comparison
        ga_results['runtime'] = ga_runtime
        if isinstance(sa_history, dict):
            sa_history['runtime'] = sa_runtime
        else:
            # If sa_history is not a dict, wrap it
            sa_history_dict = {'convergence_history': sa_history, 'runtime': sa_runtime}
            sa_history = sa_history_dict
        
        comparison = generate_comparison_report(
            (sa_solution, sa_history),
            ga_results,
            case_study_name="Case Study 2: Medium Grid"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 2 completed!")
    print(f"\n⏱️  Runtime Summary:")
    print(f"   • GA: {ga_runtime:.2f} seconds ({ga_runtime/60:.2f} minutes)")
    print(f"   • SA: {sa_runtime:.2f} seconds ({sa_runtime/60:.2f} minutes)")
    if ga_runtime > 0 and sa_runtime > 0:
        if ga_runtime > sa_runtime:
            speedup = ga_runtime / sa_runtime
            faster = "SA"
        else:
            speedup = sa_runtime / ga_runtime
            faster = "GA"
        print(f"   • Speedup: {speedup:.2f}x ({faster} is faster)")
    
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison,
        'ga_runtime': ga_runtime,
        'sa_runtime': sa_runtime
    }


def case_study_2_medium_grid_multiple_runs(num_runs=5):
    """
    Case Study 2: Medium Grid - Multiple Runs with Statistics
    Runs the case study multiple times and displays statistics (avg, std dev, best of k runs)
    
    Args:
        num_runs: Number of times to run each algorithm (default: 5)
    
    Returns:
        Dictionary with all results and comparison statistics
    """
    print("\n" + "="*80)
    print(f"CASE STUDY 2: MEDIUM GRID (6x6, 3 Robots) - {num_runs} RUNS")
    print("="*80)
    
    # Problem parameters (same as case_study_2_medium_grid)
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
        'verbose': False  # Set to False to reduce output during multiple runs
    }
    
    sa_params = {
        'initial_temp': 1500,
        'cooling_rate': 0.95,
        'max_iterations': 100
    }
    
    print(f"\nGA Parameters: {ga_params}")
    print(f"SA Parameters: {sa_params}")
    print(f"\n🔄 Running each algorithm {num_runs} times to collect statistics...")
    
    os.makedirs("results/case_study_2", exist_ok=True)
    
    # Run GA multiple times
    print("\n" + "-"*80)
    print("Running Genetic Algorithm Multiple Times...")
    print("-"*80)
    
    ga_results_list = run_algorithm_multiple_times(
        genetic_algorithm,
        num_runs,
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params
    )
    
    # Add runtime to each GA result if not already present
    for ga_res in ga_results_list:
        if isinstance(ga_res, dict) and 'runtime' not in ga_res:
            # Runtime should be tracked, but ensure it's there
            pass
    
    # Run SA multiple times
    print("\n" + "-"*80)
    print("Running Simulated Annealing Multiple Times...")
    print("-"*80)
    
    sa_results_list = run_algorithm_multiple_times(
        simulated_annealing,
        num_runs,
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params
    )
    
    # Convert SA results to tuple format (solution, history) and add runtime
    sa_results_tuples = []
    for sa_res in sa_results_list:
        if isinstance(sa_res, tuple):
            sa_solution, sa_history = sa_res
            # Ensure runtime is in history dict
            if isinstance(sa_history, dict):
                if 'runtime' not in sa_history:
                    sa_history['runtime'] = 0  # Will be calculated if needed
            sa_results_tuples.append((sa_solution, sa_history))
        else:
            # Handle if SA returns something else
            sa_results_tuples.append((sa_res, {'runtime': 0}))
    
    # Generate visualizations for the best solutions
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS (Best Solutions)")
    print("="*80)
    
    try:
        # Find best GA solution
        ga_best = min(ga_results_list, 
                     key=lambda x: x['best_solution'].combined_score 
                     if isinstance(x, dict) and 'best_solution' in x 
                     and x['best_solution'].combined_score is not None 
                     else float('inf'))
        ga_solution = ga_best['best_solution'] if isinstance(ga_best, dict) else None
        
        # Find best SA solution
        sa_best = min(sa_results_tuples,
                     key=lambda x: x[0].combined_score 
                     if x[0] and hasattr(x[0], 'combined_score') 
                     and x[0].combined_score is not None 
                     else float('inf'))
        sa_solution = sa_best[0] if sa_best else None
        
        if ga_solution:
            visualize_solution(
                ga_solution,
                title=f"Case Study 2: GA Best Solution ({num_runs} runs)",
                save_path="results/case_study_2/ga_solution_multiple.png"
            )
            print("✅ GA best solution visualization saved")
        
        if sa_solution:
            visualize_solution(
                sa_solution,
                title=f"Case Study 2: SA Best Solution ({num_runs} runs)",
                save_path="results/case_study_2/sa_solution_multiple.png"
            )
            print("✅ SA best solution visualization saved")
        
        # Plot convergence for best runs
        if isinstance(ga_best, dict) and 'convergence_history' in ga_best:
            plot_convergence_history(
                ga_best['convergence_history'],
                title=f"Case Study 2: GA Convergence (Best of {num_runs} runs)",
                save_path="results/case_study_2/ga_convergence_multiple.png"
            )
            print("✅ GA convergence plot saved")
        
        if isinstance(sa_best[1], dict) and 'convergence_history' in sa_best[1]:
            plot_convergence_history(
                sa_best[1]['convergence_history'],
                title=f"Case Study 2: SA Convergence (Best of {num_runs} runs)",
                save_path="results/case_study_2/sa_convergence_multiple.png"
            )
            print("✅ SA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        traceback.print_exc()
    
    # Generate comparison report with statistics
    print("\n" + "="*80)
    print("📊 GENERATING COMPARISON REPORT WITH STATISTICS")
    print("="*80)
    
    try:
        comparison = generate_comparison_report(
            sa_results_tuples,  # List of (solution, history) tuples
            ga_results_list,     # List of GA results dictionaries
            case_study_name=f"Case Study 2: Medium Grid ({num_runs} runs)"
        )
        print("✅ Comparison report with statistics generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    # Calculate total runtime
    total_ga_runtime = sum(r.get('runtime', 0) if isinstance(r, dict) else 0 
                           for r in ga_results_list)
    total_sa_runtime = sum(h.get('runtime', 0) if isinstance(h, dict) else 0 
                          for _, h in sa_results_tuples)
    
    print("\n✅ Case Study 2 (Multiple Runs) completed!")
    print(f"\n⏱️  Total Runtime Summary ({num_runs} runs):")
    print(f"   • GA Total: {total_ga_runtime:.2f} seconds ({total_ga_runtime/60:.2f} minutes)")
    print(f"   • SA Total: {total_sa_runtime:.2f} seconds ({total_sa_runtime/60:.2f} minutes)")
    print(f"   • GA Average: {total_ga_runtime/num_runs:.2f} seconds per run")
    print(f"   • SA Average: {total_sa_runtime/num_runs:.2f} seconds per run")
    
    return {
        'ga_results_list': ga_results_list,
        'sa_results_list': sa_results_tuples,
        'comparison': comparison,
        'num_runs': num_runs,
        'total_ga_runtime': total_ga_runtime,
        'total_sa_runtime': total_sa_runtime
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
    }
    
    sa_params = {
        'initial_temp': 2000,
        'cooling_rate': 0.97,
        'max_iterations': 200
    }
    
    print(f"\nGA Parameters: {ga_params}")
    print(f"SA Parameters: {sa_params}")
    
    os.makedirs("results/case_study_3", exist_ok=True)
    
    # Run GA with runtime measurement
    print("\n[1/2] Running GA...")
    ga_start_time = time.time()
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    ga_end_time = time.time()
    ga_runtime = ga_end_time - ga_start_time
    print(f"\n⏱️  GA Runtime: {ga_runtime:.2f} seconds ({ga_runtime/60:.2f} minutes)")
    
    # Run SA with runtime measurement
    print("\n[2/2] Running SA...")
    sa_start_time = time.time()
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
    )
    sa_end_time = time.time()
    sa_runtime = sa_end_time - sa_start_time
    print(f"\n⏱️  SA Runtime: {sa_runtime:.2f} seconds ({sa_runtime/60:.2f} minutes)")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        visualize_solution(
            ga_solution,
            title="Case Study 3: GA Best Solution",
            save_path="results/case_study_3/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        visualize_solution(
            sa_solution,
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
        # Add runtime to results for comparison
        ga_results['runtime'] = ga_runtime
        if isinstance(sa_history, dict):
            sa_history['runtime'] = sa_runtime
        else:
            # If sa_history is not a dict, wrap it
            sa_history_dict = {'convergence_history': sa_history, 'runtime': sa_runtime}
            sa_history = sa_history_dict
        
        comparison = generate_comparison_report(
            (sa_solution, sa_history),
            ga_results,
            case_study_name="Case Study 3: Large Grid"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 3 completed!")
    print(f"\n⏱️  Runtime Summary:")
    print(f"   • GA: {ga_runtime:.2f} seconds ({ga_runtime/60:.2f} minutes)")
    print(f"   • SA: {sa_runtime:.2f} seconds ({sa_runtime/60:.2f} minutes)")
    if ga_runtime > 0 and sa_runtime > 0:
        if ga_runtime > sa_runtime:
            speedup = ga_runtime / sa_runtime
            faster = "SA"
        else:
            speedup = sa_runtime / ga_runtime
            faster = "GA"
        print(f"   • Speedup: {speedup:.2f}x ({faster} is faster)")
    
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison,
        'ga_runtime': ga_runtime,
        'sa_runtime': sa_runtime
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
    }
    
    sa_params = {
        'initial_temp': 1800,
        'cooling_rate': 0.96,
        'max_iterations': 150
    }
    
    print(f"\nGA Parameters: {ga_params}")
    print(f"SA Parameters: {sa_params}")
    
    os.makedirs("results/case_study_4", exist_ok=True)
    
    # Run GA with runtime measurement
    print("\n[1/2] Running GA...")
    ga_start_time = time.time()
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    ga_end_time = time.time()
    ga_runtime = ga_end_time - ga_start_time
    print(f"\n⏱️  GA Runtime: {ga_runtime:.2f} seconds ({ga_runtime/60:.2f} minutes)")
    
    # Run SA with runtime measurement
    print("\n[2/2] Running SA...")
    sa_start_time = time.time()
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params,
    )
    sa_end_time = time.time()
    sa_runtime = sa_end_time - sa_start_time
    print(f"\n⏱️  SA Runtime: {sa_runtime:.2f} seconds ({sa_runtime/60:.2f} minutes)")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        ga_solution = ga_results['best_solution']
        
        visualize_solution(
            ga_solution,
            title="Case Study 4: GA Best Solution",
            save_path="results/case_study_4/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        visualize_solution(
            sa_solution,
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
        # Add runtime to results for comparison
        ga_results['runtime'] = ga_runtime
        if isinstance(sa_history, dict):
            sa_history['runtime'] = sa_runtime
        else:
            # If sa_history is not a dict, wrap it
            sa_history_dict = {'convergence_history': sa_history, 'runtime': sa_runtime}
            sa_history = sa_history_dict
        
        comparison = generate_comparison_report(
            (sa_solution, sa_history),
            ga_results,
            case_study_name="Case Study 4: Many Robots"
        )
        print("✅ Comparison report generated")
    except Exception as e:
        print(f"⚠️ Comparison report error: {e}")
        traceback.print_exc()
        comparison = None
    
    print("\n✅ Case Study 4 completed!")
    print(f"\n⏱️  Runtime Summary:")
    print(f"   • GA: {ga_runtime:.2f} seconds ({ga_runtime/60:.2f} minutes)")
    print(f"   • SA: {sa_runtime:.2f} seconds ({sa_runtime/60:.2f} minutes)")
    if ga_runtime > 0 and sa_runtime > 0:
        if ga_runtime > sa_runtime:
            speedup = ga_runtime / sa_runtime
            faster = "SA"
        else:
            speedup = sa_runtime / ga_runtime
            faster = "GA"
        print(f"   • Speedup: {speedup:.2f}x ({faster} is faster)")
    
    return {
        'ga': ga_results,
        'sa': (sa_solution, sa_history),
        'comparison': comparison,
        'ga_runtime': ga_runtime,
        'sa_runtime': sa_runtime
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