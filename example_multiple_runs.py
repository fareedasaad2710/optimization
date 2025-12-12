"""
Example: How to Run Case Studies with Multiple Runs

This file shows how to modify case studies to run algorithms multiple times
and display statistics (average, standard deviation, best of k runs).
"""

from case_studies import case_study_1_small_grid
from algorithm_comparison import run_algorithm_multiple_times, generate_comparison_report
from GA import genetic_algorithm
from sa_algorithm import simulated_annealing
import time

def case_study_1_with_multiple_runs(num_runs=5):
    """
    Modified version of case_study_1_small_grid that runs multiple times
    
    Args:
        num_runs: Number of times to run each algorithm (default: 5)
    """
    # Problem setup (same as original)
    grid_width, grid_height = 4, 4
    num_robots = 2
    obstacles = [5, 10]
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    # Algorithm parameters
    ga_params = {
        'population_size': 30,
        'generations': 50,
        'verbose': False  # Set to False to reduce output during multiple runs
    }
    
    sa_params = {
        'initial_temp': 1000,
        'cooling_rate': 0.95,
        'max_iterations': 50
    }
    
    print(f"\n{'='*80}")
    print(f"CASE STUDY 1: SMALL GRID (4x4, 2 Robots) - {num_runs} RUNS")
    print(f"{'='*80}\n")
    
    # ============================================================
    # METHOD 1: Using the helper function (RECOMMENDED)
    # ============================================================
    
    print("🔄 Running GA multiple times...")
    ga_results_list = run_algorithm_multiple_times(
        genetic_algorithm,
        num_runs=num_runs,
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params
    )
    
    print("🔄 Running SA multiple times...")
    sa_results_list = run_algorithm_multiple_times(
        simulated_annealing,
        num_runs=num_runs,
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **sa_params
    )
    
    # Add runtime to each result
    for ga_res in ga_results_list:
        if isinstance(ga_res, dict) and 'runtime' not in ga_res:
            # Runtime should already be in ga_results, but if not, add it
            pass
    
    # Convert SA results to tuple format (solution, history)
    sa_results_tuples = []
    for sa_res in sa_results_list:
        if isinstance(sa_res, tuple):
            sa_solution, sa_history = sa_res
            if isinstance(sa_history, dict):
                sa_history['runtime'] = sa_history.get('runtime', 0)
            sa_results_tuples.append((sa_solution, sa_history))
        else:
            # If SA returns something else, handle it
            sa_results_tuples.append((sa_res, {}))
    
    # ============================================================
    # METHOD 2: Manual loop (if you need more control)
    # ============================================================
    """
    # Alternative: Manual loop approach
    ga_results_list = []
    sa_results_list = []
    
    for run_num in range(1, num_runs + 1):
        print(f"\n🔄 Run {run_num}/{num_runs}")
        
        # Run GA
        ga_start_time = time.time()
        ga_results = genetic_algorithm(
            all_cells, free_cells, obstacles,
            grid_width, grid_height, num_robots,
            **ga_params,
            verbose=False
        )
        ga_end_time = time.time()
        ga_runtime = ga_end_time - ga_start_time
        ga_results['runtime'] = ga_runtime
        ga_results_list.append(ga_results)
        
        # Run SA
        sa_start_time = time.time()
        sa_solution, sa_history = simulated_annealing(
            all_cells, free_cells, obstacles,
            grid_width, grid_height, num_robots,
            **sa_params,
        )
        sa_end_time = time.time()
        sa_runtime = sa_end_time - sa_start_time
        
        if isinstance(sa_history, dict):
            sa_history['runtime'] = sa_runtime
        else:
            sa_history = {'convergence_history': sa_history, 'runtime': sa_runtime}
        
        sa_results_list.append((sa_solution, sa_history))
    """
    
    # Generate comparison report with statistics
    print("\n" + "="*80)
    print("📊 GENERATING COMPARISON REPORT WITH STATISTICS")
    print("="*80)
    
    comparison = generate_comparison_report(
        sa_results_tuples,  # List of (solution, history) tuples
        ga_results_list,     # List of GA results dictionaries
        case_study_name=f"Case Study 1: Small Grid ({num_runs} runs)"
    )
    
    return {
        'ga_results_list': ga_results_list,
        'sa_results_list': sa_results_tuples,
        'comparison': comparison
    }


if __name__ == "__main__":
    # Run with 5 runs (you can change this number)
    results = case_study_1_with_multiple_runs(num_runs=5)
    
    print("\n✅ Multiple runs completed!")
    print(f"   • GA runs: {len(results['ga_results_list'])}")
    print(f"   • SA runs: {len(results['sa_results_list'])}")
    print(f"   • Statistics are shown in the comparison report above")

