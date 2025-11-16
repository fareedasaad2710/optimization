"""
Simple script to run Genetic Algorithm (GA) only on a case study
No Simulated Annealing (SA) - just GA!

Usage:
    python3 run_ga_only.py
"""

from GA import genetic_algorithm
from visualization import visualize_solution, plot_convergence_history
import time
import os

def run_ga_case_study_1():
    """Run GA on Case Study 1: Small Grid (4x4, 2 Robots)"""
    
    print("\n" + "="*80)
    print("RUNNING GA ONLY - CASE STUDY 1: SMALL GRID (4x4, 2 Robots)")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 4, 4
    num_robots = 2
    obstacles = [5, 10]  # 2 obstacles
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Robots: {num_robots}")
    
    # GA parameters
    # Population creation strategy: 80% crossover, 10% mutation, 10% elite/survivors
    ga_params = {
        'population_size': 30,
        'generations': 50,
        'selection_percentage': 0.10,   # 10% elite/survivors
        'crossover_percentage': 0.80,   # 80% crossover
        'mutation_percentage': 0.10     # 10% mutation
    }
    
    print(f"\nGA Parameters: {ga_params}")
    
    # Create results directory
    os.makedirs("results/ga_only", exist_ok=True)
    
    # Run GA with runtime measurement
    print("\n" + "-"*80)
    print("Running Genetic Algorithm...")
    print("-"*80)
    
    start_time = time.time()
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\n⏱️  GA Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        best_solution = ga_results['best_solution']
        
        # Visualize GA solution
        visualize_solution(
            best_solution,
            title="GA Best Solution (Case Study 1)",
            save_path="results/ga_only/ga_solution.png"
        )
        print("✅ GA solution visualization saved")
        
        # Plot convergence history
        plot_convergence_history(
            ga_results['convergence_history'],
            title="GA Convergence (Case Study 1)",
            save_path="results/ga_only/ga_convergence.png"
        )
        print("✅ GA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print results summary
    print("\n" + "="*80)
    print("📊 GA RESULTS SUMMARY")
    print("="*80)
    
    if best_solution:
        print(f"Best Combined Score: {best_solution.combined_score:.4f}")
        if best_solution.fitness:
            print(f"Coverage: {best_solution.fitness.get('coverage_score', 'N/A')}/{len(free_cells)} cells")
            print(f"Balance: {best_solution.fitness.get('balance_score', 'N/A'):.4f}")
        if hasattr(best_solution, 'get_coverage_efficiency'):
            print(f"Coverage Efficiency: {best_solution.get_coverage_efficiency():.2f}%")
        if hasattr(best_solution, 'get_workload_balance_index'):
            print(f"Workload Balance Index: {best_solution.get_workload_balance_index():.4f}")
    
    print(f"\n✅ GA run completed!")
    print(f"Results saved to: results/ga_only/")
    
    return ga_results


def run_ga_case_study_2():
    """Run GA on Case Study 2: Medium Grid (6x6, 3 Robots)"""
    
    print("\n" + "="*80)
    print("RUNNING GA ONLY - CASE STUDY 2: MEDIUM GRID (6x6, 3 Robots)")
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
    
    # GA parameters
    # Population creation strategy: 80% crossover, 10% mutation, 10% elite/survivors
    ga_params = {
        'population_size': 50,
        'generations': 100,
        'selection_percentage': 0.10,   # 10% elite/survivors
        'crossover_percentage': 0.80,   # 80% crossover
        'mutation_percentage': 0.10     # 10% mutation
    }
    
    print(f"\nGA Parameters: {ga_params}")
    
    # Create results directory
    os.makedirs("results/ga_only", exist_ok=True)
    
    # Run GA with runtime measurement
    print("\n" + "-"*80)
    print("Running Genetic Algorithm...")
    print("-"*80)
    
    start_time = time.time()
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\n⏱️  GA Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        best_solution = ga_results['best_solution']
        
        # Visualize GA solution
        visualize_solution(
            best_solution,
            title="GA Best Solution (Case Study 2)",
            save_path="results/ga_only/ga_solution_case2.png"
        )
        print("✅ GA solution visualization saved")
        
        # Plot convergence history
        plot_convergence_history(
            ga_results['convergence_history'],
            title="GA Convergence (Case Study 2)",
            save_path="results/ga_only/ga_convergence_case2.png"
        )
        print("✅ GA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print results summary
    print("\n" + "="*80)
    print("📊 GA RESULTS SUMMARY")
    print("="*80)
    
    if best_solution:
        print(f"Best Combined Score: {best_solution.combined_score:.4f}")
        if best_solution.fitness:
            print(f"Coverage: {best_solution.fitness.get('coverage_score', 'N/A')}/{len(free_cells)} cells")
            print(f"Balance: {best_solution.fitness.get('balance_score', 'N/A'):.4f}")
        if hasattr(best_solution, 'get_coverage_efficiency'):
            print(f"Coverage Efficiency: {best_solution.get_coverage_efficiency():.2f}%")
        if hasattr(best_solution, 'get_workload_balance_index'):
            print(f"Workload Balance Index: {best_solution.get_workload_balance_index():.4f}")
    
    print(f"\n✅ GA run completed!")
    print(f"Results saved to: results/ga_only/")
    
    return ga_results


def run_ga_case_study_3():
    """Run GA on Case Study 3: Large Grid (10x10, 5 Robots)"""
    
    print("\n" + "="*80)
    print("RUNNING GA ONLY - CASE STUDY 3: LARGE GRID (10x10, 5 Robots)")
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
    
    # GA parameters
    # Population creation strategy: 80% crossover, 10% mutation, 10% elite/survivors
    ga_params = {
        'population_size': 100,
        'generations': 200,
        'selection_percentage': 0.10,   # 10% elite/survivors
        'crossover_percentage': 0.80,   # 80% crossover
        'mutation_percentage': 0.10     # 10% mutation
    }
    
    print(f"\nGA Parameters: {ga_params}")
    
    # Create results directory
    os.makedirs("results/ga_only", exist_ok=True)
    
    # Run GA with runtime measurement
    print("\n" + "-"*80)
    print("Running Genetic Algorithm...")
    print("-"*80)
    
    start_time = time.time()
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        **ga_params,
        verbose=True
    )
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\n⏱️  GA Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        best_solution = ga_results['best_solution']
        
        # Visualize GA solution
        visualize_solution(
            best_solution,
            title="GA Best Solution (Case Study 3)",
            save_path="results/ga_only/ga_solution_case3.png"
        )
        print("✅ GA solution visualization saved")
        
        # Plot convergence history
        plot_convergence_history(
            ga_results['convergence_history'],
            title="GA Convergence (Case Study 3)",
            save_path="results/ga_only/ga_convergence_case3.png"
        )
        print("✅ GA convergence plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print results summary
    print("\n" + "="*80)
    print("📊 GA RESULTS SUMMARY")
    print("="*80)
    
    if best_solution:
        print(f"Best Combined Score: {best_solution.combined_score:.4f}")
        if best_solution.fitness:
            print(f"Coverage: {best_solution.fitness.get('coverage_score', 'N/A')}/{len(free_cells)} cells")
            print(f"Balance: {best_solution.fitness.get('balance_score', 'N/A'):.4f}")
        if hasattr(best_solution, 'get_coverage_efficiency'):
            print(f"Coverage Efficiency: {best_solution.get_coverage_efficiency():.2f}%")
        if hasattr(best_solution, 'get_workload_balance_index'):
            print(f"Workload Balance Index: {best_solution.get_workload_balance_index():.4f}")
    
    print(f"\n✅ GA run completed!")
    print(f"Results saved to: results/ga_only/")
    
    return ga_results


if __name__ == "__main__":
    import sys
    
    # Check if case study number is provided
    if len(sys.argv) > 1:
        case_num = int(sys.argv[1])
    else:
        # Default to case study 1
        case_num = 1
    
    print("\n" + "="*80)
    print("🧬 GA ONLY RUNNER")
    print("="*80)
    print("\nThis script runs ONLY the Genetic Algorithm (GA)")
    print("No Simulated Annealing (SA) will be executed")
    print("="*80)
    
    if case_num == 1:
        run_ga_case_study_1()
    elif case_num == 2:
        run_ga_case_study_2()
    elif case_num == 3:
        run_ga_case_study_3()
    else:
        print(f"\n❌ Error: Case study {case_num} not available")
        print("Available case studies: 1, 2, 3")
        print("\nUsage:")
        print("  python3 run_ga_only.py          # Run case study 1 (default)")
        print("  python3 run_ga_only.py 1       # Run case study 1")
        print("  python3 run_ga_only.py 2       # Run case study 2")
        print("  python3 run_ga_only.py 3       # Run case study 3")

