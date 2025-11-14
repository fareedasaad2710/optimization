"""
Case Studies for Multi-Robot Coverage Path Planning
Compares SA and GA algorithms on different problem sizes and complexities
"""

from problem_formulation import create_grid_cells
from algorithm_comparison import compare_sa_vs_ga, generate_comparison_report
from visualization import visualize_solution
import time

def case_study_1_small_grid():
    """
    Case Study 1: Small 3x3 Grid
    - Purpose: Validate both algorithms on simple problem
    - Grid: 3x3 (9 cells)
    - Robots: 2
    - Obstacles: 1 (center cell)
    - Expected: Both algorithms should achieve 100% coverage
    """
    print("\n" + "="*70)
    print("CASE STUDY 1: Small 3x3 Grid")
    print("="*70)
    print("Purpose: Validate algorithms on simple problem")
    print("-"*70)
    
    # Configuration
    grid_width = 3
    grid_height = 3
    num_robots = 2
    obstacles = [4]  # Center cell
    
    total_cells = grid_width * grid_height
    free_cells = [i for i in range(total_cells) if i not in obstacles]
    all_cells = create_grid_cells(grid_width, grid_height)
    
    print(f"Grid Size: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Free Cells: {len(free_cells)}")
    print(f"Obstacles: {obstacles}")
    
    # Run comparison
    start_time = time.time()
    sa_solution, ga_solution, sa_history, ga_history = compare_sa_vs_ga(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        sa_params={'initial_temp': 1000, 'cooling_rate': 0.95, 'max_iterations': 500},
        ga_params={'population_size': 30, 'generations': 50, 'crossover_rate': 0.8, 'mutation_rate': 0.1}
    )
    end_time = time.time()
    
    print(f"\n⏱️  Total computation time: {end_time - start_time:.2f} seconds")
    
    # Generate report with visualizations
    generate_comparison_report(
        sa_solution, ga_solution, sa_history, ga_history,
        case_study_name="case1_small_3x3",
        save_dir="results/case1"
    )
    
    # Visualize both solutions
    print("\n📊 Visualizing SA Solution...")
    visualize_solution(sa_solution, grid_width, grid_height, "Case 1: SA Solution (3x3)")
    
    print("📊 Visualizing GA Solution...")
    visualize_solution(ga_solution, grid_width, grid_height, "Case 1: GA Solution (3x3)")
    
    return {
        'sa': {'solution': sa_solution, 'history': sa_history},
        'ga': {'solution': ga_solution, 'history': ga_history},
        'config': {'grid_width': grid_width, 'grid_height': grid_height, 'num_robots': num_robots}
    }


def case_study_2_medium_grid():
    """
    Case Study 2: Medium 5x5 Grid
    - Purpose: Test scalability with moderate complexity
    - Grid: 5x5 (25 cells)
    - Robots: 3
    - Obstacles: 3 (scattered)
    - Expected: Good coverage with balanced workload
    """
    print("\n" + "="*70)
    print("CASE STUDY 2: Medium 5x5 Grid")
    print("="*70)
    print("Purpose: Test scalability with moderate complexity")
    print("-"*70)
    
    # Configuration
    grid_width = 5
    grid_height = 5
    num_robots = 3
    obstacles = [6, 12, 18]  # Scattered obstacles
    
    total_cells = grid_width * grid_height
    free_cells = [i for i in range(total_cells) if i not in obstacles]
    all_cells = create_grid_cells(grid_width, grid_height)
    
    print(f"Grid Size: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Free Cells: {len(free_cells)}")
    print(f"Obstacles: {obstacles}")
    
    # Run comparison
    start_time = time.time()
    sa_solution, ga_solution, sa_history, ga_history = compare_sa_vs_ga(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        sa_params={'initial_temp': 1500, 'cooling_rate': 0.95, 'max_iterations': 1000},
        ga_params={'population_size': 50, 'generations': 100, 'crossover_rate': 0.8, 'mutation_rate': 0.1}
    )
    end_time = time.time()
    
    print(f"\n⏱️  Total computation time: {end_time - start_time:.2f} seconds")
    
    # Generate report
    generate_comparison_report(
        sa_solution, ga_solution, sa_history, ga_history,
        case_study_name="case2_medium_5x5",
        save_dir="results/case2"
    )
    
    # Visualize both solutions
    print("\n📊 Visualizing SA Solution...")
    visualize_solution(sa_solution, grid_width, grid_height, "Case 2: SA Solution (5x5)")
    
    print("📊 Visualizing GA Solution...")
    visualize_solution(ga_solution, grid_width, grid_height, "Case 2: GA Solution (5x5)")
    
    return {
        'sa': {'solution': sa_solution, 'history': sa_history},
        'ga': {'solution': ga_solution, 'history': ga_history},
        'config': {'grid_width': grid_width, 'grid_height': grid_height, 'num_robots': num_robots}
    }


def case_study_3_large_grid():
    """
    Case Study 3: Large 7x7 Grid
    - Purpose: Test performance on larger problem
    - Grid: 7x7 (49 cells)
    - Robots: 4
    - Obstacles: 5 (forming a barrier)
    - Expected: Algorithms should handle complexity well
    """
    print("\n" + "="*70)
    print("CASE STUDY 3: Large 7x7 Grid")
    print("="*70)
    print("Purpose: Test performance on larger problem")
    print("-"*70)
    
    # Configuration
    grid_width = 7
    grid_height = 7
    num_robots = 4
    obstacles = [21, 22, 23, 24, 25]  # Horizontal barrier
    
    total_cells = grid_width * grid_height
    free_cells = [i for i in range(total_cells) if i not in obstacles]
    all_cells = create_grid_cells(grid_width, grid_height)
    
    print(f"Grid Size: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Free Cells: {len(free_cells)}")
    print(f"Obstacles: {obstacles}")
    
    # Run comparison
    start_time = time.time()
    sa_solution, ga_solution, sa_history, ga_history = compare_sa_vs_ga(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        sa_params={'initial_temp': 2000, 'cooling_rate': 0.95, 'max_iterations': 1500},
        ga_params={'population_size': 70, 'generations': 150, 'crossover_rate': 0.8, 'mutation_rate': 0.1}
    )
    end_time = time.time()
    
    print(f"\n⏱️  Total computation time: {end_time - start_time:.2f} seconds")
    
    # Generate report
    generate_comparison_report(
        sa_solution, ga_solution, sa_history, ga_history,
        case_study_name="case3_large_7x7",
        save_dir="results/case3"
    )
    
    # Visualize both solutions
    print("\n📊 Visualizing SA Solution...")
    visualize_solution(sa_solution, grid_width, grid_height, "Case 3: SA Solution (7x7)")
    
    print("📊 Visualizing GA Solution...")
    visualize_solution(ga_solution, grid_width, grid_height, "Case 3: GA Solution (7x7)")
    
    return {
        'sa': {'solution': sa_solution, 'history': sa_history},
        'ga': {'solution': ga_solution, 'history': ga_history},
        'config': {'grid_width': grid_width, 'grid_height': grid_height, 'num_robots': num_robots}
    }


def case_study_4_complex_obstacles():
    """
    Case Study 4: Complex 6x6 Grid with Many Obstacles
    - Purpose: Test robustness with constrained environment
    - Grid: 6x6 (36 cells)
    - Robots: 3
    - Obstacles: 8 (complex pattern)
    - Expected: Algorithms should find feasible paths around obstacles
    """
    print("\n" + "="*70)
    print("CASE STUDY 4: Complex 6x6 Grid with Many Obstacles")
    print("="*70)
    print("Purpose: Test robustness with constrained environment")
    print("-"*70)
    
    # Configuration
    grid_width = 6
    grid_height = 6
    num_robots = 3
    obstacles = [7, 8, 13, 14, 19, 20, 25, 26]  # Complex L-shaped pattern
    
    total_cells = grid_width * grid_height
    free_cells = [i for i in range(total_cells) if i not in obstacles]
    all_cells = create_grid_cells(grid_width, grid_height)
    
    print(f"Grid Size: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Free Cells: {len(free_cells)}")
    print(f"Obstacles: {obstacles}")
    
    # Run comparison
    start_time = time.time()
    sa_solution, ga_solution, sa_history, ga_history = compare_sa_vs_ga(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        sa_params={'initial_temp': 1800, 'cooling_rate': 0.95, 'max_iterations': 1200},
        ga_params={'population_size': 60, 'generations': 120, 'crossover_rate': 0.8, 'mutation_rate': 0.15}
    )
    end_time = time.time()
    
    print(f"\n⏱️  Total computation time: {end_time - start_time:.2f} seconds")
    
    # Generate report
    generate_comparison_report(
        sa_solution, ga_solution, sa_history, ga_history,
        case_study_name="case4_complex_6x6",
        save_dir="results/case4"
    )
    
    # Visualize both solutions
    print("\n📊 Visualizing SA Solution...")
    visualize_solution(sa_solution, grid_width, grid_height, "Case 4: SA Solution (6x6 Complex)")
    
    print("📊 Visualizing GA Solution...")
    visualize_solution(ga_solution, grid_width, grid_height, "Case 4: GA Solution (6x6 Complex)")
    
    return {
        'sa': {'solution': sa_solution, 'history': sa_history},
        'ga': {'solution': ga_solution, 'history': ga_history},
        'config': {'grid_width': grid_width, 'grid_height': grid_height, 'num_robots': num_robots}
    }


def run_all_case_studies():
    """
    Run all case studies sequentially
    """
    print("\n" + "="*70)
    print("RUNNING ALL CASE STUDIES")
    print("="*70)
    
    all_results = {}
    
    # Case Study 1
    print("\n🔬 Starting Case Study 1...")
    all_results['case1'] = case_study_1_small_grid()
    
    # Case Study 2
    print("\n🔬 Starting Case Study 2...")
    all_results['case2'] = case_study_2_medium_grid()
    
    # Case Study 3
    print("\n🔬 Starting Case Study 3...")
    all_results['case3'] = case_study_3_large_grid()
    
    # Case Study 4
    print("\n🔬 Starting Case Study 4...")
    all_results['case4'] = case_study_4_complex_obstacles()
    
    # Summary
    print("\n" + "="*70)
    print("ALL CASE STUDIES COMPLETED!")
    print("="*70)
    
    print("\n📊 SUMMARY OF RESULTS:")
    print("-"*70)
    
    for case_name, results in all_results.items():
        sa_metrics = results['sa']['solution'].get_all_performance_metrics()
        ga_metrics = results['ga']['solution'].get_all_performance_metrics()
        config = results['config']
        
        print(f"\n{case_name.upper()}:")
        print(f"  Grid: {config['grid_width']}x{config['grid_height']}, Robots: {config['num_robots']}")
        print(f"  SA Quality Index: {sa_metrics['solution_quality_index']:.4f}")
        print(f"  GA Quality Index: {ga_metrics['solution_quality_index']:.4f}")
        
        winner = 'SA' if sa_metrics['solution_quality_index'] > ga_metrics['solution_quality_index'] else 'GA'
        print(f"  Winner: {winner}")
    
    print("\n" + "="*70 + "\n")
    
    return all_results


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific case study
        case_num = sys.argv[1]
        
        if case_num == "1":
            case_study_1_small_grid()
        elif case_num == "2":
            case_study_2_medium_grid()
        elif case_num == "3":
            case_study_3_large_grid()
        elif case_num == "4":
            case_study_4_complex_obstacles()
        elif case_num == "all":
            run_all_case_studies()
        else:
            print(f"Unknown case study: {case_num}")
            print("Usage: python case_studies.py [1|2|3|4|all]")
    else:
        # Run all by default
        run_all_case_studies()
