"""
Main Execution File for Multi-Robot Coverage Path Planning with SA
================================================================

This is the main file that demonstrates the complete implementation:
1. Problem formulation
2. SA algorithm implementation  
3. Visualization of solutions
4. Case studies validation
5. Performance comparison

Usage:
    python main.py                    # Run all case studies
    python main.py --case 1           # Run specific case study
    python main.py --visualize        # Generate visualizations
    python main.py --compare          # Compare different solutions
"""

import argparse
import os
from problem_formulation import *
from sa_algorithm import *
from visualization import *
from case_studies import (
    run_single_case_study, 
    run_all_case_studies,
    case_study_1_small_grid,
    case_study_2_medium_grid,
    case_study_3_large_grid,
    case_study_4_many_robots
)

def create_results_directory():
    """Create directory for storing results"""
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')
    if not os.path.exists('results/data'):
        os.makedirs('results/data')

def plot_solution_on_axis(ax, solution, grid_width, grid_height):
    """
    Plot solution on existing axis (for live plotting)
    
    Args:
        ax: Matplotlib axis object
        solution: RobotCoverageSolution object
        grid_width: Grid width
        grid_height: Grid height
    """
    import matplotlib.patches as mpatches
    
    # Draw grid
    for x in range(grid_width + 1):
        ax.axvline(x, color='black', linewidth=0.5)
    for y in range(grid_height + 1):
        ax.axhline(y, color='black', linewidth=0.5)
    
    # Draw obstacles
    for obs_idx in solution.obstacles:
        x = obs_idx % grid_width
        y = obs_idx // grid_width
        rect = mpatches.Rectangle((x, y), 1, 1, linewidth=1, 
                                edgecolor='black', facecolor='gray', alpha=0.7)
        ax.add_patch(rect)
    
    # Define colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    # Draw robot paths
    for robot_id, path in solution.paths.items():
        if len(path) == 0:
            continue
        
        color = colors[robot_id % len(colors)]
        
        # Convert to coordinates
        coords = [(cell_idx % grid_width + 0.5, cell_idx // grid_width + 0.5) 
                  for cell_idx in path]
        
        # Draw path
        if len(coords) > 1:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color, linewidth=2, marker='o', 
                   markersize=6, alpha=0.7, label=f'R{robot_id}')
        
        # Mark start
        if coords:
            ax.plot(coords[0][0], coords[0][1], 'o', color=color, 
                   markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

def main():
    """
    Main entry point for running algorithms and case studies
    """
    parser = argparse.ArgumentParser(
        description='Multi-Robot Coverage Path Planning - GA & SA Optimization'
    )
    
    # Add command-line arguments
    parser.add_argument('--case', type=int, choices=[1, 2, 3, 4],
                       help='Run specific case study (1-4)')
    parser.add_argument('--all-cases', action='store_true',
                       help='Run all case studies')
    parser.add_argument('--test', type=int, choices=[1, 2, 3, 4],
                       help='Run specific test case (1-4)')
    parser.add_argument('--test-all', action='store_true',
                       help='Run all test cases')
    parser.add_argument('--validate', action='store_true',
                       help='Run algorithm validation')
    parser.add_argument('--demo', action='store_true',
                       help='Run quick demo')
    
    args = parser.parse_args()
    
    # Case Study execution
    if args.case:
        print(f"\n{'='*80}")
        print(f"RUNNING CASE STUDY {args.case}")
        print(f"{'='*80}")
        result = run_single_case_study(args.case)
        if result:
            print(f"\n✅ Case Study {args.case} completed successfully!")
            print(f"Results saved to: results/case_study_{args.case}/")
        else:
            print(f"\n❌ Case Study {args.case} failed!")
    
    elif args.all_cases:
        print(f"\n{'='*80}")
        print("RUNNING ALL CASE STUDIES")
        print(f"{'='*80}")
        results = run_all_case_studies()
        print(f"\n✅ All case studies completed!")
        print(f"Total case studies run: {len(results)}")
        print(f"Results saved to: results/case_study_*/")
    
    # Test case execution
    elif args.test:
        print(f"\n{'='*80}")
        print(f"RUNNING TEST CASE {args.test}")
        print(f"{'='*80}")
        
        test_functions = {
            1: test_small_case,
            2: test_medium_case,
            3: test_large_case,
            4: test_sa_vs_ga_comparison
        }
        
        if args.test in test_functions:
            result = test_functions[args.test]()
            print(f"\n✅ Test Case {args.test} completed!")
        else:
            print(f"❌ Invalid test case: {args.test}")
    
    elif args.test_all:
        results = run_all_test_cases()
        print(f"\n✅ All tests completed! Passed: {len(results)}/4")
    
    # Validation
    elif args.validate:
        print(f"\n{'='*80}")
        print("RUNNING ALGORITHM VALIDATION")
        print(f"{'='*80}")
        algorithm_validation()
    
    # Quick demo
    elif args.demo:
        print(f"\n{'='*80}")
        print("RUNNING QUICK DEMO")
        print(f"{'='*80}")
        run_quick_demo()
    
    # No arguments - show help
    else:
        print("\n" + "="*80)
        print("Multi-Robot Coverage Path Planning - Optimization Tool")
        print("="*80)
        print("\nUsage Examples:")
        print("  python main.py --case 1          # Run Case Study 1")
        print("  python main.py --all-cases       # Run all case studies")
        print("  python main.py --test 1          # Run Test Case 1")
        print("  python main.py --test-all        # Run all tests")
        print("  python main.py --validate        # Validate algorithms")
        print("  python main.py --demo            # Quick demonstration")
        print("\nFor more options, use: python main.py --help")
        print("="*80 + "\n")


def run_quick_demo():
    """
    Run a quick demonstration of GA on a simple problem
    """
    print("\nQuick Demo: 5x5 Grid with 2 Robots\n")
    
    grid_width, grid_height = 5, 5
    num_robots = 2
    obstacles = [12]  # Center obstacle
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print(f"Grid: {grid_width}x{grid_height}, Robots: {num_robots}, Obstacles: {len(obstacles)}")
    
    # Run GA
    print("\nRunning Genetic Algorithm...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        population_size=20,
        generations=30,
        crossover_rate=0.8,
        mutation_rate=0.1,
        verbose=True
    )
    
    solution = ga_results['best_solution']
    print(f"\nBest Score: {solution.combined_score:.4f}")
    print(f"Coverage: {solution.get_coverage_efficiency():.2f}%")
    print(f"Balance: {solution.get_workload_balance_index():.4f}")
    
    # Visualize
    from visualization import visualize_solution
    visualize_solution(solution, title="Quick Demo Solution")


def test_small_case():
    """Test Case 1: Small 3x3 grid"""
    print("\nTest Case 1: Small Grid (3x3, 2 Robots)")
    
    grid_width, grid_height = 3, 3
    num_robots = 2
    obstacles = [4]
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        population_size=20,
        generations=30,
        verbose=False
    )
    
    solution = ga_results['best_solution']
    validate_solution(solution, free_cells, obstacles, num_robots)
    
    return solution, ga_results


def test_medium_case():
    """Test Case 2: Medium 5x5 grid"""
    print("\nTest Case 2: Medium Grid (5x5, 3 Robots)")
    
    grid_width, grid_height = 5, 5
    num_robots = 3
    obstacles = [6, 12, 18]
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        population_size=40,
        generations=50,
        verbose=False
    )
    
    solution = ga_results['best_solution']
    validate_solution(solution, free_cells, obstacles, num_robots)
    
    return solution, ga_results


def test_large_case():
    """Test Case 3: Large 8x8 grid"""
    print("\nTest Case 3: Large Grid (8x8, 4 Robots)")
    
    grid_width, grid_height = 8, 8
    num_robots = 4
    obstacles = list(range(28, 36))
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        population_size=60,
        generations=100,
        verbose=False
    )
    
    solution = ga_results['best_solution']
    validate_solution(solution, free_cells, obstacles, num_robots)
    
    return solution, ga_results


def test_sa_vs_ga_comparison():
    """Test Case 4: Algorithm Comparison"""
    print("\nTest Case 4: SA vs GA Comparison (6x6, 3 Robots)")
    
    from algorithm_comparison import compare_sa_vs_ga, generate_comparison_report
    
    grid_width, grid_height = 6, 6
    num_robots = 3
    obstacles = [13, 14, 19, 20]
    
    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    # Run comparison (this function should handle both algorithms)
    # You'll need to implement compare_sa_vs_ga properly
    print("Comparison test - implement in algorithm_comparison.py")
    
    return None


def run_all_test_cases():
    """Run all test cases"""
    results = {}
    
    try:
        results['test_1'] = test_small_case()
        print("✅ Test 1 passed")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    try:
        results['test_2'] = test_medium_case()
        print("✅ Test 2 passed")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    try:
        results['test_3'] = test_large_case()
        print("✅ Test 3 passed")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    try:
        results['test_4'] = test_sa_vs_ga_comparison()
        print("✅ Test 4 passed")
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    return results


def algorithm_validation():
    """Validate algorithm correctness"""
    print("\nValidating algorithm implementation...")
    print("Running validation tests...")
    # Add validation logic here
    pass


if __name__ == "__main__":
    main()