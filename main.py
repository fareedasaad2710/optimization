"""
Main Execution File for Multi-Robot Coverage Path Planning with GA & SA
======================================================================

This file:
1. Defines test cases (small, medium, large, comparison)
2. Runs BOTH GA and SA for each test
3. Saves figures (paths + convergence) into per-test folders:
      results/test_*/figures
4. Provides CLI options to run specific cases / tests / demos

Usage:
    python main.py --test 1        # Run Test Case 1 (GA + SA + images)
    python main.py --test-all      # Run all tests
    python main.py --case 1        # Run case study (if implemented)
    python main.py --demo          # Quick GA demo
"""

import argparse
import os

import matplotlib.pyplot as plt

from problem_formulation import *
from sa_algorithm import simulated_annealing, print_sa_results
from visualization import *  # if you still want to use your existing GA visual tools
from case_studies import (
    run_single_case_study,
    run_all_case_studies,
    case_study_1_small_grid,
    case_study_2_medium_grid,
    case_study_3_large_grid,
    case_study_4_many_robots,
)
from GA import genetic_algorithm  # adjust import if your GA file/module is named differently


# ----------------------------------------------------------------------
# RESULTS FOLDERS
# ----------------------------------------------------------------------

def create_results_directory():
    """Create root directory for storing results (legacy)."""
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')
    if not os.path.exists('results/data'):
        os.makedirs('results/data')


def create_test_results_directory(test_name: str):
    """
    Create per-test folders:

        results/<test_name>/figures
        results/<test_name>/data

    Returns: (base_dir, fig_dir, data_dir)
    """
    base_dir = os.path.join('results', test_name)
    fig_dir = os.path.join(base_dir, 'figures')
    data_dir = os.path.join(base_dir, 'data')

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    return base_dir, fig_dir, data_dir


# ----------------------------------------------------------------------
# PLOTTING HELPERS
# ----------------------------------------------------------------------

def plot_solution_on_axis(ax, solution, grid_width, grid_height):
    """
    Plot solution on existing axis.

    Args:
        ax: Matplotlib axis object
        solution: RobotCoverageSolution-like object
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
        rect = mpatches.Rectangle(
            (x, y), 1, 1,
            linewidth=1,
            edgecolor='black',
            facecolor='gray',
            alpha=0.7
        )
        ax.add_patch(rect)

    # Define colors
    colors = [
        'red', 'blue', 'green', 'orange',
        'purple', 'cyan', 'magenta', 'yellow'
    ]

    # Draw robot paths
    for robot_id, path in solution.paths.items() if isinstance(solution.paths, dict) else enumerate(solution.paths):
        if len(path) == 0:
            continue

        color = colors[robot_id % len(colors)]

        # Convert to coordinates
        coords = [
            (cell_idx % grid_width + 0.5, cell_idx // grid_width + 0.5)
            for cell_idx in path
        ]

        # Draw path
        if len(coords) > 1:
            xs, ys = zip(*coords)
            ax.plot(
                xs, ys,
                color=color,
                linewidth=2,
                marker='o',
                markersize=6,
                alpha=0.7,
                label=f'R{robot_id}',
            )

        # Mark start
        if coords:
            ax.plot(
                coords[0][0], coords[0][1],
                'o',
                color=color,
                markersize=10,
                markeredgecolor='black',
                markeredgewidth=2,
            )

    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def save_test_case_figures(
    test_name: str,
    grid_width: int,
    grid_height: int,
    ga_solution,
    sa_solution,
    sa_history=None,
):
    """
    Save GA & SA figures for a given test case into:

        results/<test_name>/figures/

    - GA best path: ga_best_solution.png
    - SA best path: sa_best_solution.png
    - SA convergence: sa_convergence.png
    """
    _, fig_dir, _ = create_test_results_directory(test_name)

    # --- GA path figure ---
    if ga_solution is not None:
        fig, ax = plt.subplots()
        plot_solution_on_axis(ax, ga_solution, grid_width, grid_height)
        ax.set_title(f"GA Best Solution - {test_name}")
        ga_fig_path = os.path.join(fig_dir, "ga_best_solution.png")
        plt.savefig(ga_fig_path, bbox_inches='tight')
        plt.close(fig)
        print(f"   ➜ GA path figure saved to: {ga_fig_path}")

    # --- SA path figure ---
    if sa_solution is not None:
        fig, ax = plt.subplots()
        plot_solution_on_axis(ax, sa_solution, grid_width, grid_height)
        ax.set_title(f"SA Best Solution - {test_name}")
        sa_fig_path = os.path.join(fig_dir, "sa_best_solution.png")
        plt.savefig(sa_fig_path, bbox_inches='tight')
        plt.close(fig)
        print(f"   ➜ SA path figure saved to: {sa_fig_path}")

    # --- SA convergence figure ---
    if sa_history is not None and 'iteration' in sa_history:
        fig, ax = plt.subplots()
        ax.plot(
            sa_history['iteration'],
            sa_history['best_score'],
            label='Best Score',
        )
        ax.plot(
            sa_history['iteration'],
            sa_history['current_score'],
            label='Current Score',
            alpha=0.5,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective J")
        ax.set_title(f"SA Convergence - {test_name}")
        ax.legend()
        sa_conv_path = os.path.join(fig_dir, "sa_convergence.png")
        plt.savefig(sa_conv_path, bbox_inches='tight')
        plt.close(fig)
        print(f"   ➜ SA convergence figure saved to: {sa_conv_path}")


# ----------------------------------------------------------------------
# TEST CASES (GA + SA + FIGURES)
# ----------------------------------------------------------------------

def test_small_case():
    """Test Case 1: Small 3x3 grid (GA + SA + images)."""
    print("\nTest Case 1: Small Grid (3x3, 2 Robots)")
    test_name = "test_1_small_grid"

    grid_width, grid_height = 3, 3
    num_robots = 2
    obstacles = [4]  # middle cell

    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]

    # ----- GA -----
    print(" -> Running Genetic Algorithm...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        population_size=20,
        generations=30,
        verbose=False,
    )
    ga_solution = ga_results['best_solution']

    # ----- SA -----
    print(" -> Running Simulated Annealing...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        initial_temp=1000,
        cooling_rate=0.95,
        max_iterations=1000,
    )

    # ----- Save Figures -----
    save_test_case_figures(
        test_name,
        grid_width,
        grid_height,
        ga_solution,
        sa_solution,
        sa_history,
    )

    return {
        'ga_solution': ga_solution,
        'ga_results': ga_results,
        'sa_solution': sa_solution,
        'sa_history': sa_history,
    }


def test_medium_case():
    """Test Case 2: Medium 5x5 grid (GA + SA + images)."""
    print("\nTest Case 2: Medium Grid (5x5, 3 Robots)")
    test_name = "test_2_medium_grid"

    grid_width, grid_height = 5, 5
    num_robots = 3
    obstacles = [6, 12, 18]

    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]

    # ----- GA -----
    print(" -> Running Genetic Algorithm...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        population_size=40,
        generations=50,
        verbose=False,
    )
    ga_solution = ga_results['best_solution']
    validate_solution(ga_solution, free_cells, obstacles, num_robots)

    # ----- SA -----
    print(" -> Running Simulated Annealing...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        initial_temp=1000,
        cooling_rate=0.95,
        max_iterations=1500,
    )

    # ----- Save Figures -----
    save_test_case_figures(
        test_name,
        grid_width,
        grid_height,
        ga_solution,
        sa_solution,
        sa_history,
    )

    return {
        'ga_solution': ga_solution,
        'ga_results': ga_results,
        'sa_solution': sa_solution,
        'sa_history': sa_history,
    }


def test_large_case():
    """Test Case 3: Large 8x8 grid (GA + SA + images)."""
    print("\nTest Case 3: Large Grid (8x8, 4 Robots)")
    test_name = "test_3_large_grid"

    grid_width, grid_height = 8, 8
    num_robots = 4
    obstacles = list(range(28, 36))  # middle band

    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]

    # ----- GA -----
    print(" -> Running Genetic Algorithm...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        population_size=60,
        generations=100,
        verbose=False,
    )
    ga_solution = ga_results['best_solution']
    validate_solution(ga_solution, free_cells, obstacles, num_robots)

    # ----- SA -----
    print(" -> Running Simulated Annealing...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        initial_temp=1000,
        cooling_rate=0.97,
        max_iterations=2000,
    )

    # ----- Save Figures -----
    save_test_case_figures(
        test_name,
        grid_width,
        grid_height,
        ga_solution,
        sa_solution,
        sa_history,
    )

    return {
        'ga_solution': ga_solution,
        'ga_results': ga_results,
        'sa_solution': sa_solution,
        'sa_history': sa_history,
    }


def test_sa_vs_ga_comparison():
    """
    Test Case 4: Explicit SA vs GA comparison (6x6, 3 robots).

    Saves:
      - GA best path
      - SA best path
      - SA convergence
      - A side-by-side GA vs SA comparison figure
    """
    print("\nTest Case 4: SA vs GA Comparison (6x6, 3 Robots)")
    test_name = "test_4_sa_vs_ga"

    grid_width, grid_height = 6, 6
    num_robots = 3
    obstacles = [13, 14, 19, 20]

    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]

    # ----- GA -----
    print(" -> Running Genetic Algorithm...")
    ga_results = genetic_algorithm(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        population_size=50,
        generations=80,
        verbose=False,
    )
    ga_solution = ga_results['best_solution']

    # ----- SA -----
    print(" -> Running Simulated Annealing...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        initial_temp=1000,
        cooling_rate=0.96,
        max_iterations=1500,
    )

    # ----- Per-algorithm figures -----
    save_test_case_figures(
        test_name,
        grid_width,
        grid_height,
        ga_solution,
        sa_solution,
        sa_history,
    )

    # ----- Side-by-side comparison figure -----
    _, fig_dir, _ = create_test_results_directory(test_name)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    plot_solution_on_axis(ax1, ga_solution, grid_width, grid_height)
    ax1.set_title("GA Best Solution")

    plot_solution_on_axis(ax2, sa_solution, grid_width, grid_height)
    ax2.set_title("SA Best Solution")

    comp_path = os.path.join(fig_dir, "ga_vs_sa_paths.png")
    plt.tight_layout()
    plt.savefig(comp_path, bbox_inches='tight')
    plt.close(fig)

    print(f"   ➜ GA vs SA comparison figure saved to: {comp_path}")

    return {
        'ga_solution': ga_solution,
        'ga_results': ga_results,
        'sa_solution': sa_solution,
        'sa_history': sa_history,
    }


def run_all_test_cases():
    """Run all test cases and report basic status."""
    results = {}

    try:
        results['test_1'] = test_small_case()
        print("✅ Test 1 completed")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    try:
        results['test_2'] = test_medium_case()
        print("✅ Test 2 completed")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    try:
        results['test_3'] = test_large_case()
        print("✅ Test 3 completed")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")

    try:
        results['test_4'] = test_sa_vs_ga_comparison()
        print("✅ Test 4 completed")
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")

    return results


# ----------------------------------------------------------------------
# VALIDATION / DEMO
# ----------------------------------------------------------------------

def algorithm_validation():
    """Placeholder for extra validation logic, if needed."""
    print("\nValidating algorithm implementation...")
    print("Running validation tests (you can hook test_* here if desired)...")
    # e.g., you could call run_all_test_cases() here if that’s what you want.


def run_quick_demo():
    """
    Run a quick demonstration of GA on a simple problem (unchanged).
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
        verbose=True,
    )

    solution = ga_results['best_solution']
    print(f"\nBest Score: {solution.combined_score:.4f}")
    print(f"Coverage: {solution.get_coverage_efficiency():.2f}%")
    print(f"Balance: {solution.get_workload_balance_index():.4f}")

    # Visualize via your visualization module (if you want)
    from visualization import visualize_solution
    visualize_solution(solution, title="Quick Demo Solution")


# ----------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------

def main():
    """
    Main entry point for running algorithms and case studies.
    """
    create_results_directory()

    parser = argparse.ArgumentParser(
        description='Multi-Robot Coverage Path Planning - GA & SA Optimization'
    )

    # Command-line arguments
    parser.add_argument(
        '--case',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run specific case study (1-4)',
    )
    parser.add_argument(
        '--all-cases',
        action='store_true',
        help='Run all case studies',
    )
    parser.add_argument(
        '--test',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run specific test case (1-4) [GA + SA + images]',
    )
    parser.add_argument(
        '--test-all',
        action='store_true',
        help='Run all test cases (GA + SA + images)',
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run algorithm validation',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run quick GA demo',
    )

    args = parser.parse_args()

    # Case Study execution
    if args.case:
        print(f"\n{'=' * 80}")
        print(f"RUNNING CASE STUDY {args.case}")
        print(f"{'=' * 80}")
        result = run_single_case_study(args.case)
        if result:
            print(f"\n✅ Case Study {args.case} completed successfully!")
            print(f"Results saved to: results/case_study_{args.case}/")
        else:
            print(f"\n❌ Case Study {args.case} failed!")

    elif args.all_cases:
        print(f"\n{'=' * 80}")
        print("RUNNING ALL CASE STUDIES")
        print(f"{'=' * 80}")
        results = run_all_case_studies()
        print(f"\n✅ All case studies completed!")
        print(f"Total case studies run: {len(results)}")
        print(f"Results saved to: results/case_study_*/")

    # Test case execution (GA + SA + images)
    elif args.test:
        print(f"\n{'=' * 80}")
        print(f"RUNNING TEST CASE {args.test}")
        print(f"{'=' * 80}")

        test_functions = {
            1: test_small_case,
            2: test_medium_case,
            3: test_large_case,
            4: test_sa_vs_ga_comparison,
        }

        if args.test in test_functions:
            _ = test_functions[args.test]()
            print(f"\n✅ Test Case {args.test} completed!")
        else:
            print(f"❌ Invalid test case: {args.test}")

    elif args.test_all:
        results = run_all_test_cases()
        print(f"\n✅ All tests completed!")
        print(f"Results keys: {list(results.keys())}")

    # Validation
    elif args.validate:
        print(f"\n{'=' * 80}")
        print("RUNNING ALGORITHM VALIDATION")
        print(f"{'=' * 80}")
        algorithm_validation()

    # Quick demo
    elif args.demo:
        print(f"\n{'=' * 80}")
        print("RUNNING QUICK DEMO")
        print(f"{'=' * 80}")
        run_quick_demo()

    # No arguments - show help
    else:
        print("\n" + "=" * 80)
        print("Multi-Robot Coverage Path Planning - Optimization Tool")
        print("=" * 80)
        print("\nUsage Examples:")
        print("  python main.py --test 1         # Run Test Case 1 (GA + SA)")
        print("  python main.py --test-all       # Run all test cases")
        print("  python main.py --case 1         # Run Case Study 1")
        print("  python main.py --all-cases      # Run all case studies")
        print("  python main.py --validate       # Validate algorithms")
        print("  python main.py --demo           # Quick GA demonstration")
        print("\nFor more options, use: python main.py --help")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
