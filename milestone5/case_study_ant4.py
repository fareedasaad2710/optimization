"""
Case Study 2: Medium Grid - Testing Ant Colony Optimization (ant4.py)
=======================================================================

This case study tests ant4.py (ACO with DARP + UF-STC) on a medium-sized grid.
Grid: 6x6
Robots: 3
Obstacles: Moderate number of obstacles
Purpose: Test scalability, load balancing, and multi-objective optimization
"""

import sys
import os
import time
import traceback
from datetime import datetime

# Add parent directory to path to import visualization and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ant4.py ACO implementation
from ant4 import AntColonyOptimization, create_grid_cells

# Import visualization functions
try:
    from visualization import visualize_solution, plot_convergence_history
    HAS_VISUALIZATION = True
except ImportError:
    print("⚠️  Visualization module not found. Visualization will be skipped.")
    HAS_VISUALIZATION = False

# Import problem formulation utilities
try:
    from problem_formulation import evaluate_solution
    HAS_PROBLEM_FORMULATION = True
except ImportError:
    print("⚠️  problem_formulation module not found. Some features may not work.")
    HAS_PROBLEM_FORMULATION = False


def convert_aco_solution_for_visualization(aco_solution, all_cells, free_cells, obstacles, grid_width, grid_height):
    """
    Convert ACOSolution from ant4.py to format compatible with visualization functions.
    
    Creates a wrapper object that has the same interface as RobotCoverageSolution
    expected by visualization functions.
    """
    class SolutionWrapper:
        def __init__(self, aco_sol):
            self.assignment = aco_sol.assignment
            self.paths = aco_sol.paths  # Already a dict {robot_id: [cells]}
            self.all_cells = aco_sol.all_cells
            self.free_cells = aco_sol.free_cells
            self.obstacles = aco_sol.obstacles
            self.grid_width = aco_sol.grid_width
            self.grid_height = aco_sol.grid_height
            
            # Add fitness information for visualization
            self.fitness = {
                'coverage_score': aco_sol.F1 if aco_sol.F1 is not None else 0,
                'balance_score': aco_sol.F2 if aco_sol.F2 is not None else 0,
                'robot_distances': list(aco_sol.Lr.values()) if aco_sol.Lr else []
            }
            
            # Calculate combined score for visualization
            # Normalize F1 (coverage) and F2 (imbalance) to create a combined score
            max_possible_coverage = len(free_cells) if free_cells else 1
            coverage_ratio = (aco_sol.F1 / max_possible_coverage) if max_possible_coverage > 0 else 0
            
            # For F2 (imbalance), lower is better, so invert it
            # Use a simple normalization: 1 / (1 + F2) to get a score between 0 and 1
            imbalance_score = 1.0 / (1.0 + aco_sol.F2) if aco_sol.F2 is not None and aco_sol.F2 > 0 else 1.0
            
            # Combined score: weighted average (70% coverage, 30% balance)
            self.combined_score = 0.7 * coverage_ratio + 0.3 * imbalance_score
            
            # Store original solution
            self.aco_solution = aco_sol
    
    return SolutionWrapper(aco_solution)


def convert_history_for_plotting(history):
    """
    Convert ACO history format to format expected by plot_convergence_history.
    
    ACO history format: [{'iteration': i, 'best_F1': f1, 'best_F2': f2, ...}, ...]
    Expected format: {'iteration': [...], 'best_score': [...], ...}
    """
    if not history:
        return {'iteration': [], 'best_score': []}
    
    converted = {
        'iteration': [h['iteration'] for h in history],
        'best_F1': [h.get('best_F1', 0) for h in history],
        'best_F2': [h.get('best_F2', 0) for h in history],
        'feasible_solutions': [h.get('feasible_solutions', 0) for h in history]
    }
    
    # Create combined score for plotting (lower = better, like GA/SA)
    # We want: lower score = better solution
    # F1 (coverage): higher is better → invert: (1 - F1/max_F1)
    # F2 (imbalance): lower is better → use directly: F2/max_F2
    max_F1 = max(converted['best_F1']) if converted['best_F1'] else 1
    max_F2 = max(converted['best_F2']) if converted['best_F2'] else 1
    
    if max_F1 > 0 and max_F2 > 0:
        # Score = imbalance_penalty + coverage_penalty
        # Lower score = better (less imbalance, more coverage)
        converted['best_score'] = [
            (f2 / max_F2) * 0.7 + (1.0 - f1 / max_F1) * 0.3
            for f1, f2 in zip(converted['best_F1'], converted['best_F2'])
        ]
    else:
        # Fallback: use F2 directly (lower is better)
        converted['best_score'] = converted['best_F2'] if converted['best_F2'] else [0] * len(converted['iteration'])
    
    return converted


def case_study_2_aco_ant4():
    """
    Case Study 2: Medium Grid - ACO Implementation (ant4.py)
    Grid: 6x6
    Robots: 3
    Obstacles: Moderate number of obstacles
    Purpose: Test ACO scalability, load balancing, and multi-objective optimization
    """
    print("\n" + "="*80)
    print("CASE STUDY 2: MEDIUM GRID (6x6, 3 Robots) - ANT COLONY OPTIMIZATION (ant4.py)")
    print("="*80)
    
    # Problem parameters
    grid_width, grid_height = 6, 6
    num_robots = 3
    obstacles = [1, 7, 13, 19, 25, 31]  # 6 obstacles
    
    # Create grid cells
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Total Cells: {grid_width * grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)} at cells: {obstacles}")
    print(f"  Robots: {num_robots}")
    
    # ACO algorithm parameters
    aco_params = {
        'num_ants': 15,           # Number of ants per iteration
        'initial_pheromone': 1.0,  # Initial pheromone level
        'rho': 0.5,               # Pheromone evaporation rate
        'alpha': 1.0,             # Pheromone importance
        'beta': 2.0,              # Heuristic importance (ant4 uses beta=2.0)
        'iterations': 50,         # Number of iterations
        'gamma': 1.0,             # Multi-objective workload balance weight
        'seed': 42                # Random seed for reproducibility
    }
    
    print(f"\nACO Parameters:")
    print(f"  • Number of ants: {aco_params['num_ants']}")
    print(f"  • Initial pheromone: {aco_params['initial_pheromone']}")
    print(f"  • Evaporation rate (ρ): {aco_params['rho']}")
    print(f"  • Alpha (α): {aco_params['alpha']}")
    print(f"  • Beta (β): {aco_params['beta']}")
    print(f"  • Gamma (γ): {aco_params['gamma']} (workload balance)")
    print(f"  • Iterations: {aco_params['iterations']}")
    print(f"  • Seed: {aco_params['seed']}")
    
    # Create results directory (different folder for ant4)
    results_dir = "milestone5/results/case_study_2_ant4"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"run_{timestamp}"
    
    # Run ACO with runtime measurement
    print("\n" + "="*80)
    print("🐜 RUNNING ANT COLONY OPTIMIZATION (ant4.py)")
    print("="*80)
    
    aco_start_time = time.time()
    
    # Initialize ACO
    aco = AntColonyOptimization(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        **aco_params
    )
    
    # Run optimization
    best_solution, history = aco.run(verbose=True)
    
    aco_end_time = time.time()
    aco_runtime = aco_end_time - aco_start_time
    
    print(f"\n⏱️  ACO Runtime: {aco_runtime:.2f} seconds ({aco_runtime/60:.2f} minutes)")
    
    # Display results
    if best_solution:
        print("\n" + "="*80)
        print("📊 RESULTS SUMMARY")
        print("="*80)
        print(f"\nBest Solution Found:")
        print(f"  • F1 (Coverage): {best_solution.F1}/{len(free_cells)} cells ({best_solution.F1/len(free_cells)*100:.1f}%)")
        print(f"  • F2 (Workload Imbalance): {best_solution.F2:.2f}")
        print(f"  • L̄ (Average Path Length): {best_solution.L_bar:.2f}" if hasattr(best_solution, 'L_bar') else "")
        print(f"\nRobot Path Details:")
        for robot_id in range(num_robots):
            path = best_solution.paths.get(robot_id, [])
            path_length = best_solution.Lr.get(robot_id, 0.0)
            print(f"  Robot {robot_id}:")
            print(f"    • Path length: {len(path)} cells")
            print(f"    • Distance (Lr): {path_length:.2f}")
            if len(path) <= 15:
                print(f"    • Path: {path}")
            else:
                print(f"    • Path (first 10): {path[:10]}...")
                print(f"    • Path (last 5): ...{path[-5:]}")
        
        # Check feasibility
        from ant4 import is_solution_feasible
        is_feasible, violations = is_solution_feasible(best_solution)
        print(f"\n  Feasibility Check:")
        if is_feasible:
            print(f"    ✅ Solution is feasible (no constraint violations)")
        else:
            print(f"    ❌ Solution has {len(violations)} violations:")
            for v in violations[:5]:  # Show first 5
                print(f"      - {v}")
            if len(violations) > 5:
                print(f"      ... and {len(violations) - 5} more")
    else:
        print("\n❌ No feasible solution found!")
        return None
    
    # Generate visualizations
    if HAS_VISUALIZATION and best_solution:
        print("\n" + "="*80)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*80)
        
        try:
            # Convert solution for visualization
            viz_solution = convert_aco_solution_for_visualization(
                best_solution, all_cells, free_cells, obstacles, grid_width, grid_height
            )
            
            # Visualize solution with unique filename
            solution_path = f"{results_dir}/{run_prefix}_aco_solution.png"
            visualize_solution(
                viz_solution,
                title=f"Case Study 2: ACO Best Solution (ant4.py) - {timestamp}",
                save_path=solution_path
            )
            print(f"✅ ACO solution visualization saved: {solution_path}")
            
            # Plot convergence history with unique filename
            if history:
                converted_history = convert_history_for_plotting(history)
                convergence_path = f"{results_dir}/{run_prefix}_aco_convergence.png"
                plot_convergence_history(
                    converted_history,
                    title=f"Case Study 2: ACO Convergence (ant4.py) - {timestamp}",
                    save_path=convergence_path
                )
                print(f"✅ ACO convergence plot saved: {convergence_path}")
            
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
            traceback.print_exc()
    
    # Generate detailed results report
    print("\n" + "="*80)
    print("📄 GENERATING DETAILED REPORT")
    print("="*80)
    
    try:
        report_path = f"{results_dir}/{run_prefix}_results_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CASE STUDY 2: MEDIUM GRID - ACO RESULTS (ant4.py)\n")
            f.write("="*80 + "\n\n")
            
            f.write("Problem Configuration:\n")
            f.write(f"  Grid Size: {grid_width}x{grid_height}\n")
            f.write(f"  Free Cells: {len(free_cells)}\n")
            f.write(f"  Obstacles: {obstacles}\n")
            f.write(f"  Robots: {num_robots}\n\n")
            
            f.write("ACO Parameters:\n")
            for key, value in aco_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            if best_solution:
                f.write("Best Solution:\n")
                f.write(f"  F1 (Coverage): {best_solution.F1}/{len(free_cells)} cells\n")
                f.write(f"  F2 (Workload Imbalance): {best_solution.F2:.2f}\n")
                if hasattr(best_solution, 'L_bar'):
                    f.write(f"  L̄ (Average Path Length): {best_solution.L_bar:.2f}\n")
                f.write("\n")
                
                f.write("Robot Paths:\n")
                for robot_id in range(num_robots):
                    path = best_solution.paths.get(robot_id, [])
                    path_length = best_solution.Lr.get(robot_id, 0.0)
                    f.write(f"  Robot {robot_id}:\n")
                    f.write(f"    Path length: {len(path)} cells\n")
                    f.write(f"    Distance (Lr): {path_length:.2f}\n")
                    f.write(f"    Path: {path}\n\n")
                
                # Feasibility
                from ant4 import is_solution_feasible
                is_feasible, violations = is_solution_feasible(best_solution)
                f.write(f"Feasibility: {'✅ Feasible' if is_feasible else '❌ Infeasible'}\n")
                if violations:
                    f.write(f"Violations ({len(violations)}):\n")
                    for v in violations:
                        f.write(f"  - {v}\n")
            
            f.write(f"\nRuntime: {aco_runtime:.2f} seconds ({aco_runtime/60:.2f} minutes)\n")
            
            if history:
                f.write("\nConvergence History:\n")
                f.write("Iteration | F1 (Coverage) | F2 (Imbalance) | Feasible Solutions\n")
                f.write("-" * 70 + "\n")
                for h in history:
                    f.write(f"{h['iteration']:9d} | {h.get('best_F1', 0):14d} | "
                           f"{h.get('best_F2', 0):15.2f} | {h.get('feasible_solutions', 0):17d}\n")
        
        print(f"✅ Detailed report saved to {report_path}")
        
    except Exception as e:
        print(f"⚠️  Report generation error: {e}")
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ CASE STUDY 2 (ACO ant4.py) COMPLETED!")
    print("="*80)
    print(f"\n⏱️  Runtime: {aco_runtime:.2f} seconds ({aco_runtime/60:.2f} minutes)")
    print(f"📁 Results saved in: {results_dir}/")
    
    return {
        'best_solution': best_solution,
        'history': history,
        'runtime': aco_runtime,
        'aco_params': aco_params
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ANT COLONY OPTIMIZATION - CASE STUDY 2 (ant4.py)")
    print("Testing ant4.py on Medium Grid (6x6, 3 Robots)")
    print("="*80)
    
    results = case_study_2_aco_ant4()
    
    if results:
        print("\n✅ Case study completed successfully!")
        print(f"Results saved in: milestone5/results/case_study_2_ant4/")
    else:
        print("\n❌ Case study failed!")



