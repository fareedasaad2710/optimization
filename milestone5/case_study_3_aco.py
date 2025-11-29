"""
Case Study 3: Large Grid - Testing Ant Colony Optimization (ant3.py)
=====================================================================

This case study tests ant3.py (ACO with DARP + UF-STC) on a large grid.
Grid: 10x10
Robots: 5
Obstacles: Complex environment
Purpose: Test algorithm performance on larger problems
"""

import sys
import os
import time
import traceback
from datetime import datetime

# Add parent directory to path to import visualization and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ant3.py ACO implementation
from ant3 import AntColonyOptimization, create_grid_cells

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
    Convert ACOSolution from ant3.py to format compatible with visualization functions.
    
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


def case_study_3_aco():
    """
    Case Study 3: Large Grid - ACO Implementation
    Grid: 10x10
    Robots: 5
    Obstacles: Complex environment
    Purpose: Test algorithm performance on larger problems
    """
    print("\n" + "="*80)
    print("CASE STUDY 3: LARGE GRID (10x10, 5 Robots) - ANT COLONY OPTIMIZATION")
    print("="*80)
    
    # Problem parameters (from case_studies.py)
    grid_width, grid_height = 10, 10
    num_robots = 5
    obstacles = [11, 12, 13, 22, 32, 42, 52, 62, 67, 68, 69, 77, 88]  # 13 obstacles
    
    # Create grid cells
    all_cells = create_grid_cells(grid_width, grid_height)
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]
    
    print("\nProblem Configuration:")
    print(f"  Grid Size: {grid_width}x{grid_height}")
    print(f"  Total Cells: {grid_width * grid_height}")
    print(f"  Free Cells: {len(free_cells)}")
    print(f"  Obstacles: {len(obstacles)} at cells: {obstacles}")
    print(f"  Robots: {num_robots}")
    
    # ACO algorithm parameters (adjusted for larger problem)
    aco_params = {
        'num_ants': 20,           # More ants for larger problem
        'initial_pheromone': 1.0,  # Initial pheromone level
        'rho': 0.5,               # Pheromone evaporation rate
        'alpha': 1.0,             # Pheromone importance
        'beta': 1.0,              # Heuristic importance
        'iterations': 100,        # More iterations for larger problem
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
    
    # Create results directory
    os.makedirs("milestone5/results/case_study_3_aco", exist_ok=True)
    
    # Generate unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"run_{timestamp}"
    
    # Run ACO with runtime measurement
    print("\n" + "="*80)
    print("🐜 RUNNING ANT COLONY OPTIMIZATION")
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
        print("📊 OPTIMIZATION RESULTS")
        print("="*80)
        print(f"\nBest Solution Found:")
        print(f"  • F1 (Coverage): {best_solution.F1}/{len(free_cells)} cells ({100*best_solution.F1/len(free_cells):.1f}%)")
        print(f"  • F2 (Workload Imbalance): {best_solution.F2:.2f}")
        print(f"  • Robot Path Lengths: {best_solution.Lr}")
        
        # Calculate average path length
        if best_solution.Lr:
            avg_length = sum(best_solution.Lr.values()) / len(best_solution.Lr)
            print(f"  • Average Path Length: {avg_length:.2f}")
        
        # Show path lengths per robot
        print(f"\nPath Details:")
        for robot_id, path in best_solution.paths.items():
            path_length = best_solution.Lr.get(robot_id, 0.0)
            print(f"  Robot {robot_id}: {len(path)} cells, Length={path_length:.2f}")
    else:
        print("\n⚠️  No feasible solution found!")
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
            
            # Solution visualization
            visualize_solution(
                viz_solution,
                title=f"Case Study 3: ACO Best Solution (F1={best_solution.F1}, F2={best_solution.F2:.2f})",
                save_path=f"milestone5/results/case_study_3_aco/{run_prefix}_aco_solution.png"
            )
            print(f"✅ Solution visualization saved: {run_prefix}_aco_solution.png")
            
            # Convergence plot
            converted_history = convert_history_for_plotting(history)
            plot_convergence_history(
                converted_history,
                title="Case Study 3: ACO Convergence History",
                save_path=f"milestone5/results/case_study_3_aco/{run_prefix}_aco_convergence.png"
            )
            print(f"✅ Convergence plot saved: {run_prefix}_aco_convergence.png")
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
            traceback.print_exc()
    
    # Generate results report
    try:
        report_path = f"milestone5/results/case_study_3_aco/{run_prefix}_results_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CASE STUDY 3: LARGE GRID (10x10, 5 Robots) - ACO RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write("Problem Configuration:\n")
            f.write(f"  Grid Size: {grid_width}x{grid_height}\n")
            f.write(f"  Free Cells: {len(free_cells)}\n")
            f.write(f"  Obstacles: {len(obstacles)}\n")
            f.write(f"  Robots: {num_robots}\n\n")
            
            f.write("ACO Parameters:\n")
            for key, value in aco_params.items():
                f.write(f"  • {key}: {value}\n")
            f.write("\n")
            
            if best_solution:
                f.write("Best Solution:\n")
                f.write(f"  • F1 (Coverage): {best_solution.F1}/{len(free_cells)} cells\n")
                f.write(f"  • F2 (Workload Imbalance): {best_solution.F2:.2f}\n")
                f.write(f"  • Robot Path Lengths: {best_solution.Lr}\n\n")
                
                f.write("Path Details:\n")
                for robot_id, path in best_solution.paths.items():
                    path_length = best_solution.Lr.get(robot_id, 0.0)
                    f.write(f"  Robot {robot_id}: {len(path)} cells, Length={path_length:.2f}\n")
                f.write("\n")
            
            f.write(f"Runtime: {aco_runtime:.2f} seconds ({aco_runtime/60:.2f} minutes)\n")
            f.write("\n")
            
            f.write("Convergence History:\n")
            for entry in history:
                f.write(f"  Iteration {entry['iteration']}: F1={entry.get('best_F1', 0)}, "
                       f"F2={entry.get('best_F2', 0):.2f}, "
                       f"Feasible={entry.get('feasible_solutions', 0)}/{entry.get('total_ants', 0)}\n")
        
        print(f"✅ Results report saved: {run_prefix}_results_report.txt")
        
    except Exception as e:
        print(f"⚠️ Report generation error: {e}")
        traceback.print_exc()
    
    print("\n✅ Case Study 3 ACO completed!")
    print(f"⏱️  Runtime: {aco_runtime:.2f} seconds ({aco_runtime/60:.2f} minutes)")
    
    return {
        'solution': best_solution,
        'history': history,
        'runtime': aco_runtime
    }


if __name__ == "__main__":
    try:
        results = case_study_3_aco()
        if results:
            print("\n✅ Case Study 3 execution completed successfully!")
        else:
            print("\n⚠️ Case Study 3 execution completed with warnings.")
    except Exception as e:
        print(f"\n❌ Error running Case Study 3: {e}")
        traceback.print_exc()

