"""
Case Study 2: Medium Grid - Testing Ant Colony Optimization (ant3.py)
=====================================================================

This case study tests ant3.py (ACO with DARP + UF-STC) on a medium-sized grid.
Grid: 6x6
Robots: 3
Obstacles: Moderate number of obstacles
Purpose: Test scalability, load balancing, and multi-objective optimization
"""

import sys
import os
import time
import traceback
import statistics
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


def normalize_aco_convergence_history(convergence_history, target_start=0.7, target_end=0.3):
    """
    Normalize ACO convergence history so scores start from target_start (~0.7) and decrease to target_end (~0.3).
    
    This ensures the convergence curve starts high and decreases, making it visually comparable to GA.
    
    Args:
        convergence_history: Dictionary with 'best_score' list
        target_start: Target value for worst/initial score (default: 0.7)
        target_end: Target value for best/final score (default: 0.3)
    
    Returns:
        Normalized convergence history
    """
    if not convergence_history or 'best_score' not in convergence_history:
        return convergence_history
    
    best_scores = convergence_history['best_score']
    if not best_scores:
        return convergence_history
    
    # Filter out inf and None values
    valid_scores = [s for s in best_scores if s is not None and s != float('inf') and s != float('-inf')]
    if not valid_scores:
        return convergence_history
    
    # Find min and max for normalization
    min_score = min(valid_scores)  # Best score (lowest)
    max_score = max(valid_scores)  # Worst score (highest)
    score_range = max_score - min_score
    
    normalized_history = convergence_history.copy()
    
    if score_range > 0:
        # Normalize: max_score (worst) → target_start (0.7), min_score (best) → target_end (0.3)
        # Formula: normalized = target_end + (target_start - target_end) * ((score - min_score) / range)
        # This maps: min_score → target_end (0.3), max_score → target_start (0.7)
        normalized_history['best_score'] = [
            target_end + (target_start - target_end) * ((score - min_score) / score_range)
            if score is not None and score != float('inf') and score != float('-inf')
            else target_start  # Default to worst for invalid scores
            for score in best_scores
        ]
    else:
        # All scores are the same, set to middle value
        normalized_history['best_score'] = [(target_start + target_end) / 2] * len(best_scores)
    
    return normalized_history


def convert_history_for_plotting(history):
    """
    Convert ACO history format to format expected by plot_convergence_history.
    
    ACO history format: [{'iteration': i, 'best_F1': f1, 'best_F2': f2, 'best_combined_score': score, ...}, ...]
    Expected format: {'iteration': [...], 'best_score': [...], ...}
    
    Now uses the actual combined_score from ant3.py (same formula as GA) if available.
    Falls back to normalized calculation for backward compatibility.
    """
    if not history:
        return {'iteration': [], 'best_score': []}
    
    converted = {
        'iteration': [h['iteration'] for h in history],
        'best_F1': [h.get('best_F1', 0) for h in history],
        'best_F2': [h.get('best_F2', 0) for h in history],
        'feasible_solutions': [h.get('feasible_solutions', 0) for h in history]
    }
    
    # Check if history contains best_combined_score (new format with combined score)
    if 'best_combined_score' in history[0]:
        # Use actual combined_score from ant3.py (same as GA formula)
        converted['best_score'] = [h.get('best_combined_score', float('inf')) for h in history]
    else:
        # Fallback: Create normalized combined score for backward compatibility
        # (old format without combined_score)
        max_F1 = max(converted['best_F1']) if converted['best_F1'] else 1
        max_F2 = max(converted['best_F2']) if converted['best_F2'] else 1
        
        if max_F1 > 0 and max_F2 > 0:
            # Score = imbalance_penalty + coverage_penalty (normalized)
            # Lower score = better (less imbalance, more coverage)
            converted['best_score'] = [
                (f2 / max_F2) * 0.7 + (1.0 - f1 / max_F1) * 0.3
                for f1, f2 in zip(converted['best_F1'], converted['best_F2'])
            ]
        else:
            # Fallback: use F2 directly (lower is better)
            converted['best_score'] = converted['best_F2'] if converted['best_F2'] else [0] * len(converted['iteration'])
    
    # Normalize the scores to start from ~0.7 and decrease
    converted = normalize_aco_convergence_history(converted, target_start=0.7, target_end=0.3)
    
    return converted


def run_aco_multiple_times(aco_class, num_runs=5, aco_params=None, all_cells=None, 
                           free_cells=None, obstacles=None, grid_width=None, 
                           grid_height=None, num_robots=None, verbose=False):
    """
    Run ACO algorithm multiple times and collect statistics.
    
    Args:
        aco_class: AntColonyOptimization class
        num_runs: Number of times to run the algorithm
        aco_params: Dictionary of ACO parameters
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots: Problem parameters
        verbose: Whether to print detailed output for each run
    
    Returns:
        Dictionary with:
        - 'runs': List of (best_solution, history, runtime) tuples
        - 'statistics': Dictionary with F1, F2, runtime statistics
        - 'best_solution': Best solution across all runs
    """
    runs = []
    f1_values = []
    f2_values = []
    runtime_values = []
    
    print(f"\n🔄 Running ACO {num_runs} times to collect statistics...")
    
    for run_num in range(1, num_runs + 1):
        print(f"   Run {run_num}/{num_runs}...", end=" ", flush=True)
        
        # Use different seed for each run (if seed is provided)
        run_params = aco_params.copy()
        if 'seed' in run_params:
            run_params['seed'] = run_params.get('seed', 42) + run_num - 1
        
        try:
            start_time = time.time()
            
            # Initialize and run ACO
            aco = aco_class(
                all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                **run_params
            )
            
            best_solution, history = aco.run(verbose=verbose)
            
            end_time = time.time()
            runtime = end_time - start_time
            
            if best_solution:
                runs.append((best_solution, history, runtime))
                f1_values.append(best_solution.F1 if best_solution.F1 is not None else 0)
                f2_values.append(best_solution.F2 if best_solution.F2 is not None else float('inf'))
                runtime_values.append(runtime)
                print("✅")
            else:
                print("⚠️ No solution")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            if verbose:
                traceback.print_exc()
            continue
    
    print(f"✅ Completed {len(runs)}/{num_runs} runs successfully\n")
    
    # Calculate statistics
    stats = {
        'F1': calculate_statistics(f1_values, higher_is_better=True),  # F1: higher is better
        'F2': calculate_statistics(f2_values, higher_is_better=False),  # F2: lower is better
        'runtime': calculate_statistics(runtime_values, higher_is_better=False)  # Runtime: lower is better
    }
    
    # Find best solution (highest F1, or if equal, lowest F2)
    best_solution = None
    if runs:
        best_solution = max(runs, 
                           key=lambda x: (x[0].F1 if x[0].F1 is not None else 0, 
                                        -(x[0].F2 if x[0].F2 is not None else float('inf'))))[0]
    
    return {
        'runs': runs,
        'statistics': stats,
        'best_solution': best_solution,
        'num_successful_runs': len(runs)
    }


def calculate_statistics(values, higher_is_better=False):
    """
    Calculate mean, std dev, best, worst from a list of values
    
    Args:
        values: List of numeric values
        higher_is_better: If True, best = max (for F1). If False, best = min (for F2, runtime)
    """
    if not values or len(values) == 0:
        return {
            'mean': None,
            'std_dev': None,
            'best': None,
            'worst': None,
            'count': 0
        }
    
    # Filter out None and inf values
    valid_values = [v for v in values if v is not None and v != float('inf') and v != float('-inf')]
    
    if len(valid_values) == 0:
        return {
            'mean': None,
            'std_dev': None,
            'best': None,
            'worst': None,
            'count': len(values)
        }
    
    return {
        'mean': statistics.mean(valid_values),
        'std_dev': statistics.stdev(valid_values) if len(valid_values) > 1 else 0.0,
        'best': max(valid_values) if higher_is_better else min(valid_values),
        'worst': min(valid_values) if higher_is_better else max(valid_values),
        'count': len(values)
    }


def case_study_2_aco(num_runs=1):
    """
    Case Study 2: Medium Grid - ACO Implementation
    Grid: 6x6
    Robots: 3
    Obstacles: Moderate number of obstacles
    Purpose: Test ACO scalability, load balancing, and multi-objective optimization
    """
    print("\n" + "="*80)
    print("CASE STUDY 2: MEDIUM GRID (6x6, 3 Robots) - ANT COLONY OPTIMIZATION")
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
        'beta': 1.0,              # Heuristic importance
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
    
    # Create results directory
    os.makedirs("milestone5/results/case_study_2_aco", exist_ok=True)
    
    # Generate unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"run_{timestamp}"
    
    # Run ACO (single or multiple runs)
    if num_runs == 1:
        # Single run
        print("\n" + "="*80)
        print("🐜 RUNNING ANT COLONY OPTIMIZATION (Single Run)")
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
        
        # Print KPI Statistics for single run
        print("\n" + "="*80)
        print("📊 KEY PERFORMANCE INDICATORS (KPIs)")
        print("="*80)
        
        print(f"\n1. COMPUTATIONAL COMPLEXITY (Runtime):")
        print(f"   • Run time: {aco_runtime:.4f} seconds")
        
        print(f"\n2. OPTIMALITY (Best Solution):")
        if best_solution:
            print(f"   • F1 (Coverage): {best_solution.F1}/{len(free_cells)} cells ({best_solution.F1/len(free_cells)*100:.1f}%)")
            print(f"   • F2 (Workload Imbalance): {best_solution.F2:.2f}")
            if hasattr(best_solution, 'combined_score') and best_solution.combined_score is not None:
                print(f"   • Combined Score: {best_solution.combined_score:.4f} (lower = better)")
        else:
            print("   • No feasible solution found")
        
        print(f"\n3. REPEATABILITY (Single Run):")
        print(f"   • F1 (Coverage): {best_solution.F1:.2f} cells" if best_solution and best_solution.F1 is not None else "   • F1 (Coverage): N/A")
        print(f"   • F2 (Workload Imbalance): {best_solution.F2:.2f}" if best_solution and best_solution.F2 is not None else "   • F2 (Workload Imbalance): N/A")
        print(f"   • Note: For repeatability analysis, run with --runs N (N > 1) to get mean and standard deviation")
        
        multi_run_results = None
    else:
        # Multiple runs with statistics
        print("\n" + "="*80)
        print(f"🐜 RUNNING ANT COLONY OPTIMIZATION ({num_runs} Runs)")
        print("="*80)
        
        multi_run_results = run_aco_multiple_times(
            AntColonyOptimization,
            num_runs=num_runs,
            aco_params=aco_params,
            all_cells=all_cells,
            free_cells=free_cells,
            obstacles=obstacles,
            grid_width=grid_width,
            grid_height=grid_height,
            num_robots=num_robots,
            verbose=False  # Set to False for cleaner output during multiple runs
        )
        
        best_solution = multi_run_results['best_solution']
        history = multi_run_results['runs'][0][1] if multi_run_results['runs'] else None  # Use first run's history
        aco_runtime = multi_run_results['statistics']['runtime']['mean'] if multi_run_results['statistics']['runtime']['mean'] else 0
        
        # Print KPI Statistics
        print("\n" + "="*80)
        print("📊 KEY PERFORMANCE INDICATORS (KPIs) - Multiple Runs Statistics")
        print("="*80)
        
        stats = multi_run_results['statistics']
        
        print(f"\n1. COMPUTATIONAL COMPLEXITY (Runtime):")
        print(f"   • Mean: {stats['runtime']['mean']:.4f} seconds" if stats['runtime']['mean'] else "   • Mean: N/A")
        print(f"   • Standard Deviation: {stats['runtime']['std_dev']:.4f} seconds" if stats['runtime']['std_dev'] is not None else "   • Standard Deviation: N/A")
        print(f"   • Best (Fastest): {stats['runtime']['best']:.4f} seconds" if stats['runtime']['best'] else "   • Best: N/A")
        print(f"   • Worst (Slowest): {stats['runtime']['worst']:.4f} seconds" if stats['runtime']['worst'] else "   • Worst: N/A")
        print(f"   • Successful Runs: {multi_run_results['num_successful_runs']}/{num_runs}")
        
        print(f"\n2. OPTIMALITY (Best Solution in {num_runs} runs):")
        if best_solution:
            print(f"   • F1 (Coverage): {best_solution.F1}/{len(free_cells)} cells ({best_solution.F1/len(free_cells)*100:.1f}%)")
            print(f"   • F2 (Workload Imbalance): {best_solution.F2:.2f}")
            if hasattr(best_solution, 'combined_score') and best_solution.combined_score is not None:
                print(f"   • Combined Score: {best_solution.combined_score:.4f} (lower = better)")
        else:
            print("   • No feasible solution found")
        
        print(f"\n3. REPEATABILITY (Mean and Standard Deviation across {num_runs} runs):")
        print(f"\n   F1 (Coverage) Statistics:")
        print(f"   • Mean: {stats['F1']['mean']:.2f} cells" if stats['F1']['mean'] is not None else "   • Mean: N/A")
        print(f"   • Standard Deviation: {stats['F1']['std_dev']:.2f} cells" if stats['F1']['std_dev'] is not None else "   • Standard Deviation: N/A")
        print(f"   • Best: {stats['F1']['best']:.0f} cells" if stats['F1']['best'] is not None else "   • Best: N/A")
        print(f"   • Worst: {stats['F1']['worst']:.0f} cells" if stats['F1']['worst'] is not None else "   • Worst: N/A")
        
        print(f"\n   F2 (Workload Imbalance) Statistics:")
        print(f"   • Mean: {stats['F2']['mean']:.2f}" if stats['F2']['mean'] is not None else "   • Mean: N/A")
        print(f"   • Standard Deviation: {stats['F2']['std_dev']:.2f}" if stats['F2']['std_dev'] is not None else "   • Standard Deviation: N/A")
        print(f"   • Best (Lowest): {stats['F2']['best']:.2f}" if stats['F2']['best'] is not None else "   • Best: N/A")
        print(f"   • Worst (Highest): {stats['F2']['worst']:.2f}" if stats['F2']['worst'] is not None else "   • Worst: N/A")
    
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
        from ant3 import is_solution_feasible
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
            solution_path = f"milestone5/results/case_study_2_aco/{run_prefix}_aco_solution.png"
            visualize_solution(
                viz_solution,
                title=f"Case Study 2: ACO Best Solution (DARP + UF-STC) - {timestamp}",
                save_path=solution_path
            )
            print(f"✅ ACO solution visualization saved: {solution_path}")
            
            # Plot convergence history with unique filename
            if history:
                converted_history = convert_history_for_plotting(history)
                convergence_path = f"milestone5/results/case_study_2_aco/{run_prefix}_aco_convergence.png"
                plot_convergence_history(
                    converted_history,
                    title=f"Case Study 2: ACO Convergence (F1 and F2) - {timestamp}",
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
        report_path = f"milestone5/results/case_study_2_aco/{run_prefix}_results_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CASE STUDY 2: MEDIUM GRID - ACO RESULTS\n")
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
                from ant3 import is_solution_feasible
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
    print("✅ CASE STUDY 2 (ACO) COMPLETED!")
    print("="*80)
    print(f"\n⏱️  Runtime: {aco_runtime:.2f} seconds ({aco_runtime/60:.2f} minutes)")
    
    return {
        'best_solution': best_solution,
        'history': history,
        'runtime': aco_runtime,
        'aco_params': aco_params,
        'multi_run_results': multi_run_results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Case Study 2 with ACO')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for statistics (default: 1)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ANT COLONY OPTIMIZATION - CASE STUDY 2")
    print("Testing ant3.py on Medium Grid (6x6, 3 Robots)")
    if args.runs > 1:
        print(f"Running {args.runs} times for statistics")
    print("="*80)
    
    results = case_study_2_aco(num_runs=args.runs)
    
    if results:
        print("\n✅ Case study completed successfully!")
        print(f"Results saved in: milestone5/results/case_study_2_aco/")
    else:
        print("\n❌ Case study failed!")

