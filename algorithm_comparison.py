from GA import genetic_algorithm, print_solution_summary, analyze_convergence
from sa_algorithm import simulated_annealing
import matplotlib.pyplot as plt
import numpy as np
import statistics

def convert_sa_history_to_standard(sa_history):
    """
    Convert SA history to standard format compatible with analyze_convergence
    
    SA returns: {'iteration': [...], 'best_score': [...], 'current_score': [...], 'temperature': [...]}
    GA expects: {'generation': [...], 'best_score': [...], 'avg_score': [...], 'worst_score': [...]}
    """
    if 'generation' in sa_history:
        # Already in standard format
        return sa_history
    
    # Convert SA format to standard format
    standard_history = {
        'generation': sa_history.get('iteration', list(range(len(sa_history['best_score'])))),
        'best_score': sa_history['best_score'],
        'avg_score': sa_history.get('current_score', sa_history['best_score']),  # Use current as avg proxy
        'worst_score': sa_history.get('current_score', sa_history['best_score']),  # Use current as worst proxy
        'best_coverage': sa_history.get('best_coverage', []),
        'best_balance': sa_history.get('best_balance', [])
    }
    
    return standard_history

def compare_sa_vs_ga(all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
                     sa_params=None, ga_params=None):
    """
    Compare SA and GA on same problem
    
    Args:
        sa_params: dict with SA parameters (initial_temp, cooling_rate, max_iterations)
        ga_params: dict with GA parameters (population_size, generations, crossover_rate, mutation_rate)
    """
    
    # Default parameters
    if sa_params is None:
        sa_params = {
            'initial_temp': 1000,
            'cooling_rate': 0.95,
            'max_iterations': 1000
        }
    
    if ga_params is None:
        ga_params = {
            'population_size': 50,
            'generations': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1
        }
    
    print("\n" + "="*70)
    print("COMPARING SA vs GA")
    print("="*70)
    
    # Run SA
    print("\n[1/2] Running Simulated Annealing...")
    sa_solution, sa_history = simulated_annealing(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        initial_temp=sa_params['initial_temp'],
        cooling_rate=sa_params['cooling_rate'],
        max_iterations=sa_params['max_iterations']
    )
    
    # Convert SA history to standard format
    sa_history = convert_sa_history_to_standard(sa_history)
    
    # Run GA
    print("\n[2/2] Running Genetic Algorithm...")
    ga_solution, ga_history = genetic_algorithm(
        all_cells, free_cells, obstacles, grid_width, grid_height, num_robots,
        population_size=ga_params['population_size'],
        generations=ga_params['generations'],
        crossover_rate=ga_params['crossover_rate'],
        mutation_rate=ga_params['mutation_rate']
    )
    
    # Compare metrics
    sa_metrics = sa_solution.get_all_performance_metrics()
    ga_metrics = ga_solution.get_all_performance_metrics()
    
    # Print comparison table
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: SA vs GA")
    print("="*70)
    print(f"{'Metric':<30} {'SA':>15} {'GA':>15} {'Winner':>10}")
    print("-"*70)
    
    metrics_to_compare = [
        ('Coverage Efficiency', 'coverage_efficiency', 'higher'),
        ('Workload Balance', 'workload_balance_index', 'higher'),
        ('Constraint Satisfaction', 'constraint_satisfaction_rate', 'higher'),
        ('Solution Quality', 'solution_quality_index', 'higher'),
        ('Combined Score', 'combined_score', 'lower')
    ]
    
    winners = {'SA': 0, 'GA': 0}
    
    for metric_name, metric_key, better in metrics_to_compare:
        sa_val = sa_metrics[metric_key]
        ga_val = ga_metrics[metric_key]
        
        if better == 'higher':
            winner = 'SA' if sa_val > ga_val else 'GA' if ga_val > sa_val else 'Tie'
        else:
            winner = 'SA' if sa_val < ga_val else 'GA' if ga_val < sa_val else 'Tie'
        
        if winner != 'Tie':
            winners[winner] += 1
        
        print(f"{metric_name:<30} {sa_val:>15.4f} {ga_val:>15.4f} {winner:>10}")
    
    print("-"*70)
    if winners['SA'] == winners['GA']:
        overall_winner = 'Tie'
    else:
        overall_winner = max(winners, key=winners.get)
    print(f"{'Overall Winner':<30} {'':>15} {'':>15} {overall_winner:>10}")
    print("="*70 + "\n")
    
    # Analyze convergence
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS COMPARISON")
    print("="*70)
    
    sa_conv_analysis = analyze_convergence(convert_sa_history_to_standard(sa_history))
    ga_conv_analysis = analyze_convergence(ga_history)
    
    print(f"\n{'Metric':<35} {'SA':>15} {'GA':>15}")
    print("-"*70)
    print(f"{'Total Iterations/Generations':<35} {sa_conv_analysis['total_generations']:>15} {ga_conv_analysis['total_generations']:>15}")
    print(f"{'Initial Score':<35} {sa_conv_analysis['initial_score']:>15.4f} {ga_conv_analysis['initial_score']:>15.4f}")
    print(f"{'Final Score':<35} {sa_conv_analysis['final_score']:>15.4f} {ga_conv_analysis['final_score']:>15.4f}")
    
    if sa_conv_analysis['improvement_percentage'] is not None and \
       ga_conv_analysis['improvement_percentage'] is not None:
        print(f"{'Improvement %':<35} "
              f"{sa_conv_analysis['improvement_percentage']:>14.2f}% "
              f"{ga_conv_analysis['improvement_percentage']:>14.2f}%")
    
    if sa_conv_analysis['converged_at_generation'] is not None:
        print(f"{'SA Converged at Iteration':<35} {sa_conv_analysis['converged_at_generation']:>15}")
    else:
        print(f"{'SA Converged at Iteration':<35} {'Still improving':>15}")
    
    if ga_conv_analysis['converged_at_generation'] is not None:
        print(f"{'GA Converged at Generation':<35} {ga_conv_analysis['converged_at_generation']:>15}")
    else:
        print(f"{'GA Converged at Generation':<35} {'Still improving':>15}")
    
    print("\n" + "="*70 + "\n")
    
    # Print detailed summaries
    print_solution_summary(sa_solution, sa_history, algorithm_name="SA")
    print_solution_summary(ga_solution, ga_history, algorithm_name="GA")
    
    # Plot convergence comparison
    try:
        print("📊 Generating convergence comparison plot...")
        plot_sa_vs_ga_convergence(sa_history, ga_history)
    except Exception as e:
        print(f"⚠️  Could not plot convergence: {e}")
    
    return sa_solution, ga_solution, sa_history, ga_history

def run_algorithm_multiple_times(algorithm_func, num_runs=5, *args, **kwargs):
    """
    Run an algorithm multiple times and collect all results
    
    Args:
        algorithm_func: Function to run (e.g., genetic_algorithm or simulated_annealing)
        num_runs: Number of times to run the algorithm
        *args, **kwargs: Arguments to pass to the algorithm function
    
    Returns:
        List of results from all runs
    """
    results = []
    print(f"\n🔄 Running {algorithm_func.__name__} {num_runs} times...")
    
    for run_num in range(1, num_runs + 1):
        print(f"   Run {run_num}/{num_runs}...", end=" ", flush=True)
        try:
            result = algorithm_func(*args, **kwargs)
            results.append(result)
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print(f"✅ Completed {len(results)}/{num_runs} runs successfully\n")
    return results

def safe_format_stat(value, format_str=".4f", default="N/A"):
    """
    Safely format a statistic value that might be None
    
    Args:
        value: The value to format (might be None)
        format_str: Format string (e.g., ".4f", ".2f")
        default: Default string if value is None
    
    Returns:
        Formatted string
    """
    if value is None or value == float('inf') or value == float('-inf'):
        return default
    try:
        return f"{value:{format_str}}"
    except (TypeError, ValueError):
        return default

def calculate_statistics(values):
    """
    Calculate statistics from a list of values
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with avg, std_dev, best, worst, count
    """
    if not values or len(values) == 0:
        return {
            'avg': None,
            'std_dev': None,
            'best': None,
            'worst': None,
            'count': 0
        }
    
    # Filter out None and inf values for statistics
    valid_values = [v for v in values if v is not None and v != float('inf') and v != float('-inf')]
    
    if len(valid_values) == 0:
        return {
            'avg': None,
            'std_dev': None,
            'best': None,
            'worst': None,
            'count': len(values)
        }
    
    return {
        'avg': statistics.mean(valid_values),
        'std_dev': statistics.stdev(valid_values) if len(valid_values) > 1 else 0.0,
        'best': min(valid_values),
        'worst': max(valid_values),
        'count': len(values)
    }

def generate_comparison_report(sa_results, ga_results, case_study_name="Comparison", num_runs=1):
    """
    Generate a comprehensive comparison report between SA and GA results
    
    Args:
        sa_results: Tuple or dict with (sa_solution, sa_history) or {'best_solution': ..., 'convergence_history': ...}
                   OR list of such results for multiple runs
        ga_results: Tuple or dict with (ga_solution, ga_history) or {'best_solution': ..., 'convergence_history': ...}
                   OR list of such results for multiple runs
        case_study_name: Name of the case study
        num_runs: Number of runs (for statistics display)
    
    Returns:
        Dictionary with comparison metrics
    """
    # Handle multiple runs vs single run
    is_multiple_runs = False
    sa_results_list = []
    ga_results_list = []
    
    # Check if we have multiple runs
    if isinstance(sa_results, list) and len(sa_results) > 0:
        is_multiple_runs = True
        sa_results_list = sa_results
    else:
        sa_results_list = [sa_results]
    
    if isinstance(ga_results, list) and len(ga_results) > 0:
        is_multiple_runs = True
        ga_results_list = ga_results
    else:
        ga_results_list = [ga_results]
    
    # Extract solutions and histories from all runs
    sa_solutions = []
    sa_histories = []
    ga_solutions = []
    ga_histories = []
    
    for sa_res in sa_results_list:
        if isinstance(sa_res, tuple):
            sa_sol, sa_hist = sa_res
        elif isinstance(sa_res, dict):
            sa_sol = sa_res.get('best_solution')
            sa_hist = sa_res.get('convergence_history', {})
        else:
            sa_sol = sa_res
            sa_hist = {}
        sa_solutions.append(sa_sol)
        sa_histories.append(sa_hist)
    
    for ga_res in ga_results_list:
        if isinstance(ga_res, tuple):
            ga_sol, ga_hist = ga_res
        elif isinstance(ga_res, dict):
            ga_sol = ga_res.get('best_solution')
            ga_hist = ga_res.get('convergence_history', {})
        else:
            ga_sol = ga_res
            ga_hist = {}
        ga_solutions.append(ga_sol)
        ga_histories.append(ga_hist)
    
    # Use first run for detailed comparison, or best solution if multiple runs
    if is_multiple_runs:
        # Find best solutions from all runs
        sa_scores_all = [s.combined_score if s and hasattr(s, 'combined_score') and s.combined_score is not None else float('inf') 
                        for s in sa_solutions]
        ga_scores_all = [s.combined_score if s and hasattr(s, 'combined_score') and s.combined_score is not None else float('inf') 
                        for s in ga_solutions]
        
        sa_best_idx = sa_scores_all.index(min(sa_scores_all)) if sa_scores_all else 0
        ga_best_idx = ga_scores_all.index(min(ga_scores_all)) if ga_scores_all else 0
        
        sa_solution = sa_solutions[sa_best_idx] if sa_best_idx < len(sa_solutions) else sa_solutions[0] if sa_solutions else None
        sa_history = sa_histories[sa_best_idx] if sa_best_idx < len(sa_histories) else sa_histories[0] if sa_histories else {}
        ga_solution = ga_solutions[ga_best_idx] if ga_best_idx < len(ga_solutions) else ga_solutions[0] if ga_solutions else None
        ga_history = ga_histories[ga_best_idx] if ga_best_idx < len(ga_histories) else ga_histories[0] if ga_histories else {}
    else:
        sa_solution = sa_solutions[0] if sa_solutions else None
        sa_history = sa_histories[0] if sa_histories else {}
        ga_solution = ga_solutions[0] if ga_solutions else None
        ga_history = ga_histories[0] if ga_histories else {}
    
    print(f"\n{'='*80}")
    print(f"ALGORITHM COMPARISON REPORT - {case_study_name}")
    if is_multiple_runs:
        print(f"📊 Statistics from {len(sa_results_list)} runs")
    print(f"{'='*80}\n")
    
    # Collect all scores for statistics
    sa_scores_all = [s.combined_score if s and hasattr(s, 'combined_score') and s.combined_score is not None else float('inf') 
                    for s in sa_solutions]
    ga_scores_all = [s.combined_score if s and hasattr(s, 'combined_score') and s.combined_score is not None else float('inf') 
                    for s in ga_solutions]
    
    sa_stats = calculate_statistics(sa_scores_all)
    ga_stats = calculate_statistics(ga_scores_all)
    
    # Performance Index 1: Solution Quality (Combined Score)
    print("📊 PERFORMANCE INDEX 1: SOLUTION QUALITY (Combined Score)")
    print("-" * 80)
    
    sa_score = sa_solution.combined_score if sa_solution and hasattr(sa_solution, 'combined_score') else float('inf')
    ga_score = ga_solution.combined_score if ga_solution and hasattr(ga_solution, 'combined_score') else float('inf')
    
    if is_multiple_runs:
        print(f"SA Statistics ({sa_stats['count']} runs):")
        print(f"   • Best:    {safe_format_stat(sa_stats['best'], '.4f')}")
        print(f"   • Average: {safe_format_stat(sa_stats['avg'], '.4f')}")
        print(f"   • Std Dev: {safe_format_stat(sa_stats['std_dev'], '.4f')}")
        print(f"   • Worst:   {safe_format_stat(sa_stats['worst'], '.4f')}")
        print(f"\nGA Statistics ({ga_stats['count']} runs):")
        print(f"   • Best:    {safe_format_stat(ga_stats['best'], '.4f')}")
        print(f"   • Average: {safe_format_stat(ga_stats['avg'], '.4f')}")
        print(f"   • Std Dev: {safe_format_stat(ga_stats['std_dev'], '.4f')}")
        print(f"   • Worst:   {safe_format_stat(ga_stats['worst'], '.4f')}")
        print(f"\n📌 Best Run Comparison:")
        print(f"   SA Best Score:  {safe_format_stat(sa_stats['best'], '.4f')}")
        print(f"   GA Best Score:  {safe_format_stat(ga_stats['best'], '.4f')}")
    else:
        print(f"SA Final Score:  {sa_score:.4f}")
        print(f"GA Final Score:  {ga_score:.4f}")
    
    # Compare using best scores if multiple runs, otherwise use single scores
    compare_sa = sa_stats['best'] if (is_multiple_runs and sa_stats['best'] is not None) else sa_score
    compare_ga = ga_stats['best'] if (is_multiple_runs and ga_stats['best'] is not None) else ga_score
    
    if compare_sa is not None and compare_ga is not None and compare_sa < compare_ga:
        improvement = ((compare_ga - compare_sa) / compare_ga) * 100 if compare_ga != 0 else 0
        print(f"✅ SA performs better by {improvement:.2f}%")
        winner_quality = "SA"
    elif compare_ga < compare_sa:
        improvement = ((compare_sa - compare_ga) / compare_sa) * 100 if compare_sa != 0 else 0
        print(f"✅ GA performs better by {improvement:.2f}%")
        winner_quality = "GA"
    else:
        print("🤝 Tie - Both algorithms achieved the same score")
        winner_quality = "TIE"
    
    # Performance Index 2: Coverage Efficiency
    print(f"\n📊 PERFORMANCE INDEX 2: COVERAGE EFFICIENCY")
    print("-" * 80)
    
    # Collect coverage statistics
    sa_coverages_all = [s.get_coverage_efficiency() if s and hasattr(s, 'get_coverage_efficiency') else 0 
                       for s in sa_solutions]
    ga_coverages_all = [s.get_coverage_efficiency() if s and hasattr(s, 'get_coverage_efficiency') else 0 
                       for s in ga_solutions]
    
    sa_cov_stats = calculate_statistics(sa_coverages_all)
    ga_cov_stats = calculate_statistics(ga_coverages_all)
    
    sa_coverage = sa_solution.get_coverage_efficiency() if sa_solution and hasattr(sa_solution, 'get_coverage_efficiency') else 0
    ga_coverage = ga_solution.get_coverage_efficiency() if ga_solution and hasattr(ga_solution, 'get_coverage_efficiency') else 0
    
    if is_multiple_runs:
        print(f"SA Statistics ({sa_cov_stats['count']} runs):")
        print(f"   • Best:    {safe_format_stat(sa_cov_stats['best'], '.2f')}%")
        print(f"   • Average: {safe_format_stat(sa_cov_stats['avg'], '.2f')}%")
        print(f"   • Std Dev: {safe_format_stat(sa_cov_stats['std_dev'], '.2f')}%")
        print(f"\nGA Statistics ({ga_cov_stats['count']} runs):")
        print(f"   • Best:    {safe_format_stat(ga_cov_stats['best'], '.2f')}%")
        print(f"   • Average: {safe_format_stat(ga_cov_stats['avg'], '.2f')}%")
        print(f"   • Std Dev: {safe_format_stat(ga_cov_stats['std_dev'], '.2f')}%")
        print(f"\n📌 Best Run Comparison:")
        print(f"   SA Best Coverage:  {safe_format_stat(sa_cov_stats['best'], '.2f')}%")
        print(f"   GA Best Coverage:  {safe_format_stat(ga_cov_stats['best'], '.2f')}%")
    else:
        print(f"SA Coverage:  {sa_coverage:.2f}%")
        print(f"GA Coverage:  {ga_coverage:.2f}%")
    
    compare_sa_cov = sa_cov_stats['best'] if (is_multiple_runs and sa_cov_stats['best'] is not None) else sa_coverage
    compare_ga_cov = ga_cov_stats['best'] if (is_multiple_runs and ga_cov_stats['best'] is not None) else ga_coverage
    
    if compare_sa_cov is not None and compare_ga_cov is not None and compare_sa_cov > compare_ga_cov:
        winner_coverage = "SA"
        print(f"✅ SA has better coverage (+{compare_sa_cov - compare_ga_cov:.2f}%)")
    elif compare_ga_cov > compare_sa_cov:
        winner_coverage = "GA"
        print(f"✅ GA has better coverage (+{compare_ga_cov - compare_sa_cov:.2f}%)")
    else:
        winner_coverage = "TIE"
        print("🤝 Tie - Both achieved same coverage")
    
    # Performance Index 3: Workload Balance
    print(f"\n📊 PERFORMANCE INDEX 3: WORKLOAD BALANCE INDEX")
    print("-" * 80)
    
    # Collect balance statistics
    sa_balances_all = [s.get_workload_balance_index() if s and hasattr(s, 'get_workload_balance_index') else float('inf') 
                      for s in sa_solutions]
    ga_balances_all = [s.get_workload_balance_index() if s and hasattr(s, 'get_workload_balance_index') else float('inf') 
                      for s in ga_solutions]
    
    sa_bal_stats = calculate_statistics(sa_balances_all)
    ga_bal_stats = calculate_statistics(ga_balances_all)
    
    sa_balance = sa_solution.get_workload_balance_index() if sa_solution and hasattr(sa_solution, 'get_workload_balance_index') else float('inf')
    ga_balance = ga_solution.get_workload_balance_index() if ga_solution and hasattr(ga_solution, 'get_workload_balance_index') else float('inf')
    
    if is_multiple_runs:
        print(f"SA Statistics ({sa_bal_stats['count']} runs):")
        print(f"   • Best:    {safe_format_stat(sa_bal_stats['best'], '.4f')}")
        print(f"   • Average: {safe_format_stat(sa_bal_stats['avg'], '.4f')}")
        print(f"   • Std Dev: {safe_format_stat(sa_bal_stats['std_dev'], '.4f')}")
        print(f"\nGA Statistics ({ga_bal_stats['count']} runs):")
        print(f"   • Best:    {safe_format_stat(ga_bal_stats['best'], '.4f')}")
        print(f"   • Average: {safe_format_stat(ga_bal_stats['avg'], '.4f')}")
        print(f"   • Std Dev: {safe_format_stat(ga_bal_stats['std_dev'], '.4f')}")
        print(f"\n📌 Best Run Comparison:")
        print(f"   SA Best Balance:  {safe_format_stat(sa_bal_stats['best'], '.4f')} (higher = better)")
        print(f"   GA Best Balance:  {safe_format_stat(ga_bal_stats['best'], '.4f')} (higher = better)")
    else:
        print(f"SA Balance Index:  {sa_balance:.4f} (higher = better)")
        print(f"GA Balance Index:  {ga_balance:.4f} (higher = better)")
    
    compare_sa_bal = sa_bal_stats['best'] if (is_multiple_runs and sa_bal_stats['best'] is not None) else sa_balance
    compare_ga_bal = ga_bal_stats['best'] if (is_multiple_runs and ga_bal_stats['best'] is not None) else ga_balance
    
    if compare_sa_bal is not None and compare_ga_bal is not None and compare_sa_bal > compare_ga_bal:
        winner_balance = "SA"
        print(f"✅ SA has better workload balance")
    elif compare_ga_bal > compare_sa_bal:
        winner_balance = "GA"
        print(f"✅ GA has better workload balance")
    else:
        winner_balance = "TIE"
        print("🤝 Tie - Both have same balance")
    
    # Performance Index 4: Runtime
    print(f"\n📊 PERFORMANCE INDEX 4: RUNTIME")
    print("-" * 80)
    
    # Extract runtime from all runs
    sa_runtimes_all = []
    ga_runtimes_all = []
    
    for sa_hist in sa_histories:
        if isinstance(sa_hist, dict):
            rt = sa_hist.get('runtime')
            if rt is not None:
                sa_runtimes_all.append(rt)
    
    for ga_res in ga_results_list:
        if isinstance(ga_res, dict):
            rt = ga_res.get('runtime')
            if rt is not None:
                ga_runtimes_all.append(rt)
    
    sa_rt_stats = calculate_statistics(sa_runtimes_all)
    ga_rt_stats = calculate_statistics(ga_runtimes_all)
    
    # For backward compatibility, use first run's runtime
    sa_runtime = None
    ga_runtime = None
    
    if isinstance(sa_history, dict):
        sa_runtime = sa_history.get('runtime')
    if isinstance(ga_results_list[0], dict):
        ga_runtime = ga_results_list[0].get('runtime')
    
    if is_multiple_runs and sa_rt_stats['count'] > 0 and ga_rt_stats['count'] > 0:
        sa_best_sec = safe_format_stat(sa_rt_stats['best'], '.2f')
        sa_avg_sec = safe_format_stat(sa_rt_stats['avg'], '.2f')
        sa_std_sec = safe_format_stat(sa_rt_stats['std_dev'], '.2f')
        sa_worst_sec = safe_format_stat(sa_rt_stats['worst'], '.2f')
        sa_best_min = safe_format_stat(sa_rt_stats['best']/60 if sa_rt_stats['best'] is not None else None, '.2f')
        sa_avg_min = safe_format_stat(sa_rt_stats['avg']/60 if sa_rt_stats['avg'] is not None else None, '.2f')
        sa_worst_min = safe_format_stat(sa_rt_stats['worst']/60 if sa_rt_stats['worst'] is not None else None, '.2f')
        
        ga_best_sec = safe_format_stat(ga_rt_stats['best'], '.2f')
        ga_avg_sec = safe_format_stat(ga_rt_stats['avg'], '.2f')
        ga_std_sec = safe_format_stat(ga_rt_stats['std_dev'], '.2f')
        ga_worst_sec = safe_format_stat(ga_rt_stats['worst'], '.2f')
        ga_best_min = safe_format_stat(ga_rt_stats['best']/60 if ga_rt_stats['best'] is not None else None, '.2f')
        ga_avg_min = safe_format_stat(ga_rt_stats['avg']/60 if ga_rt_stats['avg'] is not None else None, '.2f')
        ga_worst_min = safe_format_stat(ga_rt_stats['worst']/60 if ga_rt_stats['worst'] is not None else None, '.2f')
        
        print(f"SA Runtime Statistics ({sa_rt_stats['count']} runs):")
        print(f"   • Best:    {sa_best_sec} seconds ({sa_best_min} minutes)")
        print(f"   • Average: {sa_avg_sec} seconds ({sa_avg_min} minutes)")
        print(f"   • Std Dev: {sa_std_sec} seconds")
        print(f"   • Worst:   {sa_worst_sec} seconds ({sa_worst_min} minutes)")
        print(f"\nGA Runtime Statistics ({ga_rt_stats['count']} runs):")
        print(f"   • Best:    {ga_best_sec} seconds ({ga_best_min} minutes)")
        print(f"   • Average: {ga_avg_sec} seconds ({ga_avg_min} minutes)")
        print(f"   • Std Dev: {ga_std_sec} seconds")
        print(f"   • Worst:   {ga_worst_sec} seconds ({ga_worst_min} minutes)")
        
        # Compare average runtimes
        if sa_rt_stats['avg'] is not None and ga_rt_stats['avg'] is not None and sa_rt_stats['avg'] < ga_rt_stats['avg']:
            speedup = ga_rt_stats['avg'] / sa_rt_stats['avg']
            improvement = ((ga_rt_stats['avg'] - sa_rt_stats['avg']) / ga_rt_stats['avg']) * 100
            print(f"\n✅ SA is faster on average by {improvement:.2f}% ({speedup:.2f}x speedup)")
            winner_runtime = "SA"
        elif ga_rt_stats['avg'] < sa_rt_stats['avg']:
            speedup = sa_rt_stats['avg'] / ga_rt_stats['avg']
            improvement = ((sa_rt_stats['avg'] - ga_rt_stats['avg']) / sa_rt_stats['avg']) * 100
            print(f"\n✅ GA is faster on average by {improvement:.2f}% ({speedup:.2f}x speedup)")
            winner_runtime = "GA"
        else:
            print("\n🤝 Tie - Both algorithms have similar average runtime")
            winner_runtime = "TIE"
    elif sa_runtime is not None and ga_runtime is not None:
        print(f"SA Runtime:  {sa_runtime:.2f} seconds ({sa_runtime/60:.2f} minutes)")
        print(f"GA Runtime:  {ga_runtime:.2f} seconds ({ga_runtime/60:.2f} minutes)")
        
        if sa_runtime < ga_runtime:
            speedup = ga_runtime / sa_runtime
            improvement = ((ga_runtime - sa_runtime) / ga_runtime) * 100
            print(f"✅ SA is faster by {improvement:.2f}% ({speedup:.2f}x speedup)")
            winner_runtime = "SA"
        elif ga_runtime < sa_runtime:
            speedup = sa_runtime / ga_runtime
            improvement = ((sa_runtime - ga_runtime) / sa_runtime) * 100
            print(f"✅ GA is faster by {improvement:.2f}% ({speedup:.2f}x speedup)")
            winner_runtime = "GA"
        else:
            print("🤝 Tie - Both algorithms took the same time")
            winner_runtime = "TIE"
    else:
        print("⚠️  Runtime data not available")
        winner_runtime = "N/A"
    
    # Performance Index 5: Convergence Speed
    print(f"\n📊 PERFORMANCE INDEX 5: CONVERGENCE SPEED")
    print("-" * 80)
    
    # Convert SA history if needed
    sa_history_std = convert_sa_history_to_standard(sa_history) if sa_history else {}
    
    # Find convergence point (when 95% of improvement achieved)
    def find_convergence_point(history, final_score):
        if not history or 'best_score' not in history or len(history['best_score']) == 0:
            return -1
        
        initial_score = history['best_score'][0]
        if initial_score == final_score:
            return 0
        
        threshold = initial_score - 0.95 * (initial_score - final_score)
        
        for i, score in enumerate(history['best_score']):
            if score <= threshold:
                return i
        return len(history['best_score']) - 1
    
    sa_conv_point = find_convergence_point(sa_history_std, sa_score)
    ga_conv_point = find_convergence_point(ga_history, ga_score)
    
    print(f"SA converged at iteration: {sa_conv_point if sa_conv_point >= 0 else 'N/A'}")
    print(f"GA converged at generation: {ga_conv_point if ga_conv_point >= 0 else 'N/A'}")
    
    if sa_conv_point >= 0 and ga_conv_point >= 0:
        if sa_conv_point < ga_conv_point:
            winner_convergence = "SA"
            print(f"✅ SA converged faster ({ga_conv_point - sa_conv_point} iterations earlier)")
        elif ga_conv_point < sa_conv_point:
            winner_convergence = "GA"
            print(f"✅ GA converged faster ({sa_conv_point - ga_conv_point} iterations earlier)")
        else:
            winner_convergence = "TIE"
            print("🤝 Tie - Both converged at same rate")
    else:
        winner_convergence = "N/A"
        print("⚠️  Convergence comparison not available")
    
    # Overall Summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    scores = {
        'SA': sum([
            winner_quality == "SA",
            winner_coverage == "SA",
            winner_balance == "SA",
            winner_runtime == "SA",
            winner_convergence == "SA"
        ]),
        'GA': sum([
            winner_quality == "GA",
            winner_coverage == "GA",
            winner_balance == "GA",
            winner_runtime == "GA",
            winner_convergence == "GA"
        ])
    }
    
    print(f"SA wins: {scores['SA']}/5 metrics")
    print(f"GA wins: {scores['GA']}/5 metrics")
    
    if scores['SA'] > scores['GA']:
        overall_winner = "SA"
    elif scores['GA'] > scores['SA']:
        overall_winner = "GA"
    else:
        overall_winner = "TIE"
    
    print(f"\n🏆 Overall Winner: {overall_winner}")
    print(f"{'='*80}\n")
    
    # Return comparison dictionary
    result = {
        'case_study': case_study_name,
        'sa_score': sa_score,
        'ga_score': ga_score,
        'sa_coverage': sa_coverage,
        'ga_coverage': ga_coverage,
        'sa_balance': sa_balance,
        'ga_balance': ga_balance,
        'sa_convergence_point': sa_conv_point,
        'ga_convergence_point': ga_conv_point,
        'winner_quality': winner_quality,
        'winner_coverage': winner_coverage,
        'winner_balance': winner_balance,
        'winner_runtime': winner_runtime,
        'winner_convergence': winner_convergence,
        'overall_winner': overall_winner,
        'sa_wins': scores['SA'],
        'ga_wins': scores['GA'],
        'sa_runtime': sa_runtime,
        'ga_runtime': ga_runtime,
        'num_runs': len(sa_results_list) if is_multiple_runs else 1
    }
    
    # Add statistics if multiple runs
    if is_multiple_runs:
        result['sa_score_stats'] = sa_stats
        result['ga_score_stats'] = ga_stats
        result['sa_coverage_stats'] = sa_cov_stats
        result['ga_coverage_stats'] = ga_cov_stats
        result['sa_balance_stats'] = sa_bal_stats
        result['ga_balance_stats'] = ga_bal_stats
        result['sa_runtime_stats'] = sa_rt_stats
        result['ga_runtime_stats'] = ga_rt_stats
    
    return result


def plot_sa_vs_ga_convergence(sa_history, ga_history, save_path=None):
    """
    Plot convergence comparison between SA and GA
    
    Args:
        sa_history: SA convergence history
        ga_history: GA convergence history
        save_path: Optional path to save figure
    """
    # Convert SA history to standard format
    sa_history = convert_sa_history_to_standard(sa_history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Best scores comparison
    ax1.plot(sa_history['generation'], sa_history['best_score'], 
            'r-o', label='SA', linewidth=2, markersize=4)
    ax1.plot(ga_history['generation'], ga_history['best_score'], 
            'b-s', label='GA', linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration/Generation', fontsize=12)
    ax1.set_ylabel('Best Score (lower = better)', fontsize=12)
    ax1.set_title('Convergence Comparison: SA vs GA', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement percentage
    sa_initial = sa_history['best_score'][0] if sa_history['best_score'] else 1
    ga_initial = ga_history['best_score'][0] if ga_history['best_score'] else 1
    
    sa_improvement = [(sa_initial - score) / sa_initial * 100 for score in sa_history['best_score']]
    ga_improvement = [(ga_initial - score) / ga_initial * 100 for score in ga_history['best_score']]
    
    ax2.plot(sa_history['generation'], sa_improvement, 
            'r-o', label='SA', linewidth=2, markersize=4)
    ax2.plot(ga_history['generation'], ga_improvement, 
            'b-s', label='GA', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration/Generation', fontsize=12)
    ax2.set_ylabel('Improvement from Initial (%)', fontsize=12)
    ax2.set_title('Improvement Rate Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()