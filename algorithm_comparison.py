from GA import genetic_algorithm, print_solution_summary, analyze_convergence
from sa_algorithm import simulated_annealing
import matplotlib.pyplot as plt
import numpy as np

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

def generate_comparison_report(sa_results, ga_results, case_study_name="Comparison"):
    """
    Generate a comprehensive comparison report between SA and GA results
    
    Args:
        sa_results: Tuple or dict with (sa_solution, sa_history) or {'best_solution': ..., 'convergence_history': ...}
        ga_results: Tuple or dict with (ga_solution, ga_history) or {'best_solution': ..., 'convergence_history': ...}
        case_study_name: Name of the case study
    
    Returns:
        Dictionary with comparison metrics
    """
    # Handle different input formats
    if isinstance(sa_results, tuple):
        sa_solution, sa_history = sa_results
    elif isinstance(sa_results, dict):
        sa_solution = sa_results.get('best_solution')
        sa_history = sa_results.get('convergence_history', {})
    else:
        sa_solution = sa_results
        sa_history = {}
    
    if isinstance(ga_results, tuple):
        ga_solution, ga_history = ga_results
    elif isinstance(ga_results, dict):
        ga_solution = ga_results.get('best_solution')
        ga_history = ga_results.get('convergence_history', {})
    else:
        ga_solution = ga_results
        ga_history = {}
    
    print(f"\n{'='*80}")
    print(f"ALGORITHM COMPARISON REPORT - {case_study_name}")
    print(f"{'='*80}\n")
    
    # Performance Index 1: Solution Quality (Combined Score)
    print("📊 PERFORMANCE INDEX 1: SOLUTION QUALITY (Combined Score)")
    print("-" * 80)
    
    sa_score = sa_solution.combined_score if sa_solution and hasattr(sa_solution, 'combined_score') else float('inf')
    ga_score = ga_solution.combined_score if ga_solution and hasattr(ga_solution, 'combined_score') else float('inf')
    
    print(f"SA Final Score:  {sa_score:.4f}")
    print(f"GA Final Score:  {ga_score:.4f}")
    
    if sa_score < ga_score:
        improvement = ((ga_score - sa_score) / ga_score) * 100 if ga_score != 0 else 0
        print(f"✅ SA performs better by {improvement:.2f}%")
        winner_quality = "SA"
    elif ga_score < sa_score:
        improvement = ((sa_score - ga_score) / sa_score) * 100 if sa_score != 0 else 0
        print(f"✅ GA performs better by {improvement:.2f}%")
        winner_quality = "GA"
    else:
        print("🤝 Tie - Both algorithms achieved the same score")
        winner_quality = "TIE"
    
    # Performance Index 2: Coverage Efficiency
    print(f"\n📊 PERFORMANCE INDEX 2: COVERAGE EFFICIENCY")
    print("-" * 80)
    
    sa_coverage = sa_solution.get_coverage_efficiency() if sa_solution and hasattr(sa_solution, 'get_coverage_efficiency') else 0
    ga_coverage = ga_solution.get_coverage_efficiency() if ga_solution and hasattr(ga_solution, 'get_coverage_efficiency') else 0
    
    print(f"SA Coverage:  {sa_coverage:.2f}%")
    print(f"GA Coverage:  {ga_coverage:.2f}%")
    
    if sa_coverage > ga_coverage:
        winner_coverage = "SA"
        print(f"✅ SA has better coverage (+{sa_coverage - ga_coverage:.2f}%)")
    elif ga_coverage > sa_coverage:
        winner_coverage = "GA"
        print(f"✅ GA has better coverage (+{ga_coverage - sa_coverage:.2f}%)")
    else:
        winner_coverage = "TIE"
        print("🤝 Tie - Both achieved same coverage")
    
    # Performance Index 3: Workload Balance
    print(f"\n📊 PERFORMANCE INDEX 3: WORKLOAD BALANCE INDEX")
    print("-" * 80)
    
    sa_balance = sa_solution.get_workload_balance_index() if sa_solution and hasattr(sa_solution, 'get_workload_balance_index') else float('inf')
    ga_balance = ga_solution.get_workload_balance_index() if ga_solution and hasattr(ga_solution, 'get_workload_balance_index') else float('inf')
    
    print(f"SA Balance Index:  {sa_balance:.4f} (higher = better)")
    print(f"GA Balance Index:  {ga_balance:.4f} (higher = better)")
    
    if sa_balance > ga_balance:
        winner_balance = "SA"
        print(f"✅ SA has better workload balance")
    elif ga_balance > sa_balance:
        winner_balance = "GA"
        print(f"✅ GA has better workload balance")
    else:
        winner_balance = "TIE"
        print("🤝 Tie - Both have same balance")
    
    # Performance Index 4: Convergence Speed
    print(f"\n📊 PERFORMANCE INDEX 4: CONVERGENCE SPEED")
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
            winner_convergence == "SA"
        ]),
        'GA': sum([
            winner_quality == "GA",
            winner_coverage == "GA",
            winner_balance == "GA",
            winner_convergence == "GA"
        ])
    }
    
    print(f"SA wins: {scores['SA']}/4 metrics")
    print(f"GA wins: {scores['GA']}/4 metrics")
    
    if scores['SA'] > scores['GA']:
        overall_winner = "SA"
    elif scores['GA'] > scores['SA']:
        overall_winner = "GA"
    else:
        overall_winner = "TIE"
    
    print(f"\n🏆 Overall Winner: {overall_winner}")
    print(f"{'='*80}\n")
    
    # Return comparison dictionary
    return {
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
        'winner_convergence': winner_convergence,
        'overall_winner': overall_winner,
        'sa_wins': scores['SA'],
        'ga_wins': scores['GA']
    }


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