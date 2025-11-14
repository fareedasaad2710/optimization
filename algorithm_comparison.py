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