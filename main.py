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
from case_studies import *

def create_results_directory():
    """Create directory for storing results"""
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')
    if not os.path.exists('results/data'):
        os.makedirs('results/data')

def run_single_case_study(case_number):
    """Run a specific case study"""
    case_functions = {
        1: case_study_1_small_grid,
        2: case_study_2_medium_grid,
        3: case_study_3_large_grid,
        4: case_study_4_complex_grid,
        5: benchmark_case_optimal
    }
    
    if case_number not in case_functions:
        print(f"Invalid case number: {case_number}")
        print("Available cases: 1, 2, 3, 4, 5")
        return None
    
    print(f"Running Case Study {case_number}...")
    solution = case_functions[case_number]()
    
    # Generate visualization
    fig = plot_grid_solution(solution, f"Case Study {case_number} Solution")
    save_figure(fig, f'results/figures/case_study_{case_number}.png')
    
    return solution

def run_all_case_studies_with_visualization():
    """Run all case studies and generate comprehensive visualizations"""
    print("MULTI-ROBOT COVERAGE PATH PLANNING - COMPLETE ANALYSIS")
    print("=" * 80)
    
    # Create results directory
    create_results_directory()
    
    # Run all case studies
    results = run_all_case_studies()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Individual solution plots
    for case_name, solution in results.items():
        fig = plot_grid_solution(solution, f"{case_name.replace('_', ' ').title()} Solution")
        save_figure(fig, f'results/figures/{case_name}_solution.png')
    
    # Comparison plots
    solutions = list(results.values())
    titles = [name.replace('_', ' ').title() for name in results.keys()]
    
    fig1 = plot_objective_comparison(solutions, titles, "Case Studies Comparison")
    save_figure(fig1, 'results/figures/case_studies_comparison.png')
    
    fig2 = plot_robot_workload_distribution(solutions[0], "Workload Distribution Example")
    save_figure(fig2, 'results/figures/workload_distribution.png')
    
    # Results summary
    create_results_summary(solutions, titles, 'results/data/results_summary.txt')
    
    # Performance analysis
    performance_analysis(results)
    
    print("\nAll visualizations and results saved in 'results/' directory")
    return results

def algorithm_validation():
    """Validate the SA algorithm against known benchmarks"""
    print("ALGORITHM VALIDATION")
    print("=" * 40)
    
    # Test 1: Optimal solution should be found for simple cases
    print("Test 1: Optimal Solution Detection")
    benchmark_solution = benchmark_case_optimal()
    
    if benchmark_solution.combined_score < 0.1:
        print("✓ PASS: Optimal solution found for benchmark case")
    else:
        print("✗ FAIL: Optimal solution not found")
    
    # Test 2: Constraint satisfaction
    print("\nTest 2: Constraint Satisfaction")
    violations = len(benchmark_solution.fitness['problems'])
    if violations == 0:
        print("✓ PASS: All constraints satisfied")
    else:
        print(f"✗ FAIL: {violations} constraint violations found")
    
    # Test 3: Coverage maximization
    print("\nTest 3: Coverage Maximization")
    coverage_ratio = benchmark_solution.fitness['coverage_score'] / len(benchmark_solution.free_cells)
    if coverage_ratio >= 0.9:  # At least 90% coverage
        print("✓ PASS: High coverage achieved")
    else:
        print(f"✗ FAIL: Low coverage ({coverage_ratio:.2%})")
    
    # Test 4: Workload balance
    print("\nTest 4: Workload Balance")
    balance_score = benchmark_solution.fitness['balance_score']
    if balance_score < 1.0:  # Low variance
        print("✓ PASS: Good workload balance")
    else:
        print(f"✗ FAIL: Poor workload balance ({balance_score:.3f})")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Multi-Robot Coverage Path Planning with SA')
    parser.add_argument('--case', type=int, help='Run specific case study (1-5)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--compare', action='store_true', help='Compare different solutions')
    parser.add_argument('--validate', action='store_true', help='Validate algorithm')
    
    args = parser.parse_args()
    
    if args.case:
        # Run specific case study
        solution = run_single_case_study(args.case)
        if solution:
            print(f"\nCase Study {args.case} completed successfully!")
    
    elif args.validate:
        # Algorithm validation
        algorithm_validation()
    
    elif args.visualize or args.compare:
        # Run all case studies with visualization
        results = run_all_case_studies_with_visualization()
        print("\nComplete analysis finished!")
    
    else:
        # Default: Run all case studies
        print("Running all case studies...")
        results = run_all_case_studies()
        
        # Basic performance analysis
        performance_analysis(results)
        
        print("\nTo generate visualizations, run: python main.py --visualize")
        print("To validate algorithm, run: python main.py --validate")

if __name__ == "__main__":
    main()
