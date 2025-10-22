"""
Visualization Module for Multi-Robot Coverage Path Planning
==========================================================

This module provides visualization functions for:
- Grid visualization with obstacles and robot assignments
- Path visualization showing robot trajectories
- Performance plots (convergence, objective values)
- Comparison plots between different solutions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from problem_formulation import *

def plot_grid_solution(solution, title="Multi-Robot Coverage Solution"):
    """
    Create a visual plot of the grid with robot assignments and paths
    
    Args:
        solution: RobotCoverageSolution object
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    grid_width = solution.grid_width
    grid_height = solution.grid_height
    
    # Create grid
    for x in range(grid_width + 1):
        ax.axvline(x, color='black', linewidth=1)
    for y in range(grid_height + 1):
        ax.axhline(y, color='black', linewidth=1)
    
    # Define colors for different robots
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot cells
    for y in range(grid_height):
        for x in range(grid_width):
            cell_idx = y * grid_width + x
            
            if cell_idx in solution.obstacles:
                # Obstacle cell
                rect = patches.Rectangle((x, y), 1, 1, linewidth=2, 
                                      edgecolor='black', facecolor='black', alpha=0.7)
                ax.add_patch(rect)
                ax.text(x + 0.5, y + 0.5, 'X', ha='center', va='center', 
                       color='white', fontsize=16, fontweight='bold')
            else:
                # Find which robot covers this cell
                robot_id = -1
                if 1 in solution.assignment[cell_idx]:
                    robot_id = solution.assignment[cell_idx].index(1)
                
                if robot_id >= 0:
                    # Robot cell
                    color = colors[robot_id % len(colors)]
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=2, 
                                          edgecolor='black', facecolor=color, alpha=0.3)
                    ax.add_patch(rect)
                    ax.text(x + 0.5, y + 0.5, f'R{robot_id}', ha='center', va='center', 
                           fontsize=12, fontweight='bold')
                else:
                    # Unassigned cell
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=2, 
                                          edgecolor='black', facecolor='white')
                    ax.add_patch(rect)
    
    # Plot robot paths
    for robot_id, path in enumerate(solution.paths):
        if len(path) > 1:
            color = colors[robot_id % len(colors)]
            path_coords = []
            for cell_idx in path:
                x, y = solution.all_cells[cell_idx]
                path_coords.append([x + 0.5, y + 0.5])
            
            path_coords = np.array(path_coords)
            ax.plot(path_coords[:, 0], path_coords[:, 1], 
                   color=color, linewidth=3, marker='o', markersize=8, 
                   label=f'Robot {robot_id} Path')
    
    ax.set_xlim(-0.5, grid_width + 0.5)
    ax.set_ylim(-0.5, grid_height + 0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_convergence_history(iteration_data, title="SA Convergence History"):
    """
    Plot the convergence history of the SA algorithm
    
    Args:
        iteration_data: List of dictionaries with iteration, temperature, current_score, best_score
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    iterations = [d['iteration'] for d in iteration_data]
    temperatures = [d['temperature'] for d in iteration_data]
    current_scores = [d['current_score'] for d in iteration_data]
    best_scores = [d['best_score'] for d in iteration_data]
    
    # Plot temperature cooling
    ax1.plot(iterations, temperatures, 'b-', linewidth=2, label='Temperature')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Temperature Cooling Schedule')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot objective function values
    ax2.plot(iterations, current_scores, 'r-', alpha=0.7, label='Current Solution')
    ax2.plot(iterations, best_scores, 'g-', linewidth=2, label='Best Solution')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Objective Function Value')
    ax2.set_title('Objective Function Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_objective_comparison(solutions, titles, title="Objective Function Comparison"):
    """
    Compare multiple solutions using bar plots
    
    Args:
        solutions: List of RobotCoverageSolution objects
        titles: List of solution titles
        title: Plot title
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    coverage_scores = [sol.fitness['coverage_score'] for sol in solutions]
    balance_scores = [sol.fitness['balance_score'] for sol in solutions]
    combined_scores = [sol.combined_score for sol in solutions]
    
    # Coverage comparison
    bars1 = ax1.bar(titles, coverage_scores, color='skyblue', alpha=0.7)
    ax1.set_title('Coverage Score (Higher is Better)')
    ax1.set_ylabel('Cells Covered')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars1, coverage_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{score}', ha='center', va='bottom')
    
    # Balance comparison
    bars2 = ax2.bar(titles, balance_scores, color='lightcoral', alpha=0.7)
    ax2.set_title('Balance Score (Lower is Better)')
    ax2.set_ylabel('Workload Variance')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars2, balance_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    # Combined score comparison
    bars3 = ax3.bar(titles, combined_scores, color='lightgreen', alpha=0.7)
    ax3.set_title('Combined Score (Lower is Better)')
    ax3.set_ylabel('Combined Objective Value')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars3, combined_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_robot_workload_distribution(solution, title="Robot Workload Distribution"):
    """
    Plot the workload distribution among robots
    
    Args:
        solution: RobotCoverageSolution object
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    robot_distances = solution.fitness['robot_distances']
    robot_ids = list(range(len(robot_distances)))
    
    # Distance bar plot
    bars = ax1.bar(robot_ids, robot_distances, color='steelblue', alpha=0.7)
    ax1.set_title('Robot Travel Distances')
    ax1.set_xlabel('Robot ID')
    ax1.set_ylabel('Distance Traveled')
    ax1.set_xticks(robot_ids)
    
    # Add value labels on bars
    for bar, distance in zip(bars, robot_distances):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{distance:.2f}', ha='center', va='bottom')
    
    # Workload pie chart
    ax2.pie(robot_distances, labels=[f'Robot {i}' for i in robot_ids], 
           autopct='%1.1f%%', startangle=90)
    ax2.set_title('Workload Distribution')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def save_figure(fig, filename, dpi=300):
    """Save figure to file"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved as: {filename}")

def create_results_summary(solutions, titles, filename="results_summary.txt"):
    """Create a text summary of results"""
    with open(filename, 'w') as f:
        f.write("Multi-Robot Coverage Path Planning - Results Summary\n")
        f.write("=" * 60 + "\n\n")
        
        for i, (solution, title) in enumerate(zip(solutions, titles)):
            f.write(f"Solution {i+1}: {title}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Coverage Score: {solution.fitness['coverage_score']} cells\n")
            f.write(f"Balance Score: {solution.fitness['balance_score']:.3f}\n")
            f.write(f"Combined Score: {solution.combined_score:.3f}\n")
            f.write(f"Robot Distances: {solution.fitness['robot_distances']}\n")
            f.write(f"Constraint Violations: {len(solution.fitness['problems'])}\n")
            f.write("\n")
    
    print(f"Results summary saved as: {filename}")
