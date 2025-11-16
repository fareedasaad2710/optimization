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
import matplotlib.patches as mpatches
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


def plot_convergence_history(convergence_history, title="Algorithm Convergence", save_path=None):
    """
    Plot convergence history showing best, average, and worst scores over generations
    Supports both SA (with 'iteration' key) and GA (with 'generation' key) formats
    
    Args:
        convergence_history: Dictionary with convergence data
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Handle both SA and GA formats
    if 'generation' in convergence_history:
        iterations = convergence_history['generation']
        xlabel = 'Generation'
    elif 'iteration' in convergence_history:
        iterations = convergence_history['iteration']
        xlabel = 'Iteration'
    else:
        iterations = list(range(len(convergence_history.get('best_score', []))))
        xlabel = 'Iteration'
    
    best_scores = convergence_history.get('best_score', [])
    
    # Plot best score (always available)
    ax.plot(iterations, best_scores, 'g-o', label='Best Score', linewidth=2, markersize=4)
    
    # Plot average and worst if available (GA format)
    if 'avg_score' in convergence_history:
        avg_scores = convergence_history['avg_score']
        ax.plot(iterations, avg_scores, 'b-s', label='Average Score', linewidth=2, markersize=4)
    
    if 'worst_score' in convergence_history:
        worst_scores = convergence_history['worst_score']
        ax.plot(iterations, worst_scores, 'r-^', label='Worst Score', linewidth=2, markersize=4)
    
    # Plot current score if available (SA format)
    if 'current_score' in convergence_history:
        current_scores = convergence_history['current_score']
        ax.plot(iterations, current_scores, 'r-', alpha=0.5, label='Current Score', linewidth=1)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Combined Score (lower = better)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Convergence plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_robot_paths(solution, grid_size, title="Robot Paths", save_path=None):
    """
    Visualize robot paths on the grid
    
    Args:
        solution: RobotCoverageSolution object
        grid_size: Tuple (grid_width, grid_height)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    grid_width, grid_height = grid_size
    
    # Convert paths from list to dict format if needed (SA uses list, GA uses dict)
    if isinstance(solution.paths, list):
        paths_dict = {robot_id: path for robot_id, path in enumerate(solution.paths)}
    elif isinstance(solution.paths, dict):
        paths_dict = solution.paths
    else:
        paths_dict = {}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw grid
    for x in range(grid_width + 1):
        ax.axvline(x, color='black', linewidth=1)
    for y in range(grid_height + 1):
        ax.axhline(y, color='black', linewidth=1)
    
    # Draw obstacles
    for obs_idx in solution.obstacles:
        x = obs_idx % grid_width
        y = obs_idx // grid_width
        rect = mpatches.Rectangle((x, y), 1, 1, linewidth=2, 
                                   edgecolor='black', facecolor='gray', alpha=0.7)
        ax.add_patch(rect)
    
    # Define colors for robots
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    # Draw robot paths
    for robot_id, path in paths_dict.items():
        if len(path) == 0:
            continue
        
        color = colors[robot_id % len(colors)]
        
        # Convert cell indices to coordinates
        coords = []
        for cell_idx in path:
            x = (cell_idx % grid_width) + 0.5
            y = (cell_idx // grid_width) + 0.5
            coords.append((x, y))
        
        # Draw path
        if len(coords) > 1:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color, linewidth=2, marker='o', 
                   markersize=8, label=f'Robot {robot_id}', alpha=0.7)
        
        # Mark start and end
        if coords:
            start_x, start_y = coords[0]
            ax.plot(start_x, start_y, 'o', color=color, markersize=12, 
                   markeredgecolor='black', markeredgewidth=2)
            ax.text(start_x, start_y + 0.3, 'S', ha='center', va='center', 
                   fontsize=10, fontweight='bold')
            
            end_x, end_y = coords[-1]
            ax.plot(end_x, end_y, 's', color=color, markersize=12, 
                   markeredgecolor='black', markeredgewidth=2)
            ax.text(end_x, end_y + 0.3, 'E', ha='center', va='center', 
                   fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Robot paths plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_coverage_heatmap(solution, grid_size, title="Coverage Heatmap", save_path=None):
    """
    Create a heatmap showing which robot covers which cell
    
    Args:
        solution: RobotCoverageSolution object
        grid_size: Tuple (grid_width, grid_height)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    grid_width, grid_height = grid_size
    
    # Convert paths from list to dict format if needed (SA uses list, GA uses dict)
    if isinstance(solution.paths, list):
        paths_dict = {robot_id: path for robot_id, path in enumerate(solution.paths)}
    elif isinstance(solution.paths, dict):
        paths_dict = solution.paths
    else:
        paths_dict = {}
    
    # Create coverage matrix
    coverage_matrix = np.zeros((grid_height, grid_width))
    
    for robot_id, path in paths_dict.items():
        for cell_idx in path:
            x = cell_idx % grid_width
            y = cell_idx // grid_width
            coverage_matrix[y, x] = robot_id + 1  # +1 to distinguish from uncovered (0)
    
    # Mark obstacles as -1
    for obs_idx in solution.obstacles:
        x = obs_idx % grid_width
        y = obs_idx // grid_width
        coverage_matrix[y, x] = -1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap
    cmap = plt.cm.get_cmap('tab10')
    
    im = ax.imshow(coverage_matrix, cmap=cmap, interpolation='nearest', 
                   vmin=-1, vmax=len(solution.paths))
    
    # Add grid lines
    for x in range(grid_width + 1):
        ax.axvline(x - 0.5, color='black', linewidth=1)
    for y in range(grid_height + 1):
        ax.axhline(y - 0.5, color='black', linewidth=1)
    
    # Add cell labels
    for y in range(grid_height):
        for x in range(grid_width):
            value = coverage_matrix[y, x]
            
            if value == -1:
                text = 'OBS'
                color = 'white'
            elif value == 0:
                text = '—'
                color = 'black'
            else:
                text = f'R{int(value) - 1}'
                color = 'white'
            
            ax.text(x, y, text, ha='center', va='center', 
                   color=color, fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(grid_width))
    ax.set_yticks(range(grid_height))
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Robot ID', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Coverage heatmap saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_solution(solution, title="Multi-Robot Coverage Solution", save_path=None):
    """
    Comprehensive visualization of a single solution
    
    Args:
        solution: RobotCoverageSolution object
        title: Plot title
        save_path: Optional path to save figure
    """
    # Get grid dimensions
    grid_width = solution.grid_width
    grid_height = solution.grid_height
    
    # Convert paths from list to dict format if needed (SA uses list, GA uses dict)
    if isinstance(solution.paths, list):
        paths_dict = {robot_id: path for robot_id, path in enumerate(solution.paths)}
    elif isinstance(solution.paths, dict):
        paths_dict = solution.paths
    else:
        paths_dict = {}
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Subplot 1: Robot paths and assignments
    # Draw grid
    for x in range(grid_width + 1):
        ax1.axvline(x, color='black', linewidth=1)
    for y in range(grid_height + 1):
        ax1.axhline(y, color='black', linewidth=1)
    
    # Define colors for robots
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    # Draw obstacles
    for obs_idx in solution.obstacles:
        x = obs_idx % grid_width
        y = obs_idx // grid_width
        rect = mpatches.Rectangle((x, y), 1, 1, linewidth=2, 
                                   edgecolor='black', facecolor='gray', alpha=0.7)
        ax1.add_patch(rect)
        ax1.text(x + 0.5, y + 0.5, 'X', ha='center', va='center', 
                color='white', fontsize=12, fontweight='bold')
    
    # Draw robot paths
    for robot_id, path in paths_dict.items():
        if len(path) == 0:
            continue
        
        color = colors[robot_id % len(colors)]
        
        # Highlight cells covered by this robot
        for cell_idx in path:
            x = cell_idx % grid_width
            y = cell_idx // grid_width
            rect = mpatches.Rectangle((x, y), 1, 1, linewidth=1, 
                                       edgecolor='black', facecolor=color, alpha=0.3)
            ax1.add_patch(rect)
        
        # Convert cell indices to coordinates
        coords = []
        for cell_idx in path:
            x = (cell_idx % grid_width) + 0.5
            y = (cell_idx // grid_width) + 0.5
            coords.append((x, y))
        
        # Draw path line
        if len(coords) > 1:
            xs, ys = zip(*coords)
            ax1.plot(xs, ys, color=color, linewidth=2.5, marker='o', 
                    markersize=8, label=f'Robot {robot_id}', alpha=0.9)
        
        # Mark start and end positions
        if coords:
            start_x, start_y = coords[0]
            ax1.plot(start_x, start_y, 'o', color=color, markersize=15, 
                    markeredgecolor='black', markeredgewidth=2.5)
            ax1.text(start_x, start_y - 0.25, 'S', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            end_x, end_y = coords[-1]
            ax1.plot(end_x, end_y, 's', color=color, markersize=15, 
                    markeredgecolor='black', markeredgewidth=2.5)
            ax1.text(end_x, end_y - 0.25, 'E', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax1.set_xlim(0, grid_width)
    ax1.set_ylim(0, grid_height)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.set_title('Robot Paths', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax1.grid(True, alpha=0.2)
    
    # Subplot 2: Performance metrics
    ax2.axis('off')
    
    # Calculate metrics (handle both GA and SA solutions)
    # Coverage efficiency
    if hasattr(solution, 'get_coverage_efficiency'):
        coverage_efficiency = solution.get_coverage_efficiency()
    else:
        # Calculate from fitness data
        if solution.fitness and 'coverage_score' in solution.fitness:
            coverage_efficiency = (solution.fitness['coverage_score'] / len(solution.free_cells)) * 100 if len(solution.free_cells) > 0 else 0
        else:
            coverage_efficiency = 0.0
    
    # Workload balance index
    if hasattr(solution, 'get_workload_balance_index'):
        balance_index = solution.get_workload_balance_index()
    else:
        # Calculate from fitness data
        if solution.fitness and 'balance_score' in solution.fitness:
            balance_index = solution.fitness['balance_score']
        else:
            balance_index = 0.0
    
    # Count cells covered
    if isinstance(paths_dict, dict):
        cells_covered = sum(len(path) for path in paths_dict.values())
    else:
        cells_covered = 0
    
    # Format combined score safely
    combined_score_str = f"{solution.combined_score:.4f}" if solution.combined_score is not None else "N/A"
    
    # Display solution metrics
    metrics_text = f"""
    SOLUTION PERFORMANCE METRICS
    {'='*40}
    
    Combined Score: {combined_score_str}
    
    Coverage Metrics:
    • Coverage Efficiency: {coverage_efficiency:.2f}%
    • Cells Covered: {cells_covered}
    • Total Free Cells: {len(solution.free_cells)}
    
    Workload Distribution:
    • Balance Index: {balance_index:.4f}
    """
    
    # Add individual robot workloads
    if solution.fitness and 'robot_distances' in solution.fitness:
        metrics_text += "\n    Robot Workloads:\n"
        for robot_id, distance in enumerate(solution.fitness['robot_distances']):
            metrics_text += f"    • Robot {robot_id}: {distance:.2f} units\n"
    
    # Add constraint information
    if solution.fitness and 'problems' in solution.fitness:
        violations = len(solution.fitness['problems'])
        metrics_text += f"\n    Constraint Violations: {violations}\n"
    
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Solution visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return fig


def save_all_figures(solution, convergence_history, grid_size, output_dir="results/figures"):
    """Generate and save all visualization figures"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("GENERATING ALL VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_convergence_history(convergence_history, title="GA Convergence History",
                            save_path=f"{output_dir}/ga_convergence.png")
    plot_robot_paths(solution, grid_size=grid_size,
                    title=f"Best Solution - Score: {solution.combined_score:.3f}",
                    save_path=f"{output_dir}/ga_robot_paths.png")
    plot_coverage_heatmap(solution, grid_size=grid_size, title="Coverage Distribution",
                         save_path=f"{output_dir}/ga_coverage_heatmap.png")
    
    print(f"✅ All visualizations saved to {output_dir}/")
    print(f"{'='*70}\n")


# ...keep other existing functions (plot_objective_comparison, etc.)...
