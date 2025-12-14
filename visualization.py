"""
Visualization Module for Multi-Robot Coverage Path Planning
==========================================================

This module provides visualization functions for:
- Grid visualization with obstacles and robot assignments
- Path visualization showing robot trajectories
- Performance plots (convergence, objective values)
- Comparison plots between different solutions
- Real-time dynamic visualization for Dragonfly algorithm
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import ListedColormap
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


def plot_best_score_only(convergence_history, title="Best Score Convergence", save_path=None):
    """
    Plot only the best score over generations/iterations
    
    Args:
        convergence_history: Dictionary with convergence data
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    
    # Plot best score only
    ax.plot(iterations, best_scores, 'g-o', label='Best Score', linewidth=2.5, markersize=5, markevery=5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Combined Score (lower = better)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Best score plot saved to {save_path}")
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
    
    # Debug: Print path information to help diagnose visualization issues
    print(f"\n🔍 Visualization Debug Info:")
    print(f"   • Paths type: {type(solution.paths)}")
    print(f"   • Paths dict keys: {list(paths_dict.keys())}")
    print(f"   • Paths dict lengths: {[(robot_id, len(path)) for robot_id, path in paths_dict.items()]}")
    print(f"   • Total path cells: {sum(len(path) for path in paths_dict.values())}")
    print(f"   • Free cells: {len(solution.free_cells)}")
    
    # Check if paths are empty
    if not paths_dict or all(len(path) == 0 for path in paths_dict.values()):
        print(f"   ⚠️  WARNING: All paths are empty! Visualization may not show paths.")
    
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


def visualize_dragonfly_iteration(
    iteration: int,
    population,
    food,
    enemy,
    grid_width: int,
    grid_height: int,
    obstacles: list,
    save_path: str = None
):
    """
    Visualize a single iteration of the Dragonfly algorithm.
    Shows the population distribution, food (best), and enemy (worst) positions.
    
    Args:
        iteration: Current iteration number
        population: List of DragonflySolution objects
        food: Best solution (DragonflySolution)
        enemy: Worst solution (DragonflySolution)
        grid_width: Grid width
        grid_height: Grid height
        obstacles: List of obstacle cell indices
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Dragonfly Algorithm - Iteration {iteration}', fontsize=16, fontweight='bold')
    
    # Plot 6 random solutions from population (or all if < 6)
    num_to_plot = min(6, len(population))
    solutions_to_plot = np.random.choice(population, num_to_plot, replace=False) if len(population) > num_to_plot else population
    
    for idx, (ax, sol) in enumerate(zip(axes.flat, solutions_to_plot)):
        plot_single_solution(
            ax, sol.partition, sol.paths, grid_width, grid_height, obstacles,
            title=f'Dragonfly {idx+1}\nFitness: {sol.fitness:.2f}'
        )
    
    # Highlight food (best) and enemy (worst)
    for ax, sol, label in [(axes.flat[0], food, 'FOOD (Best)'), 
                           (axes.flat[-1], enemy, 'ENEMY (Worst)')]:
        if sol:
            plot_single_solution(
                ax, sol.partition, sol.paths, grid_width, grid_height, obstacles,
                title=f'{label}\nFitness: {sol.fitness:.2f}'
            )
            ax.set_facecolor('#ffffcc' if 'Best' in label else '#ffcccc')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Iteration {iteration} visualization saved: {save_path}")
    
    plt.close()


def plot_single_solution(ax, partition, paths, grid_width, grid_height, obstacles, title=""):
    """
    Plot a single solution on a given axis.
    UPDATED: More robust with better error handling and clearer visuals.
    
    Args:
        ax: Matplotlib axis
        partition: Dict {robot_id: [cells]}
        paths: Dict {robot_id: [path]}
        grid_width: Grid width
        grid_height: Grid height
        obstacles: List of obstacle cell indices
        title: Plot title
    """
    # Clear the axis
    ax.clear()
    
    # Create grid
    for x in range(grid_width + 1):
        ax.axvline(x, color='black', linewidth=0.5, alpha=0.3)
    for y in range(grid_height + 1):
        ax.axhline(y, color='black', linewidth=0.5, alpha=0.3)
    
    # Color palette for robots
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', 
              '#E056FD', '#686DE0', '#F8B500', '#78E08F']
    
    # Draw obstacles
    obstacles_set = set(obstacles)
    for cell_idx in obstacles_set:
        x = cell_idx % grid_width
        y = cell_idx // grid_width
        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, 
                                 edgecolor='black', facecolor='gray', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.5, 'X', ha='center', va='center',
                color='white', fontsize=8, fontweight='bold')
    
    # Draw robot assignments (partition) as colored backgrounds
    if partition:
        num_robots = len(partition)
        for robot_id, cells in partition.items():
            if cells:  # Check if not empty
                color = colors[robot_id % len(colors)]
                for cell_idx in cells:
                    if cell_idx not in obstacles_set:
                        x = cell_idx % grid_width
                        y = cell_idx // grid_width
                        rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5,
                                                edgecolor='lightgray', facecolor=color, alpha=0.3)
                        ax.add_patch(rect)
    
    # Draw paths with arrows
    if paths:
        for robot_id, path in paths.items():
            if not path or len(path) == 0:
                continue
            
            color = colors[robot_id % len(colors)]
            
            # Convert path indices to coordinates
            path_coords = [(cell_idx % grid_width + 0.5, cell_idx // grid_width + 0.5) 
                          for cell_idx in path]
            
            if len(path_coords) > 0:
                xs, ys = zip(*path_coords)
                
                # Draw path line
                ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.8, 
                       label=f'Robot {robot_id}', zorder=5)
                
                # Draw path points
                ax.scatter(xs, ys, color=color, s=50, alpha=0.8, 
                          edgecolors='black', linewidths=1, zorder=6)
                
                # Mark start position with larger marker
                ax.plot(xs[0], ys[0], 'o', color=color, markersize=12, 
                       markeredgecolor='black', markeredgewidth=2, zorder=7)
                ax.text(xs[0], ys[0] - 0.35, 'S', ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white',
                       bbox=dict(boxstyle='circle,pad=0.1', facecolor=color, 
                                edgecolor='black', linewidth=1.5), zorder=8)
                
                # Mark end position with square marker
                if len(xs) > 1:
                    ax.plot(xs[-1], ys[-1], 's', color=color, markersize=12, 
                           markeredgecolor='black', markeredgewidth=2, zorder=7)
                    ax.text(xs[-1], ys[-1] - 0.35, 'E', ha='center', va='center',
                           fontsize=8, fontweight='bold', color='white',
                           bbox=dict(boxstyle='square,pad=0.15', facecolor=color,
                                    edgecolor='black', linewidth=1.5), zorder=8)
    
    # Set axis properties
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.invert_yaxis()
    
    # Add legend if we have robots
    if paths and len(paths) <= 6:
        ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

# ...existing code...

def visualize_dragonfly_solution(partition, paths, grid_width, grid_height, obstacles, 
                                 title="Dragonfly Solution", save_path=None):
    """
    Visualize the final solution with robot assignments and paths (Dragonfly-specific).
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    plot_single_solution(ax, partition, paths, grid_width, grid_height, obstacles, title)
    
    # Add grid labels back for final solution
    ax.set_xticks(range(grid_width))
    ax.set_yticks(range(grid_height))
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Solution visualization saved: {save_path}")
        plt.close()
    else:
        plt.show()


class LiveDragonflyVisualizer:
    """
    Live-updating figure for Dragonfly optimization.
    Uses your existing plot_single_solution().
    """
    def __init__(self, grid_width, grid_height, obstacles, every=1, title_prefix="🐉 Dragonfly"):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles
        self.every = max(1, int(every))
        self.title_prefix = title_prefix

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.ion()                      # interactive mode ON
        self.fig.show()
        self.fig.canvas.draw()

    def update(self, iteration, partition, paths, best_fitness=None, best_score=None):
        if iteration % self.every != 0:
            return

        subtitle = []
        if best_fitness is not None:
            subtitle.append(f"fitness={best_fitness:.4f}")
        if best_score is not None:
            subtitle.append(f"score={best_score:.4f}")

        title = f"{self.title_prefix} | Iter {iteration}"
        if subtitle:
            title += " | " + " , ".join(subtitle)

        plot_single_solution(
            self.ax,
            partition=partition,
            paths=paths,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            obstacles=self.obstacles,
            title=title
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.001)               # lets the GUI breathe (non-blocking)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

def create_dragonfly_animation(
    history_snapshots: list,
    grid_width: int,
    grid_height: int,
    obstacles: list,
    save_path: str = None,
    fps: int = 2
):
    """
    Create an animated GIF showing the evolution of the best solution over iterations.
    UPDATED: Better error handling and progress feedback.
    
    Args:
        history_snapshots: List of (iteration, food_partition, food_paths) tuples
        grid_width: Grid width
        grid_height: Grid height
        obstacles: List of obstacle cell indices
        save_path: Path to save the animation (e.g., 'evolution.gif')
        fps: Frames per second
    """
    if not history_snapshots:
        print("⚠️  No snapshots to animate")
        return None
    
    print(f"📹 Creating animation with {len(history_snapshots)} frames...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        iteration, partition, paths = history_snapshots[frame]
        plot_single_solution(
            ax, partition, paths, grid_width, grid_height, obstacles,
            title=f'🏆 Best Solution - Iteration {iteration}'
        )
        
        # Print progress
        if (frame + 1) % 5 == 0 or frame == len(history_snapshots) - 1:
            print(f"  Frame {frame + 1}/{len(history_snapshots)} rendered", end='\r')
        
        return ax,
    
    try:
        anim = animation.FuncAnimation(
            fig, update, frames=len(history_snapshots),
            interval=1000//fps, blit=False, repeat=True
        )
        
        if save_path:
            print(f"\n💾 Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"✅ Animation saved: {save_path}")
        
        plt.close(fig)
        return anim
    
    except Exception as e:
        print(f"\n❌ Error creating animation: {e}")
        plt.close(fig)
        return None


def plot_dragonfly_convergence(history, save_path=None):
    """
    Plot convergence curves for Dragonfly algorithm.
    Shows best fitness, average fitness, and combined score over iterations.
    UPDATED: Better error handling and layout.
    """
    if not history or 'iteration' not in history:
        print("⚠️  No history data available")
        return
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Dragonfly Algorithm Convergence Analysis', fontsize=16, fontweight='bold')
    
    iterations = history.get('iteration', [])
    
    if not iterations:
        print("⚠️  No iteration data in history")
        plt.close(fig)
        return
    
    # Create 2x2 subplot grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Best Fitness
    if 'best_fitness' in history and history['best_fitness']:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, history['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Fitness (lower is better)', fontsize=11)
        ax1.set_title('Best Fitness Over Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
    
    # Plot 2: Average Fitness
    if 'avg_fitness' in history and history['avg_fitness']:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(iterations, history['avg_fitness'], 'g-', linewidth=2, label='Average Fitness')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Fitness', fontsize=11)
        ax2.set_title('Average Population Fitness', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
    
    # Plot 3: Combined Score
    if 'best_combined_score' in history and history['best_combined_score']:
        ax3 = fig.add_subplot(gs[1, 0])
        scores = history['best_combined_score']
        ax3.plot(iterations, scores, 'r-', linewidth=2, label='Combined Score')
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Combined Score (lower is better)', fontsize=11)
        ax3.set_title('Combined Score (Coverage + Balance)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        # Annotate best score
        if scores:
            min_score = min(scores)
            min_idx = scores.index(min_score)
            ax3.annotate(f'Best: {min_score:.4f}', 
                        xy=(iterations[min_idx], min_score),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=9)
    
    # Plot 4: Population Diversity
    if 'population_diversity' in history and history['population_diversity']:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(iterations, history['population_diversity'], 'm-', linewidth=2, label='Diversity')
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('Diversity', fontsize=11)
        ax4.set_title('Population Diversity (Exploration vs Exploitation)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Convergence plot saved: {save_path}")
    
    plt.close(fig)


def plot_weight_dynamics(history, save_path=None):
    """
    Plot how Dragonfly weights change over iterations.
    UPDATED: Better error handling and clearer labels.
    """
    if 'weights' not in history or not history['weights']:
        print("⚠️  No weight data in history")
        return
    
    if 'iteration' not in history or not history['iteration']:
        print("⚠️  No iteration data in history")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Dragonfly Weight Dynamics', fontsize=16, fontweight='bold')
    
    iterations = history['iteration']
    weights = history['weights']
    
    # Ensure we have matching lengths
    min_len = min(len(iterations), len(weights))
    iterations = iterations[:min_len]
    weights = weights[:min_len]
    
    # Extract weight values
    s_vals = [w.get('s', 0) for w in weights]
    a_vals = [w.get('a', 0) for w in weights]
    c_vals = [w.get('c', 0) for w in weights]
    f_vals = [w.get('f', 0) for w in weights]
    e_vals = [w.get('e', 0) for w in weights]
    w_vals = [w.get('w', 0) for w in weights]
    
    # Plot 1: Swarm behavior weights (Exploration)
    ax1.plot(iterations, s_vals, 'r-', linewidth=2, label='Separation (s)', marker='o', markersize=2, markevery=5)
    ax1.plot(iterations, a_vals, 'g-', linewidth=2, label='Alignment (a)', marker='s', markersize=2, markevery=5)
    ax1.plot(iterations, c_vals, 'b-', linewidth=2, label='Cohesion (c)', marker='^', markersize=2, markevery=5)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Weight Value', fontsize=12)
    ax1.set_title('Exploration Weights (Swarm Behavior)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Attraction/Repulsion weights (Exploitation)
    ax2.plot(iterations, f_vals, 'm-', linewidth=2, label='Food Attraction (f)', marker='o', markersize=2, markevery=5)
    ax2.plot(iterations, e_vals, 'c-', linewidth=2, label='Enemy Repulsion (e)', marker='s', markersize=2, markevery=5)
    ax2.plot(iterations, w_vals, 'k--', linewidth=2, label='Inertia (w)', marker='^', markersize=2, markevery=5)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Weight Value', fontsize=12)
    ax2.set_title('Exploitation Weights + Inertia', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Weight dynamics plot saved: {save_path}")
    
    plt.close(fig)


def plot_live_dragonfly_progress(iteration, best_fitness, avg_fitness, max_iterations):
    """
    Display live progress of the Dragonfly algorithm (updates in terminal).
    UPDATED: Better formatting with percentage and time estimate.
    
    Args:
        iteration: Current iteration
        best_fitness: Best fitness so far
        avg_fitness: Average fitness of population
        max_iterations: Total iterations
    """
    progress = iteration / max_iterations
    bar_length = 40
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percentage = progress * 100
    
    print(f'\r🐉 [{bar}] {percentage:5.1f}% ({iteration}/{max_iterations}) | '
          f'Best: {best_fitness:7.4f} | Avg: {avg_fitness:7.4f}', 
          end='', flush=True)
    
    if iteration == max_iterations:
        print()  # New line at the end


def plot_best_score_only(history, save_path=None):
    """
    Simple plot showing only the best combined score over iterations.
    UPDATED: Better annotations and styling.
    """
    if 'best_combined_score' not in history or not history['best_combined_score']:
        print("⚠️  No combined score data in history")
        return
    
    if 'iteration' not in history or not history['iteration']:
        print("⚠️  No iteration data in history")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = history['iteration']
    scores = history['best_combined_score']
    
    # Ensure matching lengths
    min_len = min(len(iterations), len(scores))
    iterations = iterations[:min_len]
    scores = scores[:min_len]
    
    ax.plot(iterations, scores, 'b-', linewidth=2.5, marker='o', markersize=4, markevery=5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Combined Score (lower is better)', fontsize=12)
    ax.set_title('Dragonfly Algorithm - Best Combined Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add min score annotation
    if scores:
        min_score = min(scores)
        min_iter = iterations[scores.index(min_score)]
        ax.annotate(f'Best: {min_score:.4f}\nIteration: {min_iter}', 
                    xy=(min_iter, min_score),
                    xytext=(20, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8, edgecolor='black'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2),
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Best score plot saved: {save_path}")
    
    plt.close(fig)

