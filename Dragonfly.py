"""
Dragonfly Algorithm for Multi-Robot Coverage Path Planning
===========================================================

Implementation of the Dragonfly Algorithm (DA) for optimizing
multi-robot coverage paths with DARP partitioning and UF-STC path construction.
"""

import numpy as np
import random
from typing import List, Tuple, Dict
from problem_formulation import (
    evaluate_solution,
    find_neighbors,
    calculate_robot_distances,
    distance_between_points,
    create_grid_cells
)
from DARP import darp_partition
from UF_STC import construct_spanning_tree_paths


class DragonflySolution:
    """Represents a dragonfly individual (solution)."""

    def __init__(
        self,
        partition: Dict[int, List[int]],
        paths: Dict[int, List[int]],
        fitness: float,
        grid_width: int,
        grid_height: int,
        obstacles: List[int],
    ):
        # Current position (solution)
        self.partition = {r: list(cells) for r, cells in partition.items()}
        self.paths = {r: list(path) for r, path in paths.items()}
        self.fitness = fitness

        # Environment info
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles

        # Personal best (pBest)
        self.personal_best_partition = {r: list(cells) for r, cells in partition.items()}
        self.personal_best_paths = {r: list(path) for r, path in paths.items()}
        self.personal_best_fitness = fitness

        # Discrete step vector (ΔXᵢ) — inertia memory for DA
        # Each entry is a (cell, from_robot, to_robot) move
        self.last_moves: List[Tuple[int, int, int]] = []

        # Stagnancy counter (for random re-initialization / escape)
        self.stagnancy: int = 0



class DragonflyOptimizer:
    """Dragonfly Algorithm optimizer for multi-robot coverage."""

    def __init__(
        self,
        all_cells: List[Tuple[int, int]],
        free_cells: List[int],
        obstacles: List[int],
        grid_width: int,
        grid_height: int,
        num_robots: int,
        population_size: int = 30,
        max_iterations: int = 100,
        neighbor_size: int = 5,
        w: float = 0.9,
        s_weight: float = 2.0,
        a_weight: float = 2.0,
        c_weight: float = 2.0,
        f_weight: float = 2.0,
        e_weight: float = 1.0,
        verbose: bool = True,
        neighbor_similarity_threshold: float = 0.6,  # ✅ MOVED HERE (was incorrectly in DragonflySolution)
    ):
        """
        Initialize Dragonfly Optimizer.

        Args:
            all_cells: List of all (x, y) coordinates
            free_cells: List of free cell indices
            obstacles: List of obstacle cell indices
            grid_width: Grid width
            grid_height: Grid height
            num_robots: Number of robots
            population_size: Population size (P)
            max_iterations: Maximum iterations (T)
            neighbor_size: Number of neighbors (K)
            w: Inertia weight
            s_weight: Separation weight
            a_weight: Alignment weight
            c_weight: Cohesion weight
            f_weight: Food attraction weight
            e_weight: Enemy distraction weight
            verbose: Print progress
            neighbor_similarity_threshold: threshold for similarity-based neighborhood
        """
        self.all_cells = all_cells
        self.free_cells = free_cells
        self.obstacles = obstacles
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_robots = num_robots
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.neighbor_size = neighbor_size
        self.w = w
        self.s_weight = s_weight
        self.a_weight = a_weight
        self.c_weight = c_weight
        self.f_weight = f_weight
        self.e_weight = e_weight
        self.verbose = verbose

        # ✅ Used by get_neighbors()
        self.neighbor_similarity_threshold = neighbor_similarity_threshold

        self.population: List[DragonflySolution] = []
        self.food: DragonflySolution | None = None   # Best solution
        self.enemy: DragonflySolution | None = None  # Worst solution
        self.history = {"iteration": [], "best_fitness": [], "avg_fitness": []}

    def initialize_population(self):
        """Initialize population using DARP + UF-STC."""
        if self.verbose:
            print("Initializing population...")

        for i in range(self.population_size):
            # Generate random robot starting positions
            robot_positions = random.sample(self.free_cells, self.num_robots)

            # Apply DARP partitioning
            partition = darp_partition(
                self.grid_width, self.grid_height,
                robot_positions, self.obstacles
            )

            # Build paths using UF-STC
            paths = construct_spanning_tree_paths(
                partition, self.grid_width, self.grid_height, self.obstacles
            )

            # Evaluate fitness
            fitness = self.evaluate_fitness(partition, paths)

            # Create dragonfly solution
            solution = DragonflySolution(
                partition, paths, fitness,
                self.grid_width, self.grid_height, self.obstacles
            )

            self.population.append(solution)

        # Initialize food (best) and enemy (worst)
        self.update_food_and_enemy()

    def evaluate_fitness(
        self,
        partition: Dict[int, List[int]],
        paths: Dict[int, List[int]],
    ) -> float:
        """
        Evaluate fitness of a solution (lower is better).

        Fitness = -Coverage + λ * Workload_Imbalance
        """
        # Calculate coverage
        covered_cells = set()
        for cells in partition.values():
            covered_cells.update(cells)
        coverage = len(covered_cells)

        # Calculate workload imbalance
        path_lengths = [len(path) for path in paths.values()]
        if len(path_lengths) > 0:
            avg_length = np.mean(path_lengths)
            imbalance = sum(abs(length - avg_length) for length in path_lengths)
        else:
            imbalance = 0

        # Combined fitness (minimize)
        lambda_balance = 0.5
        fitness = -coverage + lambda_balance * imbalance

        return fitness

    def update_food_and_enemy(self):
        """Update food (best) and enemy (worst) solutions."""
        self.food = min(self.population, key=lambda x: x.fitness)
        self.enemy = max(self.population, key=lambda x: x.fitness)

    def get_neighbors(self, idx: int) -> List[DragonflySolution]:
        """
        Neighborhood selection in *solution space*.
        In the original DA, neighbors are those within a radius in position space.
        Here we approximate "radius" with partition similarity.

        Similarity = (#cells assigned to same robot in both solutions) / (#free_cells)

        If similarity >= self.neighbor_similarity_threshold => neighbor
        If no one passes threshold => return []  (so we can trigger Lévy flight)
        """
        current = self.population[idx]

        # Build a quick lookup: cell -> robot for current
        cur_owner = {}
        for r, cells in current.partition.items():
            for c in cells:
                cur_owner[c] = r

        neighbors = []
        total = len(self.free_cells)

        for j, other in enumerate(self.population):
            if j == idx:
                continue

            same = 0
            # Compare ownership for all free cells
            # (fast enough for moderate grids; can be optimized later)
            oth_owner = {}
            for r, cells in other.partition.items():
                for c in cells:
                    oth_owner[c] = r

            for c in self.free_cells:
                if cur_owner.get(c, None) == oth_owner.get(c, None):
                    same += 1

            similarity = same / max(1, total)

            if similarity >= self.neighbor_similarity_threshold:
                neighbors.append(other)

        # If too many neighbors, keep the closest ones (highest similarity)
        if len(neighbors) > self.neighbor_size:
            def sim(other):
                oth_owner = {}
                for r, cells in other.partition.items():
                    for c in cells:
                        oth_owner[c] = r
                same2 = sum(1 for c in self.free_cells if cur_owner.get(c) == oth_owner.get(c))
                return same2 / max(1, total)

            neighbors.sort(key=sim, reverse=True)
            neighbors = neighbors[:self.neighbor_size]

        return neighbors

    def separation_moves(self, solution, neighbors):
        moves = []
        total_cells = len(self.free_cells)

        for neighbor in neighbors:
            same = sum(
                1 for r in range(self.num_robots)
                for c in solution.partition.get(r, [])
                if c in neighbor.partition.get(r, [])
            )

            if same / total_cells > 0.8:  # too similar
                for r, cells in solution.partition.items():
                    if cells:
                        cell = random.choice(cells)
                        other = random.choice([x for x in range(self.num_robots) if x != r])
                        moves.append((cell, r, other))

        return moves

    def alignment_moves(
        self,
        solution: DragonflySolution,
        neighbors: List[DragonflySolution],
    ) -> List[Tuple[int, int, int]]:
        """
        Generate alignment moves (imitate neighbor direction).
        In discrete DA, "velocity" ≈ neighbors' last successful move-lists (last_moves).
        So alignment = copy some of those move patterns.
        """
        if not neighbors:
            return []

        moves: List[Tuple[int, int, int]] = []

        # Prefer better neighbors (lower fitness = better)
        neighbors_sorted = sorted(neighbors, key=lambda n: n.fitness)

        # How many moves to copy in total (keep small so it doesn't dominate)
        copy_budget = max(1, self.neighbor_size // 2)

        for n in neighbors_sorted:
            if not n.last_moves:
                continue

            # Copy a couple of moves from this neighbor
            take = min(2, len(n.last_moves), copy_budget - len(moves))
            if take <= 0:
                break

            moves.extend(n.last_moves[:take])

            if len(moves) >= copy_budget:
                break

        # Optional safety: keep only moves involving free cells
        free_set = set(self.free_cells)
        moves = [m for m in moves if m[0] in free_set]

        return moves

    def cohesion_moves(
        self,
        solution: DragonflySolution,
        neighbours: List[DragonflySolution],
    ) -> List[Tuple[int, int, int]]:
        """
        Generate cohesion moves (move toward majority center).
        Assign cells based on majority voting across neighbours.
        """
        moves = []

        if not neighbours:
            return moves

        # Count which robot each cell is assigned to across neighbours
        cell_votes = {}
        for cell in self.free_cells:
            votes = [0] * self.num_robots
            for neighbour in neighbours:
                for robot_id, cells in neighbour.partition.items():
                    if cell in cells:
                        votes[robot_id] += 1
            cell_votes[cell] = votes

        # Move cells to majority robot if different from current
        for cell, votes in cell_votes.items():
            majority_robot = int(np.argmax(votes))

            # Majority threshold
            if votes[majority_robot] > len(neighbours) // 2:
                # Find current owner
                current_robot = None
                for robot_id, cells in solution.partition.items():
                    if cell in cells:
                        current_robot = robot_id
                        break

                if current_robot is not None and current_robot != majority_robot:
                    if random.random() < 0.3:  # probabilistic move
                        moves.append((cell, current_robot, majority_robot))

        return moves

    def food_moves(
        self,
        solution: DragonflySolution,
        food: DragonflySolution,
    ) -> List[Tuple[int, int, int]]:
        """
        Generate food moves (copy from best solution).
        Discrete equivalent of:
            F_i = X⁺ − X_i
        """
        moves: List[Tuple[int, int, int]] = []

        for robot_id in range(self.num_robots):
            food_cells = set(food.partition.get(robot_id, []))
            my_cells = set(solution.partition.get(robot_id, []))

            # Cells in food but not in current solution
            to_adopt = food_cells - my_cells

            if to_adopt:
                cells_to_move = random.sample(
                    list(to_adopt),
                    max(1, len(to_adopt) // 2)
                )

                for cell in cells_to_move:
                    # Find which robot currently owns this cell
                    current_robot = None
                    for r, cells in solution.partition.items():
                        if cell in cells:
                            current_robot = r
                            break

                    if current_robot is not None and current_robot != robot_id:
                        moves.append((cell, current_robot, robot_id))

        return moves

    def enemy_moves(
        self,
        solution: DragonflySolution,
        enemy: DragonflySolution,
    ) -> List[Tuple[int, int, int]]:
        """
        Generate enemy moves (move away from worst solution).
        Discrete equivalent of:
            E_i = X⁻ + X_i
        """
        moves: List[Tuple[int, int, int]] = []

        for robot_id in range(self.num_robots):
            enemy_cells = set(enemy.partition.get(robot_id, []))
            my_cells = set(solution.partition.get(robot_id, []))

            # Cells shared with enemy
            common = my_cells & enemy_cells

            # If overlap with enemy is large → push away
            if common and len(common) > 0.5 * len(my_cells):
                cells_to_move = random.sample(
                    list(common),
                    max(1, len(common) // 4)
                )

                for cell in cells_to_move:
                    other_robot = random.choice(
                        [r for r in range(self.num_robots) if r != robot_id]
                    )
                    moves.append((cell, robot_id, other_robot))

        return moves

    def apply_moves(
        self,
        solution: DragonflySolution,
        moves: List[Tuple[int, int, int]],
    ) -> Dict[int, List[int]]:
        """
        Apply moves to partition.

        Args:
            solution: Current solution
            moves: List of (cell, from_robot, to_robot) tuples

        Returns:
            New partition
        """
        new_partition = {r: list(cells) for r, cells in solution.partition.items()}

        for cell, from_robot, to_robot in moves:
            if from_robot not in new_partition or to_robot not in new_partition:
                # Ensure keys exist
                new_partition.setdefault(from_robot, [])
                new_partition.setdefault(to_robot, [])

            if cell in new_partition[from_robot]:
                new_partition[from_robot].remove(cell)
                new_partition[to_robot].append(cell)

        return new_partition

    def repair(self, partition: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Repair partition to ensure:
        1. All free cells are covered
        2. No cell is assigned to multiple robots
        3. Each robot has at least one cell
        """
        # Remove duplicates
        assigned_cells = set()
        for robot_id in range(self.num_robots):
            if robot_id in partition:
                unique_cells = []
                for cell in partition[robot_id]:
                    if cell not in assigned_cells and cell in self.free_cells:
                        unique_cells.append(cell)
                        assigned_cells.add(cell)
                partition[robot_id] = unique_cells
            else:
                partition[robot_id] = []

        # Assign unassigned cells
        unassigned = [c for c in self.free_cells if c not in assigned_cells]
        for cell in unassigned:
            robot_id = random.randint(0, self.num_robots - 1)
            partition.setdefault(robot_id, [])
            partition[robot_id].append(cell)

        # Ensure each robot has at least one cell
        for robot_id in range(self.num_robots):
            if robot_id not in partition or len(partition[robot_id]) == 0:
                # Steal a cell from the largest partition (preferred)
                largest_robot = max(partition.keys(), key=lambda r: len(partition[r]))
                if len(partition[largest_robot]) > 1:
                    cell = partition[largest_robot].pop()
                    partition[robot_id] = [cell]
                else:
                    # Fallback: assign random free cell
                    partition[robot_id] = [random.choice(self.free_cells)]

        return partition

    def update_weights(self, iteration: int):
        """
        Optional: Weight schedule over time (explore -> exploit).
        If you already have your own update_weights, keep it and delete this.
        """
        t = iteration / max(1, (self.max_iterations - 1))

        # Typical DA schedules:
        # - s,a,c decrease over time (exploration reduces)
        # - f,e increase over time (exploitation increases)
        # - w decreases slightly
        self.w = 0.9 - 0.5 * t

        self.s_weight = max(0.1, 2.0 * (1.0 - t))
        self.a_weight = max(0.1, 2.0 * (1.0 - t))
        self.c_weight = max(0.1, 2.0 * (1.0 - t))

        self.f_weight = 2.0 * t + 0.1
        self.e_weight = 1.0 * t + 0.1

    def optimize(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict]:
        """
        Run Dragonfly Algorithm optimization.

        Returns:
            Tuple of (best_solution, history)
        """
        # Step 1: Initialize population
        self.initialize_population()

        if self.verbose:
            print(f"\nStarting Dragonfly Optimization")
            print(f"Population: {self.population_size}, Iterations: {self.max_iterations}")
            print(f"Initial best fitness: {self.food.fitness:.4f}")

        # Escape / stagnancy controls (discrete DA needs this more than the vector version)
        STAGNANCY_LIMIT = 20

        # Main optimization loop
        for iteration in range(self.max_iterations):

            # Step 5 (paper): Update food (best) and enemy (worst)
            self.update_food_and_enemy()

            # Step 6 (paper): Update weights over time (explore -> exploit)
            # If you already have update_weights implemented elsewhere, keep yours.
            self.update_weights(iteration)

            for idx, solution in enumerate(self.population):
                # Get neighbors (neighborhood in solution-space)
                neighbors = self.get_neighbors(idx)

                # ------------------------------------------------------------
                # DA Factors (computed in our discrete solution-space)
                # NOTE: In the original DA paper, these are vectors:
                #   S_i (Eq 1), A_i (Eq 2), C_i (Eq 3), F_i (Eq 4), E_i (Eq 5)
                # We implement "discrete equivalents" that return move-lists
                # of the form (cell, from_robot, to_robot).
                # ------------------------------------------------------------

                # (1) Separation factor  S_i
                # Eq (1):  S_i = - Σ_{j=1..N} ( X_i - X_j )
                S = self.separation_moves(solution, neighbors)

                # (2) Alignment factor  A_i
                # Eq (2):  A_i = ( Σ_{j=1..N} V_j ) / N
                A = self.alignment_moves(solution, neighbors)

                # (3) Cohesion factor  C_i
                # Eq (3):  C_i = ( Σ_{j=1..N} X_j ) / N  -  X_i
                C = self.cohesion_moves(solution, neighbors)

                # (4) Food attraction factor  F_i
                # Eq (4):  F_i = X^+  -  X_i
                F = self.food_moves(solution, self.food)

                # (5) Enemy distraction factor  E_i
                # Eq (5):  E_i = X^-  +  X_i
                E = self.enemy_moves(solution, self.enemy)


                # ------------------------------------------------------------
                # Step update (paper Eq 6) is:
                #   ΔX_{t+1} = (s*S + a*A + c*C + f*F + e*E) + w*ΔX_t
                #
                # Here we implement a discrete Eq 6:
                #  - "ΔX_t" is solution.last_moves (list of successful moves from last step)
                #  - "w*ΔX_t" is done by keeping a fraction of last_moves
                #  - s,a,c,f,e are used as sampling budgets from S,A,C,F,E move pools
                #
                # If there are no neighbors, DA uses Lévy flight.
                # Discrete equivalent: do a small random reassignment (levy_moves)
                # ------------------------------------------------------------

                if neighbors:
                    all_moves = self.merge_moves(solution, S, A, C, F, E, max_moves=50)
                else:
                    all_moves = self.levy_moves(solution, k=10)

                # Apply moves (update "position" X in discrete form: partition)
                new_partition = self.apply_moves(solution, all_moves)

                # Repair partition to satisfy constraints (coverage/uniqueness/non-empty)
                new_partition = self.repair(new_partition)

                # Build paths using UF-STC for the new partition
                try:
                    new_paths = construct_spanning_tree_paths(
                        new_partition, self.grid_width, self.grid_height, self.obstacles
                    )
                except Exception:
                    # If path construction fails, skip this update
                    continue

                # Evaluate fitness
                new_fitness = self.evaluate_fitness(new_partition, new_paths)

                # Accept if improved
                if new_fitness < solution.fitness:
                    solution.partition = new_partition
                    solution.paths = new_paths
                    solution.fitness = new_fitness

                    # ΔX_t memory update (this makes "w*ΔX_t" meaningful next iteration)
                    solution.last_moves = list(all_moves)

                    # Reset stagnancy counter on improvement
                    solution.stagnancy = 0

                    # Update personal best
                    if new_fitness < solution.personal_best_fitness:
                        solution.personal_best_partition = {r: list(cells) for r, cells in new_partition.items()}
                        solution.personal_best_paths = {r: list(path) for r, path in new_paths.items()}
                        solution.personal_best_fitness = new_fitness
                else:
                    # No improvement => stagnancy grows
                    solution.stagnancy += 1

                    # If stuck too long, force an escape perturbation
                    if solution.stagnancy >= STAGNANCY_LIMIT:
                        shake = self.levy_moves(solution, k=25)
                        shaken_partition = self.apply_moves(solution, shake)
                        shaken_partition = self.repair(shaken_partition)

                        try:
                            shaken_paths = construct_spanning_tree_paths(
                                shaken_partition, self.grid_width, self.grid_height, self.obstacles
                            )
                        except Exception:
                            solution.stagnancy = 0
                            continue

                        shaken_fitness = self.evaluate_fitness(shaken_partition, shaken_paths)

                        if shaken_fitness < solution.fitness:
                            solution.partition = shaken_partition
                            solution.paths = shaken_paths
                            solution.fitness = shaken_fitness
                            solution.last_moves = list(shake)

                        solution.stagnancy = 0

            # Update food and enemy (safe to do again for logging correctness)
            self.update_food_and_enemy()

            # Record history
            avg_fitness = float(np.mean([s.fitness for s in self.population]))
            self.history["iteration"].append(iteration)
            self.history["best_fitness"].append(self.food.fitness)
            self.history["avg_fitness"].append(avg_fitness)

            # Print progress
            if self.verbose and (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}/{self.max_iterations} | "
                    f"Best: {self.food.fitness:.4f} | Avg: {avg_fitness:.4f}"
                )

        if self.verbose:
            print(f"\nOptimization complete!")
            print(f"Final best fitness: {self.food.fitness:.4f}")

        # Return best solution
        return self.food.partition, self.food.paths, self.history

    def dedupe_moves(self, moves: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Resolve conflicting moves.
        If multiple moves target the same cell, keep only the last one.
        Also drop moves that don't actually change robot.
        """
        last = {}
        for cell, fr, to in moves:
            if fr == to:
                continue
            last[cell] = (cell, fr, to)

        return list(last.values())

    def merge_moves(
        self,
        solution: DragonflySolution,
        S: List[Tuple[int, int, int]],
        A: List[Tuple[int, int, int]],
        C: List[Tuple[int, int, int]],
        F: List[Tuple[int, int, int]],
        E: List[Tuple[int, int, int]],
        max_moves: int = 50,
    ) -> List[Tuple[int, int, int]]:
        """
        Discrete version of Eq (6):

        ΔX_{t+1} = (s*S + a*A + c*C + f*F + e*E) + w*ΔX_t

        Here:
        - each factor returns a *move list* (cell, from_robot, to_robot)
        - w*ΔX_t is implemented by keeping a fraction of solution.last_moves
        - s,a,c,f,e are implemented by sampling budgets proportional to weights
        """
        moves: List[Tuple[int, int, int]] = []

        # ---- inertia term: w * ΔX_t  ----
        keep = int(round(self.w * len(solution.last_moves)))
        if keep > 0:
            moves.extend(solution.last_moves[:keep])

        # ---- sample from factors proportional to weights ----
        weights = {
            "S": max(0.0, self.s_weight),
            "A": max(0.0, self.a_weight),
            "C": max(0.0, self.c_weight),
            "F": max(0.0, self.f_weight),
            "E": max(0.0, self.e_weight),
        }
        pool = {"S": S, "A": A, "C": C, "F": F, "E": E}

        total_w = sum(weights.values()) + 1e-9
        remaining = max_moves - len(moves)
        if remaining <= 0:
            return self.dedupe_moves(moves)[:max_moves]

        # Compute budgets
        budgets = {k: int(round(remaining * (weights[k] / total_w))) for k in weights}

        # Small correction to ensure total budgets <= remaining
        while sum(budgets.values()) > remaining:
            k = max(budgets, key=lambda x: budgets[x])
            if budgets[k] > 0:
                budgets[k] -= 1
            else:
                break

        # Sample moves from each pool
        for k in ["S", "A", "C", "F", "E"]:
            cand = pool[k]
            b = budgets[k]
            if not cand or b <= 0:
                continue
            if len(cand) <= b:
                moves.extend(cand)
            else:
                moves.extend(random.sample(cand, b))

        # Final cleanup
        moves = self.dedupe_moves(moves)
        return moves[:max_moves]

    def levy_moves(self, solution: DragonflySolution, k: int = 10) -> List[Tuple[int, int, int]]:
        """
        Discrete analogue of Lévy flight:
        when no neighbors exist, do random perturbations to escape.

        We pick k random cells and reassign them to random robots.
        """
        moves = []
        if not self.free_cells:
            return moves

        k = min(k, len(self.free_cells))
        cells = random.sample(self.free_cells, k)

        # Build owner lookup once
        owner = {}
        for r, cs in solution.partition.items():
            for c in cs:
                owner[c] = r

        for cell in cells:
            fr = owner.get(cell, None)
            if fr is None:
                continue
            to = random.choice([r for r in range(self.num_robots) if r != fr])
            moves.append((cell, fr, to))

        return moves


def dragonfly_algorithm(
    all_cells: List[Tuple[int, int]],
    free_cells: List[int],
    obstacles: List[int],
    grid_width: int,
    grid_height: int,
    num_robots: int,
    population_size: int = 30,
    max_iterations: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Wrapper function for Dragonfly Algorithm.

    Returns:
        Dictionary with 'best_solution' and 'history'
    """
    optimizer = DragonflyOptimizer(
        all_cells, free_cells, obstacles,
        grid_width, grid_height, num_robots,
        population_size=population_size,
        max_iterations=max_iterations,
        verbose=verbose
    )

    best_solution, history = optimizer.optimize()

    return {
        "best_solution": best_solution,
        "history": history
    }

if __name__ == "__main__":
    import time
    import os
    import traceback

    # --- Case Study 2 config (same as your case_studies.py) ---
    grid_width, grid_height = 6, 6
    num_robots = 3
    obstacles = [1, 7, 13, 19, 25, 31]

    all_cells = [(x, y) for y in range(grid_height) for x in range(grid_width)]
    free_cells = [i for i in range(grid_width * grid_height) if i not in obstacles]

    # --- Dragonfly params ---
    population_size = 20
    max_iterations = 100
    verbose = True

    print("\n" + "=" * 80)
    print("RUNNING DRAGONFLY - CASE STUDY 2 (6x6, 3 robots)")
    print("=" * 80)
    print(f"Grid: {grid_width}x{grid_height}")
    print(f"Robots: {num_robots}")
    print(f"Obstacles: {len(obstacles)} -> {obstacles}")
    print(f"Free cells: {len(free_cells)}")
    print(f"Params: population_size={population_size}, max_iterations={max_iterations}")
    print("=" * 80)

    # --- Run ---
    start = time.time()
    try:
        results = dragonfly_algorithm(
            all_cells=all_cells,
            free_cells=free_cells,
            obstacles=obstacles,
            grid_width=grid_width,
            grid_height=grid_height,
            num_robots=num_robots,
            population_size=population_size,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    except Exception as e:
        print(f"❌ Dragonfly run failed: {e}")
        traceback.print_exc()
        raise
    end = time.time()

    best_solution = results["best_solution"]
    history = results["history"]

    print("\n" + "-" * 80)
    print("✅ DRAGONFLY DONE")
    print("-" * 80)
    print(f"Runtime: {end - start:.2f}s ({(end - start)/60:.2f} min)")

    # If your RobotCoverageSolution has these helpers, print them (safe checks)
    try:
        if hasattr(best_solution, "combined_score") and best_solution.combined_score is not None:
            print(f"Best combined_score: {best_solution.combined_score:.4f}")
    except Exception:
        pass

    try:
        if hasattr(best_solution, "get_coverage_efficiency"):
            print(f"Coverage efficiency: {best_solution.get_coverage_efficiency():.2f}%")
    except Exception:
        pass

    try:
        if hasattr(best_solution, "get_workload_balance_index"):
            print(f"Balance index: {best_solution.get_workload_balance_index():.4f}")
    except Exception:
        pass

    # Optional: save history to a file
    try:
        os.makedirs("results/case_study_2_dragonfly", exist_ok=True)
        out_path = "results/case_study_2_dragonfly/dragonfly_history.txt"
        with open(out_path, "w") as f:
            for it, bf, af in zip(history["iteration"], history["best_fitness"], history["avg_fitness"]):
                f.write(f"{it}\t{bf}\t{af}\n")
        print(f"📄 Saved history to: {out_path}")
    except Exception as e:
        print(f"⚠️ Could not save history file: {e}")
