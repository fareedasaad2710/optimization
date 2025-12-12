# dragonfly.py
#
# Discrete Dragonfly Algorithm for Multi-Robot Coverage Path Planning
#
# HOW TO USE:
# ----------
# from dragonfly import DragonflyOptimizer
#
# optimizer = DragonflyOptimizer(
#     num_cells=N,
#     num_robots=M,
#     population_size=20,
#     max_iterations=200,
#     generate_initial_solution=your_generate_initial,
#     evaluate=your_evaluate_cost,
#     repair=your_repair_solution,
#     compute_path_lengths=your_compute_path_lengths,  # returns list of L[r]
# )
#
# best_solution, best_cost = optimizer.run()
#
# You must define the callbacks in your main code and pass them in.
#
# A "solution" is whatever object you already use in SA (e.g. class Solution)
# but it MUST satisfy:
#   - solution.paths: list of length M; paths[r] is a list of cell indices for robot r
#   - optionally: solution.assignment[i] = robot index (if you have it).
#                 If not, we can derive assignment from paths (see below).

import copy
import random
from typing import Any, Callable, List, Tuple, Optional


class DragonflyOptimizer:
    def __init__(
        self,
        num_cells: int,
        num_robots: int,
        population_size: int,
        max_iterations: int,
        generate_initial_solution: Callable[[], Any],
        evaluate: Callable[[Any], float],
        repair: Callable[[Any], None],
        compute_path_lengths: Callable[[Any], List[float]],
        # Optional behavior weights
        w_sep: float = 0.2,
        w_align: float = 0.2,
        w_coh: float = 0.2,
        w_food: float = 0.2,
        w_enemy: float = 0.2,
        initial_radius: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.N = num_cells
        self.M = num_robots
        self.NP = population_size
        self.max_iterations = max_iterations

        self.generate_initial_solution = generate_initial_solution
        self.evaluate = evaluate
        self.repair = repair
        self.compute_path_lengths = compute_path_lengths

        # Behavior weights
        self.w_sep = w_sep
        self.w_align = w_align
        self.w_coh = w_coh
        self.w_food = w_food
        self.w_enemy = w_enemy

        self.initial_radius = initial_radius

        if seed is not None:
            random.seed(seed)

        # Internal population storage
        self.population: List[Any] = []
        self.fitness: List[float] = []
        self.best_solution: Any = None
        self.best_fitness: float = float("inf")

    # ----------------- PUBLIC API ----------------- #

    def run(self) -> Tuple[Any, float]:
        """Run the Dragonfly optimization and return (best_solution, best_cost)."""
        self._initialize_population()
        radius = self.initial_radius

        for it in range(self.max_iterations):
            for i in range(self.NP):
                current = self.population[i]
                neighbors = self._get_neighbors(i, radius)

                if not neighbors:
                    candidate = self._random_discrete_move(current)
                else:
                    candidate = self._apply_dragonfly_moves(
                        current=current,
                        neighbors=neighbors,
                        population=self.population,
                        fitness=self.fitness,
                    )

                # Repair and evaluate
                self.repair(candidate)
                new_fit = self.evaluate(candidate)

                # Greedy replacement
                if new_fit < self.fitness[i]:
                    self.population[i] = candidate
                    self.fitness[i] = new_fit

                    # Update global best
                    if new_fit < self.best_fitness:
                        self.best_fitness = new_fit
                        self.best_solution = copy.deepcopy(candidate)

            # Optional: update radius and behavior weights (can keep simple)
            radius = self._update_radius(radius, it)

        return self.best_solution, self.best_fitness

    # ----------------- INITIALIZATION ----------------- #

    def _initialize_population(self) -> None:
        self.population = []
        self.fitness = []

        for _ in range(self.NP):
            sol = self.generate_initial_solution()
            self.repair(sol)
            cost = self.evaluate(sol)
            self.population.append(sol)
            self.fitness.append(cost)

        best_idx = min(range(self.NP), key=lambda i: self.fitness[i])
        self.best_solution = copy.deepcopy(self.population[best_idx])
        self.best_fitness = self.fitness[best_idx]

    # ----------------- NEIGHBORS & DISTANCE ----------------- #

    def _get_neighbors(self, idx: int, radius: float) -> List[int]:
        """Return indices of neighbors whose distance < radius."""
        neighbors: List[int] = []
        current = self.population[idx]

        for j in range(self.NP):
            if j == idx:
                continue
            d = self._solution_distance(current, self.population[j])
            if d < radius:
                neighbors.append(j)

        return neighbors

    def _solution_distance(self, s1: Any, s2: Any) -> float:
        """
        Hamming distance on assignment vectors (normalized).
        If 'assignment' attribute doesn't exist, derive it from paths.
        """
        assign1 = self._get_assignment(s1)
        assign2 = self._get_assignment(s2)

        diff = 0
        for i in range(self.N):
            if assign1[i] != assign2[i]:
                diff += 1

        return diff / max(1, self.N)

    def _get_assignment(self, sol: Any) -> List[int]:
        """
        Returns a list 'assignment' of length N where assignment[i] is the robot index for cell i.
        Assumes cells are 0..N-1 or 1..N; adjust if needed.
        If sol has sol.assignment, we use it; otherwise we derive from paths.
        """
        if hasattr(sol, "assignment") and sol.assignment is not None:
            return list(sol.assignment)  # assume it's indexable

        # Derive from paths: we'll assume cells are in range [0, N-1] OR [1, N]
        # If your indexing is different, adapt this.
        assignment = [-1] * self.N

        for r in range(self.M):
            path = sol.paths[r]
            for c in path:
                # Guess index shift
                idx = c if 0 <= c < self.N else c - 1
                if 0 <= idx < self.N:
                    assignment[idx] = r

        # For any unassigned cells, keep -1 (or assign randomly if you prefer)
        return assignment

    # ----------------- RADIUS & WEIGHTS SCHEDULE ----------------- #

    def _update_radius(self, current_radius: float, iteration: int) -> float:
        """
        Simple schedule: slowly shrink radius, but not below a minimum.
        You can keep this very simple or even return current_radius unchanged.
        """
        # Example: linear decay
        min_radius = 0.05
        t = iteration / max(1, self.max_iterations)
        new_radius = self.initial_radius * (1.0 - t) + min_radius * t
        return new_radius

    # ----------------- RANDOM MOVE (NO NEIGHBORS) ----------------- #

    def _random_discrete_move(self, sol: Any) -> Any:
        new_sol = copy.deepcopy(sol)

        move_type = random.choice(
            ["swap_cells_between_robots", "reinsert_cell_in_path", "local_path_shuffle"]
        )

        if move_type == "swap_cells_between_robots":
            self._swap_cells_between_robots(new_sol)
        elif move_type == "reinsert_cell_in_path":
            self._reinsert_cell_in_path(new_sol)
        else:
            self._local_path_shuffle(new_sol)

        return new_sol

    # ----------------- CORE DRAGONFLY BEHAVIOR ----------------- #

    def _apply_dragonfly_moves(
        self,
        current: Any,
        neighbors: List[int],
        population: List[Any],
        fitness: List[float],
    ) -> Any:
        new_sol = copy.deepcopy(current)

        # Pick a few "good" neighbors (lower fitness)
        sorted_neighbors = sorted(neighbors, key=lambda j: fitness[j])
        k = min(3, len(sorted_neighbors))
        selected_neighbors = sorted_neighbors[:k]

        # Separation: balance workload if heavily imbalanced
        if random.random() < self.w_sep:
            self._balance_workload_move(new_sol)

        # Alignment: align workloads with average of selected neighbors
        if random.random() < self.w_align:
            self._align_workload_with_neighbors(new_sol, selected_neighbors, population)

        # Cohesion: copy subpath from a neighbor
        if random.random() < self.w_coh and selected_neighbors:
            nid = random.choice(selected_neighbors)
            self._copy_subpath_from_solution(new_sol, population[nid])

        # Attraction to food (global best)
        if random.random() < self.w_food and self.best_solution is not None:
            self._copy_subpath_from_solution(new_sol, self.best_solution)

        # Distraction from enemy (global worst)
        if random.random() < self.w_enemy:
            worst_idx = max(range(self.NP), key=lambda j: fitness[j])
            self._avoid_pattern_of_solution(new_sol, population[worst_idx])

        return new_sol

    # ----------------- DISCRETE OPERATORS ----------------- #

    def _swap_cells_between_robots(self, sol: Any) -> None:
        r1 = random.randrange(self.M)
        r2 = random.randrange(self.M)
        while r2 == r1 and self.M > 1:
            r2 = random.randrange(self.M)

        path1 = sol.paths[r1]
        path2 = sol.paths[r2]

        if not path1 or not path2:
            return

        i1 = random.randrange(len(path1))
        i2 = random.randrange(len(path2))

        c1 = path1[i1]
        c2 = path2[i2]

        path1[i1], path2[i2] = c2, c1

        if hasattr(sol, "assignment") and sol.assignment is not None:
            idx1 = c1 if 0 <= c1 < self.N else c1 - 1
            idx2 = c2 if 0 <= c2 < self.N else c2 - 1
            if 0 <= idx1 < self.N:
                sol.assignment[idx1] = r2
            if 0 <= idx2 < self.N:
                sol.assignment[idx2] = r1

    def _reinsert_cell_in_path(self, sol: Any) -> None:
        r = random.randrange(self.M)
        path = sol.paths[r]
        if len(path) < 2:
            return

        from_idx = random.randrange(len(path))
        to_idx = random.randrange(len(path))

        if from_idx == to_idx:
            return

        cell = path[from_idx]
        del path[from_idx]
        path.insert(to_idx, cell)

    def _local_path_shuffle(self, sol: Any) -> None:
        r = random.randrange(self.M)
        path = sol.paths[r]
        if len(path) < 3:
            return

        start = random.randrange(len(path) - 1)
        end = random.randrange(start + 1, len(path))

        subseq = path[start : end + 1]
        random.shuffle(subseq)
        path[start : end + 1] = subseq

    def _balance_workload_move(self, sol: Any) -> None:
        L = self.compute_path_lengths(sol)  # length M
        if not L:
            return

        r_heavy = max(range(self.M), key=lambda r: L[r])
        r_light = min(range(self.M), key=lambda r: L[r])

        if r_heavy == r_light:
            return

        path_heavy = sol.paths[r_heavy]
        if not path_heavy:
            return

        idx = random.randrange(len(path_heavy))
        cell = path_heavy[idx]
        del path_heavy[idx]
        sol.paths[r_light].append(cell)

        if hasattr(sol, "assignment") and sol.assignment is not None:
            idx_cell = cell if 0 <= cell < self.N else cell - 1
            if 0 <= idx_cell < self.N:
                sol.assignment[idx_cell] = r_light

    def _align_workload_with_neighbors(
        self, sol: Any, neighbor_indices: List[int], population: List[Any]
    ) -> None:
        """Simple alignment: try to move a cell if this robot is significantly heavier than neighbors."""
        if not neighbor_indices:
            return

        L_self = self.compute_path_lengths(sol)  # length M
        if not L_self:
            return

        # Average neighbor workloads
        L_avg = [0.0] * self.M
        for r in range(self.M):
            vals = []
            for nid in neighbor_indices:
                vals.append(self.compute_path_lengths(population[nid])[r])
            if vals:
                L_avg[r] = sum(vals) / len(vals)
            else:
                L_avg[r] = L_self[r]

        # Find robot where we are much heavier than neighbors
        diff = [L_self[r] - L_avg[r] for r in range(self.M)]
        r_heavy = max(range(self.M), key=lambda r: diff[r])

        if diff[r_heavy] <= 0:
            return

        # Try to move one cell from r_heavy to some other robot
        r_target = random.randrange(self.M)
        if r_target == r_heavy and self.M > 1:
            r_target = (r_target + 1) % self.M

        path_heavy = sol.paths[r_heavy]
        if not path_heavy:
            return

        idx = random.randrange(len(path_heavy))
        cell = path_heavy[idx]
        del path_heavy[idx]
        sol.paths[r_target].append(cell)

        if hasattr(sol, "assignment") and sol.assignment is not None:
            idx_cell = cell if 0 <= cell < self.N else cell - 1
            if 0 <= idx_cell < self.N:
                sol.assignment[idx_cell] = r_target

    def _copy_subpath_from_solution(self, target: Any, source: Any) -> None:
        r = random.randrange(self.M)
        src_path = source.paths[r]
        tgt_path = target.paths[r]

        if not src_path:
            return

        # choose subsegment
        start = random.randrange(len(src_path))
        end = random.randrange(start, len(src_path))
        subpath = src_path[start : end + 1]

        # remove those cells from all target paths
        for q in range(self.M):
            self._remove_cells_from_path(target.paths[q], subpath)

        # insert subpath into target robot path at random position
        insert_pos = random.randrange(len(tgt_path) + 1)
        for i, c in enumerate(subpath):
            tgt_path.insert(insert_pos + i, c)

        # update assignment if exists
        if hasattr(target, "assignment") and target.assignment is not None:
            for c in subpath:
                idx_cell = c if 0 <= c < self.N else c - 1
                if 0 <= idx_cell < self.N:
                    target.assignment[idx_cell] = r

    def _remove_cells_from_path(self, path: List[int], cells: List[int]) -> None:
        cell_set = set(cells)
        i = 0
        while i < len(path):
            if path[i] in cell_set:
                del path[i]
            else:
                i += 1

    def _avoid_pattern_of_solution(self, target: Any, enemy: Any) -> None:
        """
        Very simple 'distraction from enemy':
        if target shares too many cells in the same robot with enemy,
        apply a local shuffle on that robot.
        """
        # Compute similarity per robot
        for r in range(self.M):
            t_path = target.paths[r]
            e_path = enemy.paths[r]
            if not t_path or not e_path:
                continue
            common = set(t_path).intersection(e_path)
            if len(common) > len(t_path) * 0.5:
                # if more than 50% overlap, shuffle a segment
                self._local_path_shuffle(target)

    # ----------------- OPTIONAL: DEMO HOOK ----------------- #

if __name__ == "__main__":
    # This is just a tiny demo stub; replace with your real callbacks.

    class DummySolution:
        def __init__(self, M, cells_per_robot):
            self.paths: List[List[int]] = [[] for _ in range(M)]
            self.assignment: List[int] = [-1] * (M * cells_per_robot)
            cell_id = 0
            for r in range(M):
                for _ in range(cells_per_robot):
                    self.paths[r].append(cell_id)
                    self.assignment[cell_id] = r
                    cell_id += 1

    M = 3
    cells_per_robot = 4
    N = M * cells_per_robot

    def demo_generate_initial() -> DummySolution:
        # Simple initial solution: sequential cells per robot
        return DummySolution(M, cells_per_robot)

    def demo_evaluate(sol: DummySolution) -> float:
        # Fake objective: sum of squared path lengths (just as a placeholder)
        lengths = [len(p) for p in sol.paths]
        return float(sum(L * L for L in lengths))

    def demo_repair(sol: DummySolution) -> None:
        # For demo we do nothing
        pass

    def demo_compute_path_lengths(sol: DummySolution) -> List[float]:
        # For demo, length = number of cells
        return [float(len(p)) for p in sol.paths]

    optimizer = DragonflyOptimizer(
        num_cells=N,
        num_robots=M,
        population_size=10,
        max_iterations=50,
        generate_initial_solution=demo_generate_initial,
        evaluate=demo_evaluate,
        repair=demo_repair,
        compute_path_lengths=demo_compute_path_lengths,
        seed=42,
    )

    best_sol, best_cost = optimizer.run()
    print("Demo finished. Best cost:", best_cost)
    print("Best paths:", best_sol.paths)
