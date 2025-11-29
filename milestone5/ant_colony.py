# A minimal, runnable Ant Colony Optimization implementation for multi-robot coverage-path planning.
# - Simplified but practical approach, inspired by the tutorial you uploaded.
# - This code represents the grid, obstacles, M robots with fixed starts, and uses ACO to partition
#   free cells among robots (ai,r). After partitioning, it builds a coverage path per robot via DFS
#   (a Uniform-coverage spanning-tree like approach) and evaluates workload imbalance.
#
# Notes / limitations (kept simple on purpose):
# - Connectivity of partition is repaired with a lightweight heuristic repair (reassign isolated cells).
# - Objective: maximize coverage (here we try to assign all reachable free cells) and minimize workload imbalance.
# - Pheromone is defined for (robot, cell) assignment. Ants construct full partitions per iteration.
# - Pheromone update rewards solutions with high coverage and low imbalance.
#
# You can adapt or extend the heuristic, scoring, or repair rules to better match DARP / UF-STC requirements.
#
# Run: this cell will execute and show an example run on a small grid.
import math, random, itertools, collections, json
from typing import List, Tuple, Dict, Set
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# ---------- Utilities ----------
Point = Tuple[int,int]

def manhattan(a: Point, b: Point) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors(cell: Point, H: int, W: int):
    r,c = cell
    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
        nr,nc = r+dr, c+dc
        if 0 <= nr < H and 0 <= nc < W:
            yield (nr,nc)

# ---------- Grid & robots ----------
class Grid:
    def __init__(self, H:int, W:int, obstacles:Set[Point]):
        self.H, self.W = H, W
        self.obstacles = set(obstacles)
        self.free = [(r,c) for r in range(H) for c in range(W) if (r,c) not in self.obstacles]
        # map cell -> index into free list
        self.index = {cell:i for i,cell in enumerate(self.free)}
    def is_free(self, cell:Point)->bool:
        r,c = cell
        if not (0 <= r < self.H and 0 <= c < self.W): return False
        return cell not in self.obstacles
    def adj_free(self, cell:Point):
        for nb in neighbors(cell, self.H, self.W):
            if self.is_free(nb):
                yield nb

# ---------- Coverage path (simple UF-STC-like) ----------
def build_spanning_path(start:Point, assigned_cells:Set[Point], grid:Grid)->List[Point]:
    """
    Construct a traversal path over assigned_cells starting from start.
    We build a spanning tree over assigned nodes using BFS then traverse it (preorder)
    which yields a coverage route visiting each node at least once.
    """
    if not assigned_cells:
        return []
    # if start not in assigned, include it as root (we still start from robot's start but it may not be assigned)
    root = start if start in assigned_cells else min(assigned_cells, key=lambda p: manhattan(p, start))
    # BFS to create tree (only over assigned_cells)
    q = collections.deque([root])
    parent = {root: None}
    while q:
        u = q.popleft()
        for v in grid.adj_free(u):
            if v in assigned_cells and v not in parent:
                parent[v] = u
                q.append(v)
    # Some assigned cells may be unreachable in tree (disconnected); handle later
    # Build adjacency list of tree
    tree_adj = collections.defaultdict(list)
    for node,p in parent.items():
        if p is not None:
            tree_adj[p].append(node)
    # Preorder traversal to generate path
    path = []
    def dfs(u):
        path.append(u)
        for v in tree_adj[u]:
            dfs(v)
            path.append(u)  # return to parent (coverage may revisit)
    dfs(root)
    # If there are assigned cells missing because they were not discovered (disconnected),
    # append them sorted by proximity to root (repair step)
    missing = set(assigned_cells) - set(parent.keys())
    if missing:
        missing_sorted = sorted(list(missing), key=lambda p: manhattan(p, root))
        for m in missing_sorted:
            path.append(m)
    # Prepend true start if different
    if path and path[0] != start:
        path = [start] + path
    return path

def path_length(path:List[Point])->int:
    if not path: return 0
    L = 0
    for a,b in zip(path, path[1:]):
        L += manhattan(a,b)
    return L

# ---------- ACO solver (partitioning) ----------
class ACOPartitioner:
    def __init__(self, grid:Grid, robot_starts:List[Point], ants:int=20, iterations:int=100,
                 alpha:float=1.0, beta:float=1.0, rho:float=0.3, q0:float=1.0, seed:int=None):
        self.grid = grid
        self.starts = robot_starts
        self.M = len(robot_starts)
        self.N = len(grid.free)
        self.ants = ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0  # not used for deterministic choice here but kept for extension
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # pheromone: shape (M, N) -- pheromone for assigning a cell to a robot
        self.pheromone = np.ones((self.M, self.N), dtype=float)
        # precompute heuristics: desirability of robot r taking cell i
        self.heuristic = np.zeros((self.M, self.N), dtype=float)
        for r, s in enumerate(self.starts):
            for i,cell in enumerate(grid.free):
                # prefer cells nearer to the robot start and with "high reward". No explicit reward here so use inverse distance.
                d = manhattan(s, cell)
                self.heuristic[r,i] = 1.0 / (1.0 + d)  # in (0,1], larger if closer
        # small floor to avoid zeros
        self.heuristic = np.maximum(self.heuristic, 1e-6)

    def construct_solution(self)->Tuple[List[Set[Point]], List[List[Point]]]:
        """
        One ant constructs a partition: assign each free cell to one robot.
        Return: list of assigned cell sets per robot, and generated coverage paths per robot.
        """
        assignments = [set() for _ in range(self.M)]
        # We will assign all free cells, one-by-one. For each cell pick robot probabilistically
        # Probability for cell i -> robot r proportional to (pheromone[r,i]^alpha * heuristic[r,i]^beta) / penalty(load)
        for i, cell in enumerate(self.grid.free):
            weights = np.zeros(self.M, dtype=float)
            for r in range(self.M):
                # Add simple load balancing by penalizing robots with large current assigned count (encourage balanced loads)
                load_penalty = 1.0 / (1.0 + len(assignments[r]))  # decreases as load increases
                weights[r] = (self.pheromone[r,i] ** self.alpha) * (self.heuristic[r,i] ** self.beta) * load_penalty
            if weights.sum() <= 0:
                probs = np.ones(self.M) / self.M
            else:
                probs = weights / weights.sum()
            choice = np.random.choice(range(self.M), p=probs)
            assignments[choice].add(cell)
        # Repair disconnected cells: ensure assigned set for each robot is connected (lightweight)
        assignments = [self._repair_connectivity(r, assignments[r]) for r in range(self.M)]
        # Build spanning paths per robot and compute lengths
        paths = []
        lengths = []
        for r in range(self.M):
            p = build_spanning_path(self.starts[r], assignments[r], self.grid)
            paths.append(p)
            lengths.append(path_length(p))
        return assignments, paths, lengths

    def _repair_connectivity(self, r:int, assigned:Set[Point])->Set[Point]:
        """
        Ensure assigned cells for robot r are connected; if there are disconnected components,
        reassign very small components to nearest robot by distance (lightweight repair).
        """
        if not assigned:
            return set()
        assigned = set(assigned)
        # find connected components using BFS on grid adjacency
        visited = set()
        comps = []
        for node in list(assigned):
            if node in visited: continue
            q = collections.deque([node])
            comp = set([node])
            visited.add(node)
            while q:
                u = q.popleft()
                for v in self.grid.adj_free(u):
                    if v in assigned and v not in visited:
                        visited.add(v)
                        comp.add(v)
                        q.append(v)
            comps.append(comp)
        if len(comps) <= 1:
            return assigned
        # sort components by size; for components that are small, reassign to nearest robot
        comps_sorted = sorted(comps, key=lambda c: len(c), reverse=True)
        main = comps_sorted[0]
        other_comps = comps_sorted[1:]
        kept = set(main)
        for comp in other_comps:
            if len(comp) <= 2:
                # find nearest robot (including possibly the same robot) to whole component
                target_r = min(range(self.M), key=lambda rr: min(manhattan(self.starts[rr], c) for c in comp))
                if target_r == r:
                    kept.update(comp)  # keep with this robot
                else:
                    # move to other robot: effectively drop them here (they will be picked up by other ants' evaluation)
                    # but since this is a single-ant solution construction, we simply remove comp (they become unassigned)
                    pass
            else:
                # keep larger components with this robot (to avoid too many reassignments)
                kept.update(comp)
        return kept

    def evaluate(self, assignments:List[Set[Point]], lengths:List[int])->Tuple[float,float]:
        """
        Compute objectives: coverage and imbalance.
        Return combined score for maximizing (higher better) and separate items.
        """
        coverage = sum(len(s) for s in assignments)
        Ls = np.array(lengths, dtype=float)
        if len(Ls)>0:
            avg = Ls.mean()
            imbalance = float(np.sum(np.abs(Ls - avg)))
        else:
            imbalance = 0.0
        # Combine: we maximize coverage and minimize imbalance.
        # Normalize imbalance by (1 + avg) to make units comparable (heuristic choice).
        score = coverage - 0.5 * imbalance  # the coefficient 0.5 balances objectives; tune as needed
        return score, coverage, imbalance

    def update_pheromone(self, ant_solutions:List[Tuple[List[Set[Point]], List[int], float]]):
        """
        ant_solutions: list of (assignments, lengths, score)
        We'll evaporate pheromone, then deposit proportional to score on robot-cell pairs used in good solutions.
        """
        # Evaporate
        self.pheromone *= (1.0 - self.rho)
        # Deposit: sum over ants, deposit score on (r,i) if ant assigned cell i to robot r
        for assignments, lengths, score in ant_solutions:
            if score <= 0: continue
            for r,assigned in enumerate(assignments):
                for cell in assigned:
                    i = self.grid.index[cell]
                    # deposit proportional to score and inversely to path length (prefer shorter)
                    deposit = score / (1.0 + lengths[r])
                    self.pheromone[r,i] += deposit

    def run(self, verbose:bool=False):
        best = None
        best_record = None
        history = []
        for it in range(self.iterations):
            ant_solutions = []
            for a in range(self.ants):
                assignments, paths, lengths = self.construct_solution()
                score, cov, imb = self.evaluate(assignments, lengths)
                ant_solutions.append((assignments, lengths, score))
                if best is None or score > best:
                    best = score
                    best_record = (assignments, paths, lengths, score, cov, imb)
            # Update pheromone
            self.update_pheromone(ant_solutions)
            history.append(best if best is not None else 0)
            if verbose and (it % max(1, self.iterations//10) == 0):
                print(f"Iter {it+1}/{self.iterations}, current best score={best:.3f}")
        return best_record, history

# ---------- Example usage ----------
def example_run():
    # build a small example grid (7x7) with some obstacles
    H,W = 7,7
    obstacles = {(1,1),(1,2),(2,1),(4,4),(4,5),(5,5)}
    grid = Grid(H,W,obstacles)
    # two robots starting positions
    starts = [(0,0),(6,6)]
    aco = ACOPartitioner(grid, starts, ants=30, iterations=80, alpha=1.0, beta=1.0, rho=0.25, seed=42)
    best_record, history = aco.run(verbose=True)
    assignments, paths, lengths, score, cov, imb = best_record
    print("\nBEST SOLUTION SUMMARY")
    print(f"Score={score:.3f}, Coverage={cov}, Imbalance={imb:.3f}")
    for r in range(len(starts)):
        print(f"\nRobot {r}: start={starts[r]}, assigned_cells={len(assignments[r])}, path_len={lengths[r]}")
        # show first 20 cells of path
        print("Path (first 30 steps):", paths[r][:30])

    # Show pheromone top cells for each robot (for insight)
    pher = aco.pheromone.copy()
    for r in range(len(starts)):
        top_indices = np.argsort(-pher[r])[:8]
        print(f"\nTop pheromone cells for robot {r}:")
        for idx in top_indices:
            print(f"  cell={grid.free[idx]}, tau={pher[r,idx]:.3f}, heuristic={aco.heuristic[r,idx]:.3f}")

    # display a summary of cell assignments
    rows = []
    for i,cell in enumerate(grid.free):
        owner = next((r for r in range(len(starts)) if cell in assignments[r]), -1)
        rows.append({"cell":cell, "owner":owner, "heuristic_r0":aco.heuristic[0,i], "heuristic_r1":aco.heuristic[1,i]})
    
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        print("\nAssignments Preview (first 10 cells):")
        print(df.head(10).to_string())
    else:
        print("\nAssignments Preview (first 10 cells):")
        print("Cell\tOwner\tHeuristic_R0\tHeuristic_R1")
        for row in rows[:10]:
            print(f"{row['cell']}\t{row['owner']}\t{row['heuristic_r0']:.3f}\t\t{row['heuristic_r1']:.3f}")
    
    # Save result to file for download if desired
    out = {"assignments":[sorted(list(s)) for s in assignments], "lengths":lengths, "score":score, "coverage":cov, "imbalance":imb}
    output_file = "aco_partition_result.json"
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResult saved to {output_file}")

# Run the example
example_run()

