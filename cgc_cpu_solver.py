"""
Single-file CPU implementation of the Factorised Cut-Glue-Cut (FCGC) solver
for complete multicut with factorised costs c_uv = <f_u, f_v> - alpha.

Self-contained NumPy/joblib version of logic/factorised_cut_glue_and_cut.py
with its dependencies inlined; no imports from the logic/ or utils/ packages.

Instance file format:
    FACTORIZED COMPLETE MULTICUT
    alpha
    n d
    feature_node_1_dim_1 ... feature_node_1_dim_d
    ...

Usage:
    python cgc_cpu_solver.py --path instance.txt [--parallel] [--rank K] ...
"""
import argparse
import os
import time
from collections import deque

import numpy as np
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def default_rank(n: int) -> int:
    """
    Default rank for the low-rank relaxation: sqrt(2n) + 1.

    By the Barvinok-Pataki bound, the max-cut SDP always admits an optimal
    solution of rank at most sqrt(2n), so this rank loses nothing in the
    relaxation while keeping X of size n x O(sqrt(n)).
    """
    return int(np.sqrt(2 * n)) + 1


def normalize_matrix(X):
    """Normalize the rows of X to the unit sphere; zero rows are left untouched."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def calculate_factorised_multicut(P, F, alpha=None):
    """
    Multicut objective for partition P under factorised costs c_uv = <f_u, f_v> - alpha.

    Uses sum_{u in U, v in V} <f_u, f_v> = <sum_{u in U} f_u, sum_{v in V} f_v>, so the
    value is computed in O(n*d) without instantiating any |U| x |V| cost block.
    """
    if alpha is None:
        alpha = 0.0
    if len(P) < 2:
        return 0.0

    component_sums = np.array([np.sum(F[list(U), :], axis=0) for U in P])
    component_sizes = np.array([len(U) for U in P])

    total_sum = np.sum(component_sums, axis=0)
    feature_term = (total_sum @ total_sum - np.sum(component_sums * component_sums)) / 2

    n = np.sum(component_sizes)
    cut_pair_count = (n ** 2 - np.sum(component_sizes ** 2)) / 2

    return float(feature_term - alpha * cut_pair_count)


def relaxation_cut_bound(F, alpha, relaxation_value):
    """
    Convert a factorised max-cut relaxation objective <C, X X^T> (diagonal included,
    C = F F^T - alpha) into cut units: the bipartition cut value an integral solution
    with that objective would have. At the relaxation optimum this is a lower bound
    on the minimum bipartition cut of the subproblem.
    """
    m = F.shape[0]
    sq = np.sum(F ** 2)
    diag = sq - alpha * m  # sum_i <f_i, f_i> - alpha
    total = np.sum(F, axis=0)
    pair_sum = (total @ total - sq) / 2 - alpha * m * (m - 1) / 2  # sum_{i<j} c_ij
    return (pair_sum - (relaxation_value - diag) / 2) / 2


# ---------------------------------------------------------------------------
# Low-rank max-cut subproblem solver (projected gradient ascent + rounding)
# ---------------------------------------------------------------------------

def get_factorised_loss(F, X, alpha):
    """Relaxation objective ||F^T X||_F^2 - alpha * ||X^T 1||^2."""
    n = X.shape[0]
    FX = F.T @ X

    F_term = np.sum(FX ** 2)

    alpha_term = alpha * np.sum((X.T @ np.ones((n,))) ** 2)

    return F_term - alpha_term


def _factorised_projected_gradient_descent(F, k, alpha, step_size=1e-4, tol=1e-6, max_iter=10000,
                                           step_increase=1.2, step_decrease=0.5, min_step=1e-12):
    """
    Perform factorised projected gradient ascent with an adaptive step size:
    a trial step that improves the relaxation objective is accepted and the step size
    is increased; a trial step that does not improve is rejected (X unchanged) and the
    step size is decreased. The gradient is only recomputed after accepted steps, so
    rejected trials cost one loss evaluation, not a gradient.

    Stops when an accepted step improves by less than tol (relative), when the step
    size collapses below min_step, or after max_iter trials.
    """
    losses = []
    n = F.shape[0]
    FT = F.T
    ones_n = np.ones((n, 1))
    X = np.random.randn(n, k)

    X = normalize_matrix(X)

    loss = get_factorised_loss(F, X, alpha)
    losses.append(loss)

    g = F @ (FT @ X) - alpha * (ones_n @ (ones_n.T @ X))

    for _ in range(max_iter):
        X_trial = normalize_matrix(X + step_size * g)
        trial_loss = get_factorised_loss(F, X_trial, alpha=alpha)

        if trial_loss > loss:
            improvement = (trial_loss - loss) / max(abs(loss), 1e-8)
            X, loss = X_trial, trial_loss
            losses.append(loss)
            g = F @ (FT @ X) - alpha * (ones_n @ (ones_n.T @ X))
            step_size *= step_increase
            if improvement < tol:
                break
        else:
            step_size *= step_decrease
            if step_size < min_step:
                break

    return X, losses


def factorised_cut_partition(X, F, alpha, original_indices=None, num_random_hyperplanes=0):
    """
    Round X by the deterministic top-eigenvector hyperplane of X^T X (plus
    optional random hyperplanes, keeping the best) and evaluate the resulting
    bipartition exactly with calculate_factorised_multicut.
    """
    eigvals, eigvecs = np.linalg.eigh(X.T @ X)
    candidates = [eigvecs[:, -1]]
    candidates += [np.random.randn(X.shape[1]) for _ in range(num_random_hyperplanes)]

    P, cut_value = None, np.inf
    for cand in candidates:
        partition = (X @ cand) > 0
        U1 = np.where(partition)[0]
        U2 = np.where(~partition)[0]
        P_cand = [U1.tolist(), U2.tolist()]

        cut_cand = calculate_factorised_multicut(P_cand, F, alpha)

        if cut_cand < cut_value:
            P, cut_value = P_cand, cut_cand

    if original_indices is not None:
        P = [[original_indices[i] for i in partition] for partition in P]

    P = [p for p in P if p != []]

    return P, cut_value


def factorised_projected_gradient_descent(F, k, alpha, step_size=1e-4, tol=1e-6, max_iter=10000,
                                          original_indices=None, mute=True,
                                          num_random_hyperplanes=0, return_relaxation=False):
    """
    Solve + round one max-cut subproblem. With return_relaxation=True additionally
    returns the final relaxation objective <F F^T - alpha*ones, X X^T> (diagonal included).
    """
    X, losses = _factorised_projected_gradient_descent(F, k, alpha=alpha, step_size=step_size,
                                                       tol=tol, max_iter=max_iter)
    P, cut_value = factorised_cut_partition(X, F, alpha, original_indices=original_indices,
                                            num_random_hyperplanes=num_random_hyperplanes)

    if return_relaxation:
        return P, cut_value, losses[-1]
    return P, cut_value


# ---------------------------------------------------------------------------
# Factorised Cut-Glue-Cut
# ---------------------------------------------------------------------------

class FactorisedCutGlueCut:
    """
    Implementation of the Factorised Cut-Glue-Cut (FCGC) algorithm for solving the Multicut problem.
    This approach applies a factorised version of projected gradient descent to partition graphs,
    followed by a glue-and-cut step to iteratively refine the clustering.
    """
    def __init__(self, F, k=None, base_tol=1e-3, step_size=0.001, alpha=0.0, mute=False,
                 parallel=True, max_glue_rounds=10, pgd_max_iter=1000, recut_top_k=32):
        """
        Initialize the Factorised Cut-Glue-Cut solver.

        :param F: Factorised cost matrix representing the graph.
        :param k: The rank for the low-rank approximation (if None, computed from F's size).
        :param base_tol: Base tolerance for stopping criteria.
        :param step_size: Step size for projected gradient descent.
        :param alpha: Regularization parameter. Edge costs are c_uv = <f_u, f_v> - alpha.
        :param mute: If True, suppresses print output.
        :param parallel: If True, fans the subproblem solves out over processes (joblib).
        :param max_glue_rounds: Maximum number of rounds per glue-and-cut step.
        :param pgd_max_iter: Iteration budget per max-cut subproblem. Subproblems need not
                             be solved to high accuracy: moves are only accepted if the
                             exactly evaluated cut value improves, so a rough solve can at
                             worst miss a move, which later iterations can retry.
        :param recut_top_k: Number of re-cut candidate pairs solved per glue round. Merges
                            are applied exactly from pairwise cross costs without solves;
                            only re-cut attempts (re-splitting a pair's union) need a
                            max-cut solve, so candidates are ranked by |cross cost| and
                            only the top recut_top_k are solved.
        """
        self.F = F
        self.k = k if k is not None else default_rank(F.shape[0])
        self.max_glue_rounds = max_glue_rounds
        self.pgd_max_iter = pgd_max_iter
        self.recut_top_k = recut_top_k
        # Components whose max-cut attempt found no improving split. They are skipped
        # in later cut phases until they change (glue produces new node sets), so a
        # cut phase over a stable partition is a no-op and control passes directly
        # to the glue-and-cut phase.
        self.failed_cuts = set()
        self.tol = max(base_tol / np.sqrt(len(self.F)), 1e-6)
        self.step_size = step_size
        self.alpha = alpha
        self.mute = mute
        self.parallel = parallel
        self.objective = []  # Stores objective values at each iteration
        self.final_objective = 0
        self.runtimes = []  # Stores runtime at each iteration
        self.start = time.time()  # placeholder, whenever run() is started it is reset to the new time.

    def _factorised_cut(self, P):
        """
        Perform the cutting step: partitions input clusters into smaller subclusters.
        Components remembered in self.failed_cuts are skipped.
        """
        # Objective updates below are incremental; seed the baseline when the
        # phase is invoked directly, without run() having logged it.
        if not self.objective:
            self.objective.append(calculate_factorised_multicut(P, self.F, self.alpha))
            self.runtimes.append(time.perf_counter() - self.start)

        Q = deque(U for U in P if len(U) > 1 and frozenset(U) not in self.failed_cuts)
        if not Q and not self.mute:
            print("  cut phase: all components stable, skipping to glue-and-cut")

        def process_partition(U):
            if len(U) > 1:
                sub_cost = self.F[U, :]
                k = min(self.k, default_rank(len(U)))  # a rank beyond sqrt(2|U|)+1 gains nothing on the subproblem
                p, cut, relaxation = factorised_projected_gradient_descent(
                    sub_cost, k, self.alpha, self.step_size, self.tol,
                    max_iter=self.pgd_max_iter, original_indices=U, mute=self.mute,
                    return_relaxation=True
                )
                return p, cut, relaxation, U
            return None, None, None, None

        while Q:
            current_partitions = list(Q)
            Q.clear()
            if self.parallel:
                # Process partitions in parallel for efficiency
                partitions_and_values = Parallel(n_jobs=-1)(
                    delayed(process_partition)(U) for U in current_partitions
                )
            else:
                partitions_and_values = [process_partition(U) for U in current_partitions]

            for partitions, cut_value, relaxation, U in partitions_and_values:
                if partitions is None:
                    continue

                # A genuine improvement needs a real split (a one-sided rounding returns
                # [U] itself, whose true cut value is 0 up to summation noise) and a cut
                # value below fp noise, mirroring the 1e-9 epsilon of the glue phase.
                improving = cut_value < -1e-9 and len(partitions) > 1

                if not self.mute:
                    bound = relaxation_cut_bound(self.F[U, :], self.alpha, relaxation)
                    status = "accepted" if improving else "rejected"
                    print(f"  cut solve: |U|={len(U)}, cut {cut_value:.4f}, "
                          f"relaxation bound {bound:.4f} ({status})")

                if improving:
                    P.remove(U)  # Remove original cluster
                    for p in partitions:
                        P.append(p)  # Add new partitions
                        if len(p) > 1:
                            Q.append(p)  # Push to queue for further refinement

                    # A split changes the objective by exactly the cut value of its
                    # new boundary (edges elsewhere keep their cut status), so the
                    # O(nd) full recompute is deferred to once per outer iteration.
                    new_obj = self.objective[-1] + cut_value
                    self.objective.append(new_obj)
                    self.runtimes.append(time.perf_counter() - self.start)
                    if not self.mute:
                        print(f"  cut: objective {new_obj:.4f}, {len(P)} components, "
                              f"{self.runtimes[-1]:.2f}s elapsed")
                else:
                    self.failed_cuts.add(frozenset(U))

        return P

    def _factorised_glue_cut(self, P):
        """
        Perform the glue-and-cut step on the component-level (meta) problem.

        The cross cost between two components depends only on per-component summaries:
        cost(R1, R2) = <s_R1, s_R2> - alpha*|R1|*|R2| with s_R = sum_{u in R} f_u, so
        for m components ALL pairwise costs are one m x m matrix S S^T - alpha n n^T,
        computed in O(m^2 d) without any |R1| x |R2| block or max-cut solve.

        Glue: merging R1 and R2 improves the objective by exactly cost(R1, R2), so all
        profitable merges are read off the matrix and applied greedily (largest gain
        first, summaries updated incrementally) -- no solver involved.

        Cut: a pair with non-positive cross cost may still admit a cheaper split of its
        union, which does need a max-cut solve. Only the top recut_top_k candidates
        ranked by |cross cost| (the value at stake on that boundary) are solved per
        round.
        """
        # Objective updates below are incremental; seed the baseline when the
        # phase is invoked directly, without run() having logged it.
        if not self.objective:
            self.objective.append(calculate_factorised_multicut(P, self.F, self.alpha))
            self.runtimes.append(time.perf_counter() - self.start)

        Pt = [list(R) for R in P]
        failed_pairs = set()  # pairs whose re-cut solve found no improvement

        def process_pair(R1, R2, old_cut_value):
            """
            Re-cut solve on the union of R1 and R2. Returns (R1, R2, partitions,
            old_cut_value, cut), with partitions=None if no bipartition of the union
            beats the current R1|R2 split. Failure bookkeeping and diagnostics happen
            in the caller: under parallel processing this closure runs in a worker
            process, so mutating failed_pairs (or printing) here would only affect the
            worker's copy and be lost or interleaved.
            """
            SR = R1 + R2
            k = min(self.k, default_rank(len(SR)))
            p, cut = factorised_projected_gradient_descent(
                self.F[SR, :], k, self.alpha, self.step_size, self.tol,
                max_iter=self.pgd_max_iter, original_indices=SR, mute=self.mute
            )
            # len(p) > 1: profitable merges were already applied exactly in the glue
            # stage, so a one-sided rounding here is fp noise, not a move (cf. the
            # analogous guard in the cut phase).
            if cut < old_cut_value - 1e-9 and len(p) > 1:
                return R1, R2, p, old_cut_value, cut
            return R1, R2, None, old_cut_value, cut

        for glue_round in range(self.max_glue_rounds):
            # Glue stage: exact greedy merges of the largest-gain pair, from
            # all pairwise cross costs of the component summaries.
            S = np.array([self.F[R, :].sum(axis=0) for R in Pt])
            sizes = np.array([len(R) for R in Pt], dtype=float)
            cross = S @ S.T - self.alpha * np.outer(sizes, sizes)
            np.fill_diagonal(cross, -np.inf)

            # Merged-away components are masked out with -inf rows/columns
            # instead of deleting them, which would copy the whole matrix
            # per merge; the arrays are compacted once after the loop.
            merges = 0
            merge_gain = 0.0
            alive = np.ones(len(Pt), dtype=bool)
            alive_count = len(Pt)
            while alive_count > 1:
                i, j = np.unravel_index(np.argmax(cross), cross.shape)
                gain = cross[i, j]
                if gain <= 1e-9:
                    break
                Pt[i] = Pt[i] + Pt[j]
                S[i] += S[j]
                sizes[i] += sizes[j]
                Pt[j] = None
                alive[j] = False
                alive_count -= 1
                cross[j, :] = -np.inf
                cross[:, j] = -np.inf
                new_row = S @ S[i] - self.alpha * sizes * sizes[i]
                new_row[~alive] = -np.inf
                cross[i, :] = new_row
                cross[:, i] = new_row
                cross[i, i] = -np.inf
                merges += 1
                merge_gain += gain
            if merges:
                Pt = [R for R in Pt if R is not None]
                keep = np.flatnonzero(alive)
                cross = cross[np.ix_(keep, keep)]
            if merges:
                # Merging removes exactly the merged pairs' cross costs from the cut.
                merge_obj = self.objective[-1] - merge_gain
                self.objective.append(merge_obj)
                self.runtimes.append(time.perf_counter() - self.start)
                if not self.mute:
                    print(f"  glue round {glue_round + 1}: {merges} exact merges, "
                          f"gain {merge_gain:.4f}, objective {merge_obj:.4f}, "
                          f"{len(Pt)} components")

            # Cut stage: rank remaining pairs by |cross cost|, solve only the top k.
            m = len(Pt)
            iu, ju = np.triu_indices(m, k=1)
            selected = []
            skipped = 0
            for idx in np.argsort(-np.abs(cross[iu, ju])):
                if len(selected) == self.recut_top_k:
                    break
                i, j = int(iu[idx]), int(ju[idx])
                if frozenset([tuple(Pt[i]), tuple(Pt[j])]) in failed_pairs:
                    skipped += 1
                    continue
                selected.append((Pt[i], Pt[j], cross[i, j]))

            if not self.mute and selected:
                print(f"  glue round {glue_round + 1}: re-cut solving top {len(selected)} "
                      f"of {len(iu)} pairs ({skipped} skipped as known failures)")

            if self.parallel:
                results = Parallel(n_jobs=-1)(
                    delayed(process_pair)(R1, R2, c) for R1, R2, c in selected
                )
            else:
                results = [process_pair(R1, R2, c) for R1, R2, c in selected]

            new_Pt = []
            processed_regions = set()
            rejected = stale = recuts = 0
            recut_delta = 0.0

            for R1, R2, partitions, old_cut, cut in results:
                if partitions is None:
                    failed_pairs.add(frozenset([tuple(R1), tuple(R2)]))
                    rejected += 1
                    continue  # Skip failed re-cuts

                if tuple(R1) in processed_regions or tuple(R2) in processed_regions:
                    stale += 1  # A region was already consumed by an earlier accepted pair
                    continue  # Avoid double-processing same regions

                processed_regions.update([tuple(R1), tuple(R2)])
                new_Pt.extend(partitions)
                recuts += 1
                # A re-cut replaces the pair's boundary cut value old_cut by cut.
                recut_delta += cut - old_cut

                if not self.mute:
                    print(f"  glue solve: |R1|={len(R1)}, |R2|={len(R2)}, "
                          f"cross cost {old_cut:.4f} -> cut {cut:.4f} (re-cut)")

            # Keep regions that were not re-cut (including freshly merged ones)
            for R in Pt:
                if tuple(R) not in processed_regions:
                    new_Pt.append(R)

            Pt = new_Pt

            if merges == 0 and recuts == 0:
                if not self.mute:
                    print(f"  glue round {glue_round + 1}: no improving move, stopping")
                break

            new_obj = self.objective[-1] + recut_delta
            self.objective.append(new_obj)
            self.runtimes.append(time.perf_counter() - self.start)
            if not self.mute:
                print(f"  glue round {glue_round + 1}: {merges} merges, {recuts} re-cuts, "
                      f"{rejected} rejected, {stale} stale, "
                      f"objective {new_obj:.4f}, {len(Pt)} components, "
                      f"{self.runtimes[-1]:.2f}s elapsed")

        return Pt

    def run(self, max_iter=100):
        """
        Run the full Factorised Cut-Glue-Cut algorithm.
        """
        P = [list(range(self.F.shape[0]))]  # Initialize full partition
        self.start = time.perf_counter()
        self.objective.append(calculate_factorised_multicut(P, self.F, self.alpha))
        self.runtimes.append(0)

        for i in range(max_iter):
            # _factorised_cut mutates P in place, so snapshot the partition at the start
            # of the iteration to compare against in the convergence check below.
            P_prev = [list(p) for p in P]

            if not self.mute:
                print(f"Iteration {i + 1}: Cutting...")
            Pt = self._factorised_cut(P)

            if not self.mute:
                print(f"Iteration {i + 1}: Gluing and Cutting...")
            Pt = self._factorised_glue_cut(Pt)

            new_obj = calculate_factorised_multicut(Pt, self.F, self.alpha)
            self.objective.append(new_obj)
            self.runtimes.append(time.perf_counter() - self.start)
            if not self.mute:
                print(f"Iteration {i + 1}: objective {new_obj:.4f}, "
                      f"{len(Pt)} components, {self.runtimes[-1]:.2f}s elapsed")

            P = Pt
            if set(map(frozenset, P_prev)) == set(map(frozenset, Pt)):
                break

        self.final_objective = self.objective[-1]
        finish = time.perf_counter() - self.start
        if not self.mute:
            print(f"Completed in {finish} seconds")

        return P


# ---------------------------------------------------------------------------
# Instance reader and CLI
# ---------------------------------------------------------------------------

def read_factorised_matrix(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as file:
        header = file.readline().strip()
        if header != "FACTORIZED COMPLETE MULTICUT":
            raise ValueError("Invalid file format. Expected 'FACTORIZED COMPLETE MULTICUT' header.")

        alpha = float(file.readline().strip())

        n, dim = map(int, file.readline().strip().split())

        # Read the feature vectors
        F = []
        for _ in range(n):
            v = list(map(float, file.readline().strip().split()))
            if len(v) != dim:
                raise ValueError(f"Feature vector dimensionality mismatch. Expected {dim} dimensions.")
            F.append(v)

    return F, alpha, n, dim


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the CPU Cut Glue & Cut solver for factorised matrices.")
    parser.add_argument("--path", type=str, required=True, help="Path to the factorised matrix file")
    parser.add_argument("--step", type=float, default=1e-3, help="Step Size (default: 1e-3)")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance (default: 1e-3)")
    parser.add_argument("--pgd-max-iter", type=int, default=1000,
                        help="Iteration budget per max-cut subproblem; rough solves suffice "
                             "since moves are only accepted on exact improvement (default: 1000)")
    parser.add_argument("--rank", type=int, default=None,
                        help="Rank k of the low-rank max-cut variable X (n x k). Default: "
                             "sqrt(2n)+1, the Barvinok-Pataki rank; lower is faster but looser, "
                             "higher gains nothing")
    parser.add_argument("--mute", action="store_true", help="Mute output")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing (joblib)")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Read factorised matrix
    F, alpha, n, d = read_factorised_matrix(args.path)
    F = np.array(F)
    k = args.rank if args.rank is not None else default_rank(n)

    print("Starting factorised solver (CPU backend)...")
    FCGC = FactorisedCutGlueCut(
        F=F, k=k, base_tol=args.tol, step_size=args.step, alpha=alpha, mute=args.mute,
        parallel=args.parallel, pgd_max_iter=args.pgd_max_iter,
    )
    _ = FCGC.run()

    print(f"Factorised Projected Gradient Descent -> Final Cut: {FCGC.final_objective:.2f} "
          f"// Runtime {FCGC.runtimes[-1]:.2f}")


if __name__ == "__main__":
    main()
