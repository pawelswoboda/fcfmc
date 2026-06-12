"""
Single-file PyTorch implementation of the Factorised Cut-Glue-Cut (FCGC)
solver for complete multicut with factorised costs c_uv = <f_u, f_v> - alpha.

Self-contained merge of logic/factorised_cut_glue_and_cut.py (torch backend
paths) and logic/torch_pgd.py with their dependencies inlined; no imports from
the logic/ or utils/ packages.

Design rationale (docs/gpu-parallelisation.md): a single subproblem solve is far
too small to occupy a GPU, so the speedup comes from batching the independent
solves that the cut and re-cut phases fan out. Subproblems are bucketed by size
(next power of two), padded to the bucket maximum and advanced together by one
masked PGD loop, with per-element adaptive step sizes and convergence flags.

Padding is exact, not approximate: padded rows of F are zeroed and the all-ones
vector of the alpha term is replaced by the validity mask, so every element of
the batch optimises precisely its own objective and padded rows of X receive a
zero gradient (their values are inert).

The relaxation solve runs in float32 by default; exact accept/reject of the
resulting moves stays in float64 (on-device reductions over a float64 copy of
F), as in the NumPy path.

Instance file format:
    FACTORIZED COMPLETE MULTICUT
    alpha
    n d
    feature_node_1_dim_1 ... feature_node_1_dim_d
    ...

Usage:
    python cgc_torch_solver.py --path instance.txt [--device cuda] [--restarts R] ...
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch


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
# Batched PGD subproblem solver on the torch device
# ---------------------------------------------------------------------------

def _next_pow2(x: int) -> int:
    return 1 << max(x - 1, 1).bit_length()


def _normalize_rows(X):
    """Row normalisation on batched (B, n, k)."""
    return X / X.norm(dim=2, keepdim=True).clamp_min(1e-12)


def _batched_loss(F_b, X, mask, alpha):
    """
    Per-element relaxation objective:
    ||F_i^T X_i||_F^2 - alpha * ||m_i^T X_i||^2 with m_i the validity mask
    standing in for the all-ones vector (padded rows must not contribute).

    :param F_b: (B, n, d) padded feature matrices, padded rows zeroed
    :param X: (B, n, k) variable matrices
    :param mask: (B, n) 1.0 on real rows, 0.0 on padding
    :return: (B,) losses
    """
    FX = F_b.transpose(1, 2) @ X  # (B, d, k)
    F_term = FX.pow(2).sum(dim=(1, 2))
    col = mask.unsqueeze(1) @ X  # (B, 1, k) column sums over real rows
    return F_term - alpha * col.pow(2).sum(dim=(1, 2))


def _batched_grad(F_b, X, mask, alpha):
    """
    Per-element gradient F (F^T X) - alpha * m (m^T X). Rows where both F and
    the mask are zero get a zero gradient, so padded rows never move.
    """
    FtX = F_b.transpose(1, 2) @ X  # (B, d, k)
    col = mask.unsqueeze(1) @ X  # (B, 1, k)
    return F_b @ FtX - alpha * mask.unsqueeze(2) * col


class TorchPGDSolver:
    """
    Batched factorised PGD over many max-cut subproblems of one shared F.

    F is uploaded to the device once at construction; subproblem feature
    matrices are gathered on-device, so only index lists and results cross
    the host boundary.
    """

    def __init__(self, F, device=None, dtype=torch.float32):
        """
        :param F: full (n x d) feature matrix (NumPy array)
        :param device: torch device string; default "cuda" if available else "cpu"
        :param dtype: solve precision (the heuristic inner solve tolerates
                      float32; exact bookkeeping stays in the caller)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        # F64 backs the exact float64 cut-value evaluation during on-device
        # rounding; reductions are bandwidth-bound, so float64 costs little even
        # on consumer GPUs. The solve itself runs on the dtype copy.
        self.F64 = torch.as_tensor(np.asarray(F), dtype=torch.float64, device=self.device)
        self.F = self.F64.to(dtype) if dtype != torch.float64 else self.F64

    def solve(self, subproblems, alpha, step_size, tol, max_iter,
              restarts=1, step_increase=1.2, step_decrease=0.5, min_step=1e-12,
              round_solutions=False):
        """
        Solve a list of independent subproblems as size-bucketed batches.

        :param subproblems: list of (indices, k) — node index list into F and
                            requested rank. Bucket-mates share the largest
                            requested rank, which only tightens the relaxation.
        :param restarts: random initialisations per subproblem; the one with the
                         best final relaxation objective is returned. Extra
                         restarts are just extra batch elements, so they are
                         nearly free on a GPU.
        :param round_solutions: if True, also round each solution on the device
                                (deterministic top-eigenvector hyperplane) and
                                evaluate the exact cut value in float64 on the
                                GPU; X never leaves the device.
        :return: list aligned with subproblems. Without rounding: (X, relaxation)
                 with X an (n_i, k_bucket) NumPy array. With rounding:
                 (partitions, cut_value, relaxation) where partitions is a list
                 of lists of LOCAL indices into the subproblem's node list.
        """
        results = [None] * len(subproblems)
        buckets = {}
        for pos, (indices, _) in enumerate(subproblems):
            buckets.setdefault(_next_pow2(len(indices)), []).append(pos)

        for positions in buckets.values():
            self._solve_bucket(
                positions, subproblems, results, alpha=alpha,
                step_size=step_size, tol=tol, max_iter=max_iter,
                restarts=restarts, step_increase=step_increase,
                step_decrease=step_decrease, min_step=min_step,
                round_solutions=round_solutions,
            )
        return results

    def _component_sums_t(self, node_sets):
        """
        Float64 per-component feature sums s_R = sum_{u in R} f_u as a device
        tensor, computed batched: ONE host-to-device index transfer, one gather,
        and a deterministic segment reduction via differences of prefix sums
        (no atomics, so results are reproducible across runs). The prefix-sum
        difference carries the usual float64 summation-reordering noise, the
        same order of magnitude as any alternative reduction order.
        """
        d = self.F64.shape[1]
        if not node_sets:
            return torch.empty((0, d), dtype=torch.float64, device=self.device)
        lengths = np.fromiter((len(R) for R in node_sets), dtype=np.int64)
        cat = np.concatenate([np.asarray(R, dtype=np.int64) for R in node_sets])
        idx = torch.from_numpy(cat).to(self.device)
        ends = torch.from_numpy(np.cumsum(lengths) - 1).to(self.device)
        prefix = torch.cumsum(self.F64.index_select(0, idx), dim=0)
        at_ends = prefix.index_select(0, ends)
        return torch.cat([at_ends[:1], at_ends[1:] - at_ends[:-1]], dim=0)

    def component_sums(self, node_sets):
        """NumPy view of _component_sums_t, the glue phase's summary matrix S."""
        return self._component_sums_t(node_sets).cpu().numpy()

    def glue_merges(self, node_sets, alpha, eps=1e-9):
        """
        Exact greedy merge stage of the glue phase, run on the device: cross
        costs <s_R1, s_R2> - alpha*|R1|*|R2| from batched component sums, then
        repeatedly merge the largest-gain pair (gain > eps), updating one row
        per merge. Identical greedy order and arithmetic as the NumPy loop, but
        each merge costs a few kernels (argmax, GEMV, masking) plus two scalar
        syncs instead of CPU-side O(m*d + m^2) NumPy work.

        :return: (groups, merges, merge_gain, cross) — the merged node sets,
                 number of merges, total gain, and the compacted cross-cost
                 matrix over the surviving groups as a float64 NumPy array
                 (consumed by the re-cut selection stage on the host).
        """
        m = len(node_sets)
        S = self._component_sums_t(node_sets)
        sizes = torch.tensor([len(R) for R in node_sets], dtype=torch.float64,
                             device=self.device)
        cross = S @ S.T - alpha * torch.outer(sizes, sizes)
        diag = torch.arange(m, device=self.device)
        cross[diag, diag] = float("-inf")
        alive = torch.ones(m, dtype=torch.bool, device=self.device)

        groups = [list(R) for R in node_sets]
        merges, merge_gain = 0, 0.0
        alive_count = m
        while alive_count > 1:
            flat = int(torch.argmax(cross))
            i, j = flat // m, flat % m
            gain = float(cross[i, j])
            if gain <= eps:
                break
            groups[i] = groups[i] + groups[j]
            groups[j] = None
            S[i] += S[j]
            sizes[i] += sizes[j]
            alive[j] = False
            alive_count -= 1
            cross[j, :] = float("-inf")
            cross[:, j] = float("-inf")
            new_row = S @ S[i] - alpha * sizes * sizes[i]
            new_row.masked_fill_(~alive, float("-inf"))
            cross[i, :] = new_row
            cross[:, i] = new_row
            cross[i, i] = float("-inf")
            merges += 1
            merge_gain += gain

        keep = torch.nonzero(alive).squeeze(1)
        cross_np = cross.index_select(0, keep).index_select(1, keep).cpu().numpy()
        return [g for g in groups if g is not None], merges, merge_gain, cross_np

    def _round_batch(self, X, mask, idx_t, sizes, alpha):
        """
        Round a batch of solved iterates on the device with the deterministic
        top-eigenvector hyperplane: project onto the top eigenvector of X^T X
        and split by sign. The hyperplane geometry runs in the solve dtype
        (sign decisions are robust to float32), while the cut value
        <s_1, s_2> - alpha*n_1*n_2 is reduced from F64 so that the exact
        float64 acceptance semantics of the NumPy path are preserved.

        :return: list aligned with rows of (partitions, cut_value), partitions
                 holding LOCAL indices; a one-sided rounding yields a single
                 component with cut value 0.0, as in the NumPy path.
        """
        Xm = X * mask.unsqueeze(2)  # zero padded rows: they must not steer the eigenvector
        G = Xm.transpose(1, 2) @ Xm
        w = torch.linalg.eigh(G).eigenvectors[..., -1]
        proj = (Xm @ w.unsqueeze(2)).squeeze(2)
        side = (proj > 0) & mask.bool()

        out = []
        for row, n_i in enumerate(sizes):
            sel = side[row, :n_i]
            n1 = int(sel.sum())
            if n1 == 0 or n1 == n_i:
                out.append(([list(range(n_i))], 0.0))
                continue
            rows64 = self.F64.index_select(0, idx_t[row, :n_i])
            s_total = rows64.sum(dim=0)
            s1 = (rows64 * sel.unsqueeze(1).to(torch.float64)).sum(dim=0)
            cut = float(s1 @ (s_total - s1)) - alpha * n1 * (n_i - n1)
            sel_np = sel.cpu().numpy()
            local = np.arange(n_i)
            out.append(([local[sel_np].tolist(), local[~sel_np].tolist()], cut))
        return out

    def _solve_bucket(self, positions, subproblems, results, *, alpha, step_size,
                      tol, max_iter, restarts, step_increase, step_decrease,
                      min_step, round_solutions=False):
        P = len(positions)
        sizes = [len(subproblems[p][0]) for p in positions]
        n_pad = max(sizes)
        k_pad = max(subproblems[p][1] for p in positions)
        B = P * restarts

        idx_pad = np.zeros((P, n_pad), dtype=np.int64)
        mask_np = np.zeros((P, n_pad), dtype=np.float64)
        for row, p in enumerate(positions):
            indices = subproblems[p][0]
            idx_pad[row, :len(indices)] = indices
            mask_np[row, :len(indices)] = 1.0
        idx_t = torch.from_numpy(idx_pad).to(self.device)
        mask = torch.from_numpy(mask_np).to(self.device, self.dtype)
        F_b = self.F[idx_t] * mask.unsqueeze(2)  # zero the padded rows

        if restarts > 1:
            # element p occupies rows p*restarts .. p*restarts + restarts - 1
            F_b = F_b.repeat_interleave(restarts, dim=0)
            mask = mask.repeat_interleave(restarts, dim=0)

        X = _normalize_rows(
            torch.randn(B, n_pad, k_pad, device=self.device, dtype=self.dtype)
        )
        loss = _batched_loss(F_b, X, mask, alpha)
        g = _batched_grad(F_b, X, mask, alpha)
        step = torch.full((B, 1, 1), step_size, device=self.device, dtype=self.dtype)
        active = torch.ones(B, dtype=torch.bool, device=self.device)

        # Same adaptive accept/reject scheme as the sequential CPU PGD, but
        # data-parallel: per-element step sizes, with masks freezing converged
        # elements. The gradient is recomputed every iteration instead of only after
        # accepted steps (uniform control flow); for elements whose X did not change
        # this recomputes the identical g, so the trajectory per element matches the
        # sequential scheme exactly.
        for _ in range(max_iter):
            X_trial = _normalize_rows(X + step * g)
            trial_loss = _batched_loss(F_b, X_trial, mask, alpha)

            accept = active & (trial_loss > loss)
            reject = active & ~accept
            improvement = (trial_loss - loss) / loss.abs().clamp_min(1e-8)

            accept3 = accept.view(B, 1, 1)
            X = torch.where(accept3, X_trial, X)
            loss = torch.where(accept, trial_loss, loss)
            step = torch.where(accept3, step * step_increase,
                               torch.where(reject.view(B, 1, 1),
                                           step * step_decrease, step))

            active = active & ~((accept & (improvement < tol)) |
                                (reject & (step.reshape(B) < min_step)))
            if not bool(active.any()):
                break
            g = _batched_grad(F_b, X, mask, alpha)

        loss_pr = loss.reshape(P, restarts)
        best = loss_pr.argmax(dim=1)
        winners = torch.arange(P, device=self.device) * restarts + best
        X_best = X[winners]
        loss_best = loss_pr.gather(1, best.unsqueeze(1)).squeeze(1).cpu()

        if round_solutions:
            rounded = self._round_batch(X_best, mask[winners], idx_t, sizes, alpha)
            for row, p in enumerate(positions):
                partitions, cut = rounded[row]
                results[p] = (partitions, cut, float(loss_best[row]))
        else:
            X_cpu = X_best.cpu().numpy()
            for row, p in enumerate(positions):
                results[p] = (X_cpu[row, :sizes[row], :], float(loss_best[row]))


# ---------------------------------------------------------------------------
# Factorised Cut-Glue-Cut (torch backend)
# ---------------------------------------------------------------------------

class FactorisedCutGlueCut:
    """
    Implementation of the Factorised Cut-Glue-Cut (FCGC) algorithm for solving the Multicut problem.
    This approach applies a factorised version of projected gradient descent to partition graphs,
    followed by a glue-and-cut step to iteratively refine the clustering. All subproblem solves
    of a phase run as one batched PGD call on the torch device; acceptance bookkeeping stays
    exact float64.
    """
    def __init__(self, F, k=None, base_tol=1e-3, step_size=0.001, alpha=0.0, mute=False,
                 max_glue_rounds=10, pgd_max_iter=1000, recut_top_k=32,
                 device=None, pgd_restarts=1):
        """
        Initialize the Factorised Cut-Glue-Cut solver.

        :param F: Factorised cost matrix representing the graph.
        :param k: The rank for the low-rank approximation (if None, computed from F's size).
        :param base_tol: Base tolerance for stopping criteria.
        :param step_size: Step size for projected gradient descent.
        :param alpha: Regularization parameter. Edge costs are c_uv = <f_u, f_v> - alpha.
        :param mute: If True, suppresses print output.
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
        :param device: torch device (default: cuda if available).
        :param pgd_restarts: random initialisations per subproblem; the best final
                             relaxation is kept. Extra restarts are extra batch
                             elements, nearly free on a GPU.
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
        self.pgd_restarts = pgd_restarts
        self._torch_solver = TorchPGDSolver(self.F, device=device)
        self.objective = []  # Stores objective values at each iteration
        self.final_objective = 0
        self.runtimes = []  # Stores runtime at each iteration
        self.start = time.time()  # placeholder, whenever run() is started it is reset to the new time.

    def _torch_solve(self, node_sets):
        """
        Solve the max-cut subproblems for the given node sets as one batched
        PGD call on the torch device, with rounding and exact cut evaluation
        also on the device (float64 reductions over the device-resident F).
        Returns a list of (partition, cut_value, relaxation) aligned with
        node_sets; only index lists and scalars cross the host boundary, and
        the acceptance guarantees are unchanged by the float32 solve.
        """
        subproblems = [(U, min(self.k, default_rank(len(U)))) for U in node_sets]
        if not self.mute and node_sets:
            buckets = {}
            for U in node_sets:
                buckets.setdefault(_next_pow2(len(U)), []).append(len(U))
            summary = ", ".join(f"<={cap}: {len(s)}" for cap, s in sorted(buckets.items()))
            print(f"  torch solve: batch of {len(node_sets)} subproblems "
                  f"in {len(buckets)} size bucket(s) [{summary}]")
        solutions = self._torch_solver.solve(
            subproblems, alpha=self.alpha, step_size=self.step_size,
            tol=self.tol, max_iter=self.pgd_max_iter, restarts=self.pgd_restarts,
            round_solutions=True,
        )
        results = []
        for U, (partitions, cut, relaxation) in zip(node_sets, solutions):
            p = [[U[i] for i in part] for part in partitions]
            results.append((p, cut, relaxation))
        return results

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

        while Q:
            current_partitions = list(Q)
            Q.clear()
            # One batched solve over all pending components.
            partitions_and_values = [
                (p, cut, relaxation, U)
                for U, (p, cut, relaxation)
                in zip(current_partitions, self._torch_solve(current_partitions))
            ]

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
        first, summaries updated incrementally) -- no solver involved. Summaries,
        cross costs and the merge loop run on the device.

        Cut: a pair with non-positive cross cost may still admit a cheaper split of its
        union, which does need a max-cut solve. Only the top recut_top_k candidates
        ranked by |cross cost| (the value at stake on that boundary) are solved per
        round, as one batched solve.
        """
        # Objective updates below are incremental; seed the baseline when the
        # phase is invoked directly, without run() having logged it.
        if not self.objective:
            self.objective.append(calculate_factorised_multicut(P, self.F, self.alpha))
            self.runtimes.append(time.perf_counter() - self.start)

        Pt = [list(R) for R in P]
        failed_pairs = set()  # pairs whose re-cut solve found no improvement

        for glue_round in range(self.max_glue_rounds):
            # Glue stage: exact greedy merges of the largest-gain pair, from
            # all pairwise cross costs of the component summaries; runs on the device.
            Pt, merges, merge_gain, cross = self._torch_solver.glue_merges(
                Pt, self.alpha
            )
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

            # One batched solve over all candidate unions. Acceptance: a re-cut is
            # taken only if a real bipartition beats the pair's current boundary by
            # more than fp noise (profitable merges were already applied exactly in
            # the glue stage, so a one-sided rounding here is noise, not a move).
            results = []
            solved = self._torch_solve([R1 + R2 for R1, R2, _ in selected])
            for (R1, R2, old_cut_value), (p, cut, _) in zip(selected, solved):
                if cut < old_cut_value - 1e-9 and len(p) > 1:
                    results.append((R1, R2, p, old_cut_value, cut))
                else:
                    results.append((R1, R2, None, old_cut_value, cut))

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
    parser = argparse.ArgumentParser(description="Run the torch Cut Glue & Cut solver for factorised matrices.")
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
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device, e.g. cuda or cpu (default: cuda if available)")
    parser.add_argument("--restarts", type=int, default=1,
                        help="Random initialisations per subproblem; the best one is kept, "
                             "nearly free on a GPU (default: 1)")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Read factorised matrix
    F, alpha, n, d = read_factorised_matrix(args.path)
    F = np.array(F)
    k = args.rank if args.rank is not None else default_rank(n)

    print("Starting factorised solver (torch backend)...")
    FCGC = FactorisedCutGlueCut(
        F=F, k=k, base_tol=args.tol, step_size=args.step, alpha=alpha, mute=args.mute,
        pgd_max_iter=args.pgd_max_iter, device=args.device, pgd_restarts=args.restarts,
    )
    _ = FCGC.run()

    print(f"Factorised Projected Gradient Descent -> Final Cut: {FCGC.final_objective:.2f} "
          f"// Runtime {FCGC.runtimes[-1]:.2f}")


if __name__ == "__main__":
    main()
