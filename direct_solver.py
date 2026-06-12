"""
Single-file implementation of the direct low-rank relaxation of factorised
multicut, without the cut/glue decomposition into max-cut subproblems.

Self-contained version of logic/direct_lowrank_multicut.py with its
dependencies inlined; no imports from the logic/ or utils/ packages.

A partition is encoded by an assignment matrix A in {0,1}^{n x K} with one 1
per row, and minimising the multicut value (sum of costs of cut pairs) is
equivalent to maximising the joined cost <C, A A^T>. With factorised costs
C = F F^T - alpha this is the same loss the max-cut PGD already uses:

    maximise  L(X) = ||F^T X||_F^2 - alpha * ||X^T 1||^2
    subject to X >= 0 entrywise, rows of unit norm.

The feasible set is the Burer-Monteiro factorisation of the correlation
clustering SDP {M >= 0 entrywise, M PSD, diag(M) = 1}: entrywise
nonnegativity of X enforces <x_u, x_v> >= 0 for ALL pairs for free, so the
relaxation stays subquadratic (O(n*d*K) per gradient). Integral assignments
(rows on the standard basis vectors) are feasible, so the relaxed maximum
upper-bounds the best integral joined cost. K caps the number of clusters.

The relaxation can be tightened with triangle inequalities, the sparsest
members of the rank-one hypermetric family ||X^T b||^2 >= |1^T b| (integer b)
valid for all partition matrices: b = e_u + e_v - e_w gives the clustering
triangle inequality M_uw + M_vw - M_uv <= 1. Violated triangles are collected
into a working set and enforced with one-sided hinge penalties
rho_c * max(0, 1 - ||X^T b_c||^2). A linear Lagrangian term mu_c * b_c b_c^T
would be wrong here: it REWARDS over-satisfying the inequality (up to
||X^T b||^2 = 5 by making u, v parallel and w orthogonal), so the maximiser
distorts the solution instead of repairing violations. The hinge penalty is
zero once a cut is satisfied; where active it adds the same rank-one
b_c (b_c^T X) gradient term as the alpha term (the b = all-ones member with
weight -alpha), so the gradient stays cheap: O(K) extra per active cut.

Instance file format:
    FACTORIZED COMPLETE MULTICUT
    alpha
    n d
    feature_node_1_dim_1 ... feature_node_1_dim_d
    ...

Usage:
    python direct_solver.py --path instance.txt [--rank K] [--tighten] ...
"""
import os

import numpy as np
from scipy import sparse


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


def get_factorised_loss(F, X, alpha):
    """Relaxation objective ||F^T X||_F^2 - alpha * ||X^T 1||^2."""
    n = X.shape[0]
    FX = F.T @ X

    F_term = np.sum(FX ** 2)

    alpha_term = alpha * np.sum((X.T @ np.ones((n,))) ** 2)

    return F_term - alpha_term


# ---------------------------------------------------------------------------
# Relaxation solver
# ---------------------------------------------------------------------------

def project_nonneg_sphere(X):
    """
    Project each row of X onto {x >= 0, ||x|| = 1} (clamp negatives, then
    normalise). A row with no positive entry projects onto the basis vector
    of its largest (least negative) coordinate.
    """
    Xc = np.maximum(X, 0.0)
    norms = np.linalg.norm(Xc, axis=1)
    dead = norms == 0.0
    if np.any(dead):
        dead_rows = np.where(dead)[0]
        Xc[dead_rows, np.argmax(X[dead_rows], axis=1)] = 1.0
        norms[dead] = 1.0
    return Xc / norms[:, None]


def _direct_projected_gradient_ascent(F, K, alpha, step_size=1e-3, tol=1e-6, max_iter=5000,
                                      step_increase=1.2, step_decrease=0.5, min_step=1e-12,
                                      rng=None, mute=True, log_every=50, B=None, rho=None,
                                      X0=None, min_accepts=0):
    """
    Projected gradient ascent on L(X) = ||F^T X||^2 - alpha * ||X^T 1||^2 over
    rows of the nonnegative unit sphere, with an adaptive step size: accepted
    trials increase the step size, rejected trials shrink it, and the gradient
    is only recomputed after accepted steps.

    With a triangle working set (B sparse n x m whose columns are the cut
    vectors b_c, rho their penalty weights) the hinge-penalised objective
    L(X) - sum_c rho_c * max(0, 1 - ||X^T b_c||^2) is maximised instead; X0
    warm starts from the previous outer round, and min_accepts suppresses the
    tolerance-based stop until that many steps were accepted (a warm start
    near a previous optimum would otherwise trip it on the first trial).

    With mute=False, prints every log_every trials: the (penalised) loss, the
    step size, the accept count, and two integrality diagnostics. "row max" is
    the mean over nodes of the largest coordinate of x_u (1.0 means every node
    sits on a basis vector, i.e. the relaxation has become an assignment) and
    "active cols" counts the columns carrying any mass, an upper bound on the
    number of clusters argmax rounding can produce.
    """
    if rng is None:
        rng = np.random.default_rng()

    losses = []
    n = F.shape[0]
    FT = F.T
    ones_n = np.ones((n, 1))

    def loss_fn(X):
        value = get_factorised_loss(F, X, alpha)
        if B is not None:
            T = B.T @ X
            slack = np.einsum("ij,ij->i", T, T) - 1.0
            value -= float(rho @ np.maximum(0.0, -slack))
        return value

    def grad_fn(X):
        g = F @ (FT @ X) - alpha * (ones_n @ (ones_n.T @ X))
        if B is not None:
            T = B.T @ X
            active = (np.einsum("ij,ij->i", T, T) - 1.0) < 0.0
            g += B @ (T * (rho * active)[:, None])
        return g

    X = X0 if X0 is not None else project_nonneg_sphere(np.abs(rng.standard_normal((n, K))))

    loss = loss_fn(X)
    losses.append(loss)

    g = grad_fn(X)

    def log_state(trial):
        row_max = float(np.mean(np.max(X, axis=1)))
        active_cols = int(np.count_nonzero(np.max(X, axis=0) > 1e-8))
        print(f"    trial {trial}: relaxation {loss:.4f}, step {step_size:.2e}, "
              f"{len(losses) - 1} accepted, row max {row_max:.3f}, "
              f"active cols {active_cols}/{X.shape[1]}")

    termination = "max_iter reached"
    for trial in range(1, max_iter + 1):
        X_trial = project_nonneg_sphere(X + step_size * g)
        trial_loss = loss_fn(X_trial)

        if trial_loss > loss:
            improvement = (trial_loss - loss) / max(abs(loss), 1e-8)
            X, loss = X_trial, trial_loss
            losses.append(loss)
            g = grad_fn(X)
            step_size *= step_increase
            if improvement < tol and len(losses) - 1 >= min_accepts:
                termination = f"converged (relative improvement < {tol:.0e})"
                break
        else:
            step_size *= step_decrease
            if step_size < min_step:
                termination = "step size collapsed"
                break

        if not mute and trial % log_every == 0:
            log_state(trial)

    if not mute:
        log_state(trial)
        print(f"    {termination} after {trial} trials")

    return X, losses


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

def _partition_from_labels(labels):
    partition = {}
    for u, c in enumerate(labels):
        partition.setdefault(int(c), []).append(u)
    return list(partition.values())


def round_argmax(X):
    """
    Round by assigning each node to its largest coordinate. Meaningful when
    the relaxation drives rows towards (near-)disjoint supports: nonnegative
    vectors are orthogonal exactly when their supports are disjoint, so a
    well-separated solution is close to a column-supported assignment matrix
    up to no rotation ambiguity.
    """
    return _partition_from_labels(np.argmax(X, axis=1))


def round_pivot(X, threshold=0.5, rng=None):
    """
    Charikar-Guruswami-Wirth / Swamy style pivot rounding: visit nodes in
    random order; an unassigned node u becomes a pivot and absorbs every
    unassigned v with <x_u, x_v> >= threshold. O(n*K) per pivot, and the
    number of pivots is the number of clusters produced.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = X.shape[0]
    labels = np.full(n, -1, dtype=int)
    cluster = 0
    for u in rng.permutation(n):
        if labels[u] >= 0:
            continue
        members = (labels < 0) & (X @ X[u] >= threshold)
        members[u] = True  # guard against fp noise in <x_u, x_u>
        labels[members] = cluster
        cluster += 1
    return _partition_from_labels(labels)


def _evaluate_roundings(X, F, alpha, pivot_thresholds, pivot_repeats, rng):
    """
    Round X with every scheme (argmax plus pivot rounding per threshold and
    several random pivot orders), evaluate each candidate exactly with
    calculate_factorised_multicut and return ((value, partition, scheme),
    best (value, n_clusters) per scheme).
    """
    candidates = [("argmax", round_argmax(X))]
    for t in pivot_thresholds:
        for _ in range(pivot_repeats):
            candidates.append((f"pivot(t={t})", round_pivot(X, threshold=t, rng=rng)))

    scheme_best = {}
    best = (np.inf, None, None)
    for name, P in candidates:
        value = calculate_factorised_multicut(P, F, alpha)
        if name not in scheme_best or value < scheme_best[name][0]:
            scheme_best[name] = (value, len(P))
        if value < best[0]:
            best = (value, P, name)
    return best, scheme_best


# ---------------------------------------------------------------------------
# Plain direct solver
# ---------------------------------------------------------------------------

def direct_lowrank_multicut(F, alpha, K=None, restarts=1, step_size=1e-3, tol=1e-6,
                            max_iter=5000, pivot_thresholds=(0.4, 0.5, 0.6),
                            pivot_repeats=4, seed=None, mute=True):
    """
    Solve the direct nonnegative low-rank relaxation and round it.

    Every rounding candidate (argmax plus pivot rounding for each threshold
    and several random pivot orders) is evaluated exactly with
    calculate_factorised_multicut and the best partition is returned, so the
    reported value is always an exact multicut objective.

    :param F: feature matrix (n x d), costs c_uv = <f_u, f_v> - alpha
    :param K: maximum number of clusters (columns of X); default sqrt(2n) + 1
    :param restarts: independent PGD runs from random starts; best rounding wins

    :return: (partition, multicut_value, info) where info holds the best
             relaxation loss, the winning rounding scheme and cluster count.
    """
    import time

    n = F.shape[0]
    if K is None:
        K = default_rank(n)
    rng = np.random.default_rng(seed)

    best_P, best_value, best_info = None, np.inf, {}
    for r in range(restarts):
        if not mute:
            print(f"restart {r}: solving relaxation (n={n}, K={K})...")
        t0 = time.perf_counter()
        X, losses = _direct_projected_gradient_ascent(F, K, alpha, step_size=step_size, tol=tol,
                                                      max_iter=max_iter, rng=rng, mute=mute)
        t_solve = time.perf_counter() - t0

        t0 = time.perf_counter()
        (value, P, name), scheme_best = _evaluate_roundings(X, F, alpha, pivot_thresholds,
                                                            pivot_repeats, rng)
        if value < best_value:
            best_P, best_value = P, value
            best_info = {"relaxation": losses[-1], "rounding": name,
                         "clusters": len(P), "restart": r, "iterations": len(losses)}
        t_round = time.perf_counter() - t0

        if not mute:
            for name, (value, n_clusters) in scheme_best.items():
                print(f"    rounding {name}: {value:.4f} ({n_clusters} clusters)")
            print(f"restart {r}: relaxation {losses[-1]:.4f} ({t_solve:.2f}s solve, "
                  f"{t_round:.2f}s rounding), best so far {best_value:.4f} via "
                  f"{best_info['rounding']} ({best_info['clusters']} clusters)")

    return best_P, best_value, best_info


# ---------------------------------------------------------------------------
# Triangle (hypermetric) tightening
# ---------------------------------------------------------------------------

def _triangle_matrix(tri, n):
    """
    Sparse n x m matrix whose columns are the triangle cut vectors
    b_c = e_u + e_v - e_w for tri = (u, v, w) index arrays.
    """
    u, v, w = tri
    m = len(u)
    rows = np.concatenate([u, v, w])
    cols = np.tile(np.arange(m), 3)
    data = np.concatenate([np.ones(2 * m), -np.ones(m)])
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, m))


def separate_triangles(X, n_neighbors=10, viol_eps=1e-3, existing_keys=None, max_new=None,
                       block=256):
    """
    Heuristic separation for the triangle inequalities ||x_u + x_v - x_w||^2 >= 1,
    violated exactly when <x_u,x_w> + <x_v,x_w> - <x_u,x_v> > 1 (w pulled
    towards both u and v while u and v stay apart). A violated triangle needs
    both <x_u,x_w> and <x_v,x_w> large, so only pairs among the n_neighbors
    highest-inner-product neighbours of each w are checked: O(n^2 K) for the
    blockwise neighbour search plus O(n * n_neighbors^2 * K) for the checks.

    Triangles are canonicalised to u < v, deduplicated, filtered against
    existing_keys and returned sorted by violation (at most max_new), as
    ((u, v, w) index arrays, violations, int64 keys).
    """
    n = X.shape[0]
    if existing_keys is None:
        existing_keys = np.empty(0, dtype=np.int64)
    iu, iv = np.triu_indices(n_neighbors, k=1)

    all_u, all_v, all_w, all_viol = [], [], [], []
    for start in range(0, n, block):
        S = X[start:start + block] @ X.T
        b = S.shape[0]
        S[np.arange(b), np.arange(start, start + b)] = -np.inf  # exclude w itself
        nb = np.argpartition(S, -n_neighbors, axis=1)[:, -n_neighbors:]
        sims = np.take_along_axis(S, nb, axis=1)
        Xn = X[nb]
        G = Xn @ Xn.transpose(0, 2, 1)
        V = sims[:, iu] + sims[:, iv] - G[:, iu, iv] - 1.0
        hits = np.argwhere(V > viol_eps)
        if hits.size:
            bi, pi = hits[:, 0], hits[:, 1]
            all_w.append(start + bi)
            all_u.append(nb[bi, iu[pi]])
            all_v.append(nb[bi, iv[pi]])
            all_viol.append(V[bi, pi])

    empty = (np.empty(0, dtype=int),) * 3
    if not all_w:
        return empty, np.empty(0), np.empty(0, dtype=np.int64)

    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    w = np.concatenate(all_w)
    viol = np.concatenate(all_viol)
    u, v = np.minimum(u, v), np.maximum(u, v)
    keys = (u.astype(np.int64) * n + v) * n + w

    order = np.argsort(-viol)  # so np.unique keeps the most violated duplicate
    u, v, w, viol, keys = u[order], v[order], w[order], viol[order], keys[order]
    _, first = np.unique(keys, return_index=True)
    fresh = first[~np.isin(keys[first], existing_keys)]
    fresh = fresh[np.argsort(-viol[fresh])]
    if max_new is not None:
        fresh = fresh[:max_new]

    return (u[fresh], v[fresh], w[fresh]), viol[fresh], keys[fresh]


def direct_lowrank_multicut_tightened(F, alpha, K=None, rounds=8, n_neighbors=25,
                                      max_new_cuts=50000, penalty=None, penalty_growth=2.0,
                                      drop_slack=0.5, viol_eps=1e-3, step_size=1e-3, tol=1e-6,
                                      max_iter=5000, pivot_thresholds=(0.4, 0.5, 0.6),
                                      pivot_repeats=4, seed=None, mute=True):
    """
    Direct relaxation tightened with triangle inequalities via escalating
    hinge penalties. Each round maximises the penalised objective over the
    current working set (warm-started from the previous round), rounds and
    exactly evaluates the result, then escalates the penalty weight of cuts
    that are still violated (rho_c *= penalty_growth), drops cuts satisfied
    with slack > drop_slack, and separates new violated triangles into the
    working set.

    :param rounds: outer solve/separate rounds; stops early once no triangle
                   is violated
    :param n_neighbors: neighbours per node checked in separation
    :param max_new_cuts: cap on cuts added per round
    :param penalty: initial hinge weight in cost units; default is 3x the mean
                    |c_uv| over sampled pairs
    :param penalty_growth: factor applied to the weight of still-violated cuts
    :return: (partition, multicut_value, info) for the best rounding seen in
             any round.
    """
    import time

    n = F.shape[0]
    if K is None:
        K = default_rank(n)
    if penalty is None:
        sample_rng = np.random.default_rng(0)
        i = sample_rng.integers(0, n, 2048)
        j = sample_rng.integers(0, n, 2048)
        penalty = 3.0 * float(np.mean(np.abs(np.einsum("ij,ij->i", F[i], F[j]) - alpha)))
    rng = np.random.default_rng(seed)

    tri = (np.empty(0, dtype=int),) * 3
    keys = np.empty(0, dtype=np.int64)
    rho = np.empty(0)
    X = None
    best_P, best_value, best_info = None, np.inf, {}

    for rnd in range(rounds):
        B = _triangle_matrix(tri, n) if len(rho) else None
        if not mute:
            print(f"round {rnd}: solving penalised relaxation (n={n}, K={K}, {len(rho)} cuts)...")
        t0 = time.perf_counter()
        X, losses = _direct_projected_gradient_ascent(F, K, alpha, step_size=step_size, tol=tol,
                                                      max_iter=max_iter, rng=rng, mute=mute,
                                                      B=B, rho=rho if len(rho) else None, X0=X,
                                                      min_accepts=25 if rnd else 0)
        t_solve = time.perf_counter() - t0

        (value, P, name), scheme_best = _evaluate_roundings(X, F, alpha, pivot_thresholds,
                                                            pivot_repeats, rng)
        if value < best_value:
            best_P, best_value = P, value
            best_info = {"relaxation": get_factorised_loss(F, X, alpha), "rounding": name,
                         "clusters": len(P), "round": rnd, "cuts": len(rho)}
        if not mute:
            for nm, (val, n_clusters) in scheme_best.items():
                print(f"    rounding {nm}: {val:.4f} ({n_clusters} clusters)")
            print(f"round {rnd}: plain relaxation {get_factorised_loss(F, X, alpha):.4f}, "
                  f"penalised {losses[-1]:.4f} ({t_solve:.2f}s solve), best so far "
                  f"{best_value:.4f} via {best_info['rounding']} ({best_info['clusters']} clusters)")

        if rnd == rounds - 1:
            break

        # escalate still-violated cuts, drop those satisfied with comfortable slack
        n_violated_existing = 0
        n_dropped = 0
        if len(rho):
            T = B.T @ X
            slack = np.einsum("ij,ij->i", T, T) - 1.0
            n_violated_existing = int(np.count_nonzero(slack < -viol_eps))
            rho = np.where(slack < -viol_eps, rho * penalty_growth, rho)
            keep = slack < drop_slack
            n_dropped = int(np.count_nonzero(~keep))
            tri = tuple(a[keep] for a in tri)
            keys, rho = keys[keep], rho[keep]

        new_tri, new_viol, new_keys = separate_triangles(X, n_neighbors=n_neighbors,
                                                         viol_eps=viol_eps, existing_keys=keys,
                                                         max_new=max_new_cuts)
        if len(new_keys) == 0 and n_violated_existing == 0:
            if not mute:
                print(f"round {rnd}: no violated triangles, stopping")
            break

        tri = tuple(np.concatenate([a, b]) for a, b in zip(tri, new_tri))
        rho = np.concatenate([rho, np.full(len(new_keys), penalty)])
        keys = np.concatenate([keys, new_keys])
        if not mute:
            max_viol = float(new_viol[0]) if len(new_viol) else 0.0
            print(f"    separation: +{len(new_keys)} cuts (max violation {max_viol:.4f}), "
                  f"{n_violated_existing} existing still violated, {n_dropped} dropped, "
                  f"working set {len(rho)}")

    return best_P, best_value, best_info


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
    import argparse

    parser = argparse.ArgumentParser(description="Run the direct low-rank multicut relaxation.")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to a factorised matrix file (FACTORIZED COMPLETE MULTICUT "
                             "format)")
    parser.add_argument("--rank", type=int, default=None,
                        help="Rank K of the relaxation variable X (n x K); K caps the number "
                             "of clusters argmax rounding can produce. Default: sqrt(2n)+1")
    parser.add_argument("--step", type=float, default=1e-3, help="Initial step size (default: 1e-3)")
    parser.add_argument("--tol", type=float, default=1e-6, help="Relative tolerance (default: 1e-6)")
    parser.add_argument("--max-iter", type=int, default=5000,
                        help="Maximum PGD trials per restart (default: 5000)")
    parser.add_argument("--restarts", type=int, default=1,
                        help="Independent runs from random starts; best rounding wins (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--tighten", action="store_true",
                        help="Tighten the relaxation with triangle inequalities (rank-one "
                             "hypermetric cuts) via hinge-penalty cutting-plane rounds; "
                             "ignores --restarts")
    parser.add_argument("--rounds", type=int, default=8,
                        help="Outer solve/separate rounds with --tighten (default: 8)")
    parser.add_argument("--neighbors", type=int, default=25,
                        help="Neighbours per node checked in triangle separation (default: 25)")
    parser.add_argument("--max-new-cuts", type=int, default=50000,
                        help="Cap on triangle cuts added per round (default: 50000)")
    parser.add_argument("--penalty", type=float, default=None,
                        help="Initial hinge penalty weight (default: 3x mean |c_uv| over "
                             "sampled pairs)")
    return parser.parse_args()


def main():
    import time

    args = parse_arguments()

    F, alpha, n, d = read_factorised_matrix(args.path)
    F = np.array(F)
    print(f"instance {args.path}: n={n}, d={d}, alpha={alpha}")

    t0 = time.perf_counter()
    if args.tighten:
        P, value, info = direct_lowrank_multicut_tightened(
            F, alpha, K=args.rank, rounds=args.rounds, n_neighbors=args.neighbors,
            max_new_cuts=args.max_new_cuts, penalty=args.penalty, step_size=args.step,
            tol=args.tol, max_iter=args.max_iter, seed=args.seed, mute=False)
        extra = f"round {info['round']} with {info['cuts']} cuts"
    else:
        P, value, info = direct_lowrank_multicut(F, alpha, K=args.rank, restarts=args.restarts,
                                                 step_size=args.step, tol=args.tol,
                                                 max_iter=args.max_iter, seed=args.seed,
                                                 mute=False)
        extra = f"restart {info['restart']}"
    t_direct = time.perf_counter() - t0
    print(f"direct low-rank: {value:.4f} ({len(P)} clusters, "
          f"rounding {info['rounding']}, {extra}, {t_direct:.2f}s)")


if __name__ == "__main__":
    main()
