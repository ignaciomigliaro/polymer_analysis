import os
import pickle
import matplotlib.pyplot as plt
import re
import time
# ============================================================
# Cache utilities for E2E time series
# ============================================================

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

# ---------- flexible topology ----------
def _find_topology_in_prod_dir(prod_dir, prefer_prefixes=("prod",), prefer_exts=(".tpr", ".gro")):
    cands = []
    for ext in prefer_exts:
        cands.extend(glob.glob(os.path.join(prod_dir, f"*{ext}")))
    if not cands:
        raise FileNotFoundError(f"No topology (.tpr/.gro) found in {prod_dir}")

    def score(p):
        base = os.path.basename(p).lower()
        st = os.stat(p)
        is_tpr = int(base.endswith(".tpr"))
        has_prefix = int(any(base.startswith(pref.lower()) for pref in prefer_prefixes))
        return (is_tpr, has_prefix, st.st_mtime, st.st_size)

    return sorted(cands, key=score, reverse=True)[0]

def _load_universe(prod_dir, xtc_name="prod.xtc"):
    xtc = os.path.join(prod_dir, xtc_name)
    if not os.path.isfile(xtc):
        raise FileNotFoundError(f"Missing {xtc_name} in {prod_dir}")
    topo = _find_topology_in_prod_dir(prod_dir)
    u = mda.Universe(topo, xtc)
    return u, topo, xtc

# ---------- polymer + end atoms ----------
def _polymer_as_largest_fragment(u):
    frags = u.atoms.fragments
    if len(frags) == 0:
        raise RuntimeError("No fragments found. Use a .tpr topology with bonds.")
    return max(frags, key=len)

def _heavy_atoms(poly, heavy_sel="not name H*"):
    heavy = poly.select_atoms(heavy_sel)
    if len(heavy) < 2:
        raise RuntimeError(f"Heavy selection '{heavy_sel}' returned <2 atoms. Adjust heavy_sel.")
    return heavy

def _terminals_by_bonds(group):
    idx_set = set(group.indices.tolist())
    terminals = []
    for a in group:
        deg = 0
        for b in getattr(a, "bonds", []):
            a1, a2 = b.atoms
            other = a2 if a1.index == a.index else a1
            if other.index in idx_set:
                deg += 1
        if deg == 1:
            terminals.append(a)
    return terminals

def _pick_end_atoms(u, heavy_sel="not name H*"):
    u.trajectory[0]
    poly = _polymer_as_largest_fragment(u)
    heavy = _heavy_atoms(poly, heavy_sel=heavy_sel)

    terms = _terminals_by_bonds(heavy)
    if len(terms) == 2:
        return poly, terms[0], terms[1], "bond_terminals_heavy"

    # If branching or selection ambiguity: choose farthest pair among heavy atoms (PBC-aware on *this* frame)
    coords = heavy.positions
    box = u.trajectory.ts.dimensions
    D = distance_array(coords, coords, box=box)
    np.fill_diagonal(D, -np.inf)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    return poly, heavy[i], heavy[j], f"geometry_farthest_heavy (terminals_found={len(terms)})"

# ---------- unwrap + compute time series ----------
def compute_e2e_timeseries_nm(
    prod_dir,
    xtc_name="prod.xtc",
    heavy_sel="not name H*",
    use_cache=True,
    force_recompute=False,
    normalize=None,          # NEW: None | "per_carbon" | "fraction" | "relative"
    carbon_count=None,       # NEW: override if you already know N_C (e.g., 20, 200)
):
    
    u, topo, xtc = _load_universe(prod_dir, xtc_name=xtc_name)

    # pick ends once (on frame 0)
    poly, a1, a2, method = _pick_end_atoms(u, heavy_sel=heavy_sel)

    # after: poly, a1, a2, method = _pick_end_atoms(...)
    nC = int(carbon_count) if carbon_count is not None else _count_polymer_carbons(poly)

    # default: no normalization
    norm_label = "raw"
    norm_factor = 1.0

    if normalize in ("per_carbon", "perC"):
        if nC <= 0:
            raise RuntimeError("normalize='per_carbon' but carbon count is 0/unknown.")
        norm_factor = float(nC)
        norm_label = f"per_carbon (nC={nC})"

    elif normalize in ("fraction", "frac", "per_bond"):
        # typical "max" steps ~ (nC-1); avoids division by 0 for tiny chains
        denom = max(1, nC - 1)
        norm_factor = float(denom)
        norm_label = f"fraction (nC={nC})"

    elif normalize in ("relative",):
        # we’ll do relative after we compute the raw distances
        norm_label = "relative"

    elif normalize is None:
        pass
    else:
        raise ValueError("normalize must be None, 'per_carbon', 'fraction', or 'relative'")

    # UNWRAP polymer across PBC so intramolecular distance is continuous
    # This is the key fix for "crazy high" distances.
    from MDAnalysis.transformations import unwrap
    u.trajectory.add_transformations(unwrap(poly))

    times_ps = []
    dists_nm = []

    for ts in u.trajectory:
        # After unwrapping, compute simple Euclidean distance
        d_ang = np.linalg.norm(a1.position - a2.position)   # Å
        d_nm = d_ang / 10.0                                 # convert Å -> nm
        times_ps.append(ts.time)
        dists_nm.append(d_nm)

    times_ps = np.asarray(times_ps, float)
    dists_nm = np.asarray(dists_nm, float)


    if normalize == "relative":
        mu = float(np.mean(dists_nm)) if len(dists_nm) else 1.0
        if mu == 0:
            mu = 1.0
        dists_nm = dists_nm / mu
    else:
        dists_nm = dists_nm / norm_factor

    
    meta = {
        "prod_dir": prod_dir,
        "topology": topo,
        "trajectory": xtc,
        "end_atoms_0based": (int(a1.index), int(a2.index)),
        "end_atom_names": (a1.name, a2.name),
        "end_atom_types": (getattr(a1, "type", None), getattr(a2, "type", None)),
        "end_method_used": method,
        "n_frames": int(len(times_ps)),
        "mean_nm": float(np.mean(dists_nm)),
        "std_nm": float(np.std(dists_nm, ddof=1)) if len(dists_nm) > 1 else 0.0,
        "box_firstframe": u.trajectory[0].dimensions,
        "nC": int(nC),
        "normalize": normalize, 
        "normalize_label": norm_label,   
    }
    return times_ps, dists_nm, meta

# ---------- simple KDE ----------
def gaussian_kde_1d(x, gridsize=400, bw_method="scott", bw=None, cut=3.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need >=2 samples for KDE.")
    std = np.std(x, ddof=1)

    if bw is None:
        if bw_method == "silverman":
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            s = min(std, iqr / 1.34) if np.isfinite(iqr) and iqr > 0 else std
            bw = 0.9 * s * n ** (-1/5)
        else:
            bw = std * n ** (-1/5)
        if not np.isfinite(bw) or bw <= 0:
            bw = max(std, 1e-6) * 0.1

    lo = np.min(x) - cut * bw
    hi = np.max(x) + cut * bw
    grid = np.linspace(lo, hi, gridsize)
    diff = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diff**2).sum(axis=1) / (n * bw * np.sqrt(2*np.pi))
    return grid, dens, bw

# ---------- plot: time or KDE ----------
def plot_e2e(prod_dir, mode="time", burn_in=0.0, downsample=1,
             rolling_window=None, kde_bw_method="scott", kde_bw=None,
             save_plot=None, figsize=(12,5)):
    t, d, meta = compute_e2e_timeseries_nm(prod_dir)

    # burn-in fraction
    if burn_in and 0.0 < burn_in < 1.0:
        start = int(len(d) * burn_in)
        t = t[start:]
        d = d[start:]

    downsample = int(max(1, downsample))
    t_plot = t[::downsample]
    d_plot = d[::downsample]

    plt.figure(figsize=figsize)

    if mode.lower() == "time":
        plt.plot(t_plot, d_plot, lw=1.2, label="E2E (nm)")
        if rolling_window is not None and rolling_window >= 2 and len(d_plot) >= rolling_window:
            w = int(rolling_window)
            kernel = np.ones(w) / w
            d_roll = np.convolve(d_plot, kernel, mode="valid")
            t_roll = t_plot[w-1:]
            plt.plot(t_roll, d_roll, lw=2.0, label=f"Rolling mean (w={w})")
        plt.xlabel("Time (ps)")
        plt.ylabel("End-to-End Distance (nm)")
        plt.title(f"E2E vs Time | {meta['end_method_used']}")
        plt.grid(True, alpha=0.3)
        plt.legend()

    elif mode.lower() == "kde":
        grid, dens, bw = gaussian_kde_1d(d_plot, bw_method=kde_bw_method, bw=kde_bw)
        plt.plot(grid, dens, lw=2.0, label=f"KDE (bw={bw:.3g} nm)")
        plt.xlabel("End-to-End Distance (nm)")
        plt.ylabel("Density")
        plt.title(f"E2E KDE | burn_in={burn_in} | downsample={downsample} | {meta['end_method_used']}")
        plt.grid(True, alpha=0.3)
        plt.legend()

    else:
        raise ValueError("mode must be 'time' or 'kde'")

    plt.tight_layout()
    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, dpi=300)
    plt.show()

    print("end atoms (0-based):", meta["end_atoms_0based"], "| names:", meta["end_atom_names"])
    print("mean E2E (nm):", meta["mean_nm"], "| std (nm):", meta["std_nm"])
    print("box (first frame):", meta["box_firstframe"])

    return t, d, meta

def _count_polymer_carbons(poly):
    """
    Robust carbon counter for the polymer AtomGroup.
    Tries 'element C' first; falls back to name/type heuristics.
    """
    # Try element if available
    try:
        nC = len(poly.select_atoms("element C"))
        if nC > 0:
            return int(nC)
    except Exception:
        pass

    # Fallback: names
    try:
        nC = len(poly.select_atoms("name C*"))
        if nC > 0:
            return int(nC)
    except Exception:
        pass

    # Fallback: types
    try:
        nC = len(poly.select_atoms("type C"))
        if nC > 0:
            return int(nC)
    except Exception:
        pass

    # Last resort: count atoms whose name starts with "C"
    names = [getattr(a, "name", "").upper() for a in poly.atoms]
    return int(sum(n.startswith("C") for n in names))


def _cache_path(prod_dir):
    """
    Returns path to cache file for a given /prod directory.
    """
    cache_dir = os.path.join(prod_dir, ".cache_e2e")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "e2e_timeseries.pkl")

def _file_sig(path):
    """
    Lightweight file signature used to invalidate cache
    if topology or trajectory changes.
    """
    if path is None or not os.path.isfile(path):
        return None
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime, st.st_size)

def _load_cache(path):
    """
    Load cache dict from disk if it exists.
    """
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(path, obj):
    """
    Atomically save cache dict to disk.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def _log(msg, level="info", verbose=True):
    """
    Lightweight logger used throughout e2e_analysis.
    """
    if not verbose:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level.upper()} {ts}] {msg}")

# ----------------------------
# loader + end picking (same logic as before)
# ----------------------------
def _find_topology_in_prod_dir(prod_dir, prefer_prefixes=("prod",), prefer_exts=(".tpr", ".gro")):
    candidates = []
    for ext in prefer_exts:
        candidates.extend(glob.glob(os.path.join(prod_dir, f"*{ext}")))
    if not candidates:
        raise FileNotFoundError(f"No topology files found in {prod_dir} (expected .tpr or .gro)")

    def score(p):
        base = os.path.basename(p).lower()
        st = os.stat(p)
        is_tpr = int(base.endswith(".tpr"))
        has_prefix = int(any(base.startswith(pref.lower()) for pref in prefer_prefixes))
        return (is_tpr, has_prefix, st.st_mtime, st.st_size)

    return sorted(candidates, key=score, reverse=True)[0]

def _load_universe_from_prod_dir(prod_dir, xtc_name="prod.xtc"):
    xtc = os.path.join(prod_dir, xtc_name)
    if not os.path.isfile(xtc):
        raise FileNotFoundError(f"Missing {xtc_name} in {prod_dir}")
    topo = _find_topology_in_prod_dir(prod_dir)
    u = mda.Universe(topo, xtc)
    return u, topo, xtc

def _polymer_as_largest_fragment(u):
    frags = u.atoms.fragments
    if len(frags) == 0:
        raise RuntimeError("No fragments found; ensure topology has bonds (TPR recommended).")
    return max(frags, key=len)

def _heavy_atoms(polymer_atoms, heavy_sel="not name H*"):
    heavy = polymer_atoms.select_atoms(heavy_sel)
    if len(heavy) < 2:
        raise RuntimeError(f"Heavy selection '{heavy_sel}' returned <2 atoms. Adjust heavy_sel.")
    return heavy

def _terminals_by_bonds(group):
    idx_set = set(group.indices.tolist())
    terminals = []
    for a in group:
        deg = 0
        for b in getattr(a, "bonds", []):
            a1, a2 = b.atoms
            other = a2 if a1.index == a.index else a1
            if other.index in idx_set:
                deg += 1
        if deg == 1:
            terminals.append(a)
    return terminals

def _pick_end_atoms_auto(u, heavy_sel="not name H*"):
    u.trajectory[0]
    poly = _polymer_as_largest_fragment(u)
    heavy = _heavy_atoms(poly, heavy_sel=heavy_sel)

    terms = _terminals_by_bonds(heavy)
    if len(terms) == 2:
        return poly, heavy, terms[0], terms[1], "bond_terminals_heavy"

    coords = heavy.positions
    box = u.trajectory.ts.dimensions
    D = distance_array(coords, coords, box=box)
    np.fill_diagonal(D, -np.inf)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    return poly, heavy, heavy[i], heavy[j], f"geometry_farthest_heavy (terminals_found={len(terms)})"

def compute_e2e_timeseries(prod_dir, xtc_name="prod.xtc", heavy_sel="not name H*"):
    u, topo, xtc = _load_universe_from_prod_dir(prod_dir, xtc_name=xtc_name)
    poly, heavy, a1, a2, method = _pick_end_atoms_auto(u, heavy_sel=heavy_sel)

    times = []
    dists = []
    for ts in u.trajectory:
        d = calc_bonds(a1.position[None, :], a2.position[None, :], box=ts.dimensions)[0]
        times.append(ts.time)
        dists.append(d)

    times = np.asarray(times, float)
    dists = np.asarray(dists, float)

    meta = {
        "prod_dir": prod_dir,
        "topology": topo,
        "trajectory": xtc,
        "polymer_natoms": int(len(poly)),
        "heavy_natoms": int(len(heavy)),
        "end_atoms_0based": (int(a1.index), int(a2.index)),
        "end_atom_names": (a1.name, a2.name),
        "end_atom_types": (getattr(a1, "type", None), getattr(a2, "type", None)),
        "end_method_used": method,
        "n_frames": int(len(times)),
        "mean_nm": float(np.mean(dists)),
        "std_nm": float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0,
    }
    return times, dists, meta

# ----------------------------
# simple Gaussian KDE (no seaborn)
# ----------------------------
def gaussian_kde_1d(
    x,
    grid=None,
    gridsize=400,
    bw_method="scott",
    bw=None,
    cut=3.0
):
    """
    1D Gaussian KDE with optional externally supplied grid.
    Returns: grid, density, bandwidth
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need >=2 samples for KDE.")

    std = np.std(x, ddof=1)

    # --- bandwidth selection ---
    if bw is None:
        if bw_method == "silverman":
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            s = min(std, iqr / 1.34) if np.isfinite(iqr) and iqr > 0 else std
            bw = 0.9 * s * n ** (-1/5)
        else:  # scott
            bw = std * n ** (-1/5)

        if not np.isfinite(bw) or bw <= 0:
            bw = max(std, 1e-6) * 0.1

    # --- grid handling ---
    if grid is None:
        lo = np.min(x) - cut * bw
        hi = np.max(x) + cut * bw
        grid = np.linspace(lo, hi, gridsize)
    else:
        grid = np.asarray(grid, float)

    # --- KDE evaluation ---
    diff = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diff**2).sum(axis=1) / (n * bw * np.sqrt(2*np.pi))

    return grid, dens, bw

# ----------------------------
# plot wrapper: time OR kde
# ----------------------------
def plot_e2e(
    prod_dir,
    mode="time",                 # 'time' or 'kde'
    xtc_name="prod.xtc",
    heavy_sel="not name H*",
    downsample=1,
    burn_in=0.0,                 # fraction of trajectory to discard from start, e.g. 0.1
    rolling_window=None,         # only for time mode
    kde_bw_method="scott",
    kde_bw=None,
    kde_gridsize=400,
    kde_cut=3.0,
    save_plot=None,
    save_dat=None,               # saves either time series or KDE curve
    figsize=(12, 5),
    dpi=300,
):
    times, dists, meta = compute_e2e_timeseries(prod_dir, xtc_name=xtc_name, heavy_sel=heavy_sel)

    # burn-in discard
    if burn_in and 0.0 < burn_in < 1.0:
        start = int(len(dists) * burn_in)
        times = times[start:]
        dists = dists[start:]

    # downsample
    downsample = int(max(1, downsample))
    t_plot = times[::downsample]
    d_plot = dists[::downsample]

    plt.figure(figsize=figsize)

    if mode.lower() == "time":
        plt.plot(t_plot, d_plot, lw=1.2, label="E2E")

        if rolling_window is not None and rolling_window >= 2 and len(d_plot) >= rolling_window:
            w = int(rolling_window)
            kernel = np.ones(w) / w
            d_roll = np.convolve(d_plot, kernel, mode="valid")
            t_roll = t_plot[w-1:]
            plt.plot(t_roll, d_roll, lw=2.0, label=f"Rolling mean (w={w})")

        plt.xlabel("Time (ps)")
        plt.ylabel("End-to-End Distance (nm)")
        plt.title(f"E2E vs Time | {meta['end_method_used']}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_dat:
            os.makedirs(os.path.dirname(save_dat), exist_ok=True)
            np.savetxt(save_dat, np.column_stack([times, dists]), header="time_ps  end_to_end_nm")

    elif mode.lower() == "kde":
        grid, dens, bw = gaussian_kde_1d(
            d_plot,
            bw=kde_bw,
            bw_method=kde_bw_method,
            gridsize=kde_gridsize,
            cut=kde_cut,
        )
        plt.plot(grid, dens, lw=2.0, label=f"KDE (bw={bw:.4g} nm)")
        plt.xlabel("End-to-End Distance (nm)")
        plt.ylabel("Density")
        plt.title(f"E2E KDE | burn_in={burn_in} | downsample={downsample} | {meta['end_method_used']}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_dat:
            os.makedirs(os.path.dirname(save_dat), exist_ok=True)
            np.savetxt(save_dat, np.column_stack([grid, dens]), header="e2e_nm  density")

    else:
        raise ValueError("mode must be 'time' or 'kde'")

    plt.tight_layout()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, dpi=dpi)

    plt.show()

    print("prod_dir:", meta["prod_dir"])
    print("end atoms (0-based):", meta["end_atoms_0based"], "| method:", meta["end_method_used"])
    print("frames used:", len(d_plot), "(after burn-in/downsample)")
    print("mean E2E (nm):", float(np.mean(d_plot)), "| std (nm):", float(np.std(d_plot, ddof=1)) if len(d_plot) > 1 else 0.0)

    return (times, dists, meta)

def compute_e2e_timeseries_nm(
    prod_dir,
    xtc_name="prod.xtc",
    heavy_sel="not name H*",
    use_cache=True,
    force_recompute=False,
    # NEW:
    normalize=None,          # None | "per_carbon" | "fraction" | "relative"
    carbon_count=None,       # int override, e.g. 200 if counting is unreliable
    # optional UX:
    verbose=False,
    show_progress=False,
    pbar_desc=None,
    pbar_leave=False,
):
    """
    Compute end-to-end distance time series for a /prod directory.

    Returns:
      times_ps (np.ndarray), dists (np.ndarray), meta (dict)

    Distances are in nm by default (same as your existing pipeline).
    Normalization (applied AFTER E2E computed in nm):
      - None: raw nm
      - "per_carbon": nm / N_C
      - "fraction": nm / (N_C - 1)
      - "relative": d / mean(d)  (unitless)
    """
    cache_file = _cache_path(prod_dir)
    cache = _load_cache(cache_file) if use_cache else {}

    u, topo, xtc = _load_universe(prod_dir, xtc_name=xtc_name)

    signature = {
        "topology": _file_sig(topo),
        "trajectory": _file_sig(xtc),
        "heavy_sel": heavy_sel,
        "xtc_name": xtc_name,
        "normalize": normalize,
        "carbon_count": carbon_count,
        "version": "e2e_unwrap_v3_norm",
    }

    # ---- cache hit ----
    if use_cache and (not force_recompute) and cache.get("signature") == signature:
        out = cache["data"]
        if verbose:
            _log(f"Cache HIT: {prod_dir}", "info", verbose=True)
        return out["times_ps"], out["dists_nm"], out["meta"]

    if verbose:
        if use_cache:
            _log(f"Cache MISS/STALE: {prod_dir}", "warn", verbose=True)
        _log(f"  topology  : {topo}", "info", verbose=True)
        _log(f"  trajectory: {xtc}", "info", verbose=True)

    # pick ends once
    poly, a1, a2, method = _pick_end_atoms(u, heavy_sel=heavy_sel)

    # unwrap polymer for correct intramolecular distances
    from MDAnalysis.transformations import unwrap
    u.trajectory.add_transformations(unwrap(poly))

    # carbon count for normalization
    nC = int(carbon_count) if carbon_count is not None else _count_polymer_carbons(poly)

    # prealloc arrays
    n_frames = len(u.trajectory)
    times_ps = np.empty(n_frames, dtype=float)
    dists_nm = np.empty(n_frames, dtype=float)

    iterator = u.trajectory
    if show_progress:
        desc = pbar_desc or f"E2E {os.path.basename(os.path.dirname(prod_dir))}/{os.path.basename(prod_dir)}"
        iterator = tqdm(iterator, total=n_frames, desc=desc, leave=pbar_leave)

    # compute raw nm
    for i, ts in enumerate(iterator):
        # positions are in Å after unwrap
        d_ang = np.linalg.norm(a1.position - a2.position)
        dists_nm[i] = d_ang / 10.0
        times_ps[i] = ts.time

    # apply normalization
    norm_label = "raw"
    if normalize is None:
        pass
    elif normalize in ("per_carbon", "perC"):
        if nC <= 0:
            raise RuntimeError("normalize='per_carbon' but could not determine N_C (carbon count <= 0).")
        dists_nm = dists_nm / float(nC)
        norm_label = f"per_carbon (nC={nC})"
    elif normalize in ("fraction", "frac", "per_bond"):
        denom = max(1, nC - 1)
        dists_nm = dists_nm / float(denom)
        norm_label = f"fraction (nC={nC})"
    elif normalize in ("relative",):
        mu = float(np.mean(dists_nm)) if len(dists_nm) else 1.0
        if mu == 0:
            mu = 1.0
        dists_nm = dists_nm / mu
        norm_label = "relative (d/mean)"
    else:
        raise ValueError("normalize must be None, 'per_carbon', 'fraction', or 'relative'")

    # meta
    u.trajectory[0]
    box_first = u.trajectory.ts.dimensions.copy()

    meta = {
        "prod_dir": prod_dir,
        "topology": topo,
        "trajectory": xtc,
        "end_atoms_0based": (int(a1.index), int(a2.index)),
        "end_atom_names": (a1.name, a2.name),
        "end_atom_types": (getattr(a1, "type", None), getattr(a2, "type", None)),
        "end_method_used": method,
        "n_frames": int(n_frames),
        "mean_nm": float(np.mean(dists_nm)),
        "std_nm": float(np.std(dists_nm, ddof=1)) if len(dists_nm) > 1 else 0.0,
        "box_firstframe": box_first,
        "nC": int(nC),
        "normalize": normalize,
        "normalize_label": norm_label,
    }

    # save cache
    if use_cache:
        _save_cache(cache_file, {"signature": signature, "data": {"times_ps": times_ps, "dists_nm": dists_nm, "meta": meta}})
        if verbose:
            _log(f"Cache SAVE: {cache_file}", "info", verbose=True)

    return times_ps, dists_nm, meta

def expand_run_prod_dirs(run1_prod_dir, runs=(1, 2, 3)):
    """
    Given a prod_dir that contains '/run1/prod', return a list of prod_dirs for runs.
    Example input:
      /.../C200/0/run1/prod
    Output for runs=(1,2,3):
      [/.../C200/0/run1/prod, /.../C200/0/run2/prod, /.../C200/0/run3/prod]
    """
    # replace /run{n}/prod safely
    m = re.search(r"/run(\d+)/prod/?$", run1_prod_dir)
    if not m:
        raise ValueError("run1_prod_dir must end with /runX/prod, e.g. .../run1/prod")
    prefix = run1_prod_dir[:m.start()]
    suffix = "/prod"
    return [f"{prefix}/run{r}{suffix}" for r in runs]

def _apply_burnin_downsample(times, dists, burn_in=0.0, downsample=1):
    if burn_in and 0.0 < burn_in < 1.0:
        start = int(len(dists) * burn_in)
        times = times[start:]
        dists = dists[start:]
    downsample = int(max(1, downsample))
    return times[::downsample], dists[::downsample]

def compute_e2e_run_means(run1_prod_dir, runs=(1,2,3), burn_in=0.0, downsample=1,
                          heavy_sel="not name H*", xtc_name="prod.xtc",
                          use_cache=True, force_recompute=False, verbose=True):
    """
    Computes per-run mean E2E (nm) for a condition defined by run1_prod_dir.
    Returns dict with:
      prod_dirs, per_run_means_nm, mean_of_means_nm, std_across_runs_nm, sem_across_runs_nm, n_runs_used
    """
    prod_dirs = expand_run_prod_dirs(run1_prod_dir, runs=runs)

    per_run_means = []
    used = []

    for d in prod_dirs:
        if not (os.path.isdir(d) and os.path.isfile(os.path.join(d, xtc_name))):
            if verbose:
                print(f"[skip] missing {xtc_name} or dir: {d}")
            continue

        t, dist_nm, meta = compute_e2e_timeseries_nm(
            d, xtc_name=xtc_name, heavy_sel=heavy_sel,
            use_cache=use_cache, force_recompute=force_recompute
        )
        t2, dist2 = _apply_burnin_downsample(t, dist_nm, burn_in=burn_in, downsample=downsample)
        if len(dist2) < 2:
            if verbose:
                print(f"[skip] not enough frames after burn-in/downsample: {d}")
            continue

        per_run_means.append(float(np.mean(dist2)))
        used.append(d)

    per_run_means = np.array(per_run_means, float)
    n = len(per_run_means)
    if n == 0:
        raise RuntimeError("No valid runs found for averaging.")

    mean_of_means = float(np.mean(per_run_means))
    std_runs = float(np.std(per_run_means, ddof=1)) if n > 1 else 0.0
    sem_runs = float(std_runs / np.sqrt(n)) if n > 1 else 0.0

    return {
        "run1_prod_dir": run1_prod_dir,
        "prod_dirs_used": used,
        "per_run_means_nm": per_run_means,
        "n_runs_used": n,
        "mean_of_means_nm": mean_of_means,
        "std_across_runs_nm": std_runs,
        "sem_across_runs_nm": sem_runs,
    }

def plot_e2e_overlay(
    run1_prod_dirs,
    labels=None,
    runs=None,                    # e.g. (1,2,3) expands each /run1/prod
    mode="kde",                   # "kde" or "time"
    combine="pool",               # "average" | "pool" | "both"
    burn_in=0.0,
    downsample=1,
    units="A",                    # "A" or "nm" (only affects display scaling)
    show_individual_runs=False,
    # NEW:
    normalize=None,               # None | "per_carbon" | "fraction" | "relative"
    carbon_count=None,            # override N_C if desired
    kde_bw_method="scott",
    kde_bw=None,
    gridsize=400,
    cut=3.0,
    figsize=(12, 5),
    save_plot=None,               # <<<<<< keep this name
    dpi=300,
    verbose=True,
    show_progress=False,
):
    if labels is None:
        labels = [f"cond_{i+1}" for i in range(len(run1_prod_dirs))]
    if len(labels) != len(run1_prod_dirs):
        raise ValueError("labels must match length of run1_prod_dirs")
    if combine not in ("average", "pool", "both"):
        raise ValueError("combine must be 'average', 'pool', or 'both'")

    # Units scaling for display only
    if units.lower() in ["a", "å", "angstrom", "angstroms"]:
        scale = 10.0
        unit_str = "Å"
    else:
        scale = 1.0
        unit_str = "nm"

    # Label for normalized quantity
    if normalize is None:
        x_label = f"End-to-End Distance ({unit_str})"
    elif normalize in ("per_carbon", "perC"):
        x_label = f"E2E / N_C ({unit_str}/C)"
    elif normalize in ("fraction", "frac", "per_bond"):
        x_label = f"E2E / (N_C-1) ({unit_str})"
    elif normalize in ("relative",):
        x_label = "E2E / ⟨E2E⟩ (unitless)"
        scale = 1.0
        unit_str = "unitless"
    else:
        x_label = f"End-to-End Distance ({unit_str})"

    plt.figure(figsize=figsize)

    # Build shared KDE grid across all series to average densities cleanly
    shared_grid = None
    if mode.lower() == "kde":
        all_samples = []
        for base in run1_prod_dirs:
            prod_list = [base] if runs is None else expand_run_prod_dirs(base, runs=runs)
            for pd in prod_list:
                try:
                    t, d, _ = compute_e2e_timeseries_nm(
                        pd,
                        verbose=False,
                        show_progress=False,
                        normalize=normalize,
                        carbon_count=carbon_count,
                    )
                    _, d2 = _apply_burnin_downsample(t, d, burn_in=burn_in, downsample=downsample)
                    if len(d2) > 1:
                        all_samples.append(d2)
                except Exception:
                    pass

        if len(all_samples) == 0:
            raise RuntimeError("No valid data found to build KDE grid (all runs failed?).")

        pooled = np.concatenate(all_samples)
        # get a reference bandwidth to set grid padding
        _, _, bw_ref = gaussian_kde_1d(pooled, gridsize=50, bw_method=kde_bw_method, bw=kde_bw, cut=cut)
        lo = np.min(pooled) - cut * bw_ref
        hi = np.max(pooled) + cut * bw_ref
        shared_grid = np.linspace(lo, hi, gridsize)

    # Plot each condition
    for base, lab in zip(run1_prod_dirs, labels):
        prod_list = [base] if runs is None else expand_run_prod_dirs(base, runs=runs)

        series = []
        for pd in prod_list:
            if not os.path.isdir(pd):
                _log(f"[skip] {pd}: not a directory", "warn", verbose)
                continue
            try:
                t, d, meta = compute_e2e_timeseries_nm(
                    pd,
                    verbose=verbose,
                    show_progress=show_progress,
                    normalize=normalize,
                    carbon_count=carbon_count,
                )
                t2, d2 = _apply_burnin_downsample(t, d, burn_in=burn_in, downsample=downsample)
                if len(d2) >= 2:
                    series.append((pd, t2, d2, meta))
            except Exception as e:
                _log(f"[skip] {pd}: {e}", "warn", verbose)

        if len(series) == 0:
            _log(f"[skip] no valid runs for: {base}", "warn", verbose)
            continue

        if show_individual_runs and mode.lower() == "kde":
            for (pd, _, d2, _) in series:
                g, dens, _ = gaussian_kde_1d(
                    d2,
                    grid=shared_grid,
                    gridsize=gridsize,
                    bw_method=kde_bw_method,
                    bw=kde_bw,
                    cut=cut,
                )
                plt.plot(g * scale, dens, lw=1.0, alpha=0.25)

        if mode.lower() == "kde":
            if combine in ("pool", "both"):
                pooled = np.concatenate([d2 for (_, _, d2, _) in series])
                g, dens, _ = gaussian_kde_1d(
                    pooled,
                    grid=shared_grid,
                    gridsize=gridsize,
                    bw_method=kde_bw_method,
                    bw=kde_bw,
                    cut=cut,
                )
                plt.plot(g * scale, dens, lw=2.6, label=f"{lab} pool")

            if combine in ("average", "both"):
                dens_list = []
                for (_, _, d2, _) in series:
                    g, dens_r, _ = gaussian_kde_1d(
                        d2,
                        grid=shared_grid,
                        gridsize=gridsize,
                        bw_method=kde_bw_method,
                        bw=kde_bw,
                        cut=cut,
                    )
                    dens_list.append(dens_r)
                dens_avg = np.mean(np.vstack(dens_list), axis=0)
                plt.plot(g * scale, dens_avg, lw=2.2, ls="--", label=f"{lab} average")

        else:
            # time plot
            t0 = series[0][1]
            same_time = all(len(t0) == len(ti) and np.allclose(t0, ti) for (_, ti, _, _) in series[1:])

            if combine in ("average", "both") and same_time:
                D = np.vstack([d2 for (_, _, d2, _) in series])
                plt.plot(t0, np.mean(D, axis=0) * scale, lw=2.6, label=f"{lab} avg")
            else:
                pd, t2, d2, _ = series[0]
                plt.plot(t2, d2 * scale, lw=1.6, label=f"{lab} (run1)")

    # finalize
    if mode.lower() == "kde":
        plt.xlabel(x_label)
        plt.ylabel("Density")
        plt.title(f"E2E KDE overlay | combine={combine} | burn_in={burn_in} | downsample={downsample} | normalize={normalize}")
    else:
        plt.xlabel("Time (ps)")
        plt.ylabel(x_label)
        plt.title(f"E2E time overlay | combine={combine} | burn_in={burn_in} | downsample={downsample} | normalize={normalize}")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, dpi=dpi)
    plt.show()

