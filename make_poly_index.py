#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import re
from pathlib import Path
from typing import Optional


def truncate_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")


def append_log(log_path: Path, msg: str) -> None:
    with log_path.open("a") as fh:
        fh.write(msg + "\n")


def gro_contains_resname(gro: Path, resname: str) -> bool:
    """
    Check whether a GRO file contains a given resname.
    GRO fixed-width: resname is columns 6–10 (0-based 5:10).
    """
    try:
        lines = gro.read_text(errors="ignore").splitlines()
    except OSError:
        return False

    if len(lines) < 3:
        return False

    try:
        natoms = int(lines[1].strip())
    except ValueError:
        return False

    for ln in lines[2 : 2 + natoms]:
        if len(ln) >= 10 and ln[5:10].strip() == resname:
            return True
    return False


def infer_polymer_resname(gro: Path, prod_dir: Path) -> Optional[str]:
    """
    Infer polymer residue name:
      1) If LIG exists → LIG
      2) Else infer C<number> from directory path and verify it exists in GRO
    """
    if gro_contains_resname(gro, "LIG"):
        return "LIG"

    m = re.search(r"/(C\d+)/", str(prod_dir))
    if m:
        cname = m.group(1)
        if gro_contains_resname(gro, cname):
            return cname

    return None


def make_atom_filter(elem: Optional[str], name_prefix: Optional[str], group_label: str) -> str:
    """
    Build a GROMACS selection clause restricting to certain atoms.
    Prefer element filter if provided; fall back to atom name prefix.
    If neither provided, returns an empty string (no extra filter).
    """
    if elem and name_prefix:
        raise ValueError(f"Choose only one of --{group_label}-elem or --{group_label}-name-prefix")

    if elem:
        # GROMACS selection language supports: element <sym>
        # (Requires element info present/recognized in your topology.)
        return f" and element {elem}"

    if name_prefix:
        # Atom name prefix match. Common for many FF naming, but not universal.
        # Example: name \"C*\" matches C1, C2, CA, CT, etc.
        return f' and name "{name_prefix}*"'

    return ""


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> bool:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd), f"(cwd={cwd})")
        return True

    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate poly_LIG.ndx uniformly across prod dirs using gmx select.")
    ap.add_argument("--root", default="1-4-DCB", help="Root directory to search (default: 1-4-DCB)")
    ap.add_argument("--gmx", default="gmx", help="GROMACS command (default: gmx)")
    ap.add_argument("--log", default="gmx_select_failed.log", help="Failure log file")
    ap.add_argument("--dry-run", action="store_true", help="Print actions, do not run gmx")

    # Polymer atom filtering
    ap.add_argument("--poly-elem", default=None, help="Restrict polymer group to element (e.g., C)")
    ap.add_argument("--poly-name-prefix", default=None, help="Restrict polymer group to atomname prefix (e.g., C)")

    # Solvent atom filtering
    ap.add_argument("--solv-elem", default=None, help="Restrict solvent group to element (e.g., C)")
    ap.add_argument("--solv-name-prefix", default=None, help="Restrict solvent group to atomname prefix (e.g., C)")

    # Solvent residue aliases (your 1-4-DCB case: DCB or PDC)
    ap.add_argument("--solvent-resnames", default="DCB,PDC", help="Comma-separated solvent residue names (default: DCB,PDC)")

    args = ap.parse_args()

    root = Path(args.root)
    log_path = Path(args.log)
    truncate_log(log_path)

    poly_atom_filter = make_atom_filter(args.poly_elem, args.poly_name_prefix, "poly")
    solv_atom_filter = make_atom_filter(args.solv_elem, args.solv_name_prefix, "solv")

    solv_resnames = [s.strip() for s in args.solvent_resnames.split(",") if s.strip()]
    if not solv_resnames:
        print("Error: --solvent-resnames cannot be empty", file=sys.stderr)
        return 2

    # Build solvent resname clause: (resname DCB or resname PDC ...)
    solv_clause = " or ".join([f"resname {rn}" for rn in solv_resnames])
    solv_clause = f"({solv_clause})"

    prod_dirs = sorted([p for p in root.rglob("prod") if p.is_dir()])

    for prod_dir in prod_dirs:
        gro = prod_dir / "prod.gro"
        tpr = prod_dir / "prod.tpr"
        xtc = prod_dir / "prod.xtc"

        if not gro.exists() or not tpr.exists():
            continue

        traj = xtc if xtc.exists() else gro

        poly = infer_polymer_resname(gro, prod_dir)
        if poly is None:
            append_log(log_path, f"NO_POLY_RESNAME {prod_dir}")
            continue

        # Always output poly_LIG.ndx, always group names POLY_LIG and SOLV_DCB
        # Apply optional atom filters
        sel = (
            f'name "POLY_LIG" (resname {poly}{poly_atom_filter}); '
            f'name "SOLV_DCB" ({solv_clause}{solv_atom_filter})'
        )

        cmd = [
            args.gmx, "select",
            "-s", "prod.tpr",
            "-f", traj.name,
            "-select", sel,
            "-on", "poly_LIG.ndx",
        ]

        ok = run_cmd(cmd, cwd=prod_dir, dry_run=args.dry_run)
        if not ok:
            append_log(log_path, f"GMX_FAIL {prod_dir} poly={poly}")
            continue

        if args.dry_run:
            print(f"DRY-OK {prod_dir} poly={poly}")
        else:
            print(f"OK {prod_dir} poly={poly}")

    print(f"Done. Failures logged to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
