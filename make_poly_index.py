#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import re
from pathlib import Path
from typing import Optional, List


def truncate_log(path: Path) -> None:
    path.write_text("")


def append_log(path: Path, msg: str) -> None:
    with path.open("a") as fh:
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


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_name_prefix_filter(prefixes: List[str]) -> str:
    """
    Build an atomname prefix filter clause.
    prefixes=["C"] -> ' and name "C*"'
    prefixes=["Cl","CL"] -> ' and (name "Cl*" or name "CL*")'
    prefixes=[] -> '' (no filter)
    """
    if not prefixes:
        return ""

    if len(prefixes) == 1:
        return f' and name "{prefixes[0]}*"'

    ors = " or ".join([f'name "{p}*"' for p in prefixes])
    return f" and ({ors})"


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
    ap = argparse.ArgumentParser(
        description="Generate poly_LIG.ndx uniformly across prod dirs using gmx select."
    )
    ap.add_argument("--root", default="1-4-DCB", help="Root directory to search")
    ap.add_argument("--gmx", default="gmx", help="GROMACS command (default: gmx)")
    ap.add_argument("--log", default="gmx_select_failed.log", help="Failure log file")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")

    ap.add_argument(
        "--out",
        default="poly_LIG.ndx",
        help='Output index filename in each prod dir (default: "poly_LIG.ndx")',
    )

    # Atomname prefix knobs (default polymer = carbons)
    ap.add_argument(
        "--poly-prefixes",
        default="C",
        help='Comma-separated polymer atomname prefixes (default: "C")',
    )
    ap.add_argument(
        "--solv-prefixes",
        default="C",
        help='Comma-separated solvent atomname prefixes (default: "C")',
    )

    # Solvent residue aliases (your 1-4-DCB case)
    ap.add_argument(
        "--solvent-resnames",
        default="DCB,PDC",
        help='Comma-separated solvent residue names (default: "DCB,PDC")',
    )

    # ✅ NEW: polymer-only mode
    ap.add_argument(
        "--poly-only",
        action="store_true",
        help="Create only POLY_LIG group (no SOLV_DCB group)",
    )

    args = ap.parse_args()

    root = Path(args.root)
    log_path = Path(args.log)
    truncate_log(log_path)

    poly_atom_filter = build_name_prefix_filter(parse_csv_list(args.poly_prefixes))

    # Only compute solvent parts if needed
    solv_resnames = parse_csv_list(args.solvent_resnames)
    solv_clause = ""
    solv_atom_filter = ""
    if not args.poly_only:
        solv_clause = " or ".join([f"resname {r}" for r in solv_resnames])
        solv_clause = f"({solv_clause})"
        solv_atom_filter = build_name_prefix_filter(parse_csv_list(args.solv_prefixes))

    for prod_dir in sorted(p for p in root.rglob("prod") if p.is_dir()):
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

        if args.poly_only:
            sel = f'name "POLY_LIG" (resname {poly}{poly_atom_filter})'
        else:
            sel = (
                f'name "POLY_LIG" (resname {poly}{poly_atom_filter}); '
                f'name "SOLV_DCB" ({solv_clause}{solv_atom_filter})'
            )

        cmd = [
            args.gmx, "select",
            "-s", "prod.tpr",
            "-f", traj.name,
            "-select", sel,
            "-on", args.out,
        ]

        ok = run_cmd(cmd, cwd=prod_dir, dry_run=args.dry_run)
        if not ok:
            append_log(log_path, f"GMX_FAIL {prod_dir} poly={poly}")
            continue

        tag = "DRY-OK" if args.dry_run else "OK"
        print(f"{tag} {prod_dir} poly={poly}")

    print(f"Done. Failures logged to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
