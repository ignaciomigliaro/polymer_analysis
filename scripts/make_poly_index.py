#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import re
from pathlib import Path
from typing import Optional, List
import shutil
import sys


def truncate_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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

def pick_solvent_resname(gro: Path, solvent_resnames: list[str]) -> Optional[str]:
    for r in solvent_resnames:
        if gro_contains_resname(gro, r):
            return r
    return None

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
    Build an atomname prefix filter clause (valid in gmx selection language).
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


def run_gmx(cmd: list[str], cwd: Path, dry_run: bool, log_path: Path, verbose: bool) -> bool:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd), f"(cwd={cwd})")
        return True

    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)

    if proc.returncode == 0:
        return True

    with log_path.open("a") as fh:
        fh.write("\n" + "=" * 80 + "\n")
        fh.write(f"FAILED in: {cwd}\n")
        fh.write(f"CMD: {' '.join(cmd)}\n")
        fh.write("STDOUT:\n" + (proc.stdout or "") + "\n")
        fh.write("STDERR:\n" + (proc.stderr or "") + "\n")

    if verbose:
        print(f"\nFAILED in: {cwd}\nCMD: {' '.join(cmd)}\n{proc.stderr}\n")

    return False


def rename_ndx_groups(ndx_path: Path, new_names: List[str]) -> None:
    """
    Rename the first N group headers in an .ndx file to the provided names.
    Safe here because we generate either 1 group (poly-only) or 2 groups (poly+solvent)
    in a known order.
    """
    lines = ndx_path.read_text().splitlines()
    header_idx = [i for i, ln in enumerate(lines) if ln.strip().startswith("[") and ln.strip().endswith("]")]
    if len(header_idx) < len(new_names):
        raise RuntimeError(f"Expected >= {len(new_names)} groups in {ndx_path}, found {len(header_idx)}")

    for j, name in enumerate(new_names):
        lines[header_idx[j]] = f"[ {name} ]"

    ndx_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate atom index groups (polymer/solvent) and optionally COM-of-resname outputs using gmx select."
    )
    ap.add_argument("--root", default="1-4-DCB", help="Root directory to search")
    ap.add_argument("--gmx", default="gmx_mpi", help="GROMACS command (e.g., gmx_mpi)")
    ap.add_argument("--log", default="gmx_select_failed.log", help="Failure log file")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    ap.add_argument("--verbose", action="store_true", help="Print gmx stderr on failures")

    # Output index filename (atoms)
    ap.add_argument("--out", default="poly_LIG.ndx",
                    help='Atom index filename written inside each prod dir (default: "poly_LIG.ndx")')

    # Atomname prefix knobs (atoms)
    ap.add_argument("--poly-prefixes", default="C",
                    help='Comma-separated polymer atomname prefixes (default: "C")')
    ap.add_argument("--solv-prefixes", default="C",
                    help='Comma-separated solvent atomname prefixes (default: "C")')

    # Solvent residue aliases
    ap.add_argument("--solvent-resnames", default="DCB,PDC",
                    help='Comma-separated solvent residue names (default: "DCB,PDC")')

    # Polymer-only mode for atom index
    ap.add_argument("--poly-only", action="store_true",
                    help="Create only the polymer atom group (no solvent atom group)")

    # Optional: rename group headers inside the ndx file
    ap.add_argument("--rename-groups", action="store_true",
                    help="Rename index group headers to [ POLY_LIG ] and [ SOLV_DCB ]")

    # ---- Optional COM feature (positions output) ----
    # You wanted: 'com of resname DCB' style. We allow any residue name here.
    ap.add_argument("--com-solvent", action="store_true", 
                    help='If set, also run selection "com of resname <RES>" and write -oc output (e.g. DCB)')
    ap.add_argument("--com-out", default="COM.xvg",
                    help='Output XVG for COM selection (default: COM.xvg)')
    ap.add_argument("--com-traj", default="prod.xtc",
                    help='Trajectory file to use for COM (default: prod.xtc). Must exist in prod dir.')

    args = ap.parse_args()

    root = Path(args.root)
    log_path = Path(args.log)
    truncate_log(log_path)

    if shutil.which(args.gmx) is None:
        print(
            f'ERROR: GROMACS executable "{args.gmx}" not found in PATH.\n'
            f"Try: module load gromacs\n"
            f"Or pass --gmx gmx_mpi / --gmx /full/path/to/gmx",
            file=sys.stderr,
        )
        return 2

    poly_atom_filter = build_name_prefix_filter(parse_csv_list(args.poly_prefixes))

    solv_clause = ""
    solv_atom_filter = ""
    if not args.poly_only:
        solv_resnames = parse_csv_list(args.solvent_resnames)
        if not solv_resnames:
            print("ERROR: --solvent-resnames cannot be empty unless --poly-only is set", file=sys.stderr)
            return 2
        solv_clause = " or ".join([f"resname {r}" for r in solv_resnames])
        solv_clause = f"({solv_clause})"
        solv_atom_filter = build_name_prefix_filter(parse_csv_list(args.solv_prefixes))

    # Iterate over all .../prod directories
    for prod_dir in sorted(p for p in root.rglob("prod") if p.is_dir()):
        gro = prod_dir / "prod.gro"
        tpr = prod_dir / "prod.tpr"
        xtc = prod_dir / "prod.xtc"

        if not gro.exists() or not tpr.exists():
            continue

        traj_for_atoms = xtc if xtc.exists() else gro

        
        

        # 2) Atom index groups (what you need for RDF/polystat)
        poly = infer_polymer_resname(gro, prod_dir)
        if poly is None:
            append_log(log_path, f"NO_POLY_RESNAME {prod_dir}")
            continue

        poly_sel = f"(resname {poly}{poly_atom_filter})"

        if args.poly_only:
            sel = poly_sel

        elif args.com_solvent:
            solv_resnames = parse_csv_list(args.solvent_resnames)
            solv = pick_solvent_resname(gro, solv_resnames)
            if solv is None:
                append_log(log_path, f"NO_SOLVENT_MATCH {prod_dir} candidates={','.join(solv_resnames)}")
                continue
            sel = f"{poly_sel}; com of resname {solv}"

        else:
            sel = f"{poly_sel}; ({solv_clause}{solv_atom_filter})"


        cmd = [
            args.gmx, "select",
            "-s", "prod.tpr",
            "-f", traj_for_atoms.name,
            "-select", sel,
            "-on", args.out,
        ]

        ok = run_gmx(cmd, cwd=prod_dir, dry_run=args.dry_run, log_path=log_path, verbose=args.verbose)
        if not ok:
            append_log(log_path, f"GMX_FAIL {prod_dir} poly={poly}")
            continue

        if args.rename_groups and not args.dry_run:
            try:
                ndx_path = prod_dir / args.out
                if args.poly_only:
                    rename_ndx_groups(ndx_path, ["POLY_LIG"])
                else:
                    rename_ndx_groups(ndx_path, ["POLY_LIG", "SOLV_DCB"])
            except Exception as e:
                append_log(log_path, f"RENAME_FAIL {prod_dir} err={e}")

        tag = "DRY-OK" if args.dry_run else "OK"
        print(f"{tag} {prod_dir} poly={poly}")

    print(f"Done. Failures logged to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
