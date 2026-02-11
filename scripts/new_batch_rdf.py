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

    m = re.search(r"/(C\d+)/", str(prod_dir).replace("\\", "/"))
    if m:
        cname = m.group(1)
        if gro_contains_resname(gro, cname):
            return cname

    return None


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


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


def _infer_tag_from_path(prod_dir: Path) -> str:
    """
    Typical:
      .../<solvent>/<C20>/<10>/run2/prod
    Returns:
      "1-4-DCB_C20_10_run2" or "C20_10_run2"
    """
    p = str(prod_dir).replace("\\", "/")

    solv = None
    msolv = re.search(r"/([^/]*DCB[^/]*)/C\d+/", p)
    if msolv:
        solv = msolv.group(1)

    mC = re.search(r"/(C\d+)/", p)
    mdeg = re.search(r"/C\d+/(\d+)/run\d+/", p)
    mrun = re.search(r"/(run\d+)/prod/?$", p)

    parts = []
    if solv:
        parts.append(solv)
    if mC:
        parts.append(mC.group(1))
    if mdeg:
        parts.append(mdeg.group(1))
    if mrun:
        parts.append(mrun.group(1))

    if parts:
        return "_".join(parts)

    return prod_dir.as_posix().strip("/").replace("/", "_")


def build_sel_from_prefixes(resname: str, prefixes: list[str]) -> str:
    """
    Build a GROMACS selection:
      - no prefixes:  resname X
      - one:          resname X and name "C*"
      - many:         resname X and (name "C*" or name "H*")
    """
    if len(prefixes) == 0:
        return f"resname {resname}"
    if len(prefixes) == 1:
        return f'resname {resname} and name "{prefixes[0]}*"'
    ors = " or ".join([f'name "{p}*"' for p in prefixes])
    return f"resname {resname} and ({ors})"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch-run gmx rdf over many prod dirs, inferring polymer (LIG/C20/...) and solvent (DCB/PDC/...), and writing non-overwriting outputs."
    )
    ap.add_argument("--root", default="1-4-DCB", help="Root directory to search (contains C*/deg/run*/prod/...)")
    ap.add_argument("--gmx", default="gmx_mpi", help="GROMACS command (e.g., gmx_mpi)")
    ap.add_argument("--log", default="gmx_rdf_failed.log", help="Failure log file")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    ap.add_argument("--verbose", action="store_true", help="Print gmx stderr on failures")

    # selections
    ap.add_argument("--poly-prefixes", default="C", help='Comma-separated polymer atomname prefixes (default: "C")')
    ap.add_argument("--solvent-resnames", default="DCB,PDC", help='Comma-separated solvent residue names (default: "DCB,PDC")')

    # NEW: restrict which solvent atoms are used in -sel (empty => use whole solvent resname)
    ap.add_argument(
        "--solvent-prefixes",
        default="",
        help='Comma-separated solvent atomname prefixes to include in -sel (default: "" = all solvent atoms). '
             'Example: --solvent-prefixes C (only solvent atoms with names starting with C).'
    )

    # rdf knobs
    ap.add_argument("--seltype", default="mol_com", choices=[
        "atom", "res_com", "res_cog", "mol_com", "mol_cog",
        "whole_res_com", "whole_res_cog", "whole_mol_com", "whole_mol_cog",
        "part_res_com", "part_res_cog", "part_mol_com", "part_mol_cog",
        "dyn_res_com", "dyn_res_cog", "dyn_mol_com", "dyn_mol_cog",
    ], help="Selection output positions (default: mol_com)")
    ap.add_argument("--selrpos", default="atom", choices=[
        "atom", "res_com", "res_cog", "mol_com", "mol_cog",
        "whole_res_com", "whole_res_cog", "whole_mol_com", "whole_mol_cog",
        "part_res_com", "part_res_cog", "part_mol_com", "part_mol_cog",
        "dyn_res_com", "dyn_res_cog", "dyn_mol_com", "dyn_mol_cog",
    ], help="Selection reference positions (default: atom)")
    ap.add_argument("--bin", type=float, default=0.1, help="RDF bin width (default: 0.1)")
    ap.add_argument(
        "--begin",
        type=float,
        default=30.0,
        help="First frame time (ps) to read from trajectory (passed to gmx rdf as -b). Default: 30.0",
    )

    # outputs
    ap.add_argument("--out-rdf", default='rdf_Cpoly_{solv}molCOM_{tag}.xvg',
                    help='Output RDF filename template (default: rdf_Cpoly_{solv}molCOM_{tag}.xvg)')
    ap.add_argument("--out-cn", default='cn_Cpoly_{solv}molCOM_{tag}.xvg',
                    help='Output CN filename template (default: cn_Cpoly_{solv}molCOM_{tag}.xvg)')

    ap.add_argument("--ndx", default=None,
                    help='Optional index file. If provided, passed with -n.')

    ap.add_argument("--skip-existing", action="store_true",
                    help="If output file already exists, skip running RDF in that directory.")

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

    poly_prefixes = parse_csv_list(args.poly_prefixes)
    solv_resnames = parse_csv_list(args.solvent_resnames)
    solv_prefixes = parse_csv_list(args.solvent_prefixes)

    if not solv_resnames:
        print("ERROR: --solvent-resnames cannot be empty", file=sys.stderr)
        return 2

    for prod_dir in sorted(p for p in root.rglob("prod") if p.is_dir()):
        gro = prod_dir / "prod.gro"
        tpr = prod_dir / "prod.tpr"
        xtc = prod_dir / "prod.xtc"

        if not gro.exists() or not tpr.exists():
            continue
        if not xtc.exists():
            append_log(log_path, f"NO_XTC {prod_dir}")
            continue

        poly = infer_polymer_resname(gro, prod_dir)
        if poly is None:
            append_log(log_path, f"NO_POLY_RESNAME {prod_dir}")
            continue

        solv = pick_solvent_resname(gro, solv_resnames)
        if solv is None:
            append_log(log_path, f"NO_SOLVENT_MATCH {prod_dir} candidates={','.join(solv_resnames)}")
            continue

        # --- Selection strings ---
        ref_sel = build_sel_from_prefixes(poly, poly_prefixes)
        sel = build_sel_from_prefixes(solv, solv_prefixes)  # NEW

        tag = _infer_tag_from_path(prod_dir)

        out_rdf = prod_dir / args.out_rdf.format(solv=solv, tag=tag)
        out_cn = prod_dir / args.out_cn.format(solv=solv, tag=tag)

        if args.skip_existing and (out_rdf.exists() or out_cn.exists()):
            print(f"SKIP {prod_dir} (exists) -> {out_rdf.name}")
            continue

        cmd = [
            args.gmx, "rdf",
            "-s", "prod.tpr",
            "-f", "prod.xtc",
            "-ref", ref_sel,
            "-sel", sel,
            "-seltype", args.seltype,
            "-selrpos", args.selrpos,
            "-bin", str(args.bin),
            "-o", out_rdf.name,
            "-cn", out_cn.name,
            "-b", str(args.begin),
        ]

        if args.ndx:
            ndx_path = prod_dir / args.ndx
            if not ndx_path.exists():
                append_log(log_path, f"NO_NDX {prod_dir} ndx={args.ndx}")
                continue
            cmd += ["-n", args.ndx]

        ok = run_gmx(cmd, cwd=prod_dir, dry_run=args.dry_run, log_path=log_path, verbose=args.verbose)
        if not ok:
            append_log(log_path, f"GMX_FAIL {prod_dir} poly={poly} solv={solv} tag={tag}")
            continue

        tag_msg = "DRY-OK" if args.dry_run else "OK"
        print(f"{tag_msg} {prod_dir} poly={poly} solv={solv} -> {out_rdf.name}")

    print(f"Done. Failures logged to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
