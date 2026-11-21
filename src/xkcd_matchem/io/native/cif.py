# Copyright 2026 xkcd-matchem contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native parser for Crystallographic Information Files (CIF)."""

from __future__ import annotations

import pathlib
import re

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


def _strip_uncertainty(token: str) -> float:
    """Parses a CIF float stripping trailing uncertainty parentheses."""
    cleaned = re.sub(r"\([^)]*\)", "", token.strip())
    return float(cleaned)


def _clean_element_symbol(raw: str) -> str:
    """Normalizes CIF element labels such as ``'Fe3+'`` or ``'Si1'``."""
    stripped = re.sub(r"[^A-Za-z]", "", raw.strip())
    if len(stripped) >= 2:
        cand2 = stripped[:2].capitalize()
        if cand2 in elements.ELEMENTS_BY_SYMBOL:
            return cand2
    if stripped:
        cand1 = stripped[0].upper()
        if cand1 in elements.ELEMENTS_BY_SYMBOL:
            return cand1
    raise exceptions.ParseError(f"Unrecognized CIF element symbol: {raw!r}")


def _tokenize_cif(text: str) -> list[str]:
    """Tokenizes CIF syntax with comments, quotes, and multiline blocks."""
    tokens: list[str] = []
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith(";"):
            block = [line[1:]]
            idx += 1
            while idx < len(lines) and not lines[idx].startswith(";"):
                block.append(lines[idx])
                idx += 1
            tokens.append("\n".join(block).strip())
            idx += 1
            continue
        pos = 0
        n_len = len(line)
        while pos < n_len:
            ch = line[pos]
            if ch.isspace():
                pos += 1
            elif ch == "#":
                break
            elif ch in ("'", '"'):
                quote = ch
                pos += 1
                start = pos
                while pos < n_len and not (
                    line[pos] == quote
                    and (pos + 1 == n_len or line[pos + 1].isspace())
                ):
                    pos += 1
                tokens.append(line[start:pos])
                pos += 1
            else:
                start = pos
                while pos < n_len and not line[pos].isspace():
                    pos += 1
                tokens.append(line[start:pos])
        idx += 1
    return tokens


def _parse_affine_term(expr: str) -> tuple[np.ndarray, float]:
    """Parses one symmetry axis expression like ``'-x+1/2'`` without eval."""
    coeff = np.zeros(3, dtype=np.float64)
    shift = 0.0
    normalized = expr.replace(" ", "").lower()
    terms = re.findall(r"[+-]?[^+-]+", normalized)
    axis_map = {"x": 0, "y": 1, "z": 2}
    for term in terms:
        sign = -1.0 if term.startswith("-") else 1.0
        body = term.lstrip("+-")
        if body in axis_map:
            coeff[axis_map[body]] += sign
        elif body.endswith(("x", "y", "z")):
            axis = axis_map[body[-1]]
            pref = body[:-1]
            factor = float(pref) if pref else 1.0
            coeff[axis] += sign * factor
        elif "/" in body:
            num_s, den_s = body.split("/", maxsplit=1)
            shift += sign * (float(num_s) / float(den_s))
        else:
            shift += sign * float(body)
    return coeff, shift


def _apply_symops(
    frac_sites: np.ndarray,
    site_symbols: list[str],
    site_labels: list[str],
    symops: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Expands asymmetric unit sites using CIF symmetry operations."""
    if not symops:
        return frac_sites, site_symbols, site_labels
    parsed_ops: list[tuple[np.ndarray, np.ndarray]] = []
    for op_str in symops:
        parts = op_str.split(",")
        if len(parts) != 3:
            continue
        rot = np.zeros((3, 3), dtype=np.float64)
        trans = np.zeros(3, dtype=np.float64)
        for row_idx, expr in enumerate(parts):
            c, s = _parse_affine_term(expr)
            rot[row_idx] = c
            trans[row_idx] = s
        parsed_ops.append((rot, trans))
    if not parsed_ops:
        return frac_sites, site_symbols, site_labels

    expanded_frac: list[np.ndarray] = []
    expanded_syms: list[str] = []
    expanded_lbls: list[str] = []
    for pos, sym, lbl in zip(
        frac_sites, site_symbols, site_labels, strict=True
    ):
        for rot, trans in parsed_ops:
            cand = np.mod(rot @ pos + trans, 1.0)
            cand = np.where(np.isclose(cand, 1.0, atol=1e-5), 0.0, cand)
            duplicate = False
            for existing in expanded_frac:
                diff = np.abs(cand - existing)
                diff = np.minimum(diff, 1.0 - diff)
                if np.linalg.norm(diff) < 1e-3:
                    duplicate = True
                    break
            if not duplicate:
                expanded_frac.append(cand)
                expanded_syms.append(sym)
                expanded_lbls.append(lbl)
    return np.vstack(expanded_frac), expanded_syms, expanded_lbls


def _extract_loops_and_scalars(
    tokens: list[str],
) -> tuple[dict[str, str], list[tuple[list[str], list[list[str]]]]]:
    """Splits CIF tokens into scalar tag map and loop tables."""
    scalars: dict[str, str] = {}
    loops: list[tuple[list[str], list[list[str]]]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = tok.lower()
        if low == "loop_":
            i += 1
            headers: list[str] = []
            while i < len(tokens) and tokens[i].startswith("_"):
                headers.append(tokens[i].lower())
                i += 1
            rows: list[list[str]] = []
            n_cols = len(headers)
            while i < len(tokens):
                nxt = tokens[i].lower()
                if nxt == "loop_" or nxt.startswith(("_", "data_")):
                    break
                if n_cols == 0 or i + n_cols > len(tokens):
                    i += 1
                    continue
                rows.append(tokens[i : i + n_cols])
                i += n_cols
            loops.append((headers, rows))
        elif tok.startswith("_") and i + 1 < len(tokens):
            scalars[low] = tokens[i + 1]
            i += 2
        else:
            i += 1
    return scalars, loops


def _collect_cif_sites_and_symops(
    loops: list[tuple[list[str], list[list[str]]]],
) -> tuple[list[str], list[list[float]], list[str], list[str]]:
    """Extracts symmetry operations and fractional sites from CIF loops."""
    symops: list[str] = []
    frac_list: list[list[float]] = []
    syms: list[str] = []
    labels: list[str] = []
    for headers, rows in loops:
        for sym_tag in (
            "_space_group_symop_operation_xyz",
            "_symmetry_equiv_pos_as_xyz",
        ):
            if sym_tag in headers:
                col_idx = headers.index(sym_tag)
                symops.extend(row[col_idx] for row in rows)
        if not (
            "_atom_site_fract_x" in headers
            and "_atom_site_fract_y" in headers
            and "_atom_site_fract_z" in headers
        ):
            continue
        ix = headers.index("_atom_site_fract_x")
        iy = headers.index("_atom_site_fract_y")
        iz = headers.index("_atom_site_fract_z")
        it = (
            headers.index("_atom_site_type_symbol")
            if "_atom_site_type_symbol" in headers
            else None
        )
        il = (
            headers.index("_atom_site_label")
            if "_atom_site_label" in headers
            else None
        )
        for row in rows:
            raw_sym = row[it] if it is not None else row[il or 0]
            syms.append(_clean_element_symbol(raw_sym))
            labels.append(row[il] if il is not None else raw_sym)
            frac_list.append(
                [
                    _strip_uncertainty(row[ix]),
                    _strip_uncertainty(row[iy]),
                    _strip_uncertainty(row[iz]),
                ]
            )
    return symops, frac_list, syms, labels


def read_cif(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads a Crystallographic Information File (CIF) into ``AtomsData``."""
    p = pathlib.Path(path)
    tokens = _tokenize_cif(p.read_text(encoding="utf-8"))
    scalars, loops = _extract_loops_and_scalars(tokens)
    try:
        a = _strip_uncertainty(scalars["_cell_length_a"])
        b = _strip_uncertainty(scalars["_cell_length_b"])
        c = _strip_uncertainty(scalars["_cell_length_c"])
        alpha = _strip_uncertainty(scalars.get("_cell_angle_alpha", "90"))
        beta = _strip_uncertainty(scalars.get("_cell_angle_beta", "90"))
        gamma = _strip_uncertainty(scalars.get("_cell_angle_gamma", "90"))
    except (KeyError, ValueError) as exc:
        raise exceptions.ParseError(
            f"Missing or invalid unit-cell tags in CIF: {p}"
        ) from exc

    symops, frac_list, syms, labels = _collect_cif_sites_and_symops(loops)

    if not frac_list:
        raise exceptions.ParseError(
            f"No _atom_site_fract_* records in CIF: {p}"
        )

    frac_arr = np.array(frac_list, dtype=np.float64)
    exp_frac, exp_syms, exp_labels = _apply_symops(
        frac_arr, syms, labels, symops
    )
    sg_name = scalars.get("_space_group_name_h-m_alt", "")
    return structure.AtomsData.from_cell_parameters(
        symbols=exp_syms,
        frac_positions=exp_frac,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        pbc=(True, True, True),
        atom_labels=exp_labels,
        metadata={"source": str(p), "space_group": sg_name},
    )


registry.register_format(
    registry.FormatSpec(
        name="cif",
        extensions=(".cif",),
        filenames=(),
        reader=read_cif,
        description="Crystallographic Information File (CIF)",
    )
)
