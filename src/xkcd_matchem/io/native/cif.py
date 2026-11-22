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
import warnings

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.data import spacegroups
from xkcd_matchem.io import registry
from xkcd_matchem.io.native import _limits


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


def _split_cif_data_blocks(
    tokens: list[str],
) -> list[tuple[str, list[str]]]:
    """Partitions CIF tokens into named ``data_<name>`` blocks."""
    blocks: list[tuple[str, list[str]]] = []
    cur_name = ""
    cur_tokens: list[str] = []
    for tok in tokens:
        if tok.lower().startswith("data_"):
            if cur_tokens:
                blocks.append((cur_name, cur_tokens))
            cur_name = tok[5:]
            cur_tokens = []
        else:
            cur_tokens.append(tok)
    if cur_tokens or not blocks:
        blocks.append((cur_name, cur_tokens))
    return blocks


def _parse_cif_symops(symops: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parses CIF affine symmetry strings into rotation/translation pairs."""
    parsed_ops: list[tuple[np.ndarray, np.ndarray]] = []
    for op_str in symops:
        parsed = spacegroups.parse_symop_string(op_str)
        if parsed is not None:
            parsed_ops.append(parsed)
    return parsed_ops


def _apply_symops(
    frac_sites: np.ndarray,
    site_symbols: list[str],
    site_labels: list[str],
    parsed_ops: list[tuple[np.ndarray, np.ndarray]],
    cell: np.ndarray,
    site_occupancies: list[float] | None = None,
    source: str | pathlib.Path = "",
) -> tuple[np.ndarray, list[str], list[str], list[float] | None]:
    """Expands asymmetric unit sites using CIF symmetry operations."""
    if not parsed_ops:
        return frac_sites, site_symbols, site_labels, site_occupancies

    atom_limit = _limits.max_atoms()
    occs_in = (
        site_occupancies
        if site_occupancies is not None
        else [1.0] * len(site_symbols)
    )
    expanded_frac: list[np.ndarray] = []
    expanded_syms: list[str] = []
    expanded_lbls: list[str] = []
    expanded_occs: list[float] = []
    by_sym_frac: dict[str, list[np.ndarray]] = {}

    def _is_duplicate(cand: np.ndarray, sym: str) -> bool:
        bucket = by_sym_frac.get(sym)
        if not bucket:
            return False
        existing_arr = np.vstack(bucket)
        df = np.mod(cand[None, :] - existing_arr + 0.5, 1.0) - 0.5
        dists_sq = np.einsum("ij,ij->i", df @ cell, df @ cell)
        return bool(np.any(dists_sq < 1e-4))

    for pos, sym, lbl, occ in zip(
        frac_sites, site_symbols, site_labels, occs_in, strict=True
    ):
        wrapped = np.mod(pos, 1.0)
        wrapped = np.where(
            np.isclose(wrapped, 1.0, rtol=0.0, atol=1e-5), 0.0, wrapped
        )
        if not _is_duplicate(wrapped, sym):
            if len(expanded_frac) >= atom_limit:
                _limits.check_atom_count(len(expanded_frac) + 1, source=source)
            expanded_frac.append(wrapped)
            by_sym_frac.setdefault(sym, []).append(wrapped)
            expanded_syms.append(sym)
            expanded_lbls.append(lbl)
            expanded_occs.append(occ)

    for pos, sym, lbl, occ in zip(
        frac_sites, site_symbols, site_labels, occs_in, strict=True
    ):
        for rot, trans in parsed_ops:
            cand = np.mod(rot @ pos + trans, 1.0)
            cand = np.where(
                np.isclose(cand, 1.0, rtol=0.0, atol=1e-5), 0.0, cand
            )
            if not _is_duplicate(cand, sym):
                if len(expanded_frac) >= atom_limit:
                    _limits.check_atom_count(
                        len(expanded_frac) + 1, source=source
                    )
                expanded_frac.append(cand)
                by_sym_frac.setdefault(sym, []).append(cand)
                expanded_syms.append(sym)
                expanded_lbls.append(lbl)
                expanded_occs.append(occ)
    return (
        np.vstack(expanded_frac),
        expanded_syms,
        expanded_lbls,
        expanded_occs if site_occupancies is not None else None,
    )


def _min_new_site_distance(
    expanded_frac: np.ndarray,
    n_input: int,
    cell: np.ndarray,
) -> float:
    """Returns min periodic Cartesian distance involving generated sites."""
    n_tot = len(expanded_frac)
    if n_tot <= 1 or n_tot <= n_input:
        return float("inf")
    new_frac = expanded_frac[n_input:]
    raw_df = new_frac[:, None, :] - expanded_frac[None, :, :]
    df = np.mod(raw_df + 0.5, 1.0) - 0.5
    cart_df = df.reshape(-1, 3) @ cell
    dists = np.linalg.norm(cart_df, axis=1).reshape(len(new_frac), n_tot)
    for k in range(len(new_frac)):
        dists[k, n_input + k] = float("inf")
    return float(np.min(dists))


def _expand_sites_with_table(
    frac_arr: np.ndarray,
    syms: list[str],
    labels: list[str],
    occs: list[float] | None,
    scalars: dict[str, str],
    cell: np.ndarray,
    source: pathlib.Path,
) -> tuple[np.ndarray, list[str], list[str], list[float] | None]:
    """Expands CIF sites via implicit space-group lookup and origin checks."""
    sg_name = (
        scalars.get("_space_group_name_h-m_alt")
        or scalars.get("_symmetry_space_group_name_h-m")
        or ""
    ).strip()
    raw_num = (
        scalars.get("_space_group_it_number")
        or scalars.get("_symmetry_int_tables_number")
        or ""
    ).strip()
    sg_num: int | None = int(raw_num) if raw_num.isdigit() else None
    raw_input = (frac_arr, syms, labels, occs)
    if (
        (sg_num is None and not sg_name)
        or sg_num == 1
        or spacegroups.normalize_hm_symbol(sg_name) == "p1"
    ):
        return raw_input

    candidates = spacegroups.lookup_spacegroup_candidates(
        sg_name=sg_name, sg_number=sg_num
    )
    if not candidates:
        warnings.warn(
            f"CIF {source.name} declares space group {sg_name or sg_num!r} "
            "without symmetry operations; returning asymmetric unit only. Use "
            "pymatgen/ASE for full non-standard symmetry expansion.",
            UserWarning,
            stacklevel=3,
        )
        return raw_input

    evaluated: list[
        tuple[
            float,
            tuple[np.ndarray, list[str], list[str], list[float] | None],
        ]
    ] = []
    for cand_ops in candidates:
        exp = _apply_symops(
            frac_arr,
            syms,
            labels,
            cand_ops,
            cell,
            site_occupancies=occs,
            source=source,
        )
        if len(exp[0]) == len(frac_arr):
            return exp
        d_min = _min_new_site_distance(exp[0], len(frac_arr), cell)
        evaluated.append((d_min, exp))

    valid = [pair for pair in evaluated if pair[0] >= 0.85]
    if valid:
        return min(valid, key=lambda pair: len(pair[1][0]))[1]
    best_d, best_exp = max(evaluated, key=lambda pair: pair[0])
    if best_d >= 0.75:
        return best_exp
    warnings.warn(
        f"CIF {source.name} implicit space-group expansion for "
        f"{sg_name or sg_num!r} produced overlapping sites "
        f"(d_min={best_d:.2f} A); returning input sites unexpanded.",
        UserWarning,
        stacklevel=3,
    )
    return raw_input


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


def _parse_cif_occ(raw_occ: str) -> float:
    """Parses optional CIF site occupancy string."""
    cleaned = raw_occ.strip()
    if cleaned in ("?", "."):
        return 1.0
    return float(_strip_uncertainty(cleaned))


def _collect_cif_sites_and_symops(
    loops: list[tuple[list[str], list[list[str]]]],
    source: str | pathlib.Path = "",
) -> tuple[
    list[str], list[list[float]], list[str], list[str], list[float] | None
]:
    """Extracts symmetry operations, fractional sites, and occupancies."""
    symops: list[str] = []
    frac_list: list[list[float]] = []
    syms: list[str] = []
    labels: list[str] = []
    occs: list[float] = []
    has_occ = False
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
        _limits.check_atom_count(len(frac_list) + len(rows), source=source)
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
        io_occ = (
            headers.index("_atom_site_occupancy")
            if "_atom_site_occupancy" in headers
            else None
        )
        if io_occ is not None:
            has_occ = True
        for row in rows:
            raw_sym = row[it] if it is not None else row[il or 0]
            syms.append(_clean_element_symbol(raw_sym))
            labels.append(row[il] if il is not None else raw_sym)
            occs.append(
                _parse_cif_occ(row[io_occ]) if io_occ is not None else 1.0
            )
            frac_list.append(
                [
                    _strip_uncertainty(row[ix]),
                    _strip_uncertainty(row[iy]),
                    _strip_uncertainty(row[iz]),
                ]
            )
    return symops, frac_list, syms, labels, occs if has_occ else None


def _parse_cif_block(
    block_name: str,
    block_tokens: list[str],
    source: pathlib.Path,
) -> structure.AtomsData:
    """Parses a single CIF data block into ``AtomsData``."""
    scalars, loops = _extract_loops_and_scalars(block_tokens)
    try:
        a = _strip_uncertainty(scalars["_cell_length_a"])
        b = _strip_uncertainty(scalars["_cell_length_b"])
        c = _strip_uncertainty(scalars["_cell_length_c"])
        alpha = _strip_uncertainty(scalars.get("_cell_angle_alpha", "90"))
        beta = _strip_uncertainty(scalars.get("_cell_angle_beta", "90"))
        gamma = _strip_uncertainty(scalars.get("_cell_angle_gamma", "90"))
    except (KeyError, ValueError) as exc:
        raise exceptions.ParseError(
            f"Missing or invalid unit-cell tags in CIF: {source}"
        ) from exc

    symops, frac_list, syms, labels, occs = _collect_cif_sites_and_symops(
        loops, source=source
    )
    if not frac_list:
        raise exceptions.ParseError(
            f"No _atom_site_fract_* records in CIF: {source}"
        )
    _limits.check_atom_count(len(frac_list), source=source)

    cell = geometry.cell_vectors_from_parameters(a, b, c, alpha, beta, gamma)
    frac_arr = np.array(frac_list, dtype=np.float64)
    parsed_ops = _parse_cif_symops(symops) if symops else []
    if parsed_ops:
        exp_frac, exp_syms, exp_labels, exp_occs = _apply_symops(
            frac_arr,
            syms,
            labels,
            parsed_ops,
            cell,
            site_occupancies=occs,
            source=source,
        )
    else:
        exp_frac, exp_syms, exp_labels, exp_occs = _expand_sites_with_table(
            frac_arr, syms, labels, occs, scalars, cell, source
        )
    sg_name = (
        scalars.get("_space_group_name_h-m_alt")
        or scalars.get("_symmetry_space_group_name_h-m")
        or ""
    )
    metadata: dict[str, object] = {
        "source": str(source),
        "space_group": sg_name,
        "data_block": block_name,
    }
    occ_arr = (
        np.asarray(exp_occs, dtype=np.float64)
        if exp_occs is not None
        else None
    )
    return structure.AtomsData.from_fractional(
        symbols=exp_syms,
        frac_positions=exp_frac,
        cell=cell,
        pbc=(True, True, True),
        atom_labels=exp_labels,
        occupancies=occ_arr,
        metadata=metadata,
    )


def read_cif_all(path: str | pathlib.Path) -> list[structure.AtomsData]:
    """Reads all structural ``data_`` blocks from a multi-structure CIF."""
    p = pathlib.Path(path)
    _limits.check_file_size(p)
    tokens = _tokenize_cif(p.read_text(encoding="utf-8"))
    blocks = _split_cif_data_blocks(tokens)
    structures: list[structure.AtomsData] = []
    for block_name, block_tokens in blocks:
        _, loops = _extract_loops_and_scalars(block_tokens)
        if any("_atom_site_fract_x" in hdrs for hdrs, _ in loops):
            _limits.check_frame_count(len(structures) + 1, source=p)
            structures.append(_parse_cif_block(block_name, block_tokens, p))
    if not structures:
        raise exceptions.ParseError(
            f"No _atom_site_fract_* records in CIF: {p}"
        )
    return structures


def read_cif(
    path: str | pathlib.Path,
    block: int | str | None = None,
) -> structure.AtomsData:
    """Reads a Crystallographic Information File (CIF) into ``AtomsData``.

    Args:
      path: Filesystem path to the ``.cif`` file.
      block: Optional 0-based structural block index or ``data_<name>``
        identifier when reading multi-block CIFs (defaults to the first block
        containing atomic coordinates).

    Returns:
      Parsed ``AtomsData`` crystal structure.
    """
    p = pathlib.Path(path)
    _limits.check_file_size(p)
    tokens = _tokenize_cif(p.read_text(encoding="utf-8"))
    blocks = _split_cif_data_blocks(tokens)
    struct_blocks: list[tuple[str, list[str]]] = []
    for block_name, block_tokens in blocks:
        _, loops = _extract_loops_and_scalars(block_tokens)
        if any("_atom_site_fract_x" in hdrs for hdrs, _ in loops):
            struct_blocks.append((block_name, block_tokens))
    if not struct_blocks:
        raise exceptions.ParseError(
            f"No _atom_site_fract_* records in CIF: {p}"
        )
    if block is None:
        return _parse_cif_block(*struct_blocks[0], p)
    if isinstance(block, int):
        if not 0 <= block < len(struct_blocks):
            raise exceptions.ParseError(
                f"CIF block index {block} out of range "
                f"[0, {len(struct_blocks)})."
            )
        return _parse_cif_block(*struct_blocks[block], p)
    key = str(block).strip().removeprefix("data_").lower()
    for b_name, b_tokens in struct_blocks:
        if b_name.lower() == key:
            return _parse_cif_block(b_name, b_tokens, p)
    raise exceptions.ParseError(f"CIF data block {block!r} not found in {p}.")


registry.register_format(
    registry.FormatSpec(
        name="cif",
        extensions=(".cif",),
        filenames=(),
        reader=read_cif,
        description="Crystallographic Information File (CIF)",
    )
)
