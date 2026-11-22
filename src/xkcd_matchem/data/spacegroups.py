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
"""Zero-dependency crystallographic space-group symmetry lookup."""

from __future__ import annotations

from collections.abc import Sequence
import functools
import re

import numpy as np


_CENTERING_VECTORS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "P": ((0.0, 0.0, 0.0),),
    "A": ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5)),
    "B": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.5)),
    "C": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0)),
    "I": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    "F": (
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ),
    "R": (
        (0.0, 0.0, 0.0),
        (2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
    ),
}


# Generator specs: ITA number -> (expected_order, centering, generators).
_SPACE_GROUP_SPECS: dict[int, tuple[int, str, tuple[str, ...]]] = {
    1: (1, "P", ("x,y,z",)),
    2: (2, "P", ("-x,-y,-z",)),
    4: (2, "P", ("-x,y+1/2,-z",)),
    5: (4, "C", ("-x,y,-z",)),
    12: (8, "C", ("-x,y,-z", "-x,-y,-z")),
    14: (4, "P", ("-x,y+1/2,-z+1/2", "-x,-y,-z")),
    15: (8, "C", ("-x,y,-z+1/2", "-x,-y,-z")),
    19: (4, "P", ("-x+1/2,-y,z+1/2", "-x,y+1/2,-z+1/2")),
    26: (4, "P", ("-x,y,z", "x,-y,z+1/2")),
    29: (4, "P", ("-x,-y,z+1/2", "x+1/2,-y,z")),
    33: (4, "P", ("-x,-y,z+1/2", "-x+1/2,y+1/2,z+1/2")),
    36: (8, "C", ("-x,-y,z+1/2", "-x,y,z")),
    47: (8, "P", ("-x,-y,z", "-x,y,-z", "-x,-y,-z")),
    55: (8, "P", ("-x,-y,z", "-x+1/2,y+1/2,-z", "-x,-y,-z")),
    57: (8, "P", ("-x,-y,z+1/2", "-x,y+1/2,-z+1/2", "-x,-y,-z")),
    58: (8, "P", ("-x,-y,z", "-x+1/2,y+1/2,-z+1/2", "-x,-y,-z")),
    59: (8, "P", ("-x+1/2,-y+1/2,z", "-x+1/2,y,-z", "-x,-y,-z")),
    60: (8, "P", ("-x,-y,z+1/2", "-x+1/2,y+1/2,-z", "-x,-y,-z")),
    61: (8, "P", ("-x+1/2,-y,z+1/2", "-x,y+1/2,-z+1/2", "-x,-y,-z")),
    62: (8, "P", ("-x+1/2,-y,z+1/2", "-x,y+1/2,-z", "-x,-y,-z")),
    63: (16, "C", ("-x,-y,z+1/2", "-x,y,-z+1/2", "-x,-y,-z")),
    64: (16, "C", ("-x,-y,z+1/2", "-x,y+1/2,-z+1/2", "-x,-y,-z")),
    65: (16, "C", ("-x,-y,z", "-x,y,-z", "-x,-y,-z")),
    71: (16, "I", ("-x,-y,z", "-x,y,-z", "-x,-y,-z")),
    74: (16, "I", ("-x,-y+1/2,z", "-x,y+1/2,-z", "-x,-y,-z")),
    82: (8, "I", ("-y,x,-z",)),
    87: (16, "I", ("-y,x,z", "-x,-y,-z")),
    88: (16, "I", ("-y+3/4,x+1/4,z+1/4", "-x,-y,-z")),
    99: (8, "P", ("-y,x,z", "-x,y,z")),
    123: (16, "P", ("-y,x,z", "-x,y,-z", "-x,-y,-z")),
    129: (16, "P", ("-y+1/2,x,z", "-x+1/2,y,-z", "-x,-y,-z")),
    136: (16, "P", ("-y+1/2,x+1/2,z+1/2", "-x+1/2,y+1/2,-z+1/2", "-x,-y,-z")),
    139: (32, "I", ("-y,x,z", "-x,y,-z", "-x,-y,-z")),
    140: (32, "I", ("-y,x,z", "-x,y,-z+1/2", "-x,-y,-z")),
    141: (32, "I", ("-y+3/4,x+1/4,z+1/4", "-x,y,z", "-x,-y,-z")),
    147: (6, "P", ("-y,x-y,z", "-x,-y,-z")),
    148: (18, "R", ("-y,x-y,z", "-x,-y,-z")),
    160: (18, "R", ("-y,x-y,z", "-y,-x,z")),
    161: (18, "R", ("-y,x-y,z", "-y,-x,z+1/2")),
    164: (12, "P", ("-y,x-y,z", "-y,-x,-z", "-x,-y,-z")),
    166: (36, "R", ("-y,x-y,z", "y,x,-z", "-x,-y,-z")),
    167: (36, "R", ("-y,x-y,z", "y,x,-z+1/2", "-x,-y,-z")),
    176: (12, "P", ("-y,x-y,z", "-x,-y,z+1/2", "-x,-y,-z")),
    186: (12, "P", ("-y,x-y,z", "-x,-y,z+1/2", "-y,-x,z")),
    187: (12, "P", ("-y,x-y,z", "x,y,-z", "-y,-x,z")),
    191: (24, "P", ("-y,x-y,z", "-x,-y,z", "-y,-x,-z", "-x,-y,-z")),
    193: (24, "P", ("-y,x-y,z", "-x,-y,z+1/2", "-y,-x,-z+1/2", "-x,-y,-z")),
    194: (24, "P", ("-y,x-y,z", "-x,-y,z+1/2", "-y,-x,-z", "-x,-y,-z")),
    198: (12, "P", ("-x+1/2,-y,z+1/2", "-x,y+1/2,-z+1/2", "z,x,y")),
    204: (48, "I", ("-x,-y,z", "z,x,y", "-x,-y,-z")),
    205: (24, "P", ("-x+1/2,-y,z+1/2", "-x,y+1/2,-z+1/2", "z,x,y", "-x,-y,-z")),
    206: (48, "I", ("-x+1/2,-y,z+1/2", "-x,y+1/2,-z+1/2", "z,x,y", "-x,-y,-z")),
    216: (96, "F", ("-x,-y,z", "z,x,y", "y,x,z")),
    217: (48, "I", ("-x,-y,z", "z,x,y", "y,x,z")),
    221: (48, "P", ("-x,-y,z", "z,x,y", "y,x,z", "-x,-y,-z")),
    223: (48, "P", ("-x,-y,z", "z,x,y", "y+1/2,x+1/2,z+1/2", "-x,-y,-z")),
    225: (192, "F", ("-x,-y,z", "z,x,y", "y,x,z", "-x,-y,-z")),
    226: (192, "F", ("-x,-y,z", "z,x,y", "y+1/2,x+1/2,z+1/2", "-x,-y,-z")),
    227: (
        192,
        "F",
        (
            "-x+3/4,-y+3/4,z",
            "z,x,y",
            "-y+3/4,x,-z+3/4",
            "-x,-y,-z",
        ),
    ),
    229: (96, "I", ("-x,-y,z", "z,x,y", "y,x,z", "-x,-y,-z")),
    230: (
        96,
        "I",
        (
            "-x+1/2,-y,z+1/2",
            "-x,y+1/2,-z+1/2",
            "z,x,y",
            "y+1/4,x+1/4,z+1/4",
            "-x,-y,-z",
        ),
    ),
}

# Origin Choice 1 generators for two-origin ITA groups whose standard setting
# in _SPACE_GROUP_SPECS uses Origin Choice 2 (inversion at origin).
_ORIGIN_1_SPECS: dict[int, tuple[int, str, tuple[str, ...]]] = {
    227: (
        192,
        "F",
        ("-x,-y,z", "z,x,y", "-y,x,-z", "-x+1/4,-y+1/4,-z+1/4"),
    ),
}

_HM_TO_IT: dict[str, int] = {
    "p1": 1,
    "p-1": 2,
    "p21": 4,
    "c2": 5,
    "c2/m": 12,
    "p21/c": 14,
    "p21/n": 14,
    "c2/c": 15,
    "p212121": 19,
    "pmc21": 26,
    "pca21": 29,
    "pna21": 33,
    "cmc21": 36,
    "pmmm": 47,
    "pbam": 55,
    "pbcm": 57,
    "pnnm": 58,
    "pmmn": 59,
    "pbcn": 60,
    "pbca": 61,
    "pnma": 62,
    "pbnm": 62,
    "cmcm": 63,
    "cmce": 64,
    "cmca": 64,
    "cmmm": 65,
    "immm": 71,
    "imma": 74,
    "i-4": 82,
    "i4/m": 87,
    "i41/a": 88,
    "p4mm": 99,
    "p4/mmm": 123,
    "p4/nmm": 129,
    "p42/mnm": 136,
    "i4/mmm": 139,
    "i4/mcm": 140,
    "i41/amd": 141,
    "p-3": 147,
    "r-3": 148,
    "r3m": 160,
    "r3c": 161,
    "p-3m1": 164,
    "r-3m": 166,
    "r-3c": 167,
    "p63/m": 176,
    "p63mc": 186,
    "p-6m2": 187,
    "p6/mmm": 191,
    "p63/mcm": 193,
    "p63/mmc": 194,
    "p213": 198,
    "im-3": 204,
    "pa-3": 205,
    "ia-3": 206,
    "f-43m": 216,
    "i-43m": 217,
    "pm-3m": 221,
    "pm-3n": 223,
    "fm-3m": 225,
    "fm-3c": 226,
    "fd-3m": 227,
    "im-3m": 229,
    "ia-3d": 230,
}


def parse_affine_term(expr: str) -> tuple[np.ndarray, float]:
    """Parses one CIF affine axis expression such as ``'-x+1/2'``."""
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
            denom = float(den_s)
            if abs(denom) < 1e-12:
                raise ValueError(f"Zero denominator in symop term {expr!r}")
            shift += sign * (float(num_s) / denom)
        else:
            shift += sign * float(body)
    return coeff, shift


def parse_symop_string(op_str: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Parses a 3-axis symmetry operation string into ``(rot, trans)``."""
    parts = op_str.split(",")
    if len(parts) != 3:
        return None
    rot = np.zeros((3, 3), dtype=np.float64)
    trans = np.zeros(3, dtype=np.float64)
    try:
        for row_idx, expr in enumerate(parts):
            c, s = parse_affine_term(expr)
            rot[row_idx] = c
            trans[row_idx] = s
    except ValueError:
        return None
    return rot, np.mod(trans, 1.0)


def _op_signature(rot: np.ndarray, trans: np.ndarray) -> tuple[int, ...]:
    """Returns exact quantized hash key for a crystallographic symop."""
    r_int = tuple(int(round(v)) for v in rot.ravel())
    t_wrapped = np.mod(trans, 1.0)
    t_wrapped = np.where(
        np.isclose(t_wrapped, 1.0, rtol=0.0, atol=1e-6), 0.0, t_wrapped
    )
    t_int = tuple(int(round(float(v) * 24.0)) % 24 for v in t_wrapped)
    return r_int + t_int


def expand_generators(
    centering: str,
    gen_ops: Sequence[str],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Closes a space-group generator set under composition and centering."""
    ops: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()

    def _add(rot: np.ndarray, trans: np.ndarray) -> None:
        t_mod = np.mod(trans, 1.0)
        t_mod = np.where(
            np.isclose(t_mod, 1.0, rtol=0.0, atol=1e-6), 0.0, t_mod
        )
        sig = _op_signature(rot, t_mod)
        if sig not in seen and len(ops) < 192:
            seen.add(sig)
            ops.append((rot, t_mod))

    _add(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    for op_s in gen_ops:
        parsed = parse_symop_string(op_s)
        if parsed is not None:
            _add(*parsed)
    for cvec in _CENTERING_VECTORS.get(centering.upper(), ()):
        _add(np.eye(3, dtype=np.float64), np.asarray(cvec, dtype=np.float64))

    idx = 0
    while idx < len(ops) and len(ops) < 192:
        r_a, t_a = ops[idx]
        cur_snapshot = list(ops)
        for r_b, t_b in cur_snapshot:
            _add(r_a @ r_b, r_a @ t_b + t_a)
            _add(r_b @ r_a, r_b @ t_a + t_b)
        idx += 1
    return ops


def _extract_origin_choice(sg_name: str) -> str | None:
    """Extracts explicit ':1' or ':2' origin-choice suffix when present."""
    cleaned = re.sub(r"[\s_]+", "", sg_name.strip().lower())
    if ":" not in cleaned:
        return None
    suffix = cleaned.split(":", maxsplit=1)[1]
    return suffix if suffix in ("1", "2") else None


def normalize_hm_symbol(sg_name: str) -> str:
    """Normalizes a Hermann-Mauguin symbol for lookup."""
    cleaned = re.sub(r"[\s_]+", "", sg_name.strip().lower())
    if ":" in cleaned:
        cleaned = cleaned.split(":", maxsplit=1)[0]
    return cleaned


@functools.lru_cache(maxsize=256)
def _cached_group_ops(it_num: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Expands and caches symmetry operations for an IT space-group number."""
    _, centering, gens = _SPACE_GROUP_SPECS[it_num]
    return tuple(expand_generators(centering, gens))


@functools.lru_cache(maxsize=64)
def _cached_origin1_ops(
    it_num: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Expands and caches Origin Choice 1 symmetry operations."""
    _, centering, gens = _ORIGIN_1_SPECS[it_num]
    return tuple(expand_generators(centering, gens))


def _resolve_it_number(
    sg_name: str = "",
    sg_number: int | None = None,
) -> int | None:
    """Resolves ITA number from space-group number or H-M symbol."""
    if sg_number is not None and int(sg_number) in _SPACE_GROUP_SPECS:
        return int(sg_number)
    if sg_name.strip():
        return _HM_TO_IT.get(normalize_hm_symbol(sg_name))
    return None


def _copy_ops(
    ops: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns fresh mutable array copies of cached symmetry operations."""
    return [(r.copy(), t.copy()) for r, t in ops]


def lookup_spacegroup_symops(
    sg_name: str = "",
    sg_number: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Resolves standard symmetry operations from H-M symbol or IT number.

    Args:
      sg_name: Hermann-Mauguin space-group string (e.g. ``"Fm-3m"``).
      sg_number: International Tables space-group number (``1..230``).

    Returns:
      List of ``(rot_3x3, trans_3)`` pairs, or ``None`` if not in table.
    """
    it_num = _resolve_it_number(sg_name=sg_name, sg_number=sg_number)
    if it_num is None:
        return None
    if _extract_origin_choice(sg_name) == "1" and it_num in _ORIGIN_1_SPECS:
        return _copy_ops(_cached_origin1_ops(it_num))
    return _copy_ops(_cached_group_ops(it_num))


def lookup_spacegroup_candidates(
    sg_name: str = "",
    sg_number: int | None = None,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Returns candidate origin-choice symmetry tables for a space group."""
    it_num = _resolve_it_number(sg_name=sg_name, sg_number=sg_number)
    if it_num is None:
        return []
    origin = _extract_origin_choice(sg_name)
    if origin == "1" and it_num in _ORIGIN_1_SPECS:
        return [_copy_ops(_cached_origin1_ops(it_num))]
    if origin == "2" or it_num not in _ORIGIN_1_SPECS:
        return [_copy_ops(_cached_group_ops(it_num))]
    return [
        _copy_ops(_cached_group_ops(it_num)),
        _copy_ops(_cached_origin1_ops(it_num)),
    ]
