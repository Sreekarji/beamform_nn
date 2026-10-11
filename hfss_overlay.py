"""
Overlay exported Ansys HFSS far-field patterns with the analytical MVDR
pattern and the saved null-aligned neural-network prediction.

The default (and simplest) workflow expects one already-combined HFSS report per
case:

    hfss_exports/case_01.csv
    ...
    hfss_exports/case_05.csv

HFSS report exports are commonly CSV, TAB, or plain text.  The parser accepts
metadata before a table and common columns such as Theta, Phi,
dB(GainTotal), dB(RealizedGainTotal), and dB(DirectivityTotal).

For a coherent embedded-element workflow, use --mode embedded and export one
file per port with complex Etheta or Ephi data.  The script then combines the
complex fields; it deliberately refuses to combine dB-only embedded data.

The Python steering convention is
    a_n(theta) = exp(1j * n * pi * sin(theta))
and the analytical response is w^H a(theta).  HFSS source conventions can
reverse the sign or conjugate the source vector, so calibrate that choice with
a broadside/single-element case before interpreting residuals.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N = 8
SIGMA_N = 1.0
SIGMA_S = 100.0
SIGMA_I = 1000.0
DEFAULT_FLOOR_DB = -120.0


@dataclass(frozen=True)
class Case:
    case_id: int
    theta_s: float
    theta_i: float


DEFAULT_CASES = (
    Case(1, 0.0, 30.0),
    Case(2, 30.0, -40.0),
    Case(3, -25.0, 45.0),
    Case(4, 5.0, -55.0),
    Case(5, 58.0, 55.0),
)


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]
    path: Path


@dataclass
class Curve:
    angle: np.ndarray
    db: np.ndarray
    source: str


# ---------------------------------------------------------------------------
# Analytical reference


def steering(theta_deg: Sequence[float] | float, n: int = N) -> np.ndarray:
    """Return steering vectors with shape (..., n)."""
    theta = np.asarray(theta_deg, dtype=float)
    elements = np.arange(n, dtype=float)
    return np.exp(1j * np.pi * np.sin(np.deg2rad(theta))[..., None] * elements)


def mvdr_weights(
    theta_s: float,
    theta_i: float,
    n: int = N,
    sigma_s: float = SIGMA_S,
    sigma_i: float = SIGMA_I,
    sigma_n: float = SIGMA_N,
) -> np.ndarray:
    """Compute the float64 two-source MVDR weights without a full inverse."""
    a_s = steering(theta_s, n)
    a_i = steering(theta_i, n)
    G = np.stack([a_s, a_i], axis=1)
    M = sigma_n * np.diag([1.0 / sigma_s, 1.0 / sigma_i]) + G.conj().T @ G
    v = a_s - G @ np.linalg.solve(M, G.conj().T @ a_s)
    denominator = a_s.conj() @ v
    if abs(denominator) < 1e-14:
        raise ValueError(
            f"ill-conditioned MVDR normalization for ({theta_s}, {theta_i})"
        )
    return v / denominator


def response_pattern_db(weights: np.ndarray, angles: np.ndarray, floor_db: float) -> np.ndarray:
    """Return a peak-normalized analytical power pattern in dB."""
    a = steering(angles, len(weights))
    response = a @ np.conj(weights)
    power = np.maximum(np.abs(response) ** 2, 10.0 ** (floor_db / 10.0))
    db = 10.0 * np.log10(power)
    return db - np.max(db)


class NeuralPredictor:
    """Lazy PyTorch wrapper for a saved nullaligned BeamNet checkpoint."""

    def __init__(self, checkpoint: Path, n: int = N):
        try:
            import torch
            from models import BeamNet
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PyTorch is required to load the NN checkpoint; use --no-nn "
                "for an HFSS-versus-MVDR-only comparison."
            ) from exc

        self.torch = torch
        self.model = BeamNet(N=n, mode="nullaligned")
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions before the weights_only argument.
            state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    def weights(self, theta_s: float, theta_i: float) -> np.ndarray:
        x = self.torch.tensor([[theta_s, theta_i]], dtype=self.torch.float32)
        with self.torch.no_grad():
            w = self.model(x)[0].cpu().numpy()
        return np.asarray(w, dtype=np.complex128)


# ---------------------------------------------------------------------------
# HFSS table parsing


def _normalise(text: str) -> str:
    text = text.replace("\ufeff", "").strip().strip('"').strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _number(text: str) -> float:
    """Parse a numeric HFSS cell, including Fortran D exponents."""
    value = text.strip().strip('"').replace("D", "E").replace("d", "e")
    # HFSS sometimes appends a unit to a header, but data cells should only
    # contain a number.  Removing a trailing degree symbol is harmless.
    value = value.replace("°", "").strip()
    return float(value)


def _delimiter(lines: Iterable[str]) -> str:
    sample = "\n".join(line for line in lines if line.strip())[:8192]
    if "\t" in sample:
        return "\t"
    if ";" in sample and sample.count(";") >= sample.count(","):
        return ";"
    if "," in sample:
        return ","
    # This fallback is mainly for simple whitespace-delimited exports.
    return " "


def _split(line: str, delimiter: str) -> list[str]:
    if delimiter == " ":
        return re.split(r"\s+", line.strip())
    return next(csv.reader([line], delimiter=delimiter))


def read_table(path: Path) -> Table:
    """Find and read the first HFSS-style header/data table in *path*."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    delimiter = _delimiter(lines)
    header_index: Optional[int] = None
    headers: list[str] = []

    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith(("#", "%", "!")):
            continue
        cells = _split(line, delimiter)
        normal = [_normalise(cell) for cell in cells]
        has_angle = any(
            token in {"theta", "thetadeg", "angle", "angledeg", "phi"}
            or token.startswith("theta")
            for token in normal
        )
        has_value = any(
            any(key in token for key in ("gain", "directivity", "realized", "etheta", "ephi", "magnitude", "phase", "db"))
            for token in normal
        )
        if has_angle and has_value:
            header_index = index
            headers = [cell.strip().strip('"') for cell in cells]
            break

    if header_index is None:
        raise ValueError(
            f"Could not find an HFSS table header in {path}. Expected a Theta "
            "column and a dB/gain/field column."
        )

    rows: list[list[str]] = []
    width = len(headers)
    for line in lines[header_index + 1 :]:
        if not line.strip() or line.lstrip().startswith(("#", "%", "!")):
            continue
        cells = _split(line, delimiter)
        if len(cells) < width:
            continue
        cells = cells[:width]
        try:
            if not any(np.isfinite(_number(cell)) for cell in cells):
                raise ValueError
        except (TypeError, ValueError):
            # Metadata/footer lines after the table.
            continue
        rows.append(cells)

    if not rows:
        raise ValueError(f"No numeric rows found after the header in {path}")
    return Table(headers, rows, path)


def _find_column(
    headers: Sequence[str],
    explicit: Optional[str],
    predicates: Sequence,
    description: str,
) -> int:
    normal = [_normalise(header) for header in headers]
    if explicit:
        target = _normalise(explicit)
        for index, token in enumerate(normal):
            if token == target or target in token:
                return index
        raise ValueError(f"Column {explicit!r} was not found in {headers}")
    for predicate in predicates:
        for index, token in enumerate(normal):
            if predicate(token):
                return index
    raise ValueError(f"Could not identify {description} column in {headers}")


def _angle_column(headers: Sequence[str], explicit: Optional[str]) -> int:
    return _find_column(
        headers,
        explicit,
        (
            lambda t: t in {"theta", "thetadeg", "angle", "angledeg"},
            lambda t: t.startswith("theta"),
            lambda t: t.startswith("angle"),
        ),
        "angle",
    )


def _phi_column(headers: Sequence[str], explicit: Optional[str]) -> Optional[int]:
    normal = [_normalise(header) for header in headers]
    if explicit:
        return _find_column(headers, explicit, (lambda t: True,), "phi")
    for index, token in enumerate(normal):
        if token in {"phi", "phideg"} or token.startswith("phi"):
            return index
    return None


def _row_numbers(table: Table, column: int) -> np.ndarray:
    values = []
    for row in table.rows:
        try:
            values.append(_number(row[column]))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def _select_and_map_angles(
    table: Table,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected HFSS row indices and Python signed angles."""
    theta_index = _angle_column(table.headers, args.theta_column)
    theta = _row_numbers(table, theta_index)
    phi_index = _phi_column(table.headers, args.phi_column)
    phi = _row_numbers(table, phi_index) if phi_index is not None else None
    row_index = np.isfinite(theta)

    if args.convention == "python":
        angle = theta + args.angle_offset
    else:
        if phi is None:
            raise ValueError(
                f"{table.path} has no Phi column; --convention hfss_spherical "
                "needs Phi to map the x-z cuts."
            )
        phi_arg = str(args.phi).strip().lower()
        if phi_arg == "both":
            select_phi = np.isclose(np.mod(phi, 360.0), 0.0, atol=args.phi_tolerance)
            select_phi |= np.isclose(np.mod(phi, 360.0), 180.0, atol=args.phi_tolerance)
        elif phi_arg == "all":
            select_phi = np.ones_like(phi, dtype=bool)
        else:
            try:
                requested = float(phi_arg)
            except ValueError as exc:
                raise ValueError("--phi must be a number, 'both', or 'all'") from exc
            select_phi = np.isclose(phi, requested, atol=args.phi_tolerance)
        row_index &= select_phi
        if not np.any(row_index):
            available = np.unique(np.round(phi[np.isfinite(phi)], 5))
            raise ValueError(
                f"No rows at requested Phi={args.phi} in {table.path}; "
                f"available values include {available[:12]}"
            )

        phi_mod = np.mod(phi, 360.0)
        # For an x-z cut, HFSS spherical Theta is unsigned.  Phi=0 maps to
        # +Python angle and Phi=180 maps to -Python angle.  For 'all', use the
        # nearest x-axis half-plane and retain only the principal [-90, 90]
        # angle.  This is an explicit convention, never an implicit guess.
        sign = np.where(np.cos(np.deg2rad(phi_mod)) >= 0.0, 1.0, -1.0)
        principal_theta = np.minimum(np.mod(theta, 360.0), 180.0 - np.mod(theta, 360.0))
        angle = sign * principal_theta + args.angle_offset

    row_index &= np.isfinite(angle)
    return np.flatnonzero(row_index), np.asarray(angle[row_index], dtype=float)


def _unique_curve(angle: np.ndarray, value_db: np.ndarray, source: str) -> Curve:
    """Sort a curve and average duplicate angles from multi-frequency/report rows."""
    finite = np.isfinite(angle) & np.isfinite(value_db)
    angle = np.asarray(angle[finite], dtype=float)
    value_db = np.asarray(value_db[finite], dtype=float)
    if angle.size == 0:
        raise ValueError(f"No finite pattern samples in {source}")
    order = np.argsort(angle)
    angle, value_db = angle[order], value_db[order]
    unique, inverse = np.unique(angle, return_inverse=True)
    sums = np.bincount(inverse, weights=value_db)
    counts = np.bincount(inverse)
    return Curve(unique, sums / counts, source)


def _pattern_column(headers: Sequence[str], explicit: Optional[str]) -> tuple[int, bool]:
    """Return (column index, is_db) for a combined HFSS pattern."""
    normal = [_normalise(header) for header in headers]
    if explicit:
        index = _find_column(headers, explicit, (lambda t: True,), "pattern")
        return index, ("db" in normal[index] or "decibel" in normal[index])

    # Prefer total gain/realized gain/directivity over component fields.
    db_predicates = (
        lambda t: "db" in t and "realizedgaintotal" in t,
        lambda t: "db" in t and "gaintotal" in t,
        lambda t: "db" in t and "directivitytotal" in t,
        lambda t: "db" in t and ("etheta" in t or "ephi" in t or "etotal" in t),
        lambda t: "db" in t and any(k in t for k in ("gain", "directivity", "field")),
    )
    linear_predicates = (
        lambda t: t in {"realizedgaintotal", "gaintotal", "directivitytotal"},
        lambda t: any(k in t for k in ("gain", "directivity")) and "db" not in t,
    )
    for predicate in db_predicates:
        for index, token in enumerate(normal):
            if predicate(token):
                return index, True
    for predicate in linear_predicates:
        for index, token in enumerate(normal):
            if predicate(token):
                return index, False
    raise ValueError(
        f"Could not identify a dB/linear pattern column in {headers}. "
        "Pass --pattern-column explicitly."
    )


def read_combined_pattern(path: Path, args: argparse.Namespace) -> Curve:
    table = read_table(path)
    indices, angle = _select_and_map_angles(table, args)
    column, is_db = _pattern_column(table.headers, args.pattern_column)
    values = _row_numbers(table, column)[indices]
    if is_db:
        db = values
    else:
        if np.any(values <= 0):
            raise ValueError(f"Linear pattern column contains non-positive values in {path}")
        db = 10.0 * np.log10(values)
    return _unique_curve(angle, db, str(path))


# ---------------------------------------------------------------------------
# Embedded complex-field parsing


def _component_columns(
    headers: Sequence[str],
    component: str,
    explicit_real: Optional[str],
    explicit_imag: Optional[str],
    explicit_mag: Optional[str],
    explicit_phase: Optional[str],
) -> tuple[str, int, int]:
    normal = [_normalise(header) for header in headers]
    comp = _normalise(component)

    def has_component(token: str) -> bool:
        return comp in token or (comp == "etheta" and "etheta" in token) or (comp == "ephi" and "ephi" in token)

    def find(explicit: Optional[str], predicate, desc: str) -> int:
        return _find_column(headers, explicit, (predicate,), desc)

    if explicit_real or explicit_imag:
        ri = find(explicit_real, lambda t: has_component(t) and ("real" in t or t.startswith("re")), "real field")
        ii = find(explicit_imag, lambda t: has_component(t) and ("imag" in t or t.startswith("im")), "imaginary field")
        return "ri", ri, ii

    real_candidates = [
        i for i, token in enumerate(normal)
        if has_component(token) and ("real" in token or token.startswith("re"))
    ]
    imag_candidates = [
        i for i, token in enumerate(normal)
        if has_component(token) and ("imag" in token or token.startswith("im"))
    ]
    if real_candidates and imag_candidates:
        return "ri", real_candidates[0], imag_candidates[0]

    if explicit_mag or explicit_phase:
        mi = find(explicit_mag, lambda t: has_component(t) and ("mag" in t or "magnitude" in t or "abs" in t), "field magnitude")
        pi = find(explicit_phase, lambda t: has_component(t) and ("phase" in t or "ang" in t), "field phase")
        return "mp", mi, pi

    mag_candidates = [
        i for i, token in enumerate(normal)
        if has_component(token) and ("mag" in token or "magnitude" in token or "abs" in token)
    ]
    phase_candidates = [
        i for i, token in enumerate(normal)
        if has_component(token) and ("phase" in token or "ang" in token)
    ]
    if mag_candidates and phase_candidates:
        return "mp", mag_candidates[0], phase_candidates[0]

    # A single complex-valued column is accepted as a last resort.  It is
    # useful for custom exports, although standard HFSS reports usually use
    # real/imaginary or magnitude/phase columns.
    single = [
        i for i, token in enumerate(normal)
        if has_component(token) and not any(k in token for k in ("db", "gain", "phase", "real", "imag", "mag", "magnitude"))
    ]
    if single:
        return "complex", single[0], -1
    raise ValueError(
        f"Could not find complex {component} columns in {headers}. "
        "Embedded-element recombination requires Re/Im or magnitude/phase data."
    )


def _complex_cell(text: str) -> complex:
    value = text.strip().strip('"').replace("I", "j").replace("i", "j")
    value = value.replace(" ", "")
    return complex(value)


def read_embedded_component(
    path: Path,
    args: argparse.Namespace,
    component: str,
) -> Curve:
    table = read_table(path)
    indices, angle = _select_and_map_angles(table, args)
    kind, first, second = _component_columns(
        table.headers,
        component,
        args.real_column,
        args.imag_column,
        args.magnitude_column,
        args.phase_column,
    )

    values: list[complex] = []
    for row_number in indices:
        row = table.rows[row_number]
        if kind == "ri":
            values.append(complex(_number(row[first]), _number(row[second])))
        elif kind == "mp":
            magnitude = _number(row[first])
            phase = _number(row[second])
            if args.phase_unit == "deg":
                phase = np.deg2rad(phase)
            values.append(magnitude * np.exp(1j * phase))
        else:
            values.append(_complex_cell(row[first]))
    values_array = np.asarray(values, dtype=np.complex128)
    # Curve.db stores power dB.  The coherent sum is performed before this
    # conversion by combine_embedded_fields().
    return Curve(angle, values_array, str(path))  # type: ignore[arg-type]


def combine_embedded_fields(
    paths: Sequence[Path],
    weights: np.ndarray,
    args: argparse.Namespace,
) -> Curve:
    components = [read_embedded_component(path, args, args.field_component) for path in paths]
    reference = components[0]
    for component in components[1:]:
        if component.angle.shape != reference.angle.shape or not np.allclose(component.angle, reference.angle):
            raise ValueError(
                "Embedded port exports do not share the same angle grid; "
                "resample them in HFSS or export a common Infinite Sphere."
            )

    excitation = np.conj(weights) if not args.no_source_conjugate else weights
    field = np.zeros(reference.angle.shape, dtype=np.complex128)
    for coefficient, component in zip(excitation, components):
        field += coefficient * np.asarray(component.db, dtype=np.complex128)
    power = np.maximum(np.abs(field) ** 2, 10.0 ** (args.floor_db / 10.0))
    db = 10.0 * np.log10(power)
    return _unique_curve(reference.angle, db, ";".join(str(path) for path in paths))


# ---------------------------------------------------------------------------
# Cases, metrics, and plotting


def _case_path(directory: Path, template: str, case: Case) -> Path:
    rendered = template.format(
        case_id=case.case_id,
        id=case.case_id,
        theta_s=case.theta_s,
        theta_i=case.theta_i,
    )
    path = directory / rendered
    if path.exists():
        return path
    # Make the default forgiving of an extension change made by the HFSS UI.
    stem = path.stem
    candidates = sorted(
        candidate for candidate in directory.glob(stem + ".*")
        if candidate.suffix.lower() in {".csv", ".tab", ".txt", ".dat"}
    )
    if candidates:
        return candidates[0]
    return path


def _load_cases(path: Optional[Path]) -> tuple[Case, ...]:
    if path is None:
        return DEFAULT_CASES
    import json

    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("cases", payload)
    if isinstance(payload, dict):
        payload = [payload]
    cases = []
    for index, item in enumerate(payload, start=1):
        cases.append(
            Case(
                int(item.get("case_id", item.get("id", index))),
                float(item["theta_s"]),
                float(item["theta_i"]),
            )
        )
    return tuple(cases)


def _interpolate(curve: Curve, angles: np.ndarray) -> np.ndarray:
    return np.interp(angles, curve.angle, curve.db, left=np.nan, right=np.nan)


def _value_at(curve: Curve, angle: float) -> float:
    if angle < curve.angle.min() or angle > curve.angle.max():
        return float("nan")
    return float(np.interp(angle, curve.angle, curve.db))


def _null_depth(curve: Curve, case: Case) -> float:
    """Return pattern(theta_i) - pattern(theta_s), in dB."""
    return _value_at(curve, case.theta_i) - _value_at(curve, case.theta_s)


def _normalise_db(curve: Curve) -> Curve:
    return Curve(curve.angle, curve.db - np.nanmax(curve.db), curve.source)


def _metric_row(
    case: Case,
    hfss: Curve,
    mvdr: Curve,
    nn: Optional[Curve],
    floor_db: float,
) -> dict[str, float | str]:
    hfss = Curve(hfss.angle, np.maximum(hfss.db, floor_db), hfss.source)
    mvdr = Curve(mvdr.angle, np.maximum(mvdr.db, floor_db), mvdr.source)
    hfss = _normalise_db(hfss)
    mvdr = _normalise_db(mvdr)
    mask = (hfss.angle >= mvdr.angle.min()) & (hfss.angle <= mvdr.angle.max())
    hfss_values = hfss.db[mask]
    mvdr_values = _interpolate(mvdr, hfss.angle[mask])
    valid = np.isfinite(mvdr_values) & np.isfinite(hfss_values)
    if not np.any(valid):
        raise ValueError(
            f"No overlapping angle range between HFSS export {hfss.source} "
            "and the analytical Python pattern"
        )

    row: dict[str, float | str] = {
        "case_id": case.case_id,
        "theta_s_deg": case.theta_s,
        "theta_i_deg": case.theta_i,
        "hfss_source": hfss.source,
        "samples": int(valid.sum()),
        "mvdr_rmse_db": float(np.sqrt(np.mean((hfss_values[valid] - mvdr_values[valid]) ** 2))),
        "mvdr_max_abs_error_db": float(np.max(np.abs(hfss_values[valid] - mvdr_values[valid]))),
        "hfss_null_depth_db": _null_depth(hfss, case),
        "mvdr_null_depth_db": _null_depth(mvdr, case),
    }
    if nn is not None:
        nn = _normalise_db(nn)
        nn_values = _interpolate(nn, hfss.angle[mask])
        valid_nn = valid & np.isfinite(nn_values)
        row["nn_rmse_db"] = float(np.sqrt(np.mean((hfss_values[valid_nn] - nn_values[valid_nn]) ** 2)))
        row["nn_max_abs_error_db"] = float(np.max(np.abs(hfss_values[valid_nn] - nn_values[valid_nn])))
        row["nn_null_depth_db"] = _null_depth(nn, case)
    else:
        row["nn_rmse_db"] = ""
        row["nn_max_abs_error_db"] = ""
        row["nn_null_depth_db"] = ""
    return row


def plot_case(
    case: Case,
    hfss: Curve,
    mvdr_weights_value: np.ndarray,
    nn_weights_value: Optional[np.ndarray],
    output: Path,
    floor_db: float,
) -> tuple[Curve, Optional[Curve]]:
    hfss = _normalise_db(hfss)
    # Plot on the imported grid so no interpolation hides HFSS sampling gaps.
    angles = hfss.angle
    mvdr = Curve(angles, response_pattern_db(mvdr_weights_value, angles, floor_db), "analytical MVDR")
    nn = None
    if nn_weights_value is not None:
        nn = Curve(angles, response_pattern_db(nn_weights_value, angles, floor_db), "nullaligned NN")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(hfss.angle, np.maximum(hfss.db, floor_db), "k.", ms=3, label="HFSS")
    ax.plot(mvdr.angle, np.maximum(mvdr.db, floor_db), "tab:blue", lw=2, label="analytical MVDR")
    if nn is not None:
        ax.plot(nn.angle, np.maximum(nn.db, floor_db), "tab:orange", lw=1.7, ls="--", label="nullaligned NN")
    ax.axvline(case.theta_s, color="tab:green", ls=":", lw=1, label="desired angle")
    ax.axvline(case.theta_i, color="tab:red", ls=":", lw=1, label="interferer angle")
    ax.set_xlim(float(np.min(angles)), float(np.max(angles)))
    ax.set_ylim(floor_db, 3)
    ax.set_xlabel("Python array angle (deg)")
    ax.set_ylabel("peak-normalized pattern (dB)")
    ax.set_title(f"Case {case.case_id:02d}: theta_s={case.theta_s:g}°, theta_i={case.theta_i:g}°")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return mvdr, nn


def run(args: argparse.Namespace) -> list[dict[str, float | str]]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(Path(args.cases_json) if args.cases_json else None)

    nn_predictor: Optional[NeuralPredictor] = None
    if not args.no_nn:
        checkpoint = Path(args.nn_checkpoint) if args.nn_checkpoint else Path("data/model_nullaligned_full.pth")
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parent / checkpoint
        if checkpoint.exists():
            try:
                nn_predictor = NeuralPredictor(checkpoint)
                print(f"loaded NN checkpoint: {checkpoint}")
            except Exception as exc:
                if args.require_nn:
                    raise
                print(f"warning: NN checkpoint was not loaded ({exc}); continuing without NN")
        elif args.require_nn:
            raise FileNotFoundError(f"NN checkpoint not found: {checkpoint}")
        else:
            print(f"warning: NN checkpoint not found: {checkpoint}; continuing without NN")

    rows: list[dict[str, float | str]] = []
    for case in cases:
        if args.mode == "combined":
            path = _case_path(input_dir, args.file_template, case)
            if not path.exists():
                message = f"missing HFSS export for case {case.case_id}: {path}"
                if args.allow_missing:
                    print("warning: " + message)
                    continue
                raise FileNotFoundError(message)
            hfss = read_combined_pattern(path, args)
        else:
            paths = []
            for port in range(args.port_start, args.port_start + N):
                rendered = args.port_template.format(
                    case_id=case.case_id,
                    id=case.case_id,
                    theta_s=case.theta_s,
                    theta_i=case.theta_i,
                    port=port,
                    port_index=port - args.port_start,
                )
                path = input_dir / rendered
                if not path.exists():
                    raise FileNotFoundError(f"missing embedded export: {path}")
                paths.append(path)
            mvdr_value = mvdr_weights(case.theta_s, case.theta_i)
            hfss = combine_embedded_fields(paths, mvdr_value, args)

        mvdr_value = mvdr_weights(case.theta_s, case.theta_i)
        nn_value = nn_predictor.weights(case.theta_s, case.theta_i) if nn_predictor else None
        plot_path = output_dir / f"case_{case.case_id:02d}_overlay.png"
        mvdr_curve, nn_curve = plot_case(case, hfss, mvdr_value, nn_value, plot_path, args.floor_db)
        row = _metric_row(case, hfss, mvdr_curve, nn_curve, args.floor_db)
        rows.append(row)
        print(
            f"case {case.case_id:02d}: MVDR RMSE={row['mvdr_rmse_db']:.3f} dB, "
            f"null(HFSS/MVDR)={row['hfss_null_depth_db']:.1f}/{row['mvdr_null_depth_db']:.1f} dB"
            + (f", NN RMSE={row['nn_rmse_db']:.3f} dB" if nn_curve is not None else "")
        )

    metrics_path = output_dir / "metrics.csv"
    if rows:
        keys = list(rows[0].keys())
        with metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {metrics_path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="hfss_exports", help="HFSS export directory")
    parser.add_argument("--output-dir", default="plots/hfss", help="overlay output directory")
    parser.add_argument("--mode", choices=("combined", "embedded"), default="combined")
    parser.add_argument("--file-template", default="case_{case_id:02d}.csv", help="combined-file template")
    parser.add_argument("--port-template", default="case_{case_id:02d}_port_{port:02d}.csv", help="embedded-file template")
    parser.add_argument("--port-start", type=int, default=1, help="first HFSS port number")
    parser.add_argument("--cases-json", help="JSON list of {case_id, theta_s, theta_i}")
    parser.add_argument("--nn-checkpoint", help="nullaligned BeamNet .pth; defaults to data/model_nullaligned_full.pth")
    parser.add_argument("--no-nn", action="store_true", help="skip NN overlay")
    parser.add_argument("--require-nn", action="store_true", help="fail if the NN checkpoint cannot be loaded")
    parser.add_argument("--convention", choices=("python", "hfss_spherical"), default="python", help="angle convention")
    parser.add_argument("--angle-offset", type=float, default=0.0, help="degrees added after angle mapping")
    parser.add_argument("--phi", default="0", help="HFSS spherical Phi in degrees, or 'both'/'all'")
    parser.add_argument("--phi-tolerance", type=float, default=1e-3)
    parser.add_argument("--theta-column", help="exact or partial angle-column name")
    parser.add_argument("--phi-column", help="exact or partial Phi-column name")
    parser.add_argument("--pattern-column", help="exact or partial dB/linear pattern-column name")
    parser.add_argument("--field-component", choices=("etheta", "ephi"), default="etheta")
    parser.add_argument("--real-column", help="embedded field real column")
    parser.add_argument("--imag-column", help="embedded field imaginary column")
    parser.add_argument("--magnitude-column", help="embedded field magnitude column")
    parser.add_argument("--phase-column", help="embedded field phase column")
    parser.add_argument("--phase-unit", choices=("deg", "rad"), default="deg")
    parser.add_argument("--no-source-conjugate", action="store_true", help="do not conjugate MVDR weights for embedded fields")
    parser.add_argument("--floor-db", type=float, default=DEFAULT_FLOOR_DB)
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
