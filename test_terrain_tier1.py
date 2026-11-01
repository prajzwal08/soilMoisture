"""
Tier-1 validation (§32.5): synthetic DEMs with analytic answers, plus the one exact
test in the pipeline — mass conservation.

These run in seconds and catch what real terrain hides. On a real DEM a
direction-encoding bug looks like plausible terrain; on a symmetric V-valley it
shows up as an asymmetric TWI field, and there is nothing else it could be.

  surface                required result
  inclined plane         a per row = distance from the top edge; slope exactly s;
                         TWI linear downslope
  cone (divergent)       a = r/2 exactly  (see the correction below)
  V-valley               TWI maximal on the axis AND symmetric
  single pit in a plane  after breaching, zero cells lack a downstream neighbour
  any                    D8 accumulation summed over outlets == valid cell count

Two corrections to §32.5's table, both found by running it:

  §32.5 says a cone should give 'a ~ one cell everywhere; TWI low and near-uniform'.
  That is wrong. On a cone the upslope area at radius r is pi*r^2 and the contour
  width is 2*pi*r, so the specific catchment area is r/2 — it grows linearly, and
  measured median accumulation was 17.4 cells, not ~1. 'Divergent' bounds the
  contour-width ratio, it does not bound a. The replacement is strictly stronger:
  r/2 is an exact analytic value, not an approximate ceiling.

  §32.5's pit test cannot be posed through pyflwdir's pit list, because
  pyflwdir.from_dem fills internally (§32.4's own API caveat), so the raw DEM
  reports zero pits before conditioning and the test passes vacuously. It is posed
  here against terrain_ops.interior_sinks, which counts sinks directly.

Slope, and therefore TWI, is checked on the INTERIOR only. horn_slope replicates
edge rows, so the outermost row's gradient is half the true one; that is documented
behaviour, and the 10 km region buffer means no station tile ever sits on an edge.

Usage:
  conda activate terramind
  python test_terrain_tier1.py
"""

import sys

import numpy as np

import terrain_ops as T

RES = 30.0
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


# ─────────────────────────────────────────────────────────────────────────────

def test_inclined_plane(n: int = 80, slope: float = 0.10) -> None:
    """
    Plane tilted along y only. Row j (0-based, from the top) has j+1 cells draining
    through it per column, so a = (j+1)*res = distance from the top edge.

    Interior columns only: the left and right columns lose water off the side under
    MFD's diagonal splitting, which is correct behaviour, not an error.
    """
    print("\ninclined plane, slope 0.10")
    y = np.arange(n)[:, None] * np.ones((1, n))
    dem = (100.0 - slope * RES * y).astype(np.float32)   # decreasing southward

    wd = T.scratch_dir("t1_plane_")
    try:
        cond = T.condition_dem(dem, RES, wd)
        # a plane has no depressions: conditioning must not move a single cell
        check("conditioning leaves a plane untouched",
              np.allclose(cond, dem, atol=1e-4),
              f"max |dz| = {np.nanmax(np.abs(cond - dem)):.2e} m")

        beta = T.horn_slope(dem, RES)
        inner = slice(1, n - 1)          # horn_slope replicates edge rows by design
        check("Horn slope equals the analytic slope (interior)",
              np.allclose(np.tan(beta[inner, inner]), slope, atol=1e-6),
              f"tan(beta) = {np.tan(beta[inner, inner]).mean():.8f} vs {slope}")
        check("edge rows are half-slope, as replication implies",
              np.allclose(np.tan(beta[0, inner]), slope / 2, atol=1e-6),
              f"tan(beta) row 0 = {np.tan(beta[0, n//2]):.8f}")

        acc = T.flow_accum_mfd(cond, RES, wd)
        mid = slice(n // 4, 3 * n // 4)
        rows = np.nanmean(acc[:, mid], axis=1)
        expect = np.arange(1, n + 1, dtype=float)
        check("MFD a per row = distance from the top edge",
              np.allclose(rows, expect, rtol=0.02),
              f"max rel err {np.nanmax(np.abs(rows - expect) / expect):.4f}")

        twi, floored = T.twi_from(acc, beta, RES)
        col = twi[inner, n // 2]
        d = np.diff(col)
        check("TWI increases monotonically downslope (interior)", bool(np.all(d > 0)),
              f"min step {d.min():.4f}")
        want = np.log(expect[1:n - 1] * RES / slope)
        check("TWI is linear in ln(row)  (a linear, slope constant)",
              np.allclose(col, want, atol=1e-3),
              f"max dev {np.nanmax(np.abs(col - want)):.2e}")
        check("no cells hit the tan-slope floor", floored == 0.0, f"floored {floored:.3%}")

        flw = T.d8_network(cond, RES)
        got, want = T.mass_conservation_d8(flw, int(np.isfinite(dem).sum()))
        check("D8 mass conservation, to the integer", got == want, f"{got:.0f} vs {want:.0f}")
    finally:
        T.cleanup(wd)


def test_cone(n: int = 101, slope: float = 0.15) -> None:
    """
    A cone: upslope area at radius r is pi*r^2, contour width is 2*pi*r, so the
    specific catchment area is exactly r/2 — equivalently, accumulation in cells is
    r_cells/2. That is the analytic answer, and it is what this checks.

    §32.5's table asks instead for 'a ~ one cell everywhere; TWI low and near-uniform'.
    That is wrong: divergence bounds the contour-width ratio, not a. Measured median
    accumulation on this surface is 17.4 cells, exactly r_cells/2 for the sampled
    annulus, so the original expectation would have failed on correct code.

    Restricted to an annulus: the apex has a degenerate 3x3 neighbourhood, and beyond
    the inscribed circle the upslope sector is truncated by the grid corners so
    pi*r^2 no longer counts the contributing area.
    """
    print(f"\ncone (divergent), slope {slope}")
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r_cells = np.hypot(xx - c, yy - c)
    dem = (500.0 - slope * r_cells * RES).astype(np.float32)

    wd = T.scratch_dir("t1_cone_")
    try:
        cond = T.condition_dem(dem, RES, wd)
        acc = T.flow_accum_mfd(cond, RES, wd)
        beta = T.horn_slope(dem, RES)
        twi, _ = T.twi_from(acc, beta, RES)

        m = (r_cells > 5) & (r_cells < c - 2)      # annulus inside the inscribed circle
        ratio = acc[m] / (r_cells[m] / 2.0)
        check("MFD accumulation = r/2 on a cone",
              abs(float(np.nanmedian(ratio)) - 1.0) < 0.10,
              f"median acc/(r/2) = {np.nanmedian(ratio):.4f}, "
              f"median acc {np.nanmedian(acc[m]):.2f} cells")
        check("slope on the cone flank equals the analytic slope",
              abs(float(np.nanmedian(np.tan(beta[m]))) - slope) < 0.01,
              f"median tan(beta) {np.nanmedian(np.tan(beta[m])):.5f} vs {slope}")
        # TWI = ln((r/2)/tan(beta)): grows as ln r, so it is NOT near-uniform, but it
        # must be radially symmetric — the same statement the V-valley test makes.
        twi_by_r = [np.nanmean(twi[(r_cells > k - 0.5) & (r_cells < k + 0.5)])
                    for k in range(10, c - 3, 5)]
        check("TWI increases with radius (a grows, slope constant)",
              bool(np.all(np.diff(twi_by_r) > 0)),
              f"min step {np.min(np.diff(twi_by_r)):.4f}")
        ring = (r_cells > c // 2 - 0.5) & (r_cells < c // 2 + 0.5)
        check("TWI is radially symmetric on the cone",
              float(np.nanstd(twi[ring])) < 0.15,
              f"sd around the ring {np.nanstd(twi[ring]):.4f}")
    finally:
        T.cleanup(wd)


def test_v_valley(n: int = 101) -> None:
    """
    Symmetric V-valley along y. TWI must be maximal on the axis and symmetric about
    it: asymmetry means a direction-encoding bug, and no real-terrain test would
    separate that from genuine landscape asymmetry.
    """
    print("\nV-valley (symmetric)")
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dem = (200.0 + 0.20 * RES * np.abs(xx - c) - 0.02 * RES * yy).astype(np.float32)

    wd = T.scratch_dir("t1_valley_")
    try:
        cond = T.condition_dem(dem, RES, wd)
        acc = T.flow_accum_mfd(cond, RES, wd)
        beta = T.horn_slope(dem, RES)
        twi, _ = T.twi_from(acc, beta, RES)

        row = twi[3 * n // 4, :]
        check("TWI is maximal on the valley axis",
              int(np.nanargmax(row)) == c,
              f"argmax col {int(np.nanargmax(row))}, axis {c}")

        # compare the two flanks of the same row
        left  = row[c - 40:c]
        right = row[c + 1:c + 41][::-1]
        asym = float(np.nanmax(np.abs(left - right)))
        check("TWI is symmetric about the axis", asym < 1e-3,
              f"max flank difference {asym:.3e}")

        aleft  = np.nanmean(acc[:, c - 40:c])
        aright = np.nanmean(acc[:, c + 1:c + 41])
        check("accumulation is symmetric about the axis",
              abs(aleft - aright) / max(aleft, aright) < 1e-3,
              f"{aleft:.3f} vs {aright:.3f} cells")
    finally:
        T.cleanup(wd)


def test_pit(n: int = 60, slope: float = 0.05, depth: float = 20.0) -> None:
    """
    A single deep pit in a plane, the canopy-dam analogue. After breaching, no cell
    may lack a downstream neighbour.

    Counted with terrain_ops.interior_sinks, NOT with pyflwdir's pit list:
    pyflwdir.from_dem fills internally, so it reports zero pits on the raw DEM and
    the test would pass without conditioning ever having run.
    """
    print(f"\nsingle pit in a plane, {depth:.0f} m deep")
    y = np.arange(n)[:, None] * np.ones((1, n))
    dem = (100.0 - slope * RES * y).astype(np.float32)
    pj, pi = n // 2, n // 2
    dem[pj - 1:pj + 2, pi - 1:pi + 2] -= depth

    wd = T.scratch_dir("t1_pit_")
    try:
        sinks_raw = T.interior_sinks(dem)
        cond = T.condition_dem(dem, RES, wd)
        flw = T.d8_network(cond, RES)
        sinks_cond = T.interior_sinks(cond)

        check("the raw DEM does contain the pit (test is not vacuous)",
              sinks_raw > 0, f"{sinks_raw} interior sink(s) before conditioning")
        check("zero interior sinks after breaching", sinks_cond == 0,
              f"{sinks_cond} remain")
        check("pyflwdir.from_dem hides the raw pit (§32.4's API caveat holds)",
              len([k for k in T.d8_network(dem, RES).idxs_pit
                   if 0 < k // n < n - 1 and 0 < k % n < n - 1]) == 0,
              "raw DEM reports 0 interior pits through pyflwdir despite having one")

        carved = dem - cond
        check("breaching carves DOWN through the obstruction, not up",
              float(np.nanmax(carved)) > 0.0,
              f"max carved {np.nanmax(carved):.2f} m, max raised {np.nanmax(-carved):.2f} m")
        # filling would raise the pit floor by ~depth over a wide flooded area;
        # breaching should touch few cells
        touched = int(np.sum(np.abs(carved) > 0.01))
        check("breaching touches few cells (not a flooded lake)",
              touched < 0.05 * dem.size, f"{touched} of {dem.size} cells ({touched/dem.size:.2%})")

        got, want = T.mass_conservation_d8(flw, int(np.isfinite(cond).sum()))
        check("D8 mass conservation after conditioning", got == want,
              f"{got:.0f} vs {want:.0f}")
    finally:
        T.cleanup(wd)


def test_hand(n: int = 101) -> None:
    """
    HAND on the V-valley: zero on the stream, >= 0 everywhere, and non-decreasing as
    you climb the flank. A negative HAND is a conditioning bug by definition.
    """
    print("\nHAND on a V-valley")
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dem = (200.0 + 0.20 * RES * np.abs(xx - c) - 0.02 * RES * yy).astype(np.float32)

    wd = T.scratch_dir("t1_hand_")
    try:
        cond = T.condition_dem(dem, RES, wd)
        acc = T.flow_accum_mfd(cond, RES, wd)
        flw = T.d8_network(cond, RES)
        streams = T.stream_mask(acc, RES, stream_ha=1.0)
        hand = T.hand_from(flw, dem, streams)

        check("streams are non-empty and on the axis",
              streams.any() and abs(int(np.median(np.where(streams)[1])) - c) <= 1,
              f"{streams.sum()} stream cells, median col {int(np.median(np.where(streams)[1]))}")
        check("HAND >= 0 everywhere", bool(np.all(np.nan_to_num(hand, nan=0.0) >= -1e-4)),
              f"min {np.nanmin(hand):.4f} m")
        check("HAND == 0 on stream cells",
              float(np.nanmax(np.abs(hand[streams]))) < 1e-3,
              f"max |HAND| on streams {np.nanmax(np.abs(hand[streams])):.2e} m")

        prof = hand[3 * n // 4, c:c + 40]
        d = np.diff(prof)
        check("HAND is non-decreasing climbing the flank",
              bool(np.all(d >= -1e-4)), f"min step {d.min():.4f} m")
    finally:
        T.cleanup(wd)


def main() -> int:
    print("Tier 1 — synthetic DEMs with analytic answers (§32.5)")
    print(f"pyflwdir/whitebox derivation at {RES:.0f} m")
    test_inclined_plane()
    test_cone()
    test_v_valley()
    test_pit()
    test_hand()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"TIER 1 FAILED — {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("TIER 1 PASSED — all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
