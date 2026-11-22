//! Compact-support slope-oriented wave packets, inspired by Grenier et al.
//! (2024). This is a deliberately simple approximation, not their solver.
use super::{Grid, hash, smooth};

fn unit(key: u64) -> f64 {
    // Upper 24 bits fit exactly in f64 after narrowing to u32.
    let bits = (hash(key) >> 40) as u32;
    f64::from(bits) / 16_777_216.0
}

fn wave(seed: u64, point: [f64; 2], transverse: [f64; 2]) -> f64 {
    let [px, py] = point;
    // This example has a fixed positive local region, but signed lattice
    // indices are needed around the origin. Each kernel is compactly supported
    // on the unit disk so a 3x3 neighborhood covers all nonzero contributions.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "bounded 64 km laboratory coordinates in lattice units"
    )]
    let (ix, iy) = (px.floor() as i32, py.floor() as i32);
    let mut sum = [0.0, 0.0];
    let mut weights = 0.0;
    for y in iy - 1..=iy + 1 {
        for x in ix - 1..=ix + 1 {
            let key = hash(
                seed ^ u64::from(u32::from_ne_bytes(x.to_ne_bytes())) << 32
                    ^ u64::from(u32::from_ne_bytes(y.to_ne_bytes())),
            );
            let dx = px - (f64::from(x) + unit(key).mul_add(0.6, 0.2));
            let dy = py - (f64::from(y) + unit(key ^ 1).mul_add(0.6, 0.2));
            let w = (1.0 - dx.mul_add(dx, dy * dy)).max(0.0).powi(3);
            if w == 0.0 {
                continue;
            }
            let phase = std::f64::consts::TAU
                * (2.0_f64.mul_add(dx.mul_add(transverse[0], dy * transverse[1]), unit(key ^ 2)));
            let (s, c) = phase.sin_cos();
            sum[0] = c.mul_add(w, sum[0]);
            sum[1] = s.mul_add(w, sum[1]);
            weights += w;
        }
    }
    // Soft normalized phase: avoids singular spikes where opposite
    // phasors cancel. Negative lobes form channels rather than mounds.
    let norm = sum[0]
        .mul_add(sum[0], sum[1].mul_add(sum[1], 0.10 * weights * weights))
        .sqrt();
    if norm > 0.0 {
        (sum[0] / norm).max(0.0).powi(2)
    } else {
        0.0
    }
}

pub fn apply(grid: &mut Grid, seed: u64, amplitude: f64) {
    let mut residual = vec![0.0; grid.heights.len()];
    for (i, &height) in grid.heights.iter().enumerate() {
        let (x, y) = grid.coordinates(i);
        let xm = x.saturating_sub(1);
        let xp = (x + 1).min(grid.cells);
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(grid.cells);
        let gx = (grid.at(xp, y) - grid.at(xm, y)) / (f64::from(xp - xm) * grid.spacing);
        let gy = (grid.at(x, yp) - grid.at(x, ym)) / (f64::from(yp - ym) * grid.spacing);
        let slope = gx.hypot(gy);
        let mask = smooth((slope / 0.35).min(1.0))
            * smooth((height / 100.0).clamp(0.0, 1.0))
            * grid.boundary_weight(x, y);
        if mask > 0.0 {
            // Waves oscillate perpendicular to the gradient; their troughs
            // are approximately parallel to downhill water flow.
            let transverse = [-gy / slope, gx / slope];
            let point = [f64::from(x) / 8.0, f64::from(y) / 8.0];
            residual[i] = -amplitude * mask * wave(seed, point, transverse);
        }
    }
    for (height, detail) in grid.heights.iter_mut().zip(residual) {
        *height += detail;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flat_field_is_invariant_and_slope_field_repeatable() {
        let mut grid = Grid::new(16, 1000.0);
        grid.heights.fill(300.0);
        let original = grid.clone();
        apply(&mut grid, 42, 90.0);
        assert_eq!(grid.heights, original.heights);
        for y in 0..=grid.cells {
            for x in 0..=grid.cells {
                let i = grid.index(x, y);
                grid.heights[i] = 100.0 + 100.0 * f64::from(x);
            }
        }
        let mut second = grid.clone();
        apply(&mut grid, 42, 90.0);
        apply(&mut second, 42, 90.0);
        assert_eq!(grid.heights, second.heights);
        assert_ne!(grid.heights, original.heights);
    }
}
