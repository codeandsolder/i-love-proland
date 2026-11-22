//! Concurrent uplift, stream-power incision and hillslope transport. Uplift
//! is a smooth function of the inherited geographical and lithology priors,
//! not a random height perturbation. All coefficients are experimental.
use super::{Climate, Grid, Rivers, SEED, blur, hash, noise, smooth, upsample};

fn step(grid: &mut Grid, climate: &Climate, reference: &Grid, uplift: &[f64]) {
    let flow = Rivers::route(grid, climate);
    let mut bed = grid.heights.clone();
    for &i in &flow.order {
        let j = flow.receiver[i];
        let h = grid.heights[i];
        if j == usize::MAX || reference.heights[i] <= 0.0 {
            continue;
        }
        let (x, y) = grid.coordinates(i);
        let (jx, jy) = grid.coordinates(j);
        let distance = grid.spacing * f64::from(x.abs_diff(jx)).hypot(f64::from(y.abs_diff(jy)));
        let [_, rock, arid, _, _] = climate.at(i);
        let w = grid.boundary_weight(x, y) * smooth((reference.heights[i] / 120.0).clamp(0.0, 1.0));
        let rise =
            grid.spacing * uplift[i] * (1.0 - (h - reference.heights[i]) / 3600.0).clamp(0.0, 1.5);
        let lifted = h + w * rise;
        let k = (570.0 / distance)
            * flow.discharge[i].powf(0.45)
            * (1.0 - 0.48 * rock)
            * (1.0 - 0.30 * arid);
        let downstream = bed[j].max(0.0);
        let cut = (lifted - downstream).max(0.0) * k / (1.0 + k);
        bed[i] = (lifted - w * cut).max(0.02);
    }

    // Hillslope transport distributes material conservatively, with much
    // stronger relaxation above the critical slope of the local rock/soil.
    let mut transport = vec![0.0; bed.len()];
    for i in 0..bed.len() {
        if reference.heights[i] <= 0.0 {
            continue;
        }
        let (x, y) = grid.coordinates(i);
        let a = grid.boundary_weight(x, y);
        if a == 0.0 {
            continue;
        }
        for (j, d) in grid.neighbors(i).filter(|(j, _)| *j > i) {
            if reference.heights[j] <= 0.0 {
                continue;
            }
            let (jx, jy) = grid.coordinates(j);
            let w = a.min(grid.boundary_weight(jx, jy));
            let rock = climate.at(i)[1].midpoint(climate.at(j)[1]);
            let delta = bed[i] - bed[j];
            let critical = (0.75 + 0.45 * rock) * d;
            let excess = (delta.abs() - critical).max(0.0);
            let flux = w * ((0.034 - 0.018 * rock) * delta + 0.085 * excess.copysign(delta));
            transport[i] -= flux;
            transport[j] += flux;
        }
    }
    for (i, value) in grid.heights.iter_mut().enumerate() {
        if reference.heights[i] > 0.0 {
            *value = (bed[i] + transport[i]).max(0.02);
        }
    }
}

/// Use the original native terrain as a low-frequency uplift guide; the
/// drainage and transport model is allowed to reshape the mountains, while
/// sea and lowland are preserved. The final level is detail only.
pub(super) fn generate(base: &Grid, mut climate: Climate, strength: f64) -> (Grid, Climate) {
    let mut reference = base.clone();
    let mut grid = super::initial_relief(base, &climate, 0.8);
    let mut field = Grid::new(base.cells, base.spacing);
    let regional = blur(base, &base.heights, 4);
    for (i, r) in field.heights.iter_mut().enumerate() {
        let (x, y) = base.coordinates(i);
        let [_, rock, arid, _, _] = climate.at(i);
        let high = smooth(((regional[i] - 400.0) / 1800.0).clamp(0.0, 1.0));
        let uplift = 0.025 + strength * (0.30 * high + 0.15 * rock) * (1.0 - 0.4 * arid);
        // Heterogeneity of uplift over several kilometres, not height noise.
        *r = uplift
            * (1.0 + 0.42 * noise(hash(SEED ^ 90), x, y))
            * smooth((base.heights[i] / 180.0).clamp(0.0, 1.0));
    }
    for (level, steps) in [28, 14, 6, 0].into_iter().enumerate() {
        grid = upsample(&grid, 0, 0.0);
        reference = upsample(&reference, 0, 0.0);
        field = upsample(&field, 0, 0.0);
        climate.refine();
        for _ in 0..steps {
            step(&mut grid, &climate, &reference, &field.heights);
        }
        let seed = hash(SEED ^ u64::try_from(level).unwrap_or(0) ^ 0x4556);
        for i in 0..grid.heights.len() {
            let (x, y) = grid.coordinates(i);
            let h = grid.heights[i];
            let rock = climate.at(i)[1];
            let amplitude = grid.spacing * (0.016 + 0.045 * rock);
            grid.heights[i] += amplitude
                * noise(seed, x, y)
                * grid.boundary_weight(x, y)
                * smooth((h / 130.0).clamp(0.0, 1.0));
        }
    }
    (grid, climate)
}
