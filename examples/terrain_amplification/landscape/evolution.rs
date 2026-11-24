//! Concurrent uplift, stream-power incision and hillslope transport. Uplift
//! is a smooth function of the inherited geographical and lithology priors,
//! not a random height perturbation. All coefficients are experimental.
use super::{Climate, Grid, Rivers, SEED, blur, hash, noise, smooth, upsample};

fn step(grid: &mut Grid, climate: &Climate, reference: &Grid, uplift: &[f64]) {
    let flow = Rivers::route(grid, climate);
    let mut bed = grid.heights.clone();
    for &index in &flow.order {
        let receiver = flow.receiver[index];
        let height = grid.heights[index];
        if receiver == usize::MAX || reference.heights[index] <= 0.0 {
            continue;
        }
        let (x, y) = grid.coordinates(index);
        let (jx, jy) = grid.coordinates(receiver);
        let distance = grid.spacing * f64::from(x.abs_diff(jx)).hypot(f64::from(y.abs_diff(jy)));
        let [_, rock, arid, _, _] = climate.at(index);
        let boundary =
            grid.boundary_weight(x, y) * smooth((reference.heights[index] / 120.0).clamp(0.0, 1.0));
        let rise = grid.spacing
            * uplift[index]
            * (1.0 - (height - reference.heights[index]) / 3600.0).clamp(0.0, 1.5);
        let lifted = boundary.mul_add(rise, height);
        let incision = (570.0 / distance)
            * flow.discharge[index].powf(0.45)
            * 0.48f64.mul_add(-rock, 1.0)
            * 0.30f64.mul_add(-arid, 1.0);
        let downstream = bed[receiver].max(0.0);
        let cut = (lifted - downstream).max(0.0) * incision / (1.0 + incision);
        bed[index] = boundary.mul_add(-cut, lifted).max(0.02);
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
            let critical = 0.45f64.mul_add(rock, 0.75) * d;
            let excess = (delta.abs() - critical).max(0.0);
            let flux = w * 0.085f64.mul_add(
                excess.copysign(delta),
                0.018f64.mul_add(-rock, 0.034) * delta,
            );
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
        let uplift = (strength * 0.15f64.mul_add(rock, 0.30 * high))
            .mul_add(0.4f64.mul_add(-arid, 1.0), 0.025);
        // Heterogeneity of uplift over several kilometres, not height noise.
        *r = uplift
            * 0.42f64.mul_add(noise(hash(SEED ^ 0x5a), x, y), 1.0)
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
            let amplitude = grid.spacing * 0.045f64.mul_add(rock, 0.016);
            grid.heights[i] = (amplitude * noise(seed, x, y) * grid.boundary_weight(x, y))
                .mul_add(smooth((h / 130.0).clamp(0.0, 1.0)), grid.heights[i]);
        }
    }
    (grid, climate)
}
