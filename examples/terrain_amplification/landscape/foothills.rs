//! Widen the slope break at the foot of mountain valleys with a compact
//! drainage-conditioned colluvial apron. Leave channels, summits and sea
//! level intact. This is a regional geomorphic filter, not a fluid solver.
use super::{Climate, Grid, Rivers, blur, gradient, smooth};

fn unit(x: f64) -> f64 {
    smooth(x.clamp(0.0, 1.0))
}

pub(super) fn blend(grid: &mut Grid, climate: &Climate) {
    let flow = Rivers::route(grid, climate);
    let mut distance: Vec<f64> = grid
        .heights
        .iter()
        .zip(&flow.discharge)
        .map(|(&h, &q)| if h <= 0.0 || q >= 14.0 { 0.0 } else { 1e9 })
        .collect();
    let mut stage = grid.heights.clone();
    for forward in [true, false] {
        for rank in 0..distance.len() {
            let i = if forward {
                rank
            } else {
                distance.len() - 1 - rank
            };
            for (j, d) in grid.neighbors(i) {
                let candidate = distance[j] + d;
                if candidate < distance[i] {
                    distance[i] = candidate;
                    stage[i] = stage[j];
                }
            }
        }
    }
    let broad = blur(grid, &grid.heights, 32);
    let local = blur(grid, &grid.heights, 6);
    let original = grid.clone();
    for (i, h) in grid.heights.iter_mut().enumerate() {
        let (x, y) = original.coordinates(i);
        let old = original.heights[i];
        let [gx, gy] = gradient(&original, x, y);
        let slope = gx.hypot(gy);
        let above = (old - stage[i]).max(0.0);
        let valley = unit((above - 12.0) / 90.0) * (1.0 - unit((above - 320.0) / 480.0));
        let lateral =
            unit((distance[i] - 100.0) / 220.0) * (1.0 - unit((distance[i] - 1700.0) / 1600.0));
        let foot = valley * lateral * unit(old / 100.0) * original.boundary_weight(x, y);
        // Concave slope bases accumulate debris, while convex lower spurs
        // relax; no flattening of high ridges and no changing channel beds.
        let delta = (0.68 * (broad[i] - old) + 0.32 * (local[i] - old)).clamp(-85.0, 125.0);
        let activity = unit((slope - 0.035) / 0.27);
        *h = old + foot * (0.45 + 0.55 * activity) * delta;
    }
}
