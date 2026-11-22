//! Regional water features extracted from the generated spill surface and
//! connected drainage network. This is a geometric preview, not fluid dynamics.
pub(super) mod weathering;

use super::{Grid, Rivers, lerp};
use std::{fmt::Write, fs, io, path::Path};

pub(super) struct Water {
    pub(super) lake_levels: Vec<f64>,
    pub(super) river: Vec<f64>,
    pub(super) segments: Vec<Segment>,
    pub(super) lakes: usize,
}
pub(super) struct Segment {
    start: usize,
    end: usize,
    width: f64,
    start_xy: [f64; 2],
    end_xy: [f64; 2],
}
fn xy(grid: &Grid, i: usize) -> [f64; 2] {
    let (x, y) = grid.coordinates(i);
    [f64::from(x), f64::from(y)]
}
fn mix(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    std::array::from_fn(|i| lerp(a[i], b[i], t))
}
fn lakes(grid: &Grid, flow: &Rivers) -> (Vec<f64>, usize) {
    let mut visited = vec![false; grid.heights.len()];
    let mut result = vec![0.0; grid.heights.len()];
    let mut count = 0;
    for i in 0..grid.heights.len() {
        let depth = flow.filled[i] - grid.heights[i];
        if visited[i] || grid.heights[i] <= 0.0 || depth <= 0.25 {
            continue;
        }
        visited[i] = true;
        let mut component = vec![i];
        let mut cursor = 0;
        let mut max_depth = depth;
        let level = flow.filled[i];
        while cursor < component.len() {
            let node = component[cursor];
            cursor += 1;
            for (j, _) in grid.neighbors(node) {
                let d = level - grid.heights[j];
                if visited[j]
                    || grid.heights[j] <= 0.0
                    || d <= 0.25
                    || (flow.filled[j] - level).abs() > 0.01
                {
                    continue;
                }
                visited[j] = true;
                max_depth = max_depth.max(d);
                component.push(j);
            }
        }
        // Avoid treating individual trapped terrain samples as ponds.
        if component.len() >= 16 && max_depth >= 6.0 {
            count += 1;
            for node in component {
                result[node] = level;
            }
        }
    }
    (result, count)
}

fn splat(grid: &Grid, start: [f64; 2], end: [f64; 2], width_m: f64, mask: &mut [f64]) {
    let radius = width_m / (2.0 * grid.spacing);
    let reach = radius + 1.0;
    let axis = [end[0] - start[0], end[1] - start[1]];
    let length_sq = axis[0].mul_add(axis[0], axis[1] * axis[1]).max(1e-9);
    #[expect(clippy::cast_possible_truncation, reason = "bounded grid-space box")]
    let (xmin, xmax, ymin, ymax) = (
        (start[0].min(end[0]) - reach).floor() as i32,
        (start[0].max(end[0]) + reach).ceil() as i32,
        (start[1].min(end[1]) - reach).floor() as i32,
        (start[1].max(end[1]) + reach).ceil() as i32,
    );
    for y in ymin.max(0)..=ymax.min(i32::from(grid.cells)) {
        for x in xmin.max(0)..=xmax.min(i32::from(grid.cells)) {
            let delta = [f64::from(x) - start[0], f64::from(y) - start[1]];
            let fraction =
                (delta[0].mul_add(axis[0], delta[1] * axis[1]) / length_sq).clamp(0.0, 1.0);
            let dist = fraction
                .mul_add(-axis[0], delta[0])
                .hypot(fraction.mul_add(-axis[1], delta[1]));
            let coverage = (reach - dist).clamp(0.0, 1.0) * (2.0 * radius).min(1.0);
            if let (Ok(x), Ok(y)) = (u16::try_from(x), u16::try_from(y)) {
                let index = grid.index(x, y);
                mask[index] = mask[index].max(coverage);
            }
        }
    }
}

pub(super) fn extract(grid: &Grid, flow: &Rivers) -> Water {
    let (lake_levels, lakes) = lakes(grid, flow);
    let mut river = vec![0.0; grid.heights.len()];
    let mut segments = Vec::new();
    let mut main_inflow = vec![usize::MAX; grid.heights.len()];
    let threshold = 14.0;
    for (i, &j) in flow.receiver.iter().enumerate() {
        if j == usize::MAX || flow.discharge[i] < threshold || grid.heights[i] <= 0.0 {
            continue;
        }
        let current = main_inflow[j];
        if current == usize::MAX || flow.discharge[i] > flow.discharge[current] {
            main_inflow[j] = i;
        }
    }
    for (i, &j) in flow.receiver.iter().enumerate() {
        if j == usize::MAX || flow.discharge[i] < threshold || grid.heights[i] <= 0.0 {
            continue;
        }
        let a = xy(grid, i);
        let b = xy(grid, j);
        let width = (4.5 * flow.discharge[i].sqrt()).clamp(3.0, 180.0);
        let start = if main_inflow[i] == usize::MAX {
            a
        } else {
            mix(xy(grid, main_inflow[i]), a, 0.5)
        };
        let end = mix(a, b, 0.5);
        // Smooth the node corner using a quadratic Bézier within the
        // upstream/downstream cells. Tributary segments share the end
        // midpoint with their receiver; explicit connector at confluences.
        let mut last = start;
        for k in 1..=4 {
            let t = f64::from(k) * 0.25;
            let point = mix(mix(start, a, t), mix(a, end, t), t);
            splat(grid, last, point, width, &mut river);
            segments.push(Segment {
                start: i,
                end: j,
                width,
                start_xy: last,
                end_xy: point,
            });
            last = point;
        }
        if main_inflow[j] != i || flow.receiver[j] == usize::MAX || grid.heights[j] <= 0.0 {
            splat(grid, end, b, width, &mut river);
            segments.push(Segment {
                start: i,
                end: j,
                width,
                start_xy: end,
                end_xy: b,
            });
        }
    }
    Water {
        lake_levels,
        river,
        segments,
        lakes,
    }
}

pub(super) fn save_graph(grid: &Grid, water: &Water, path: &Path) -> io::Result<()> {
    let mut data = String::from(
        "upstream_node,downstream_node,x0_m,y0_m,x1_m,y1_m,width_m,upstream_bed_m,downstream_bed_m\n",
    );
    for s in &water.segments {
        let _ = writeln!(
            data,
            "{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            s.start,
            s.end,
            s.start_xy[0] * grid.spacing,
            s.start_xy[1] * grid.spacing,
            s.end_xy[0] * grid.spacing,
            s.end_xy[1] * grid.spacing,
            s.width,
            grid.heights[s.start],
            grid.heights[s.end]
        );
    }
    fs::write(path, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connected_depression_has_one_lake_level() {
        let mut grid = Grid::new(12, 100.0);
        grid.heights.fill(20.0);
        for y in 3..=8 {
            for x in 3..=8 {
                let i = grid.index(x, y);
                grid.heights[i] = 10.0;
            }
        }
        let c = super::super::Climate {
            fields: std::array::from_fn(|_| {
                let mut g = Grid::new(12, 100.0);
                g.heights.fill(1.0);
                g
            }),
        };
        let flow = Rivers::route(&grid, &c);
        let water = extract(&grid, &flow);
        assert_eq!(water.lakes, 1);
        assert_eq!(water.lake_levels.iter().filter(|&&v| v == 20.0).count(), 36);
    }
}

/// Small regional index over the existing curved river segments. Used to
/// evaluate projected water coverage independently of height-grid spacing.
/// This is a rendering query, not a river-bed displacement or water mesh.
pub(super) struct RiverCoverage<'a> {
    segments: &'a [Segment],
    buckets: Vec<Vec<usize>>,
    side: usize,
    spacing: f64,
}

impl<'a> RiverCoverage<'a> {
    const BIN_M: f64 = 250.0;
    const MAX_FILTER_M: f64 = 125.0;

    pub(super) fn new(grid: &Grid, water: &'a Water) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "regional bin count is bounded by grid extent"
        )]
        #[expect(clippy::cast_sign_loss, reason = "grid extent is positive")]
        let side = (f64::from(grid.cells) * grid.spacing / Self::BIN_M).ceil() as usize + 1;
        let mut index = Self {
            segments: &water.segments,
            buckets: vec![Vec::new(); side * side],
            side,
            spacing: grid.spacing,
        };
        for (i, s) in water.segments.iter().enumerate() {
            let margin = f64::mul_add(s.width, 0.5, Self::MAX_FILTER_M);
            let [xmin, ymin] = std::array::from_fn(|c| {
                index.bin(f64::mul_add(
                    s.start_xy[c].min(s.end_xy[c]),
                    grid.spacing,
                    -margin,
                ))
            });
            let [xmax, ymax] = std::array::from_fn(|c| {
                index.bin(f64::mul_add(
                    s.start_xy[c].max(s.end_xy[c]),
                    grid.spacing,
                    margin,
                ))
            });
            for y in ymin..=ymax {
                for x in xmin..=xmax {
                    index.buckets[y * side + x].push(i);
                }
            }
        }
        index
    }

    fn bin(&self, value: f64) -> usize {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounded regional bin index"
        )]
        #[expect(clippy::cast_sign_loss, reason = "clamped nonnegative bin index")]
        let result = (value / Self::BIN_M).max(0.0).floor() as usize;
        result.min(self.side - 1)
    }

    pub(super) fn at(&self, point: [f64; 2], footprint: f64) -> f64 {
        let [x, y] = point.map(|v| self.bin(v));
        let half = (footprint * 0.5).clamp(0.4, Self::MAX_FILTER_M);
        let mut coverage = 0.0_f64;
        for &index in &self.buckets[y * self.side + x] {
            let segment = &self.segments[index];
            let start = segment.start_xy.map(|v| v * self.spacing);
            let end = segment.end_xy.map(|v| v * self.spacing);
            let axis = [end[0] - start[0], end[1] - start[1]];
            let delta = [point[0] - start[0], point[1] - start[1]];
            let fraction = (delta[0].mul_add(axis[0], delta[1] * axis[1])
                / axis[0].mul_add(axis[0], axis[1] * axis[1]).max(1e-9))
            .clamp(0.0, 1.0);
            let distance = f64::mul_add(fraction, -axis[0], delta[0])
                .hypot(f64::mul_add(fraction, -axis[1], delta[1]));
            let radius = segment.width * 0.5;
            // Approximate pixel integral of an infinite strip locally:
            // start subpixel river cannot cover the entire pixel, even at its
            // centerline. The union of segment capsules is approximated by max.
            let alpha = (radius + half - distance).clamp(0.0, (2.0 * radius).min(2.0 * half))
                / (2.0 * half);
            coverage = coverage.max(alpha);
        }
        coverage
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    #[test]
    fn coverage_tracks_metric_width_and_subpixel_filter() {
        let grid = Grid::new(32, 62.5);
        let water = Water {
            lake_levels: Vec::new(),
            river: Vec::new(),
            lakes: 0,
            segments: vec![Segment {
                start: 0,
                end: 1,
                width: 10.0,
                start_xy: [3.0, 4.0],
                end_xy: [9.0, 4.0],
            }],
        };
        let index = RiverCoverage::new(&grid, &water);
        assert!((index.at([250.0, 250.0], 2.0) - 1.0).abs() < 1e-12);
        assert!((index.at([250.0, 255.0], 2.0) - 0.5).abs() < 1e-12);
        assert!(index.at([250.0, 260.0], 2.0) < 1e-12);
        assert!((index.at([250.0, 250.0], 20.0) - 0.5).abs() < 1e-12);
    }
}
