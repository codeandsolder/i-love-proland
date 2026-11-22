//! Regional water features extracted from the generated spill surface and
//! connected drainage network. This is a geometric preview, not fluid dynamics.
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

fn splat(grid: &Grid, a: [f64; 2], b: [f64; 2], width_m: f64, mask: &mut [f64]) {
    let r = width_m / (2.0 * grid.spacing);
    let reach = r + 1.0;
    let d = [b[0] - a[0], b[1] - a[1]];
    let length_sq = d[0].mul_add(d[0], d[1] * d[1]).max(1e-9);
    #[expect(clippy::cast_possible_truncation, reason = "bounded grid-space box")]
    let (xmin, xmax, ymin, ymax) = (
        (a[0].min(b[0]) - reach).floor() as i32,
        (a[0].max(b[0]) + reach).ceil() as i32,
        (a[1].min(b[1]) - reach).floor() as i32,
        (a[1].max(b[1]) + reach).ceil() as i32,
    );
    for y in ymin.max(0)..=ymax.min(i32::from(grid.cells)) {
        for x in xmin.max(0)..=xmax.min(i32::from(grid.cells)) {
            let p = [f64::from(x) - a[0], f64::from(y) - a[1]];
            let t = (p[0].mul_add(d[0], p[1] * d[1]) / length_sq).clamp(0.0, 1.0);
            let dist = (p[0] - t * d[0]).hypot(p[1] - t * d[1]);
            let coverage = (reach - dist).clamp(0.0, 1.0) * (2.0 * r).min(1.0);
            if let (Ok(x), Ok(y)) = (u16::try_from(x), u16::try_from(y)) {
                let i = grid.index(x, y);
                mask[i] = mask[i].max(coverage);
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
