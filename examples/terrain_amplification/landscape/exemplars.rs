//! Small, external terrain example atlas. Match local slope/curvature and
//! synthesize only the residual, with overlap blending and protected streams.
use super::{Climate, Grid, Rivers, SEED, blur, hash, smooth};
use std::{fs, io, path::Path};

const CELLS: u16 = 512;

#[derive(Clone, Copy)]
struct Descriptor([f64; 3]);
#[derive(Clone, Copy)]
struct Patch {
    x: u16,
    y: u16,
    orientation: u8,
    descriptor: Descriptor,
}

const fn rotate(o: u8, x: i32, y: i32) -> (i32, i32) {
    match o {
        0 => (x, y),
        1 => (-y, x),
        2 => (-x, -y),
        3 => (y, -x),
        4 => (-x, y),
        5 => (-y, -x),
        6 => (x, -y),
        _ => (y, x),
    }
}

fn offset(g: &Grid, x: u16, y: u16, dx: i32, dy: i32) -> usize {
    let clamp = |v: i32| u16::try_from(v.clamp(0, i32::from(g.cells))).unwrap_or_default();
    g.index(clamp(i32::from(x) + dx), clamp(i32::from(y) + dy))
}
fn describe(
    grid: &Grid,
    heights: &[f64],
    x: u16,
    y: u16,
    orientation: u8,
    radius: u16,
) -> Descriptor {
    let sample = |dx, dy| {
        let (dx, dy) = rotate(orientation, dx, dy);
        heights[offset(grid, x, y, dx, dy)]
    };
    let half = i32::from(radius / 2);
    let west = sample(-half, 0);
    let east = sample(half, 0);
    let south = sample(0, -half);
    let north = sample(0, half);
    let length = f64::from(radius) * grid.spacing;
    Descriptor([
        (east - west) / length,
        (north - south) / length,
        f64::mul_add(west + east + south + north, -0.25, sample(0, 0)) / (length * 0.5),
    ])
}

pub(super) struct Atlas {
    source: Grid,
    residual: Vec<f64>,
    patches: Vec<Patch>,
    radius: u16,
    limit: f64,
}
impl Atlas {
    pub(super) fn read(path: &Path) -> io::Result<Self> {
        Self::read_band(path, 32, 64, 220.0)
    }
    pub(super) fn read_broad(path: &Path) -> io::Result<Self> {
        Self::read_band(path, 64, 256, 550.0)
    }
    fn read_band(path: &Path, radius: u16, passes: u32, limit: f64) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let n = usize::from(CELLS + 1).pow(2);
        if bytes.len() != n * 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected 513x513 LE16 signed elevations, 62.5m spacing",
            ));
        }
        let mut source = Grid::new(CELLS, 62.5);
        for (height, chunk) in source.heights.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *height = f64::from(i16::from_le_bytes(*chunk));
        }
        let broad = blur(&source, &source.heights, passes);
        let residual: Vec<_> = source
            .heights
            .iter()
            .zip(&broad)
            .map(|(h, b)| h - b)
            .collect();
        let mut patches = Vec::new();
        for y in (radius..=CELLS - radius).step_by(usize::from(radius * 3 / 4)) {
            for x in (radius..=CELLS - radius).step_by(usize::from(radius * 3 / 4)) {
                for orientation in 0..8 {
                    patches.push(Patch {
                        x,
                        y,
                        orientation,
                        descriptor: describe(&source, &broad, x, y, orientation, radius),
                    });
                }
            }
        }
        let rms =
            (residual.iter().map(|h| h * h).sum::<f64>() / f64::from(CELLS + 1).powi(2)).sqrt();
        println!(
            "DEM atlas: {} bytes, {} patches, residual RMS {:.1}m",
            bytes.len(),
            patches.len(),
            rms
        );
        Ok(Self {
            source,
            residual,
            patches,
            radius,
            limit,
        })
    }
    pub(super) const fn source(&self) -> &Grid {
        &self.source
    }
    fn choose(&self, target: Descriptor, key: u64) -> Option<Patch> {
        self.patches
            .iter()
            .copied()
            .enumerate()
            .min_by(|(ia, a), (ib, b)| {
                let score = |i: usize, c: Patch| {
                    let [x, y, k] = std::array::from_fn(|j| c.descriptor.0[j] - target.0[j]);
                    let bits = u32::try_from(
                        hash(key ^ u64::try_from(i).unwrap_or_default()) & 0xffff_ffff,
                    )
                    .unwrap_or_default();
                    f64::mul_add(2.0 * k, k, x.mul_add(x, y * y))
                        + 0.045 * f64::from(bits) / f64::from(u32::MAX)
                };
                score(*ia, *a).total_cmp(&score(*ib, *b))
            })
            .map(|(_, p)| p)
    }
    pub(super) fn apply(&self, grid: &mut Grid, climate: &Climate, strength: f64) {
        self.apply_inner(grid, climate, strength, false);
    }
    pub(super) fn apply_constrained(&self, grid: &mut Grid, climate: &Climate, strength: f64) {
        self.apply_inner(grid, climate, strength, true);
    }
    fn apply_inner(&self, grid: &mut Grid, climate: &Climate, strength: f64, constrain: bool) {
        let original = grid.clone();
        let broad = blur(grid, &grid.heights, u32::from(self.radius).pow(2) / 64);
        let radius = i32::from(self.radius);
        let flow = Rivers::route(grid, climate);
        let water = super::hydro::extract(grid, &flow);
        let mut distance: Vec<f64> = grid
            .heights
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                if h <= 0.0 || water.lake_levels[i] > 0.0 || flow.discharge[i] >= 14.0 {
                    0.0
                } else {
                    1.0e9
                }
            })
            .collect();
        for forward in [true, false] {
            for rank in 0..distance.len() {
                let i = if forward {
                    rank
                } else {
                    distance.len() - 1 - rank
                };
                distance[i] = grid
                    .neighbors(i)
                    .fold(distance[i], |v, (j, d)| v.min(distance[j] + d));
            }
        }
        let mut result = vec![0.0; grid.heights.len()];
        let mut weights = vec![0.0; grid.heights.len()];
        for cy in (0..=grid.cells).step_by(usize::from(self.radius)) {
            for cx in (0..=grid.cells).step_by(usize::from(self.radius)) {
                let target = describe(grid, &broad, cx, cy, 0, self.radius);
                let key = hash(SEED ^ u64::from(cx).rotate_left(21) ^ u64::from(cy));
                let Some(patch) = self.choose(target, key) else {
                    continue;
                };
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let (Ok(x), Ok(y)) = (
                            u16::try_from(i32::from(cx) + dx),
                            u16::try_from(i32::from(cy) + dy),
                        ) else {
                            continue;
                        };
                        if x > grid.cells || y > grid.cells {
                            continue;
                        }
                        let (rx, ry) = rotate(patch.orientation, dx, dy);
                        let source = offset(&self.source, patch.x, patch.y, rx, ry);
                        let i = grid.index(x, y);
                        let w = (1.0 - smooth(f64::from(dx.abs()) / f64::from(radius)))
                            * (1.0 - smooth(f64::from(dy.abs()) / f64::from(radius)));
                        result[i] = w.mul_add(self.residual[source], result[i]);
                        weights[i] += w;
                    }
                }
            }
        }
        for i in 0..grid.heights.len() {
            if weights[i] <= 0.0 {
                continue;
            }
            let (x, y) = grid.coordinates(i);
            let [_, rock, arid, _, _] = climate.at(i);
            let h = grid.heights[i];
            let [gx, gy] = super::gradient(&original, x, y);
            let relief = smooth((gx.hypot(gy) / 0.55).min(1.0));
            let amplitude = strength
                * 0.84f64.mul_add(rock.max(relief), 0.16)
                * 0.8f64.mul_add(-arid, 1.0)
                * smooth((h / 600.0).clamp(0.0, 1.0))
                * smooth((distance[i] / 320.0).min(1.0))
                * grid.boundary_weight(x, y);
            grid.heights[i] = amplitude.mul_add(
                (result[i] / weights[i]).clamp(-self.limit, self.limit),
                grid.heights[i],
            );
        }
        if constrain {
            let (corrected, max_lift) =
                super::preserve_depression_depth(grid, &original, &flow.filled);
            println!(
                "drainage constraint corrected {corrected} points, maximum lift {max_lift:.2}m"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transform_preserves_grid_distance() {
        for orientation in 0..8 {
            let (x, y) = rotate(orientation, 7, -11);
            assert_eq!(x * x + y * y, 170);
        }
    }
}
