//! Regional terrain amplification A/B. An experimental heightfield generator,
//! not yet a canonical planet surface or a reproduction of a published solver.
mod directed;
mod landscape;

use std::{
    cmp::Ordering, collections::BinaryHeap, error::Error, fs, io, path::Path, time::Instant,
};

const BASE_CELLS: u16 = 64;
const LEVELS: u32 = 4;
const RADIUS_M: f64 = 600_000.0;
const SEED: u64 = 0x6b65_7262_696e;

#[derive(Clone, Debug)]
struct Grid {
    cells: u16,
    spacing: f64,
    heights: Vec<f64>,
}

impl Grid {
    fn new(cells: u16, spacing: f64) -> Self {
        let n = usize::from(cells) + 1;
        Self {
            cells,
            spacing,
            heights: vec![0.0; n * n],
        }
    }

    fn index(&self, x: u16, y: u16) -> usize {
        usize::from(y) * (usize::from(self.cells) + 1) + usize::from(x)
    }

    fn coordinates(&self, i: usize) -> (u16, u16) {
        // Invariant: i is a valid grid index, cells < u16::MAX.
        let n = usize::from(self.cells) + 1;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "coordinates are bounded by u16 cell count"
        )]
        let result = ((i % n) as u16, (i / n) as u16);
        result
    }

    fn at(&self, x: u16, y: u16) -> f64 {
        self.heights[self.index(x, y)]
    }

    fn boundary_weight(&self, x: u16, y: u16) -> f64 {
        let distance = x.min(y).min(self.cells - x).min(self.cells - y);
        smooth((f64::from(distance) * self.spacing / 4000.0).min(1.0))
    }

    fn neighbors(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let (x, y) = self.coordinates(i);
        let x = i32::from(x);
        let y = i32::from(y);
        (-1_i32..=1).flat_map(move |dy| {
            (-1_i32..=1).filter_map(move |dx| {
                if dx == 0 && dy == 0 {
                    return None;
                }
                let nx = u16::try_from(x + dx).ok()?;
                let ny = u16::try_from(y + dy).ok()?;
                if nx > self.cells || ny > self.cells {
                    return None;
                }
                Some((
                    self.index(nx, ny),
                    self.spacing * f64::from(dx * dx + dy * dy).sqrt(),
                ))
            })
        })
    }
}

fn smooth(t: f64) -> f64 {
    t * t * 2.0_f64.mul_add(-t, 3.0)
}

fn hash(mut key: u64) -> u64 {
    key = (key ^ (key >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    key = (key ^ (key >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    key ^ (key >> 31)
}

// Seeded gradient noise with 8 evenly spaced unit gradients.
fn gradient(seed: u64, x: u16, y: u16, dx: f64, dy: f64) -> f64 {
    const D: f64 = std::f64::consts::FRAC_1_SQRT_2;
    match hash(seed ^ (u64::from(x) << 16) ^ u64::from(y)) & 7 {
        0 => dx,
        1 => dy,
        2 => -dx,
        3 => -dy,
        4 => D * (dx + dy),
        5 => D * (dx - dy),
        6 => D * (dy - dx),
        _ => -D * (dx + dy),
    }
}

fn noise(seed: u64, x: u16, y: u16) -> f64 {
    // One lattice period spans four fine-grid intervals.
    let (ix, iy) = (x / 4, y / 4);
    let (dx, dy) = (f64::from(x % 4) * 0.25, f64::from(y % 4) * 0.25);
    let a = lerp(
        gradient(seed, ix, iy, dx, dy),
        gradient(seed, ix + 1, iy, dx - 1.0, dy),
        smooth(dx),
    );
    let b = lerp(
        gradient(seed, ix, iy + 1, dx, dy - 1.0),
        gradient(seed, ix + 1, iy + 1, dx - 1.0, dy - 1.0),
        smooth(dx),
    );
    lerp(a, b, smooth(dy))
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    (b - a).mul_add(t, a)
}

fn midpoint(a: f64, b: f64, c: f64, d: f64) -> f64 {
    // Cubic interpolation clamped to the enclosing samples (no new extrema).
    ((b + c) * 0.5625 - (a + d) * 0.0625).clamp(b.min(c), b.max(c))
}

fn upsample(parent: &Grid, seed: u64, amplitude: f64) -> Grid {
    let mut child = Grid::new(parent.cells * 2, parent.spacing * 0.5);
    let row_sample = |x: u16, y: u16| {
        let i = x / 2;
        if x.is_multiple_of(2) {
            parent.at(i, y)
        } else {
            midpoint(
                parent.at(i.saturating_sub(1), y),
                parent.at(i, y),
                parent.at(i + 1, y),
                parent.at((i + 2).min(parent.cells), y),
            )
        }
    };
    for y in 0..=child.cells {
        for x in 0..=child.cells {
            let j = y / 2;
            let h = if y.is_multiple_of(2) {
                row_sample(x, j)
            } else {
                midpoint(
                    row_sample(x, j.saturating_sub(1)),
                    row_sample(x, j),
                    row_sample(x, j + 1),
                    row_sample(x, (j + 2).min(parent.cells)),
                )
            };
            let i = child.index(x, y);
            child.heights[i] = h + noise(seed, x, y)
                * amplitude
                * child.boundary_weight(x, y)
                * (h / 100.0).clamp(0.0, 1.0);
        }
    }
    child
}

#[derive(Clone, Copy, Debug)]
struct FloodEntry {
    height: f64,
    index: usize,
}

impl PartialEq for FloodEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for FloodEntry {}
impl PartialOrd for FloodEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FloodEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .height
            .total_cmp(&self.height)
            .then_with(|| other.index.cmp(&self.index))
    }
}

struct Drainage {
    filled: Vec<f64>,
    parent: Vec<usize>,
    order: Vec<usize>,
    area: Vec<f64>,
    slope: Vec<f64>,
}

impl Drainage {
    fn compute(grid: &Grid) -> Self {
        let n = grid.heights.len();
        let mut filled = vec![f64::INFINITY; n];
        let mut parent = vec![usize::MAX; n];
        let mut order = Vec::with_capacity(n);
        let mut queue = BinaryHeap::new();
        // Fixed external outlets and ocean cells. Priority flood gives a
        // coherent spill path even where raw terrain has closed depressions.
        for (i, &height) in grid.heights.iter().enumerate() {
            let (x, y) = grid.coordinates(i);
            if x == 0 || y == 0 || x == grid.cells || y == grid.cells || height <= 0.0 {
                filled[i] = height;
                queue.push(FloodEntry { height, index: i });
            }
        }
        while let Some(FloodEntry { height, index }) = queue.pop() {
            order.push(index);
            for (neighbor, _) in grid.neighbors(index) {
                if filled[neighbor].is_infinite() {
                    filled[neighbor] = height.max(grid.heights[neighbor]);
                    parent[neighbor] = index;
                    queue.push(FloodEntry {
                        height: filled[neighbor],
                        index: neighbor,
                    });
                }
            }
        }
        let mut area = vec![1.0; n];
        let mut slope = vec![0.0; n];
        // Multiple-flow-direction routing across strictly descending filled
        // slopes; flats follow their priority-flood parent. Reverse flood
        // order is topological for both types of edge.
        for &i in order.iter().rev() {
            if parent[i] == usize::MAX {
                continue;
            }
            let weight = |j: usize, distance: f64| {
                let s = ((filled[i] - filled[j]) / distance).max(0.0);
                s * s
            };
            let total: f64 = grid.neighbors(i).map(|(j, d)| weight(j, d)).sum();
            if total > 0.0 {
                let upstream = area[i];
                let mut mean = 0.0;
                for (j, d) in grid.neighbors(i) {
                    let fraction = weight(j, d) / total;
                    area[j] += upstream * fraction;
                    mean += fraction * ((grid.heights[i] - grid.heights[j]) / d).max(0.0);
                }
                slope[i] = mean;
            } else {
                let j = parent[i];
                area[j] += area[i];
            }
        }
        Self {
            filled,
            parent,
            order,
            area,
            slope,
        }
    }

    fn outlet_area(&self) -> f64 {
        self.parent
            .iter()
            .zip(&self.area)
            .filter_map(|(&p, &a)| (p == usize::MAX).then_some(a))
            .sum()
    }
}

fn erode(grid: &mut Grid, steps: u32) {
    for _ in 0..steps {
        let flow = Drainage::compute(grid);
        let mut change = vec![0.0; grid.heights.len()];
        for (i, &height) in grid.heights.iter().enumerate() {
            let (x, y) = grid.coordinates(i);
            let w = grid.boundary_weight(x, y);
            if height <= 0.0 || w == 0.0 {
                continue;
            }
            // Relative stream-power incision with a depth limiter. Discharge
            // is area under spatially uniform rain; sediment leaves the bed.
            // Depressions route runoff but are not automatically incised.
            if flow.filled[i] <= height {
                let incision = (0.035
                    * grid.spacing
                    * (flow.area[i] * 0.25).powf(0.42)
                    * flow.slope[i].sqrt())
                .min(0.2 * grid.spacing);
                change[i] -= w * incision.min(height);
            }
            // Conservative pairwise talus relaxation across land cells;
            // each pair visited once and fixed boundary stays fixed.
            for (j, d) in grid.neighbors(i).filter(|(j, _)| *j > i) {
                if grid.heights[j] <= 0.0 {
                    continue;
                }
                let (jx, jy) = grid.coordinates(j);
                let pair_weight = w.min(grid.boundary_weight(jx, jy));
                let delta = height - grid.heights[j];
                let excess = (delta.abs() - 0.40 * d).max(0.0);
                let flux = pair_weight * 0.05 * excess.copysign(delta);
                change[i] -= flux;
                change[j] += flux;
            }
        }
        for (h, dh) in grid.heights.iter_mut().zip(change) {
            *h += dh;
        }
    }
}

#[derive(Clone, Copy)]
enum Method {
    Noise,
    Erosion,
    Directed,
    Hybrid,
}

fn amplify(base: &Grid, method: Method) -> Grid {
    let mut grid = base.clone();
    for level in 0..LEVELS {
        let amplitude = 0.18 * grid.spacing;
        let seed = hash(SEED ^ u64::from(level));
        let directed =
            matches!(method, Method::Directed) || matches!(method, Method::Hybrid) && level >= 2;
        if directed {
            grid = upsample(&grid, seed, 0.0);
            directed::apply(&mut grid, seed, amplitude);
        } else {
            grid = upsample(&grid, seed, amplitude);
            if matches!(method, Method::Erosion | Method::Hybrid) {
                erode(&mut grid, 6);
            }
        }
    }
    grid
}

struct StockMap {
    values: Vec<u8>,
    width: u16,
    height: u16,
}
impl StockMap {
    fn read(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let invalid = || {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "expected stock 4096x2048 grayscale MapSO",
            )
        };
        if bytes.len() != 16 + 4096 * 2048
            || bytes.get(..16) != Some(&[0, 16, 0, 0, 0, 8, 0, 0, 1, 0, 0, 0, 0, 0, 128, 0])
        {
            return Err(invalid());
        }
        Ok(Self {
            values: bytes[16..].to_vec(),
            width: 4096,
            height: 2048,
        })
    }
    fn at(&self, x: u16, y: u16) -> f64 {
        f64::from(self.values[usize::from(y) * usize::from(self.width) + usize::from(x)])
            * (5000.0 / 255.0)
            - 1500.0
    }
    fn sample(&self, x: f64, y: f64) -> f64 {
        let x = x.rem_euclid(f64::from(self.width));
        let y = y.clamp(0.0, f64::from(self.height - 1));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "coordinates explicitly constrained to u16 map extent"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "coordinates explicitly constrained to u16 map extent"
        )]
        let (ix, iy) = (x.floor() as u16, y.floor() as u16);
        let jx = (ix + 1) % self.width;
        let jy = (iy + 1).min(self.height - 1);
        lerp(
            lerp(self.at(ix, iy), self.at(jx, iy), x.fract()),
            lerp(self.at(ix, jy), self.at(jx, jy), x.fract()),
            y.fract(),
        )
    }
    fn select_region(&self) -> (u16, u16) {
        // Deterministic search for a mountainous inland patch between
        // latitudes -25 and +25 degrees; no image-specific hand tuning.
        let mut best = (f64::NEG_INFINITY, 0, 0);
        for y in (750..1300).step_by(48) {
            for x in (48..self.width - 48).step_by(48) {
                let (mut low, mut high, mut sum, mut count) =
                    (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0);
                for sy in (y - 32..=y + 32).step_by(8) {
                    for sx in (x - 32..=x + 32).step_by(8) {
                        let h = self.at(sx, sy);
                        low = low.min(h);
                        high = high.max(h);
                        sum += h;
                        count += 1.0;
                    }
                }
                let score = if low > 100.0 {
                    (high - low) + sum / count * 0.3
                } else {
                    f64::NEG_INFINITY
                };
                if score > best.0 {
                    best = (score, x, y);
                }
            }
        }
        (best.1, best.2)
    }

    fn region(&self, cx: u16, cy: u16) -> Grid {
        let mut grid = Grid::new(BASE_CELLS, 1000.0);
        let latitude = (f64::from(cy) / f64::from(self.height) - 0.5) * std::f64::consts::PI;
        let step = std::f64::consts::TAU * RADIUS_M / f64::from(self.width);
        // Local tangent metric approximation adequate for this regional
        // comparison; this is NOT the spherical canonical sampler.
        for y in 0..=BASE_CELLS {
            for x in 0..=BASE_CELLS {
                let dx = (f64::from(x) - 32.0) * grid.spacing / (step * latitude.cos());
                let dy = (f64::from(y) - 32.0) * grid.spacing / step;
                let i = grid.index(x, y);
                grid.heights[i] = self.sample(f64::from(cx) + dx, f64::from(cy) + dy);
            }
        }
        grid
    }
}

fn shade(grid: &Grid, x: u16, y: u16) -> [u8; 3] {
    let h = grid.at(x, y);
    let xm = x.saturating_sub(1);
    let xp = (x + 1).min(grid.cells);
    let ym = y.saturating_sub(1);
    let yp = (y + 1).min(grid.cells);
    let gx = (grid.at(xp, y) - grid.at(xm, y)) / (f64::from(xp - xm) * grid.spacing);
    let gy = (grid.at(x, yp) - grid.at(x, ym)) / (f64::from(yp - ym) * grid.spacing);
    let steepness = gx.hypot(gy);
    let illumination = (0.68 + 0.48 * gx - 0.56 * gy) / (1.0 + steepness * steepness).sqrt();
    let rock = smooth(((steepness - 0.15) / 0.65).clamp(0.0, 1.0));
    let alpine = smooth(((h - 2200.0) / 900.0).clamp(0.0, 1.0));
    let base = [
        lerp(87.0, 170.0, rock),
        lerp(119.0, 148.0, rock),
        lerp(72.0, 119.0, rock),
    ];
    let brightness = 0.28 + 0.85 * illumination.max(0.0);
    base.map(|c| {
        let value = if h <= 0.0 {
            70.0
        } else {
            lerp(c, 209.0, alpine) * brightness
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "quantize clamped RGB channel"
        )]
        #[expect(clippy::cast_sign_loss, reason = "quantize clamped RGB channel")]
        let result = value.round().clamp(0.0, 255.0) as u8;
        result
    })
}

fn save_ppm(grid: &Grid, path: &Path) -> io::Result<()> {
    let side = grid.cells + 1;
    let mut bytes = format!("P6\n{side} {side}\n255\n").into_bytes();
    for y in (0..=grid.cells).rev() {
        for x in 0..=grid.cells {
            bytes.extend(shade(grid, x, y));
        }
    }
    fs::write(path, bytes)
}

fn statistics(grid: &Grid, label: &str) {
    let flow = Drainage::compute(grid);
    let mut gradient_sq = 0.0;
    let mut pit_area = 0.0;
    let mut max_flow = 0.0_f64;
    for (i, &h) in grid.heights.iter().enumerate() {
        let (x, y) = grid.coordinates(i);
        let dx = (grid.at((x + 1).min(grid.cells), y) - h) / grid.spacing;
        let dy = (grid.at(x, (y + 1).min(grid.cells)) - h) / grid.spacing;
        gradient_sq += dx * dx + dy * dy;
        if flow.filled[i] > h + 1.0 {
            pit_area += 1.0;
        }
        max_flow = max_flow.max(flow.area[i]);
    }
    let samples = f64::from(grid.cells + 1).powi(2);
    println!(
        "{label}: slope_rms={:.4} filled_area_pct={:.3} largest_drainage_km2={:.2} outlet_balance={:.9} visited={}",
        (gradient_sq / samples).sqrt(),
        100.0 * pit_area / samples,
        max_flow * grid.spacing.powi(2) / 1e6,
        flow.outlet_area() / samples,
        flow.order.len()
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if let [map, output, biome, dem, mode] = args.as_slice() {
        if mode != "natural" {
            return Err("expected natural".into());
        }
        let output = Path::new(output);
        fs::create_dir_all(output)?;
        return landscape::run_natural(
            &StockMap::read(Path::new(map))?,
            Path::new(biome),
            Path::new(dem),
            output,
        );
    }
    if let [map, output, biome, dem] = args.as_slice() {
        let output = Path::new(output);
        fs::create_dir_all(output)?;
        return landscape::run_exemplars(
            &StockMap::read(Path::new(map))?,
            Path::new(biome),
            Path::new(dem),
            output,
        );
    }
    if let [map, output, biome] = args.as_slice() {
        let output = Path::new(output);
        fs::create_dir_all(output)?;
        return landscape::run(&StockMap::read(Path::new(map))?, Path::new(biome), output);
    }
    let [map, output] = args.as_slice() else {
        return Err(
            "usage: terrain_amplification KERBIN_MAP OUTPUT_DIR [BIOME_RGB [DEM_RAW [natural]]]"
                .into(),
        );
    };
    let output = Path::new(output);
    fs::create_dir_all(output)?;
    let stock = StockMap::read(Path::new(map))?;
    let (cx, cy) = stock.select_region();
    let base = stock.region(cx, cy);
    println!(
        "stock center: ({cx},{cy}); 64 km region; {}m initial spacing; seed={SEED:x}",
        base.spacing
    );
    let start = Instant::now();
    let noise = amplify(&base, Method::Noise);
    println!("gradient noise: {:.3}s", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let eroded = amplify(&base, Method::Erosion);
    println!(
        "multiscale incision+talus: {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let directed = amplify(&base, Method::Directed);
    println!(
        "slope-directed packets: {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let hybrid = amplify(&base, Method::Hybrid);
    println!(
        "coarse erosion + directed detail: {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let mut coarse_only = base;
    for _ in 0..LEVELS {
        coarse_only = upsample(&coarse_only, 0, 0.0);
    }
    for (grid, name) in [
        (&coarse_only, "coarse"),
        (&noise, "noise"),
        (&eroded, "erosion"),
        (&directed, "directed"),
        (&hybrid, "hybrid"),
    ] {
        statistics(grid, name);
        save_ppm(grid, &output.join(format!("{name}.ppm")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drainage_conserves_area_and_routes_closed_pit() {
        let mut grid = Grid::new(8, 1.0);
        grid.heights.fill(10.0);
        let pit = grid.index(4, 4);
        grid.heights[pit] = 1.0;
        let flow = Drainage::compute(&grid);
        assert_eq!(flow.order.len(), 81);
        assert!((flow.outlet_area() - 81.0).abs() < 1e-9);
        assert!((flow.filled[pit] - 10.0).abs() < 1e-9);
        assert_ne!(flow.parent[pit], usize::MAX);
    }

    #[test]
    fn erosion_is_repeatable_and_boundary_unchanged() {
        let mut sample = Grid::new(16, 1000.0);
        for y in 0..=sample.cells {
            for x in 0..=sample.cells {
                let i = sample.index(x, y);
                sample.heights[i] = 300.0 + f64::from(x) * 12.0 + noise(SEED, x, y) * 100.0;
            }
        }
        let mut a = sample.clone();
        let mut b = sample.clone();
        erode(&mut a, 5);
        erode(&mut b, 5);
        assert_eq!(a.heights, b.heights);
        for (i, &h) in a.heights.iter().enumerate() {
            let (x, y) = a.coordinates(i);
            assert!(h.is_finite());
            if a.boundary_weight(x, y) == 0.0 {
                assert_eq!(h.to_bits(), sample.heights[i].to_bits());
            }
        }
        assert_ne!(a.heights, sample.heights);
    }
}
