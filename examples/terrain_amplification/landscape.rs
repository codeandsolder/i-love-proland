//! Biome-conditioned regional landscape experiment. The source biome labels
//! are priors, not painted colors or fixed final surface classifications.
mod evolution;
mod exemplars;
mod foothills;
mod hydro;
mod view;

use super::{Drainage, Grid, RADIUS_M, SEED, StockMap, hash, lerp, noise, smooth, upsample};
use std::{error::Error, fs, io, path::Path, time::Instant};

const PALETTE: [[u8; 3]; 11] = [
    [55, 98, 171],
    [131, 188, 46],
    [93, 133, 42],
    [250, 242, 183],
    [167, 167, 167],
    [234, 191, 111],
    [151, 79, 35],
    [199, 143, 223],
    [255, 255, 255],
    [228, 253, 255],
    [216, 216, 216],
];

// rainfall, uplift/rock, aridity, cold, vegetation potential
const PRIORS: [[f64; 5]; 11] = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.06, 0.0, 0.0, 0.9],
    [1.2, 0.65, 0.0, 0.1, 0.7],
    [0.85, 0.0, 0.1, 0.0, 0.4],
    [1.1, 1.0, 0.0, 0.4, 0.3],
    [0.08, 0.15, 1.0, 0.0, 0.03],
    [0.27, 0.5, 0.8, 0.0, 0.15],
    [0.65, 0.1, 0.0, 0.8, 0.25],
    [0.1, 0.2, 0.0, 1.0, 0.0],
    [0.1, 0.0, 0.0, 1.0, 0.0],
    [0.1, 0.0, 0.0, 1.0, 0.0],
];

struct Biomes {
    labels: Vec<u8>,
}
impl Biomes {
    fn read(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() != 4096 * 2048 * 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected 4096x2048 RGB8 stock biome export",
            ));
        }
        let labels = bytes
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| {
                PALETTE
                    .iter()
                    .position(|c| c == p)
                    .and_then(|i| u8::try_from(i).ok())
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, format!("unknown biome {p:?}"))
                    })
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self { labels })
    }

    fn prior(&self, x: f64, y: f64) -> [f64; 5] {
        // Separable 7x7 binomial kernel, ~1.2 source pixels sigma;
        // interpolate category probabilities, never arbitrary palette RGB.
        const W: [f64; 7] = [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0];
        #[expect(
            clippy::cast_possible_truncation,
            reason = "stock-map coordinates fit i32"
        )]
        let ix = x.round() as i32;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "stock-map coordinates fit i32"
        )]
        let iy = y.round() as i32;
        let mut result = [0.0; 5];
        for (dy, wy) in (-3..=3).zip(W) {
            for (dx, wx) in (-3..=3).zip(W) {
                #[expect(
                    clippy::cast_sign_loss,
                    reason = "wrapped/clamped coordinates are nonnegative"
                )]
                let i =
                    (iy + dy).clamp(0, 2047) as usize * 4096 + (ix + dx).rem_euclid(4096) as usize;
                let w = wx * wy / 4096.0;
                for (out, value) in result.iter_mut().zip(PRIORS[usize::from(self.labels[i])]) {
                    *out = w.mul_add(value, *out);
                }
            }
        }
        result
    }
}

fn blur(grid: &Grid, input: &[f64], passes: u32) -> Vec<f64> {
    let mut source = input.to_vec();
    let mut destination = vec![0.0; source.len()];
    for _ in 0..passes {
        for horizontal in [true, false] {
            for (i, v) in destination.iter_mut().enumerate() {
                let (x, y) = grid.coordinates(i);
                let (x, y) = (i32::from(x), i32::from(y));
                *v = (-2..=2)
                    .zip([1.0, 4.0, 6.0, 4.0, 1.0])
                    .map(|(d, w)| {
                        let clamp = |v: i32| {
                            #[expect(
                                clippy::cast_possible_truncation,
                                reason = "clamped to u16 grid extent"
                            )]
                            #[expect(
                                clippy::cast_sign_loss,
                                reason = "clamped to nonnegative extent"
                            )]
                            let v = v.clamp(0, i32::from(grid.cells)) as u16;
                            v
                        };
                        let j = if horizontal {
                            grid.index(clamp(x + d), clamp(y))
                        } else {
                            grid.index(clamp(x), clamp(y + d))
                        };
                        source[j] * w / 16.0
                    })
                    .sum();
            }
            std::mem::swap(&mut source, &mut destination);
        }
    }
    source
}

struct Climate {
    fields: [Grid; 5],
}
impl Climate {
    fn new(map: &StockMap, biomes: &Biomes, center: (u16, u16)) -> Self {
        let mut fields = std::array::from_fn(|_| Grid::new(64, 1000.0));
        let latitude = (f64::from(center.1) / f64::from(map.height) - 0.5) * std::f64::consts::PI;
        let step = std::f64::consts::TAU * RADIUS_M / f64::from(map.width);
        for y in 0..=64 {
            for x in 0..=64 {
                let dx = (f64::from(x) - 32.0) * 1000.0 / (step * latitude.cos());
                let dy = (f64::from(y) - 32.0) * 1000.0 / step;
                for (field, val) in fields
                    .iter_mut()
                    .zip(biomes.prior(f64::from(center.0) + dx, f64::from(center.1) + dy))
                {
                    let i = field.index(x, y);
                    field.heights[i] = val;
                }
            }
        }
        // Additional blur on the coarse physical fields smooths biome class
        // boundaries without flattening the elevation source.
        for field in &mut fields {
            field.heights = blur(field, &field.heights, 2);
        }
        Self { fields }
    }
    fn refine(&mut self) {
        for field in &mut self.fields {
            *field = upsample(field, 0, 0.0);
        }
    }
    fn at(&self, i: usize) -> [f64; 5] {
        std::array::from_fn(|c| self.fields[c].heights[i])
    }
}

/// Raising each modified cell to its new spill envelope minus the original
/// depression depth guarantees that the final depression cannot be deeper.
/// Raising to at most the current envelope cannot increase that envelope.
fn preserve_depression_depth(
    grid: &mut Grid,
    original: &Grid,
    original_filled: &[f64],
) -> (usize, f64) {
    let after = Drainage::compute(grid);
    let mut corrected = 0;
    let mut max_lift = 0.0_f64;
    for (index, height) in grid.heights.iter_mut().enumerate() {
        if original.heights[index] <= 0.0 {
            continue;
        }
        let old_depth = (original_filled[index] - original.heights[index]).max(0.0);
        let low = after.filled[index] - old_depth;
        let lift = (low - *height).max(0.0);
        if lift > 1e-8 {
            corrected += 1;
            max_lift = max_lift.max(lift);
        }
        *height = height.max(low);
    }
    (corrected, max_lift)
}

fn initial_relief(base: &Grid, climate: &Climate, strength: f64) -> Grid {
    let local = blur(base, &base.heights, 1);
    let broad = blur(base, &local, 12);
    let mut result = base.clone();
    for (i, h) in result.heights.iter_mut().enumerate() {
        let (x, y) = base.coordinates(i);
        let original = base.heights[i];
        let [_, mountain, _, _, _] = climate.at(i);
        let land = smooth((original / 150.0).clamp(0.0, 1.0));
        let highland = smooth(((original - 400.0) / 1300.0).clamp(0.0, 1.0));
        let weight = base.boundary_weight(x, y) * land;
        // Counter broad rounded stock summits; coherent uplift amplifies
        // relief on mountain belts, not everywhere with random spikes.
        let uplift =
            strength * 0.8f64.mul_add(mountain.max(highland), 0.2) * (local[i] - broad[i]).max(0.0);
        *h = weight.mul_add(local[i] - original + uplift, original);
    }
    result
}

struct Rivers {
    receiver: Vec<usize>,
    discharge: Vec<f64>,
    filled: Vec<f64>,
    order: Vec<usize>,
}
impl Rivers {
    fn route(grid: &Grid, climate: &Climate) -> Self {
        let drainage = Drainage::compute(grid);
        let mut receiver = drainage.parent;
        let mut discharge: Vec<_> = (0..grid.heights.len())
            .map(|i| climate.at(i)[0] * grid.spacing.powi(2) / 1e6)
            .collect();
        for &i in drainage.order.iter().rev() {
            if receiver[i] == usize::MAX {
                continue;
            }
            // Route the actual river through one receiver (steepest descent
            // on the spill surface); keep diffuse MFD out of the river map.
            let mut steepest = 0.0;
            for (j, d) in grid.neighbors(i) {
                let slope = (drainage.filled[i] - drainage.filled[j]) / d;
                if slope > steepest {
                    steepest = slope;
                    receiver[i] = j;
                }
            }
            discharge[receiver[i]] += discharge[i];
        }
        Self {
            receiver,
            discharge,
            filled: drainage.filled,
            order: drainage.order,
        }
    }
}

fn evolve(grid: &mut Grid, climate: &Climate, steps: usize) {
    for _ in 0..steps {
        let rivers = Rivers::route(grid, climate);
        let mut bed = grid.heights.clone();
        // Implicit downstream-first stream-power update, n=1. Cap the
        // difference per geomorphic step; no subtraction on the ocean floor.
        // Water surfaces/depressions are routed but not blindly trenched.
        for &i in &rivers.order {
            let j = rivers.receiver[i];
            let height = grid.heights[i];
            if j == usize::MAX || height <= 0.0 {
                continue;
            }
            let (x, y) = grid.coordinates(i);
            let [_, rock, arid, _, _] = climate.at(i);
            let distance = {
                let (jx, jy) = grid.coordinates(j);
                grid.spacing * f64::from(x.abs_diff(jx)).hypot(f64::from(y.abs_diff(jy)))
            };
            let k = (260.0 / distance)
                * rivers.discharge[i].powf(0.42)
                * 0.45f64.mul_add(-rock, 1.0)
                * 0.35f64.mul_add(-arid, 1.0);
            let downstream = bed[j].max(0.0);
            if height > downstream {
                let lowered = k.mul_add(downstream, height) / (1.0 + k);
                bed[i] = (height - lowered)
                    .min(grid.spacing * 0.16)
                    .mul_add(-grid.boundary_weight(x, y), height);
            }
        }
        // Widen incision into a valley instead of leaving a one-pixel trench.
        // Mix narrow channel incision with nearby hillslope response; blend
        // armored rock less than softer lowland soils.
        let incision: Vec<_> = grid.heights.iter().zip(&bed).map(|(a, b)| a - b).collect();
        let broad = blur(grid, &incision, 2);
        let smoothed = blur(grid, &bed, 1);
        for i in 0..grid.heights.len() {
            let (x, y) = grid.coordinates(i);
            let [_, rock, _, _, _] = climate.at(i);
            let w = smooth((grid.heights[i] / 100.0).clamp(0.0, 1.0));
            let border = x.min(y).min(grid.cells - x).min(grid.cells - y);
            let edge = smooth((f64::from(border) * grid.spacing / 4000.0).min(1.0));
            let incision = 0.35_f64.mul_add(incision[i], 0.65 * broad[i]);
            let diffusion = (smoothed[i] - bed[i]) * 0.15f64.mul_add(-rock, 0.22);
            grid.heights[i] = (edge * w).mul_add(diffusion - incision, grid.heights[i]);
        }
    }
}

// Distance to an established stream network is a useful ridge prior:
// the high ground should sit between valleys, not atop arbitrary noise bumps.
fn divide_uplift(grid: &mut Grid, climate: &Climate, level: u32) {
    let flow = Rivers::route(grid, climate);
    let threshold = 1.5 / 2.0_f64.powi(i32::try_from(level).unwrap_or(0));
    let mut distance: Vec<f64> = grid
        .heights
        .iter()
        .zip(&flow.discharge)
        .map(|(&h, &a)| {
            if h <= 0.0 || a >= threshold {
                0.0
            } else {
                1.0e6
            }
        })
        .collect();
    // Two-pass 8-neighbor chamfer approximation to Euclidean distance.
    for forward in [true, false] {
        for rank in 0..distance.len() {
            let i = if forward {
                rank
            } else {
                distance.len() - 1 - rank
            };
            let value = grid
                .neighbors(i)
                .fold(distance[i], |v, (j, d)| v.min(distance[j] + d));
            distance[i] = value;
        }
    }
    let broad = blur(grid, &grid.heights, 3);
    for i in 0..grid.heights.len() {
        let (x, y) = grid.coordinates(i);
        let h = grid.heights[i];
        let mountain = climate.at(i)[1];
        let high = smooth(((h - 500.0) / 1450.0).clamp(0.0, 1.0));
        let ridge = smooth(((h - broad[i] + 90.0) / 220.0).clamp(0.0, 1.0));
        let factor = 0.7f64.mul_add(high, 0.3 * mountain) * ridge * grid.boundary_weight(x, y);
        grid.heights[i] =
            (factor * 0.55).mul_add(distance[i].min(3.0 * grid.spacing), grid.heights[i]);
    }
}

fn generate(base: &Grid, mut climate: Climate, strength: f64, divides: bool) -> (Grid, Climate) {
    let mut grid = initial_relief(base, &climate, strength);
    for level in 0_u32..4 {
        let spacing = grid.spacing;
        grid = upsample(&grid, 0, 0.0);
        climate.refine();
        // Modest roughness, confined to appropriate rock/relief. The river
        // process, not noise, should be responsible for valley structure.
        for i in 0..grid.heights.len() {
            let (x, y) = grid.coordinates(i);
            let [_, mountain, arid, _, _] = climate.at(i);
            let h = grid.heights[i];
            let roughness = 0.025f64.mul_add(arid, 0.065f64.mul_add(mountain, 0.018));
            grid.heights[i] = (roughness
                * spacing
                * noise(hash(SEED ^ u64::from(level)), x, y)
                * smooth((h / 120.0).clamp(0.0, 1.0)))
            .mul_add(grid.boundary_weight(x, y), grid.heights[i]);
        }
        if divides && level < 3 {
            divide_uplift(&mut grid, &climate, level);
        }
        if level < 3 {
            evolve(&mut grid, &climate, if level == 0 { 12 } else { 5 });
        }
    }
    (grid, climate)
}

fn coast_region(stock: &StockMap) -> (u16, u16) {
    let mut best = (f64::NEG_INFINITY, 0, 0);
    for y in (750..1300).step_by(32) {
        for x in (48..stock.width - 48).step_by(32) {
            let mut sea = 0;
            let mut high = 0.0_f64;
            for sy in (y - 32..=y + 32).step_by(8) {
                for sx in (x - 32..=x + 32).step_by(8) {
                    let h = stock.at(sx, sy);
                    if h <= 0.0 {
                        sea += 1;
                    }
                    high = high.max(h);
                }
            }
            if (8..=30).contains(&sea) {
                let score = f64::mul_add(f64::from((sea - 18_i32).abs()), -20.0, high);
                if score > best.0 {
                    best = (score, x, y);
                }
            }
        }
    }
    (best.1, best.2)
}

fn gradient(grid: &Grid, x: u16, y: u16) -> [f64; 2] {
    let xm = x.saturating_sub(1);
    let xp = (x + 1).min(grid.cells);
    let ym = y.saturating_sub(1);
    let yp = (y + 1).min(grid.cells);
    [
        (grid.at(xp, y) - grid.at(xm, y)) / (f64::from(xp - xm) * grid.spacing),
        (grid.at(x, yp) - grid.at(x, ym)) / (f64::from(yp - ym) * grid.spacing),
    ]
}

struct Surface {
    colors: Vec<[f64; 3]>,
    water: Vec<f64>,
    water_level: Vec<f64>,
    cover: Vec<[u8; 3]>,
    hydro: hydro::Water,
    river_cells: usize,
    lake_cells: usize,
}
fn materials(grid: &Grid, climate: &Climate) -> Surface {
    let flow = Rivers::route(grid, climate);
    let mut colors = Vec::with_capacity(grid.heights.len());
    let mut water = Vec::with_capacity(grid.heights.len());
    let mut water_level = Vec::with_capacity(grid.heights.len());
    let hydro = hydro::extract(grid, &flow);
    let mut cover = Vec::with_capacity(grid.heights.len());
    let mut river_cells = 0;
    let mut lake_cells = 0;
    let broad_heights = blur(grid, &grid.heights, 12);
    // Persistent water coverage derived from connected drainage, not biome
    // color. Current grid is too coarse for meter-wide streams, so this
    // map represents pixel-area coverage (not a full water mesh).
    for (i, &h) in grid.heights.iter().enumerate() {
        let (x, y) = grid.coordinates(i);
        let [gx, gy] = gradient(grid, x, y);
        let slope = gx.hypot(gy);
        let [rain, mountain, arid, cold, veg] = climate.at(i);
        let moist = smooth((rain / 1.2).clamp(0.0, 1.0));
        let variation = noise(SEED ^ 9, x / 4, y / 4).mul_add(0.12, noise(SEED ^ 8, x, y) * 0.06);
        let rock = smooth((variation.mul_add(0.18, slope - 0.20) / 0.70).clamp(0.0, 1.0));
        let talus = smooth(((broad_heights[i] - h) / 70.0).clamp(0.0, 1.0))
            * smooth((slope / 0.45).clamp(0.0, 1.0))
            * (1.0 - rock);
        let snowline = 280.0f64.mul_add(gy.clamp(-1.0, 1.0), 900.0f64.mul_add(-cold, 3100.0));
        let snow = smooth((variation.mul_add(450.0, h - snowline) / 500.0).clamp(0.0, 1.0))
            * 0.65f64.mul_add(-rock, 1.0);
        let floodplain =
            smooth((flow.discharge[i] / 25.0).min(1.0)) * (1.0 - smooth((slope / 0.16).min(1.0)));
        let forest = smooth(
            (veg * moist)
                .mul_add(0.6f64.mul_add(-mountain, 1.0), variation)
                .clamp(0.0, 1.0),
        );
        let grass = [0.37, 0.43, 0.25];
        let forest_color = [0.19, 0.28, 0.14];
        let dry = [0.54, 0.40, 0.23];
        let stone = [0.46, 0.43, 0.40];
        let wet = [0.29, 0.40, 0.20];
        let color = std::array::from_fn(|c| {
            let vegetated = lerp(grass[c], forest_color[c], forest);
            let soil = lerp(lerp(vegetated, dry[c], arid), wet[c], floodplain * 0.6);
            let rocky = lerp(
                lerp(soil, [0.46, 0.425, 0.34][c], talus * 0.45),
                stone[c] * (1.0 + variation),
                rock,
            );
            (lerp(rocky, [0.82, 0.86, 0.87][c], snow) * (1.0 + variation)).clamp(0.0, 1.0)
        });
        let lake = hydro.lake_levels[i] > 0.0;
        let river = h > 0.0 && !lake && hydro.river[i] > 0.0;
        river_cells += usize::from(river);
        lake_cells += usize::from(lake);
        let coverage = if h <= 0.0 || lake {
            1.0
        } else {
            hydro.river[i]
        };
        let class = if h <= 0.0 || lake || coverage > 0.65 {
            [43, 105, 147]
        } else if snow > 0.50 {
            [230, 240, 245]
        } else if rock > 0.50 {
            [165, 153, 143]
        } else if arid > 0.50 {
            [201, 161, 91]
        } else if floodplain > 0.15 {
            [69, 160, 127]
        } else if forest > 0.50 {
            [37, 100, 65]
        } else if cold > 0.6 {
            [158, 142, 172]
        } else {
            [121, 169, 78]
        };
        cover.push(class);
        colors.push(color);
        water.push(coverage);
        water_level.push(hydro.lake_levels[i]);
    }
    Surface {
        colors,
        water,
        water_level,
        cover,
        hydro,
        river_cells,
        lake_cells,
    }
}

fn rgb(value: f64) -> u8 {
    #[expect(clippy::cast_possible_truncation, reason = "bounded RGB sample")]
    #[expect(clippy::cast_sign_loss, reason = "bounded nonnegative RGB sample")]
    let result = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    result
}
fn light(grid: &Grid, x: u16, y: u16) -> f64 {
    let [gx, gy] = gradient(grid, x, y);
    let diffuse = 0.56f64.mul_add(-gy, 0.48f64.mul_add(gx, 0.67))
        / f64::mul_add(gy, gy, f64::mul_add(gx, gx, 1.0)).sqrt();
    // Approximate terrain self-shadow along sun direction using exponentially
    // spaced samples. Soft transition avoids integer-grid shadow streaks.
    let h = grid.at(x, y);
    let mut horizon = 0.0_f64;
    for d in [2_i32, 4, 8, 16, 32, 64, 128] {
        let sx = i32::from(x) - d * 3 / 4;
        let sy = i32::from(y) + d;
        if sx < 0 || sy > i32::from(grid.cells) {
            break;
        }
        let (Ok(sx), Ok(sy)) = (u16::try_from(sx), u16::try_from(sy)) else {
            break;
        };
        horizon = horizon.max((grid.at(sx, sy) - h) / (f64::from(d) * grid.spacing * 1.25));
    }
    let shadow = 1.0 - smooth(((horizon - 0.82) / 0.22).clamp(0.0, 1.0));
    f64::mul_add(0.70 * diffuse.max(0.0), shadow, 0.52)
}
fn save(grid: &Grid, surface: &Surface, path: &Path, water_enabled: bool) -> io::Result<()> {
    let side = grid.cells + 1;
    let mut data = format!("P6\n{side} {side}\n255\n").into_bytes();
    for y in (0..=grid.cells).rev() {
        for x in 0..=grid.cells {
            let i = grid.index(x, y);
            let illum = light(grid, x, y);
            let water = if water_enabled { surface.water[i] } else { 0.0 };
            let color: [u8; 3] = std::array::from_fn(|c| {
                rgb(lerp(
                    surface.colors[i][c] * illum,
                    [0.095, 0.34, 0.42][c] * 0.15f64.mul_add(illum, 0.85),
                    water,
                ))
            });
            data.extend(color);
        }
    }
    fs::write(path, data)
}

fn export_layers(grid: &Grid, surface: &Surface, output: &Path, label: &str) -> io::Result<()> {
    let side = grid.cells + 1;
    let mut albedo = format!("P6\n{side} {side}\n255\n").into_bytes();
    let mut landcover = albedo.clone();
    let mut hydro = albedo.clone();
    for y in (0..=grid.cells).rev() {
        for x in 0..=grid.cells {
            let i = grid.index(x, y);
            let color: [u8; 3] = std::array::from_fn(|c| {
                rgb(lerp(
                    surface.colors[i][c],
                    [0.095, 0.34, 0.42][c],
                    surface.water[i],
                ))
            });
            albedo.extend(color);
            landcover.extend(surface.cover[i]);
            let water = surface.water[i];
            let shade = 0.35f64.mul_add(light(grid, x, y), 0.30);
            let color: [u8; 3] =
                std::array::from_fn(|c| rgb(lerp(shade, [0.12, 0.55, 0.90][c], water)));
            hydro.extend(color);
        }
    }
    fs::write(output.join(format!("{label}-albedo.ppm")), albedo)?;
    fs::write(output.join(format!("{label}-cover.ppm")), landcover)?;
    fs::write(output.join(format!("{label}-hydrology.ppm")), hydro)?;
    hydro::save_graph(
        grid,
        &surface.hydro,
        &output.join(format!("{label}-rivers.csv")),
    )
}

pub fn run(stock: &StockMap, biome: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let biomes = Biomes::read(biome)?;
    for (label, center) in [
        ("mountains", stock.select_region()),
        ("coast", coast_region(stock)),
    ] {
        let base = stock.region(center.0, center.1);
        let mut raw = base.clone();
        let mut climate = Climate::new(stock, &biomes, center);
        for _ in 0..4 {
            raw = upsample(&raw, 0, 0.0);
            climate.refine();
        }
        let camera = view::Camera::choose(&raw);
        let source_surface = materials(&raw, &climate);
        save(
            &raw,
            &source_surface,
            &output.join(format!("{label}-source.ppm")),
            true,
        )?;
        view::render(
            &raw,
            &source_surface,
            &camera,
            &output.join(format!("{label}-source-flight.ppm")),
        )?;
        println!("{label} center={center:?}");
        for (kind, strength, divides) in [
            ("grounded", 0.75, false),
            ("alpine", 1.7, false),
            ("divides", 1.7, true),
            ("tectonic", 1.0, false),
            ("tectonic-strong", 1.65, false),
        ] {
            let start = Instant::now();
            let climate = Climate::new(stock, &biomes, center);
            let (grid, climate) = if kind.starts_with("tectonic") {
                evolution::generate(&base, climate, strength)
            } else {
                generate(&base, climate, strength, divides)
            };
            println!(
                "{label}-{kind}: generation {:.3}s",
                start.elapsed().as_secs_f64()
            );
            let surface = materials(&grid, &climate);
            let high = grid
                .heights
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            println!(
                "max elevation {high:.1} m; river cells {}, lake cells {}",
                surface.river_cells, surface.lake_cells
            );
            println!("connected lakes {}", surface.hydro.lakes);
            super::statistics(&grid, &format!("{label}-{kind}"));
            if kind.starts_with("tectonic") {
                export_layers(&grid, &surface, output, &format!("{label}-{kind}"))?;
            }
            save(
                &grid,
                &surface,
                &output.join(format!("{label}-{kind}.ppm")),
                true,
            )?;
            view::render(
                &grid,
                &surface,
                &camera,
                &output.join(format!("{label}-{kind}-flight.ppm")),
            )?;
        }
    }
    Ok(())
}

pub fn run_exemplars(
    stock: &StockMap,
    biome: &Path,
    dem: &Path,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let biomes = Biomes::read(biome)?;
    let atlas = exemplars::Atlas::read(dem)?;
    let reference = atlas.source();
    let reference_climate = Climate {
        fields: std::array::from_fn(|channel| {
            let mut g = Grid::new(reference.cells, reference.spacing);
            g.heights.fill(PRIORS[4][channel]);
            g
        }),
    };
    super::statistics(reference, "alps-reference");
    let reference_surface = materials(reference, &reference_climate);
    save(
        reference,
        &reference_surface,
        &output.join("alps-reference.ppm"),
        true,
    )?;
    view::render(
        reference,
        &reference_surface,
        &view::Camera::choose(reference),
        &output.join("alps-reference-flight.ppm"),
    )?;
    for (label, center) in [
        ("mountains", stock.select_region()),
        ("coast", coast_region(stock)),
    ] {
        let base = stock.region(center.0, center.1);
        let mut raw = base.clone();
        for _ in 0..4 {
            raw = upsample(&raw, 0, 0.0);
        }
        let camera = view::Camera::choose(&raw);
        let start = Instant::now();
        let (grid, climate) =
            evolution::generate(&base, Climate::new(stock, &biomes, center), 0.75);
        println!(
            "{label}-moderate generation {:.3}s",
            start.elapsed().as_secs_f64()
        );
        for (name, strength) in [
            ("moderate", 0.0),
            ("exemplar", 1.0),
            ("exemplar-strong", 1.65),
        ] {
            let mut variant = grid.clone();
            let start = Instant::now();
            if strength > 0.0 {
                atlas.apply(&mut variant, &climate, strength);
            }
            println!(
                "{label}-{name} amplification {:.3}s",
                start.elapsed().as_secs_f64()
            );
            let surface = materials(&variant, &climate);
            let tag = format!("{label}-{name}");
            super::statistics(&variant, &tag);
            println!(
                "{tag} lakes={} river_cells={} max_elevation={:.1}",
                surface.hydro.lakes,
                surface.river_cells,
                variant
                    .heights
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            );
            save(&variant, &surface, &output.join(format!("{tag}.ppm")), true)?;
            view::render(
                &variant,
                &surface,
                &camera,
                &output.join(format!("{tag}-flight.ppm")),
            )?;
            if name == "exemplar" {
                export_layers(&variant, &surface, output, &tag)?;
            }
        }
    }
    Ok(())
}

pub fn run_natural(
    stock: &StockMap,
    biome: &Path,
    dem: &Path,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let biomes = Biomes::read(biome)?;
    let atlas = exemplars::Atlas::read_broad(dem)?;
    for (label, center) in [
        ("mountains", stock.select_region()),
        ("coast", coast_region(stock)),
    ] {
        let base = stock.region(center.0, center.1);
        let mut raw = base.clone();
        for _ in 0..4 {
            raw = upsample(&raw, 0, 0.0);
        }
        let camera = view::Camera::choose(&raw);
        for (name, uplift, amplitude) in [("gentle-natural", 0.0, 1.75), ("natural", 0.45, 1.15)] {
            let start = Instant::now();
            let climate = Climate::new(stock, &biomes, center);
            let (mut grid, climate) = if uplift == 0.0 {
                generate(&base, climate, 1.45, false)
            } else {
                evolution::generate(&base, climate, uplift)
            };
            println!(
                "{label}-{name} generation {:.3}s",
                start.elapsed().as_secs_f64()
            );
            let before = grid.clone();
            let start = Instant::now();
            atlas.apply_constrained(&mut grid, &climate, amplitude);
            println!(
                "{label}-{name} amplification {:.3}s",
                start.elapsed().as_secs_f64()
            );
            let old = Drainage::compute(&before);
            let new = Drainage::compute(&grid);
            let excess = new
                .filled
                .iter()
                .zip(&grid.heights)
                .zip(old.filled.iter().zip(&before.heights))
                .map(|((a, b), (c, d))| ((a - b) - (c - d)).max(0.0))
                .fold(0.0_f64, f64::max);
            println!("maximum new depression depth beyond original {excess:.6}m");
            let surface = materials(&grid, &climate);
            let tag = format!("{label}-{name}");
            super::statistics(&grid, &tag);
            println!(
                "{tag} lakes={} river_cells={} max_elevation={:.1}",
                surface.hydro.lakes,
                surface.river_cells,
                grid.heights
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            );
            save(&grid, &surface, &output.join(format!("{tag}.ppm")), true)?;
            view::render(
                &grid,
                &surface,
                &camera,
                &output.join(format!("{tag}-flight.ppm")),
            )?;
            export_layers(&grid, &surface, output, &tag)?;
        }
    }
    Ok(())
}

fn export_region(
    grid: &Grid,
    surface: &Surface,
    channels: &hydro::weathering::ChannelField,
    stock: &StockMap,
    center: (u16, u16),
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    use i_love_proland::terrain::{Region, RegionCell, RegionChart};
    use std::io::Write;
    let latitude = (f64::from(center.1) / f64::from(stock.height) - 0.5) * std::f64::consts::PI;
    let longitude = ((1.75 - f64::from(center.0) / f64::from(stock.width)) * std::f64::consts::TAU)
        .rem_euclid(std::f64::consts::TAU)
        - std::f64::consts::PI;
    let chart = RegionChart {
        radius_m: RADIUS_M,
        latitude_rad: latitude,
        longitude_rad: longitude,
        spacing_m: grid.spacing,
        cells: grid.cells,
    };
    let cells = grid
        .heights
        .iter()
        .zip(&surface.water_level)
        .zip(&surface.colors)
        .map(|((&h, &lake), &rgb)| RegionCell::new(h, (lake > 0.0).then_some(lake), rgb))
        .collect::<io::Result<Vec<_>>>()?;
    let region = Region::new(chart, cells, channels.reaches().to_vec())?;
    let now = Instant::now();
    let mut sum = 0.0;
    for y in 0..=grid.cells {
        for x in 0..=grid.cells {
            let p = [f64::from(x) * grid.spacing, f64::from(y) * grid.spacing];
            let d = chart.direction(p);
            if let Some(sample) = region.sample(std::hint::black_box(d)) {
                sum += sample.weather.bed;
            }
        }
    }
    println!(
        "regional direction query: {:.3}s for {} samples, checksum {sum:.3}",
        now.elapsed().as_secs_f64(),
        grid.heights.len()
    );
    let mut file = io::BufWriter::new(fs::File::create(path)?);
    region.write(&mut file)?;
    file.flush()?;
    println!(
        "region artifact {} bytes, {} reaches",
        fs::metadata(path)?.len(),
        region.reaches().len()
    );
    Ok(())
}

pub fn run_sharp(
    stock: &StockMap,
    biome: &Path,
    dem: &Path,
    output: &Path,
    full: bool,
    weathered: bool,
) -> Result<(), Box<dyn Error>> {
    let biomes = Biomes::read(biome)?;
    let atlas = exemplars::Atlas::read_broad(dem)?;
    for (label, center) in [
        ("mountains", stock.select_region()),
        ("coast", coast_region(stock)),
    ] {
        let base = stock.region(center.0, center.1);
        let mut raw = base.clone();
        for _ in 0..4 {
            raw = upsample(&raw, 0, 0.0);
        }
        let camera = view::Camera::choose(&raw);
        let (mut grid, climate) =
            generate(&base, Climate::new(stock, &biomes, center), 1.45, false);
        atlas.apply_constrained(&mut grid, &climate, 1.75);
        if weathered {
            foothills::blend(&mut grid, &climate);
        }
        let surface = materials(&grid, &climate);
        super::statistics(&grid, label);
        let start = Instant::now();
        if weathered {
            let flow = Rivers::route(&grid, &climate);
            let channels = hydro::weathering::build(&grid, &flow)?;
            export_region(
                &grid,
                &surface,
                &channels,
                stock,
                center,
                &output.join(format!("{label}.plterrain")),
            )?;
            view::compare_weathering(
                &grid,
                &surface,
                &camera,
                atlas.source(),
                &channels,
                output,
                label,
            )?;
        } else {
            view::compare_shading(
                &grid,
                &surface,
                &camera,
                output,
                label,
                atlas.source(),
                full,
            )?;
        }
        println!(
            "{label}: reference renders {:.3}s",
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blur_preserves_constant() {
        let grid = Grid::new(4, 100.0);
        assert_eq!(blur(&grid, &[17.0; 25], 3), vec![17.0; 25]);
    }
    #[test]
    fn rivers_are_topological_and_conserve_discharge() {
        let mut grid = Grid::new(8, 1000.0);
        for i in 0..grid.heights.len() {
            grid.heights[i] = 100.0 + f64::from(u16::try_from(i % 9).unwrap()) * 100.0;
        }
        let climate = Climate {
            fields: std::array::from_fn(|_| {
                let mut g = Grid::new(8, 1000.0);
                g.heights.fill(1.0);
                g
            }),
        };
        let flow = Rivers::route(&grid, &climate);
        let mut rank = vec![0; 81];
        for (i, &node) in flow.order.iter().enumerate() {
            rank[node] = i;
        }
        let mut total = 0.0;
        for (i, &j) in flow.receiver.iter().enumerate() {
            if j == usize::MAX {
                total += flow.discharge[i];
            } else {
                assert!(rank[j] < rank[i]);
            }
        }
        assert_eq!(total, 81.0);
    }
}
