//! Small deterministic reference renderer for evaluating relief from a cockpit
//! instead of relying exclusively on overhead hillshades. Not the game renderer.
use super::{Grid, Surface, lerp, light, rgb};
use std::{fs, io, path::Path};

type V3 = [f64; 3];
const WIDTH: u16 = 800;
const HEIGHT: u16 = 450;

fn dot(a: V3, b: V3) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn normalize(a: V3) -> V3 {
    let n = dot(a, a).sqrt();
    a.map(|x| x / n)
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn sub(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] - b[i])
}

struct Sample {
    ids: [usize; 4],
    tx: f64,
    ty: f64,
}
impl Sample {
    fn at(grid: &Grid, x: f64, y: f64) -> Option<Self> {
        let gx = x / grid.spacing;
        let gy = y / grid.spacing;
        if !(0.0..f64::from(grid.cells)).contains(&gx)
            || !(0.0..f64::from(grid.cells)).contains(&gy)
        {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "grid coordinates in validated u16 range"
        )]
        #[expect(clippy::cast_sign_loss, reason = "grid coordinates nonnegative")]
        let ix = gx.floor() as u16;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "grid coordinates in validated u16 range"
        )]
        #[expect(clippy::cast_sign_loss, reason = "grid coordinates nonnegative")]
        let iy = gy.floor() as u16;
        Some(Self {
            ids: [
                grid.index(ix, iy),
                grid.index(ix + 1, iy),
                grid.index(ix, iy + 1),
                grid.index(ix + 1, iy + 1),
            ],
            tx: gx.fract(),
            ty: gy.fract(),
        })
    }
    fn blend(&self, f: impl Fn(usize) -> f64) -> f64 {
        let [a, b, c, d] = self.ids.map(f);
        lerp(lerp(a, b, self.tx), lerp(c, d, self.tx), self.ty)
    }
    fn elevation(&self, grid: &Grid, surface: &Surface) -> f64 {
        self.blend(|i| grid.heights[i].max(surface.water_level[i]).max(0.0))
    }
}

pub(super) struct Camera {
    eye: V3,
    forward: V3,
    right: V3,
    up: V3,
}
impl Camera {
    pub(super) fn choose(grid: &Grid) -> Self {
        let limit = grid.cells / 6;
        let mut peak = (f64::NEG_INFINITY, grid.cells / 2, grid.cells / 2);
        for y in limit..grid.cells - limit {
            for x in limit..grid.cells - limit {
                let h = grid.at(x, y);
                if h > peak.0 {
                    peak = (h, x, y);
                }
            }
        }
        // Choose a low position around the mountain, looking at the peak.
        let mut valley = (f64::INFINITY, 0.0, 0.0);
        let px = f64::from(peak.1) * grid.spacing;
        let py = f64::from(peak.2) * grid.spacing;
        let r = 11_000.0_f64;
        for i in 0..24 {
            let angle = f64::from(i) * std::f64::consts::TAU / 24.0;
            let x = r.mul_add(angle.cos(), px);
            let y = r.mul_add(angle.sin(), py);
            if let Some(sample) = Sample::at(grid, x, y) {
                let h = sample.blend(|j| grid.heights[j]).max(0.0);
                if h < valley.0 {
                    valley = (h, x, y);
                }
            }
        }
        let eye = [valley.1, valley.2, valley.0 + 1400.0];
        let target = [px, py, peak.0 * 0.72];
        let forward = normalize(sub(target, eye));
        let right = normalize(cross(forward, [0.0, 0.0, 1.0]));
        let up = cross(right, forward);
        Self {
            eye,
            forward,
            right,
            up,
        }
    }
}

/// Bound the slopes of bilinearly interpolated rendered heights, allowing
/// variable conservative ray steps that do not skip a sharp ridge.
fn max_slope(grid: &Grid, surface: &Surface) -> f64 {
    let mut m = 0.0_f64;
    for y in 0..grid.cells {
        for x in 0..grid.cells {
            let at = |x, y| {
                let i = grid.index(x, y);
                grid.heights[i].max(surface.water_level[i]).max(0.0)
            };
            let h = at(x, y);
            m = m
                .max((at(x + 1, y) - h).abs())
                .max((at(x, y + 1) - h).abs());
        }
    }
    m / grid.spacing * std::f64::consts::SQRT_2
}

fn trace(
    grid: &Grid,
    surface: &Surface,
    eye: V3,
    ray: V3,
    max_slope: f64,
) -> Option<(Sample, f64)> {
    let speed = ray[2].abs() + max_slope * ray[0].hypot(ray[1]);
    let mut distance = 0.0_f64;
    let mut previous = 0.0_f64;
    let sample_at = |t: f64| Sample::at(grid, ray[0].mul_add(t, eye[0]), ray[1].mul_add(t, eye[1]));
    while distance < 95_000.0 {
        let sample = sample_at(distance)?;
        let gap = ray[2].mul_add(distance, eye[2]) - sample.elevation(grid, surface);
        if gap <= 0.0 {
            let mut lo = previous;
            let mut hi = distance;
            for _ in 0..12 {
                let mid = lo.midpoint(hi);
                let at = sample_at(mid)?;
                if ray[2].mul_add(mid, eye[2]) > at.elevation(grid, surface) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return Some((sample_at(hi)?, hi));
        }
        previous = distance;
        distance += (gap / speed).clamp(0.5, 400.0);
    }
    None
}

pub(super) fn render(
    grid: &Grid,
    surface: &Surface,
    camera: &Camera,
    path: &Path,
) -> io::Result<()> {
    let slope_bound = max_slope(grid, surface);
    let shaded: Vec<_> = (0..grid.heights.len())
        .map(|i| {
            let (x, y) = grid.coordinates(i);
            let illum = light(grid, x, y);
            std::array::from_fn::<_, 3, _>(|c| {
                lerp(
                    surface.colors[i][c] * illum,
                    [0.095, 0.34, 0.42][c] * (0.85 + 0.15 * illum),
                    surface.water[i],
                )
            })
        })
        .collect();
    let mut bytes = format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let u = (2.0 * (f64::from(x) + 0.5) / f64::from(WIDTH) - 1.0) * 0.76;
            let v =
                (1.0 - 2.0 * (f64::from(y) + 0.5) / f64::from(HEIGHT)) * 0.76 * f64::from(HEIGHT)
                    / f64::from(WIDTH);
            let ray = normalize(std::array::from_fn(|i| {
                camera.forward[i] + camera.right[i] * u + camera.up[i] * v
            }));
            let sky = [0.53, 0.70, 0.82];
            let horizon = [0.68, 0.77, 0.80];
            let color: [f64; 3] = if let Some((sample, distance)) =
                trace(grid, surface, camera.eye, ray, slope_bound)
            {
                let fog = 1.0 - (-distance / 29_000.0).exp();
                std::array::from_fn(|c| lerp(sample.blend(|i| shaded[i][c]), horizon[c], fog))
            } else {
                let h = (ray[2] * 2.4).clamp(0.0, 1.0);
                std::array::from_fn(|c| lerp(horizon[c], sky[c], h))
            };
            bytes.extend(color.map(rgb));
        }
    }
    fs::write(path, bytes)
}
