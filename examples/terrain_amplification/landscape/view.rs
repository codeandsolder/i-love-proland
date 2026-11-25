//! Small deterministic reference renderer for evaluating relief from a cockpit
//! instead of relying exclusively on overhead hillshades. Not the game renderer.
mod channel_mesh;
mod rock;
mod shading;

use super::{
    Grid, Surface,
    hydro::weathering::{ChannelField, Weathered},
    lerp, light, rgb,
};
use std::{fs, io, path::Path};

pub(super) fn export_relief(source: &Grid, path: &Path) -> io::Result<()> {
    rock::RockAtlas::new(source).export(path)
}

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
        a[2].mul_add(-b[1], a[1] * b[2]),
        a[0].mul_add(-b[2], a[2] * b[0]),
        a[1].mul_add(-b[0], a[0] * b[1]),
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
        let [southwest, southeast, northwest, northeast] = self.ids.map(f);
        lerp(
            lerp(southwest, southeast, self.tx),
            lerp(northwest, northeast, self.tx),
            self.ty,
        )
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
                .max((at(x, y + 1) - h).abs())
                .max((at(x + 1, y + 1) - at(x, y + 1)).abs())
                .max((at(x + 1, y + 1) - at(x + 1, y)).abs());
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
    let speed = max_slope.mul_add(ray[0].hypot(ray[1]), ray[2].abs());
    let mut distance = 0.0_f64;
    let mut previous = 0.0_f64;
    let sample_at = |t: f64| Sample::at(grid, ray[0].mul_add(t, eye[0]), ray[1].mul_add(t, eye[1]));
    loop {
        if !distance.is_finite() || distance >= 95_000.0 {
            return None;
        }
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
}

fn weather_at(
    grid: &Grid,
    surface: &Surface,
    channels: &ChannelField,
    sample: &Sample,
    p: [f64; 2],
) -> Weathered {
    let base = sample.blend(|i| grid.heights[i]);
    let standing = sample.blend(|i| grid.heights[i].max(surface.water_level[i]).max(0.0));
    if standing > base + 0.02 {
        return Weathered {
            bed: base,
            water: Some(standing),
            depth: standing - base,
            ..Weathered::default()
        };
    }
    channels.at(p, base)
}

// Far above the terrain use the conservative coarse slope bound and
// the guaranteed +6 m upper envelope; near the surface use cell traversal.
fn trace_weathered(
    grid: &Grid,
    surface: &Surface,
    mesh: &mut channel_mesh::ChannelMesh<'_>,
    eye: V3,
    ray: V3,
    bound: f64,
) -> Option<channel_mesh::Hit> {
    let horizontal = ray[0].hypot(ray[1]);
    let speed = bound.mul_add(horizontal, ray[2].abs());
    let mut distance = 0.0_f64;
    loop {
        if !distance.is_finite() || distance >= 95_000.0 {
            return None;
        }
        let x = ray[0].mul_add(distance, eye[0]);
        let y = ray[1].mul_add(distance, eye[1]);
        let sample = Sample::at(grid, x, y)?;
        let gap = ray[2].mul_add(distance, eye[2]) - sample.elevation(grid, surface);
        if gap <= ChannelField::MAX_RAISE + 2.0 {
            return mesh.trace(eye, ray, distance, 95_000.0);
        }
        distance += ((gap - ChannelField::MAX_RAISE) / speed).min(400.0);
    }
}

fn render_weathered(
    grid: &Grid,
    surface: &Surface,
    camera: &Camera,
    channels: &ChannelField,
    atlas: &rock::RockAtlas,
    path: &Path,
    dimensions: [u16; 2],
) -> io::Result<()> {
    let [width, height] = dimensions;
    let bound = max_slope(grid, surface);
    let subdiv = if width <= 960 { 32 } else { 16 };
    let mut mesh = channel_mesh::ChannelMesh::new(grid, surface, channels, subdiv);
    let pixel_angle = 1.52 / f64::from(width);
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            let u = f64::mul_add(f64::from(x) + 0.5, pixel_angle, -0.76);
            let v = (f64::mul_add(f64::from(height), 0.5, -f64::from(y)) - 0.5) * pixel_angle;
            let ray = normalize(std::array::from_fn(|i| {
                camera.up[i].mul_add(v, camera.right[i].mul_add(u, camera.forward[i]))
            }));
            let color = if let Some(hit) =
                trace_weathered(grid, surface, &mut mesh, camera.eye, ray, bound)
            {
                let p = std::array::from_fn(|i| ray[i].mul_add(hit.distance, camera.eye[i]));
                shading::shade_weathered(
                    grid,
                    surface,
                    &hit,
                    shading::IncidentRay {
                        position: p,
                        direction: ray,
                        distance: hit.distance,
                        pixel_angle,
                    },
                    atlas,
                )
            } else {
                shading::sky(ray)
            };
            bytes.extend(color.map(rgb));
        }
    }
    println!(
        "weathered fine cells: {} ({subdiv}x{subdiv} subcells)",
        mesh.cells()
    );
    fs::write(path, bytes)
}

pub(super) fn compare_weathering(
    grid: &Grid,
    surface: &Surface,
    camera: &Camera,
    source: &Grid,
    channels: &ChannelField,
    path: &Path,
    label: &str,
) -> io::Result<()> {
    let atlas = rock::RockAtlas::new(source);
    render_weathered(
        grid,
        surface,
        camera,
        channels,
        &atlas,
        &path.join(format!("{label}-weathered.ppm")),
        [1280, 720],
    )?;
    if let Some((center, direction)) = super::hydro::weathering::viewpoint(
        channels,
        [camera.eye[0], camera.eye[1]],
        f64::from(grid.cells) * grid.spacing,
    ) {
        let norm = direction[0].hypot(direction[1]);
        if norm > 0.0 {
            let tangent = direction.map(|v| v / norm);
            let eye_xy = [
                f64::mul_add(
                    tangent[1],
                    220.0,
                    f64::mul_add(tangent[0], -380.0, center[0]),
                ),
                f64::mul_add(
                    tangent[0],
                    -220.0,
                    f64::mul_add(tangent[1], -380.0, center[1]),
                ),
            ];
            if let (Some(eye_sample), Some(target_sample)) = (
                Sample::at(grid, eye_xy[0], eye_xy[1]),
                Sample::at(grid, center[0], center[1]),
            ) {
                let eye = [
                    eye_xy[0],
                    eye_xy[1],
                    eye_sample.elevation(grid, surface) + 320.0,
                ];
                let target = [center[0], center[1], target_sample.elevation(grid, surface)];
                let forward = normalize(sub(target, eye));
                let right = normalize(cross(forward, [0.0, 0.0, 1.0]));
                let up = cross(right, forward);
                let close = Camera {
                    eye,
                    forward,
                    right,
                    up,
                };
                render_weathered(
                    grid,
                    surface,
                    &close,
                    channels,
                    &atlas,
                    &path.join(format!("{label}-river-close.ppm")),
                    [960, 540],
                )?;
            }
        }
    }
    Ok(())
}

pub(super) fn render(
    grid: &Grid,
    surface: &Surface,
    camera: &Camera,
    path: &Path,
) -> io::Result<()> {
    render_legacy(grid, surface, camera, path, WIDTH, HEIGHT)
}

fn render_legacy(
    grid: &Grid,
    surface: &Surface,
    camera: &Camera,
    path: &Path,
    width: u16,
    height: u16,
) -> io::Result<()> {
    let slope_bound = max_slope(grid, surface);
    let shaded: Vec<_> = (0..grid.heights.len())
        .map(|i| {
            let (x, y) = grid.coordinates(i);
            let illum = light(grid, x, y);
            std::array::from_fn::<_, 3, _>(|c| {
                lerp(
                    surface.colors[i][c] * illum,
                    [0.095, 0.34, 0.42][c] * 0.15f64.mul_add(illum, 0.85),
                    surface.water[i],
                )
            })
        })
        .collect();
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            let u = (2.0 * (f64::from(x) + 0.5) / f64::from(width) - 1.0) * 0.76;
            let v =
                (1.0 - 2.0 * (f64::from(y) + 0.5) / f64::from(height)) * 0.76 * f64::from(height)
                    / f64::from(width);
            let ray = normalize(std::array::from_fn(|i| {
                camera.up[i].mul_add(v, camera.right[i].mul_add(u, camera.forward[i]))
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

// Same camera and raster dimensions for all comparisons. Ray intersections
// continue to use the unmodified heightfield, including lake surfaces.
pub(super) fn compare_shading(
    grid: &Grid,
    surface: &Surface,
    camera: &Camera,
    output: &Path,
    label: &str,
    source: &Grid,
    full: bool,
) -> io::Result<()> {
    const W: u16 = 1280;
    const H: u16 = 720;
    if full {
        render_legacy(
            grid,
            surface,
            camera,
            &output.join(format!("{label}-legacy.ppm")),
            W,
            H,
        )?;
    }
    let bound = max_slope(grid, surface);
    let pixel_angle = 1.52 / f64::from(W);
    let atlas = rock::RockAtlas::new(source);
    let rivers = super::hydro::RiverCoverage::new(grid, &surface.hydro);
    for (name, style) in [
        ("lighting", shading::Style::Lighting),
        ("detail", shading::Style::Procedural),
        (
            "geologic",
            shading::Style::Geologic {
                atlas: &atlas,
                rivers: &rivers,
            },
        ),
    ] {
        if !full && !matches!(style, shading::Style::Geologic { .. }) {
            continue;
        }
        let mut bytes = format!("P6\n{W} {H}\n255\n").into_bytes();
        for y in 0..H {
            for x in 0..W {
                let u = f64::mul_add(f64::from(x) + 0.5, pixel_angle, -0.76);
                let v = (f64::mul_add(f64::from(H), 0.5, -f64::from(y)) - 0.5) * pixel_angle;
                let ray = normalize(std::array::from_fn(|i| {
                    camera.up[i].mul_add(v, camera.right[i].mul_add(u, camera.forward[i]))
                }));
                let color = if let Some((sample, distance)) =
                    trace(grid, surface, camera.eye, ray, bound)
                {
                    let p = std::array::from_fn(|i| ray[i].mul_add(distance, camera.eye[i]));
                    shading::shade(
                        grid,
                        surface,
                        &sample,
                        shading::IncidentRay {
                            position: p,
                            direction: ray,
                            distance,
                            pixel_angle,
                        },
                        style,
                    )
                } else {
                    shading::sky(ray)
                };
                bytes.extend(color.map(rgb));
            }
        }
        fs::write(output.join(format!("{label}-{name}.ppm")), bytes)?;
    }
    Ok(())
}
