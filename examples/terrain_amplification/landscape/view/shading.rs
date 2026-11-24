//! Per-hit lighting and continuous, footprint-filtered surface microdetail.
//! The displacement here is a shading normal only: it does not alter
//! canonical terrain, drainage, or the ray intersection.
use super::super::hydro::RiverCoverage;
use super::{Grid, Sample, Surface, V3, dot, normalize, rock::RockAtlas};
use crate::{hash, lerp, smooth};

const SUN: V3 = [-0.48, 0.56, 0.67];
const SKY: V3 = [0.53, 0.70, 0.82];
const HORIZON: V3 = [0.68, 0.77, 0.80];

pub(super) fn cubic(before: f64, lower: f64, upper: f64, after: f64, fraction: f64) -> (f64, f64) {
    let cubic_term = f64::midpoint(
        3.0f64.mul_add(-upper, 3.0f64.mul_add(lower, -before)),
        after,
    );
    let quadratic = 0.5f64.mul_add(
        -after,
        2.0f64.mul_add(upper, 2.5f64.mul_add(-lower, before)),
    );
    let linear = 0.5 * (upper - before);
    (
        cubic_term
            .mul_add(fraction, quadratic)
            .mul_add(fraction, linear)
            .mul_add(fraction, lower),
        2.0f64
            .mul_add(quadratic, 3.0 * cubic_term * fraction)
            .mul_add(fraction, linear),
    )
}

fn normal(grid: &Grid, sample: &Sample) -> V3 {
    let (column, row_origin) = grid.coordinates(sample.ids[0]);
    let clamp = |v: i32| u16::try_from(v.clamp(0, i32::from(grid.cells))).unwrap_or_default();
    let data: [(f64, f64); 4] = std::array::from_fn(|row| {
        let yy = clamp(i32::from(row_origin) + i32::try_from(row).unwrap_or_default() - 1);
        let [before, lower, upper, after] =
            [-1, 0, 1, 2].map(|dx| grid.at(clamp(i32::from(column) + dx), yy));
        cubic(before, lower, upper, after, sample.tx)
    });
    let [before, lower, upper, after] = data.map(|p| p.0);
    let (_, dy) = cubic(before, lower, upper, after, sample.ty);
    let [before, lower, upper, after] = data.map(|p| p.1);
    let (dx, _) = cubic(before, lower, upper, after, sample.ty);
    normalize([-dx / grid.spacing, -dy / grid.spacing, 1.0])
}

/// Smooth scalar lattice field and its analytic gradient in world units.
/// All octaves are spatially keyed; no mutable RNG or camera-dependent phase.
fn field(point: V3, size: V3, seed: u64) -> (f64, V3) {
    let chart: V3 = std::array::from_fn(|i| point[i] / size[i]);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "regional world-space lattice coordinates fit i32"
    )]
    let cell = chart.map(|amplitude| amplitude.floor() as i32);
    let fraction: V3 = std::array::from_fn(|i| chart[i] - f64::from(cell[i]));
    let fade = fraction.map(|amplitude| {
        amplitude
            * amplitude
            * amplitude
            * f64::mul_add(amplitude, 6.0f64.mul_add(amplitude, -15.0), 10.0)
    });
    let df = fraction.map(|amplitude| 30.0 * amplitude * amplitude * (amplitude - 1.0).powi(2));
    let mut value = 0.0;
    let mut deriv = [0.0; 3];
    for z in 0..=1_i32 {
        for y in 0..=1_i32 {
            for x in 0..=1_i32 {
                let bits = [x, y, z];
                let key = hash(
                    seed ^ u64::from(u32::from_ne_bytes((cell[0] + x).to_ne_bytes()))
                        .rotate_left(7)
                        ^ u64::from(u32::from_ne_bytes((cell[1] + y).to_ne_bytes()))
                            .rotate_left(29)
                        ^ u64::from(u32::from_ne_bytes((cell[2] + z).to_ne_bytes()))
                            .rotate_left(47),
                );
                let amplitude = f64::mul_add(
                    f64::from(u32::try_from(key >> 32).unwrap_or_default()) / f64::from(u32::MAX),
                    2.0,
                    -1.0,
                );
                let weights: V3 =
                    std::array::from_fn(|i| if bits[i] == 0 { 1.0 - fade[i] } else { fade[i] });
                value = (amplitude * weights[0] * weights[1]).mul_add(weights[2], value);
                for axis in 0..3 {
                    let sign = if bits[axis] == 0 { -1.0 } else { 1.0 };
                    deriv[axis] += amplitude
                        * sign
                        * df[axis]
                        * weights[(axis + 1) % 3]
                        * weights[(axis + 2) % 3]
                        / size[axis];
                }
            }
        }
    }
    (value, deriv)
}

fn microdetail(p: V3, n: V3, footprint: f64, rock: f64) -> (V3, f64) {
    let mut grad = [0.0; 3];
    let mut tint = 0.0;
    // Larger horizontal scale on the exposed rock band creates weathered
    // layers instead of isotropic speckle. Distance filtering removes
    // subpixel octaves smoothly, preventing distant shimmering.
    for (size, height, color, seed) in [
        ([180.0, 140.0, 65.0], 4.5, 0.105, 11),
        ([58.0, 52.0, 25.0], 1.6, 0.060, 17),
        ([17.0, 19.0, 12.0], 0.42, 0.045, 23),
        ([5.0, 6.0, 5.0], 0.11, 0.025, 31),
    ] {
        let scale = size.into_iter().fold(f64::INFINITY, f64::min);
        let band = 1.0 - smooth(((footprint / scale - 0.30) / 0.70).clamp(0.0, 1.0));
        if band <= 0.0 {
            continue;
        }
        let (v, g) = field(p, size, seed);
        let strength = band * 0.75f64.mul_add(rock, 0.25);
        tint = (strength * color).mul_add(v, tint);
        for (out, derivative) in grad.iter_mut().zip(g) {
            *out = (strength * height).mul_add(derivative, *out);
        }
    }
    // Gradient of the scalar relief projected into the surface tangent
    // plane; this also works on steep slopes without stretched UVs.
    let along = dot(n, grad);
    let bumped = normalize(std::array::from_fn(|i| {
        n[i] - along.mul_add(-n[i], grad[i])
    }));
    (bumped, 1.0 + tint)
}

fn visibility(grid: &Grid, point: V3) -> f64 {
    let horiz = SUN[0].hypot(SUN[1]);
    let sun_slope = SUN[2] / horiz;
    let mut horizon = f64::NEG_INFINITY;
    for distance in [
        45.0, 90.0, 180.0, 360.0, 720.0, 1440.0, 2880.0, 5760.0, 11520.0,
    ] {
        let sample_x = point[0] + distance * SUN[0] / horiz;
        let sample_y = point[1] + distance * SUN[1] / horiz;
        let Some(sample) = Sample::at(grid, sample_x, sample_y) else {
            break;
        };
        let height = sample.blend(|i| grid.heights[i]);
        horizon = horizon.max((height - point[2]) / distance);
    }
    1.0 - smooth(((horizon - sun_slope + 0.035) / 0.07).clamp(0.0, 1.0))
}

fn linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
fn srgb(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055f64.mul_add(c.powf(1.0 / 2.4), -0.055)
    }
}

pub(super) fn sky(ray: V3) -> V3 {
    let h = (ray[2] * 2.4).clamp(0.0, 1.0);
    std::array::from_fn(|i| lerp(HORIZON[i], SKY[i], h))
}

#[derive(Clone, Copy)]
pub(super) enum Style<'a> {
    Lighting,
    Procedural,
    Geologic {
        atlas: &'a RockAtlas,
        rivers: &'a RiverCoverage<'a>,
    },
}

#[derive(Clone, Copy)]
pub(super) struct IncidentRay {
    pub(super) position: V3,
    pub(super) direction: V3,
    pub(super) distance: f64,
    pub(super) pixel_angle: f64,
}

pub(super) fn shade(
    grid: &Grid,
    surface: &Surface,
    sample: &Sample,
    incident: IncidentRay,
    style: Style<'_>,
) -> V3 {
    let IncidentRay {
        position: p,
        direction: ray,
        distance,
        pixel_angle,
    } = incident;
    let n = normal(grid, sample);
    let footprint = distance * pixel_angle / (-dot(n, ray)).abs().max(0.24);
    let rock = smooth((((1.0 - n[2]) - 0.035) / 0.27_f64).clamp(0.0, 1.0));
    let water = match style {
        Style::Geologic { rivers, .. } => {
            let standing = sample.blend(|i| {
                if grid.heights[i] <= 0.0 || surface.water_level[i] > 0.0 {
                    1.0
                } else {
                    0.0
                }
            });
            standing.max(rivers.at([p[0], p[1]], footprint))
        }
        _ => sample.blend(|i| surface.water[i]),
    };
    let (bump, tint) = match style {
        Style::Lighting => (n, 1.0),
        Style::Procedural => microdetail(p, n, footprint, rock),
        Style::Geologic { atlas, .. } => atlas.sample(p, n, footprint, rock),
    };
    let geologic = matches!(style, Style::Geologic { .. });
    let mut albedo: V3 = std::array::from_fn(|i| sample.blend(|j| surface.colors[j][i]));
    if geologic {
        albedo = geologic_albedo(p, bump, albedo);
    }
    let shade = visibility(grid, p);
    let sun = dot(bump, normalize(SUN)).max(0.0) * shade;
    // Physically motivated linear-light diffuse shading. A hemisphere
    // ambient term keeps the shadow faces readable without washing out
    // the relief. Water bypasses rock microdetail.
    let ambient = if geologic {
        // Directional sky fill reveals small-scale relief even on faces
        // looking away from the sun, unlike a constant ambient intensity.
        0.23f64.mul_add(
            dot(bump, normalize([0.76, -0.41, 0.5])).max(0.0),
            0.21f64.mul_add(bump[2].max(0.0), 0.12),
        )
    } else {
        0.25f64.mul_add(bump[2].max(0.0), 0.19)
    };
    let illumination = 1.05f64.mul_add(sun, ambient);
    let fog = 1.0 - (-distance / if geologic { 128_000.0 } else { 78_000.0 }).exp();
    std::array::from_fn(|i| {
        let land = linear(albedo[i]) * illumination * tint;
        let wet = linear([0.095, 0.34, 0.42][i]) * 0.2f64.mul_add(shade, 0.8);
        srgb(lerp(lerp(land, wet, water), linear(HORIZON[i]), fog))
    })
}

fn geologic_albedo(p: V3, bump: V3, mut albedo: V3) -> V3 {
    let (v, _) = field(p, [230.0, 185.0, 170.0], 201);
    let (small, _) = field(p, [76.0, 92.0, 68.0], 207);
    let local_slope = bump[2].clamp(0.0, 1.0);
    // Green valleys break into vegetation and exposed soils;
    // talus / bedrock transitions depend on the local normal.
    let snow = smooth(((albedo[0].min(albedo[1]).min(albedo[2]) - 0.5) / 0.25).clamp(0.0, 1.0));
    let exposed = smooth((0.12f64.mul_add(v, 1.0 - local_slope - 0.075) / 0.4).clamp(0.0, 1.0))
        * (1.0 - snow);
    let green = smooth(((albedo[1] - albedo[0] + 0.01) / 0.09).clamp(0.0, 1.0));
    let patch = smooth(((v + 0.3) / 0.75).clamp(0.0, 1.0));
    let variation = 0.08f64.mul_add(small, 0.14f64.mul_add(v, 1.0));
    for (i, c) in albedo.iter_mut().enumerate() {
        let soil = lerp(*c, [0.40, 0.34, 0.24][i], green * patch * 0.22);
        let stone = [0.48, 0.455, 0.42][i] * variation;
        *c = lerp(soil, stone, exposed * 0.36);
    }
    albedo
}

fn water_fine_band(footprint: f64) -> f64 {
    1.0 - smooth(((footprint / 4.0 - 0.25) / 0.75).clamp(0.0, 1.0))
}

fn fluvial_albedo(p: V3, n: V3, footprint: f64, weather: super::Weathered, albedo: &mut V3) -> f64 {
    let (v, _) = field(p, [105.0, 85.0, 47.0], 0x414c_4c55);
    let (grain, _) = field(p, [21.0, 17.0, 13.0], 0x0053_4544);
    // Spatially coherent riparian vegetation clumps and sorted gravel.
    // Fade each small-scale band before it becomes smaller than a pixel.
    let fine_band = water_fine_band(footprint);
    let mid_band = 1.0 - smooth(((footprint / 16.0 - 0.25) / 0.75).clamp(0.0, 1.0));
    let (fine, _) = field(p, [4.1, 3.2, 2.8], 0x4752_4156);
    let (clumps, _) = field(p, [12.0, 17.0, 9.0], 0x5249_5041);
    let sorted = (clumps * mid_band).mul_add(0.75, fine * fine_band * 0.25);
    let flat = smooth(((n[2] - 0.72) / 0.28).clamp(0.0, 1.0));
    let patch = smooth(((v + 0.22) / 0.85).clamp(0.0, 1.0));
    let grain_mask = smooth(((grain + 0.15) / 0.75).clamp(0.0, 1.0));
    let gravel = 0.12f64.mul_add(
        smooth((weather.deposit / 2.5).min(1.0)),
        1.1f64.mul_add(weather.point_bar, 0.32 * weather.gravel * patch),
    ) * flat;
    let bank = weather
        .scour
        .mul_add(0.29, weather.bank * 0.24f64.mul_add(grain_mask, 0.13));
    let flood = weather.floodplain
        * flat
        * 0.60f64.mul_add(
            smooth(0.32f64.mul_add(sorted, v + 0.4).clamp(0.0, 1.0)),
            0.12,
        );
    for (i, c) in albedo.iter_mut().enumerate() {
        let sediment =
            [0.50, 0.465, 0.385][i] * 0.29f64.mul_add(sorted, 0.23f64.mul_add(grain, 1.0));
        let damp = [0.25, 0.245, 0.20][i] * 0.19f64.mul_add(v, 1.0);
        let riparian = [0.18, 0.30, 0.115][i]
            * 0.30f64.mul_add(sorted, 0.18f64.mul_add(grain, 0.25f64.mul_add(v, 1.0)));
        *c = lerp(
            lerp(
                lerp(*c, riparian, flood.clamp(0.0, 1.0)),
                sediment,
                gravel.clamp(0.0, 1.0),
            ),
            damp,
            bank.clamp(0.0, 1.0),
        );
    }
    sorted
}

pub(super) fn shade_weathered(
    grid: &Grid,
    surface: &Surface,
    hit: &super::channel_mesh::Hit,
    incident: IncidentRay,
    atlas: &RockAtlas,
) -> V3 {
    let IncidentRay {
        position: p,
        direction: ray,
        distance,
        pixel_angle,
    } = incident;
    let sample = &hit.sample;
    let weather = hit.weather;
    let fine_normal = hit.normal;
    let coarse = normal(grid, sample);
    let mask = weather.bank.max(weather.gravel).max(weather.floodplain);
    let blend = smooth((mask * 4.0).clamp(0.0, 1.0));
    let n = fine_normal.map_or(coarse, |n| {
        normalize(std::array::from_fn(|i| lerp(coarse[i], n[i], blend)))
    });
    let footprint = distance * pixel_angle / (-dot(n, ray)).abs().max(0.24);
    let rock = smooth((((1.0 - n[2]) - 0.035) / 0.27).clamp(0.0, 1.0));
    let (bump, tint) = atlas.sample(p, n, footprint, rock.max(weather.gravel * 0.75));
    let mut albedo = geologic_albedo(
        p,
        bump,
        std::array::from_fn(|i| sample.blend(|j| surface.colors[j][i])),
    );
    let sorted = fluvial_albedo(p, n, footprint, weather, &mut albedo);
    let fine_band = water_fine_band(footprint);
    let sun = normalize(SUN);
    let shade = visibility(grid, p);
    let ambient = 0.23f64.mul_add(
        dot(bump, normalize([0.76, -0.41, 0.5])).max(0.0),
        0.21f64.mul_add(bump[2].max(0.0), 0.12),
    );
    let illumination = (1.05 * dot(bump, sun).max(0.0)).mul_add(shade, ambient);
    let fog = 1.0 - (-distance / 128_000.0).exp();
    let standing = weather.water.is_some();
    let water_color = if standing {
        // A water surface reflects the sky at grazing angles and transmits
        // the submerged sediment at normal incidence. Depth controls
        // attenuation; do not illuminate water like a blue terrain texture.
        let wave_filter = 1.0 - smooth(((footprint / 12.0 - 0.25) / 0.75).clamp(0.0, 1.0));
        let (_, g) = field(p, [12.0, 8.0, 30.0], 0x5741_5645);
        let (_, g2) = field(p, [4.0, 5.7, 18.0], 0x5741_5646);
        let wn = normalize([
            (0.16 * fine_band).mul_add(-g2[0], -0.54 * wave_filter * g[0]),
            (0.16 * fine_band).mul_add(-g2[1], -0.54 * wave_filter * g[1]),
            1.0,
        ]);
        let fresnel = 0.975f64.mul_add((1.0 - (-dot(wn, ray)).clamp(0.0, 1.0)).powi(5), 0.025);
        let reflect = std::array::from_fn(|i| (2.0 * dot(wn, ray)).mul_add(-wn[i], ray[i]));
        let reflected = sky(reflect);
        let sparkle = dot(normalize(std::array::from_fn(|i| sun[i] - ray[i])), wn)
            .max(0.0)
            .powi(96)
            * shade;
        let depth = weather.depth;
        std::array::from_fn(|i| {
            let bed = linear([0.47, 0.435, 0.335][i] * 0.22f64.mul_add(sorted, 1.0));
            let transmission =
                (-depth / ([1.7, 3.8, 3.0][i] * (-dot(wn, ray)).abs().max(0.25))).exp();
            let transmitted = lerp(linear([0.09, 0.245, 0.225][i]), bed, transmission);
            0.14f64.mul_add(
                sparkle,
                lerp(
                    transmitted * 0.55f64.mul_add(shade, 0.45),
                    linear(reflected[i]),
                    fresnel,
                ),
            )
        })
    } else {
        [0.0; 3]
    };
    std::array::from_fn(|i| {
        let land = linear(albedo[i]) * illumination * tint;
        let radiance = if standing { water_color[i] } else { land };
        srgb(lerp(radiance, linear(HORIZON[i]), fog))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cubic_reconstructs_linear_plane() {
        let (h, d) = cubic(1.0, 3.0, 5.0, 7.0, 0.25);
        assert!((h - 3.5).abs() < 1e-12);
        assert!((d - 2.0).abs() < 1e-12);
    }
    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let p = [89.1, 56.2, 12.3];
        let s = [17.0, 23.0, 13.0];
        let (_, g) = field(p, s, 41);
        for (i, &expected) in g.iter().enumerate() {
            let mut a = p;
            let mut b = p;
            a[i] -= 1e-3;
            b[i] += 1e-3;
            let numerical = (field(b, s, 41).0 - field(a, s, 41).0) / 2e-3;
            assert!((numerical - expected).abs() < 1e-7);
        }
        let n = normalize([0.2, -0.4, 1.0]);
        let (bump, tint) = microdetail(p, n, 1e6, 1.0);
        assert!(bump.iter().zip(n).all(|(a, b)| (a - b).abs() < 1e-12));
        assert!((tint - 1.0).abs() < 1e-12);
    }
}
