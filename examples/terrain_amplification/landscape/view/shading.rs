//! Per-hit lighting and continuous, footprint-filtered surface microdetail.
//! The displacement here is a shading normal only: it does not alter
//! canonical terrain, drainage, or the ray intersection.
use super::super::hydro::RiverCoverage;
use super::{Grid, Sample, Surface, V3, dot, normalize, rock::RockAtlas};
use crate::{hash, lerp, smooth};

const SUN: V3 = [-0.48, 0.56, 0.67];
const SKY: V3 = [0.53, 0.70, 0.82];
const HORIZON: V3 = [0.68, 0.77, 0.80];

pub(super) fn cubic(a: f64, b: f64, c: f64, d: f64, t: f64) -> (f64, f64) {
    let p = 0.5 * (-a + 3.0 * b - 3.0 * c + d);
    let q = a - 2.5 * b + 2.0 * c - 0.5 * d;
    let r = 0.5 * (c - a);
    (
        ((p * t + q) * t + r) * t + b,
        (3.0 * p * t + 2.0 * q) * t + r,
    )
}

fn normal(grid: &Grid, sample: &Sample) -> V3 {
    let (x, y) = grid.coordinates(sample.ids[0]);
    let clamp = |v: i32| u16::try_from(v.clamp(0, i32::from(grid.cells))).unwrap_or_default();
    let data: [(f64, f64); 4] = std::array::from_fn(|row| {
        let yy = clamp(i32::from(y) + i32::try_from(row).unwrap_or_default() - 1);
        let [a, b, c, d] = [-1, 0, 1, 2].map(|dx| grid.at(clamp(i32::from(x) + dx), yy));
        cubic(a, b, c, d, sample.tx)
    });
    let [a, b, c, d] = data.map(|p| p.0);
    let (_, dy) = cubic(a, b, c, d, sample.ty);
    let [a, b, c, d] = data.map(|p| p.1);
    let (dx, _) = cubic(a, b, c, d, sample.ty);
    normalize([-dx / grid.spacing, -dy / grid.spacing, 1.0])
}

/// Smooth scalar lattice field and its analytic gradient in world units.
/// All octaves are spatially keyed; no mutable RNG or camera-dependent phase.
fn field(p: V3, size: V3, seed: u64) -> (f64, V3) {
    let q: V3 = std::array::from_fn(|i| p[i] / size[i]);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "regional world-space lattice coordinates fit i32"
    )]
    let cell = q.map(|v| v.floor() as i32);
    let t: V3 = std::array::from_fn(|i| q[i] - f64::from(cell[i]));
    let f = t.map(|v| v * v * v * (v * (6.0 * v - 15.0) + 10.0));
    let df = t.map(|v| 30.0 * v * v * (v - 1.0).powi(2));
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
                let v = f64::from(u32::try_from(key >> 32).unwrap_or_default())
                    / f64::from(u32::MAX)
                    * 2.0
                    - 1.0;
                let w: V3 = std::array::from_fn(|i| if bits[i] == 0 { 1.0 - f[i] } else { f[i] });
                value += v * w[0] * w[1] * w[2];
                for axis in 0..3 {
                    let sign = if bits[axis] == 0 { -1.0 } else { 1.0 };
                    deriv[axis] +=
                        v * sign * df[axis] * w[(axis + 1) % 3] * w[(axis + 2) % 3] / size[axis];
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
        let strength = band * (0.25 + 0.75 * rock);
        tint += strength * color * v;
        for (out, derivative) in grad.iter_mut().zip(g) {
            *out += strength * height * derivative;
        }
    }
    // Gradient of the scalar relief projected into the surface tangent
    // plane; this also works on steep slopes without stretched UVs.
    let along = dot(n, grad);
    let bumped = normalize(std::array::from_fn(|i| n[i] - (grad[i] - along * n[i])));
    (bumped, 1.0 + tint)
}

fn visibility(grid: &Grid, p: V3) -> f64 {
    let horiz = SUN[0].hypot(SUN[1]);
    let sun_slope = SUN[2] / horiz;
    let mut horizon = f64::NEG_INFINITY;
    for d in [
        45.0, 90.0, 180.0, 360.0, 720.0, 1440.0, 2880.0, 5760.0, 11520.0,
    ] {
        let x = p[0] + d * SUN[0] / horiz;
        let y = p[1] + d * SUN[1] / horiz;
        let Some(s) = Sample::at(grid, x, y) else {
            break;
        };
        let h = s.blend(|i| grid.heights[i]);
        horizon = horizon.max((h - p[2]) / d);
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
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
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

pub(super) fn shade(
    grid: &Grid,
    surface: &Surface,
    sample: &Sample,
    p: V3,
    ray: V3,
    distance: f64,
    pixel_angle: f64,
    style: Style<'_>,
) -> V3 {
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
        let (v, _) = field(p, [230.0, 185.0, 170.0], 201);
        let (small, _) = field(p, [76.0, 92.0, 68.0], 207);
        let local_slope = bump[2].clamp(0.0, 1.0);
        // Green valleys break into vegetation and exposed soils;
        // talus / bedrock transitions depend on the local normal.
        let snow = smooth(((albedo[0].min(albedo[1]).min(albedo[2]) - 0.5) / 0.25).clamp(0.0, 1.0));
        let exposed =
            smooth(((1.0 - local_slope - 0.075 + 0.12 * v) / 0.4).clamp(0.0, 1.0)) * (1.0 - snow);
        let green = smooth(((albedo[1] - albedo[0] + 0.01) / 0.09).clamp(0.0, 1.0));
        let patch = smooth(((v + 0.3) / 0.75).clamp(0.0, 1.0));
        let variation = 1.0 + 0.14 * v + 0.08 * small;
        for (i, c) in albedo.iter_mut().enumerate() {
            let soil = lerp(*c, [0.40, 0.34, 0.24][i], green * patch * 0.22);
            let stone = [0.48, 0.455, 0.42][i] * variation;
            *c = lerp(soil, stone, exposed * 0.36);
        }
    }
    let shade = visibility(grid, p);
    let sun = dot(bump, normalize(SUN)).max(0.0) * shade;
    // Physically motivated linear-light diffuse shading. A hemisphere
    // ambient term keeps the shadow faces readable without washing out
    // the relief. Water bypasses rock microdetail.
    let ambient = if geologic {
        // Directional sky fill reveals small-scale relief even on faces
        // looking away from the sun, unlike a constant ambient intensity.
        0.12 + 0.21 * bump[2].max(0.0) + 0.23 * dot(bump, normalize([0.76, -0.41, 0.5])).max(0.0)
    } else {
        0.19 + 0.25 * bump[2].max(0.0)
    };
    let illumination = ambient + 1.05 * sun;
    let fog = 1.0 - (-distance / if geologic { 128_000.0 } else { 78_000.0 }).exp();
    std::array::from_fn(|i| {
        let land = linear(albedo[i]) * illumination * tint;
        let wet = linear([0.095, 0.34, 0.42][i]) * (0.8 + 0.2 * shade);
        srgb(lerp(lerp(land, wet, water), linear(HORIZON[i]), fog))
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
