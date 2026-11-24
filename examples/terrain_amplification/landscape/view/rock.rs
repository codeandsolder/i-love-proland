//! Small real-terrain residual atlas, used at reduced scales as a material
//! normal field. No additional source data or heightmap storage is needed.
use super::{Grid, V3, dot, normalize, shading};
use crate::{hash, smooth};

const PATCH: i32 = 32;

pub(super) struct RockAtlas {
    residual: Grid,
}

impl RockAtlas {
    pub(super) fn new(source: &Grid) -> Self {
        let broad = super::super::blur(source, &source.heights, 8);
        let mut residual = source.clone();
        for (h, b) in residual.heights.iter_mut().zip(broad) {
            *h -= b;
        }
        Self { residual }
    }

    // Translate overlapping source patches deterministically. Each source
    // patch has a full support halo so no atlas wrap seam is ever sampled.
    fn plane(&self, point: [f64; 2], pixel: f64, seed: u64) -> (f64, [f64; 2]) {
        let chart = point.map(|value| value / (pixel * f64::from(PATCH)));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounded regional material coordinates"
        )]
        let cell = chart.map(|value| value.floor() as i32);
        let fraction: [f64; 2] = std::array::from_fn(|i| chart[i] - f64::from(cell[i]));
        let fade = fraction.map(smooth);
        let df = fraction.map(|value| 6.0 * value * (1.0 - value) / (pixel * f64::from(PATCH)));
        let mut value = 0.0;
        let mut deriv = [0.0; 2];
        let margin = PATCH + 2;
        let range = u64::try_from(i32::from(self.residual.cells) - 2 * margin + 1).unwrap_or(1);
        for y in 0..=1_i32 {
            for x in 0..=1_i32 {
                let key = hash(
                    seed ^ u64::from(u32::from_ne_bytes((cell[0] + x).to_ne_bytes()))
                        .rotate_left(9)
                        ^ u64::from(u32::from_ne_bytes((cell[1] + y).to_ne_bytes()))
                            .rotate_left(36),
                );
                let ox = margin + i32::try_from(key % range).unwrap_or_default();
                let oy = margin + i32::try_from(hash(key) % range).unwrap_or_default();
                let ax = (fraction[0] - f64::from(x)).mul_add(f64::from(PATCH), f64::from(ox));
                let ay = (fraction[1] - f64::from(y)).mul_add(f64::from(PATCH), f64::from(oy));
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "patch halo bounds atlas coordinates"
                )]
                #[expect(clippy::cast_sign_loss, reason = "atlas coordinates are positive")]
                let (ix, iy) = (ax.floor() as u16, ay.floor() as u16);
                let (tx, ty) = (ax.fract(), ay.fract());
                let rows: [(f64, f64); 4] = std::array::from_fn(|r| {
                    let yy = iy - 1 + u16::try_from(r).unwrap_or_default();
                    let [before, lower, upper, after] =
                        [ix - 1, ix, ix + 1, ix + 2].map(|xx| self.residual.at(xx, yy));
                    shading::cubic(before, lower, upper, after, tx)
                });
                let [before, lower, upper, after] = rows.map(|r| r.0);
                let (relief, gy) = shading::cubic(before, lower, upper, after, ty);
                let [before, lower, upper, after] = rows.map(|r| r.1);
                let (gx, _) = shading::cubic(before, lower, upper, after, ty);
                let (wx, dwx) = if x == 0 {
                    (1.0 - fade[0], -df[0])
                } else {
                    (fade[0], df[0])
                };
                let (wy, dwy) = if y == 0 {
                    (1.0 - fade[1], -df[1])
                } else {
                    (fade[1], df[1])
                };
                // Preserve dimensionless source relief after spatial scaling.
                let height = relief * pixel / self.residual.spacing;
                value = (height * wx).mul_add(wy, value);
                deriv[0] += (height * dwx).mul_add(wy, gx / self.residual.spacing * wx * wy);
                deriv[1] += (height * wx).mul_add(dwy, gy / self.residual.spacing * wx * wy);
            }
        }
        (value, deriv)
    }

    pub(super) fn sample(&self, point: V3, normal: V3, footprint: f64, rock: f64) -> (V3, f64) {
        let weights = normal.map(|v| v.powi(4));
        let total: f64 = weights.iter().sum();
        let mut gradient = [0.0; 3];
        let mut tint = 0.0;
        for (pixel, amp, seed) in [(39.0, 0.8, 81), (12.0, 0.65, 97), (3.5, 0.4, 113)] {
            let band = 1.0 - smooth(((footprint / (pixel * 2.4) - 0.35) / 0.65).clamp(0.0, 1.0));
            if band == 0.0 {
                continue;
            }
            let strength = band * 0.78f64.mul_add(rock, 0.22) * amp;
            for (axis, weight) in weights.into_iter().enumerate() {
                if weight < 0.002 {
                    continue;
                }
                let axis_u = (axis + 1) % 3;
                let axis_v = (axis + 2) % 3;
                let (height, derivative) = self.plane(
                    [point[axis_u], point[axis_v]],
                    pixel,
                    seed ^ u64::try_from(axis).unwrap_or_default(),
                );
                let weight = weight / total * strength;
                gradient[axis_u] = weight.mul_add(derivative[0], gradient[axis_u]);
                gradient[axis_v] = weight.mul_add(derivative[1], gradient[axis_v]);
                // Exposed crests and weathered recesses differ in albedo.
                // Use bounded residual rather than an arbitrary color noise.
                tint = (weight * (height / (pixel * 0.45)).clamp(-1.0, 1.0)).mul_add(0.23, tint);
            }
        }
        let along = dot(normal, gradient);
        (
            normalize(std::array::from_fn(|i| {
                normal[i] - along.mul_add(-normal[i], gradient[i])
            })),
            1.0 + tint,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_overlap_is_continuous() {
        let mut grid = Grid::new(128, 62.5);
        for (i, h) in grid.heights.iter_mut().enumerate() {
            *h = (f64::from(u32::try_from(i).unwrap_or_default()) * 0.031).sin() * 60.0;
        }
        let atlas = RockAtlas::new(&grid);
        let pixel = 12.0;
        let boundary = pixel * f64::from(PATCH);
        let (a, da) = atlas.plane([boundary - 1e-5, 112.3], pixel, 13);
        let (b, db) = atlas.plane([boundary + 1e-5, 112.3], pixel, 13);
        assert!((a - b).abs() < 1e-3);
        assert!(da.iter().zip(db).all(|(a, b)| (a - b).abs() < 1e-3));
    }
}
