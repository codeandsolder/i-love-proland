//! Metric local tangent coordinates for spherical contact and rendering.
//!
//! The horizontal origin remains the nominal southwest grid corner; the
//! tangent basis is anchored at the centre. All geometry uses small offsets
//! rather than subtracting planet-sized f32 Cartesian coordinates.
use super::{TICKS_PER_METRE, TerrainRequest, invalid};
use crate::terrain::RegionChart;
use std::io;

/// Shared local spherical vertex evaluator for GPU physics and drawing.
pub const GRID_PROJECTION_WGSL: &str = include_str!("projection.wgsl");

/// Uniform ABI shared by collision and rendering shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridProjection {
    // radius, sin(centre latitude), cos(centre latitude), cos(chart latitude)
    sphere: [f32; 4],
    // spacing, half span, maximum horizontal displacement in x/y
    grid: [f32; 4],
}

impl GridProjection {
    pub(super) fn new(
        chart: RegionChart,
        request: TerrainRequest,
        side: u32,
        elevation_bound: f64,
    ) -> io::Result<Self> {
        let spacing = f64::from(request.spacing_ticks) / TICKS_PER_METRE;
        let half = f64::from(side - 1) * spacing * 0.5;
        let radius = chart.radius_m;
        let centre_y = f64::from(request.origin_ticks[1]) / TICKS_PER_METRE + half;
        let lat = (centre_y - chart.extent() * 0.5) / radius + chart.latitude_rad;
        let (sin_lat, cos_lat) = lat.sin_cos();
        let cos_chart = chart.latitude_rad.cos();
        let dlat = half / radius;
        let dlon = dlat / cos_chart;
        // The entire triangular mesh lies within these componentwise
        // displacements from its nominal rectangular lattice. Linear
        // interpolation preserves vertex bounds, including at cell edges.
        // Use |sin(t)-t| <= |t|^3/6, |1-cos(t)| <= t^2/2,
        // and the unit Lipschitz constant of cosine.
        // One metre covers rounding of the bounded f32 height field.
        let h = elevation_bound + 1.0;
        let x = h * dlon
            + radius * ((lat - chart.latitude_rad).abs() + dlat) * dlon
            + radius * dlon.powi(3) / 6.0;
        let y = h * dlat + radius * dlat.powi(3) / 6.0 + (radius + h) * dlon.powi(2) * 0.5;
        // All projection operations are on offsets. Include a relative
        // rounding allowance for f32 trigonometry and arithmetic.
        let rounding = 64.0 * f64::from(f32::EPSILON) * (half + x.max(y));
        let bounds = [x + rounding, y + rounding];
        if radius <= h
            || half <= 0.0
            || lat.abs() + dlat >= std::f64::consts::FRAC_PI_2
            || dlon >= std::f64::consts::FRAC_PI_2
            || bounds.iter().any(|&v| !v.is_finite() || v >= half)
        {
            return Err(invalid(
                "terrain grid exceeds local spherical contact bounds",
            ));
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "local GPU uniform, validated finite bounds"
        )]
        let projection = Self {
            sphere: [radius, sin_lat, cos_lat, cos_chart].map(|v| v as f32),
            grid: [spacing, half, bounds[0], bounds[1]].map(|v| v as f32),
        };
        if !projection
            .sphere
            .iter()
            .chain(&projection.grid)
            .all(|v| v.is_finite())
            || projection.sphere[0] <= 0.0
            || projection.sphere[3] <= 0.0
            || projection.grid[0] <= 0.0
        {
            return Err(invalid(
                "local spherical contact uniform cannot be represented",
            ));
        }
        Ok(projection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_projection_bounds_include_spherical_vertices() -> io::Result<()> {
        for latitude in [0.0, 0.7, -1.2, 1.4] {
            let chart = RegionChart {
                radius_m: 600_000.0,
                latitude_rad: latitude,
                longitude_rad: 0.3,
                spacing_m: 100.0,
                cells: 100,
            };
            let request = TerrainRequest {
                origin_ticks: [16_000, 112_000],
                spacing_ticks: 160,
                slot: 0,
            };
            let projection = GridProjection::new(chart, request, 65, 3000.0)?;
            let half = 320.0;
            let centre = [1320.0, 7320.0];
            let up = chart.direction(centre);
            let lon = up[2].atan2(up[0]);
            let lat = up[1].asin();
            let west = [lon.sin(), 0.0, -lon.cos()];
            let north = [-lat.sin() * lon.cos(), lat.cos(), -lat.sin() * lon.sin()];
            let dot =
                |a: [f64; 3], b: [f64; 3]| a.into_iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
            for row in 0..=64 {
                for col in 0..=64 {
                    let xy = [f64::from(col) * 10.0, f64::from(row) * 10.0];
                    let direction = chart.direction([1000.0 + xy[0], 7000.0 + xy[1]]);
                    for bed in [-3000.0, 0.0, 3000.0] {
                        let r = chart.radius_m + bed;
                        let expected = [
                            half + r * dot(direction, west),
                            half + r * dot(direction, north),
                            r * dot(direction, up) - chart.radius_m,
                        ];
                        // Check both horizontal broadphase margins against
                        // independent body-fixed double precision geometry.
                        for axis in 0..2 {
                            assert!(
                                (expected[axis] - xy[axis]).abs()
                                    <= f64::from(projection.grid[axis + 2])
                            );
                        }
                        // Mirror the small-offset shader expression in f64
                        // to verify the basis and the sagitta's sign.
                        let dlat = (xy[1] - half) / chart.radius_m;
                        let dlon = (xy[0] - half) / (chart.radius_m * chart.latitude_rad.cos());
                        let coslat = lat.cos() * dlat.cos() - lat.sin() * dlat.sin();
                        let sag_lon = 2.0 * (dlon * 0.5).sin().powi(2);
                        let sag_lat = 2.0 * (dlat * 0.5).sin().powi(2);
                        let actual = [
                            half + r * coslat * dlon.sin(),
                            half + r * (dlat.sin() + lat.sin() * coslat * sag_lon),
                            bed - r * (sag_lat + lat.cos() * coslat * sag_lon),
                        ];
                        for (a, b) in actual.into_iter().zip(expected) {
                            assert!((a - b).abs() < 1e-8);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
