//! Collision geometry is reconstructed from the GPU's stored elevations.
//! The host's double-precision geodetic projection is not a second terrain
//! algorithm. The triangle order is shared with the intended grid renderer.
use super::{CollisionGrid, TICKS_PER_METRE, invalid};
use crate::terrain::RegionChart;
use std::io;

impl CollisionGrid {
    /// Reconstruct the solid terrain vertices in body-fixed metres.
    /// Water elevations and materials never become collision surfaces.
    ///
    /// # Errors
    /// Rejects invalid grid dimensions, sample counts, chart coordinates
    /// or nonfinite GPU elevations.
    pub fn body_vertices(&self, chart: RegionChart) -> io::Result<Vec<[f64; 3]>> {
        let count = self
            .side
            .checked_mul(self.side)
            .ok_or_else(|| invalid("collision grid overflow"))?;
        let span = self
            .side
            .checked_sub(1)
            .and_then(|n| n.checked_mul(self.request.spacing_ticks))
            .ok_or_else(|| invalid("collision grid span overflow"))?;
        if self.side < 2
            || self.request.spacing_ticks == 0
            || usize::try_from(count).ok() != Some(self.samples.len())
            || self.request.origin_ticks.iter().any(|&origin| {
                origin
                    .checked_add(span)
                    .is_none_or(|end| f64::from(end) / TICKS_PER_METRE > chart.extent())
            })
        {
            return Err(invalid("invalid collision grid geometry"));
        }
        let mut vertices = Vec::with_capacity(self.samples.len());
        for (index, sample) in (0..count).zip(&self.samples) {
            let coordinates = [index % self.side, index / self.side];
            let point = std::array::from_fn(|axis| {
                f64::from(
                    self.request.origin_ticks[axis]
                        + coordinates[axis] * self.request.spacing_ticks,
                ) / TICKS_PER_METRE
            });
            let radius = chart.radius_m + f64::from(sample.bed_m);
            if !(radius.is_finite() && radius > 0.0) {
                return Err(invalid("invalid GPU collision elevation"));
            }
            vertices.push(chart.direction(point).map(|v| v * radius));
        }
        Ok(vertices)
    }

    /// Indexed triangles with outward-facing winding on the spherical
    /// chart. This is the fixed southwest-to-northeast diagonal used for
    /// both fine collision and future chart-grid rendering.
    #[must_use = "Use this topology with the reconstructed solid vertices"]
    pub fn triangles(&self) -> impl Iterator<Item = [u32; 3]> + '_ {
        grid_triangles(self.side)
    }
}

pub(super) fn grid_triangles(side: u32) -> impl Iterator<Item = [u32; 3]> {
    let cells = side.saturating_sub(1);
    (0..cells).flat_map(move |row| {
        (0..cells).flat_map(move |column| {
            let sw = row * side + column;
            let se = sw + 1;
            let nw = sw + side;
            let ne = nw + 1;
            // West x north is outward in RegionChart coordinates.
            [[sw, se, ne], [sw, ne, nw]]
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::gpu::{TerrainRequest, TerrainSample};

    #[test]
    fn collision_uses_solid_bed_and_outward_triangles() -> io::Result<()> {
        let chart = RegionChart {
            radius_m: 600_000.0,
            latitude_rad: 0.3,
            longitude_rad: 1.2,
            spacing_m: 250.0,
            cells: 8,
        };
        let grid = CollisionGrid {
            request: TerrainRequest {
                origin_ticks: [16_000, 16_000],
                spacing_ticks: 16,
                slot: 2,
            },
            side: 2,
            samples: vec![
                TerrainSample {
                    bed_m: 100.0,
                    water_m: 120.0,
                    rgba: 0,
                    water_kind: 2
                };
                4
            ],
        };
        let vertices = grid.body_vertices(chart)?;
        assert!(
            vertices
                .iter()
                .all(|v| (v.iter().map(|c| c * c).sum::<f64>().sqrt() - 600_100.0).abs() < 1e-8)
        );
        let triangles: Vec<_> = grid.triangles().collect();
        assert_eq!(triangles, [[0, 1, 3], [0, 3, 2]]);
        let [origin, second, third] = triangles[0].map(|i| vertices[i as usize]);
        let edge_a = std::array::from_fn::<_, 3, _>(|i| second[i] - origin[i]);
        let edge_b = std::array::from_fn::<_, 3, _>(|i| third[i] - origin[i]);
        let normal = [
            edge_a[1].mul_add(edge_b[2], -edge_a[2] * edge_b[1]),
            edge_a[2].mul_add(edge_b[0], -edge_a[0] * edge_b[2]),
            edge_a[0].mul_add(edge_b[1], -edge_a[1] * edge_b[0]),
        ];
        assert!(normal.iter().zip(origin).map(|(n, r)| n * r).sum::<f64>() > 0.0);
        Ok(())
    }
}
