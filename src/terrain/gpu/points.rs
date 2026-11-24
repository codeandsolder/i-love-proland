//! Irregular addresses for curved planet tiles, scatter and custom meshes.
//! The caller owns the topology and projection; the field only returns
//! canonical samples in the requested order.
use super::{Kernel, Resident, TerrainGpu, TerrainRequest, WGSL, invalid, storage};
use crate::terrain::RegionChart;
use std::io;

/// A slot populated with caller-supplied chart lattice coordinates.
#[derive(Clone, Copy, Debug)]
pub struct TerrainPointRequest<'a> {
    pub slot: u32,
    pub addresses: &'a [[u32; 2]],
}

impl RegionChart {
    /// Project a body-fixed direction onto the canonical 1/16 metre lattice.
    ///
    /// Rejects points outside the chart. The last partial lattice interval
    /// (if any) is rounded inward, never beyond the region's control bounds.
    #[must_use = "Use identical addresses for shared spherical tile vertices"]
    pub fn project_ticks(self, direction: [f64; 3]) -> Option<[u32; 2]> {
        let point = self.project(direction)?;
        let extent = (self.extent() * super::TICKS_PER_METRE).floor();
        if !(16.0..=4_000_000.0).contains(&extent) {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "rounded coordinates fit the validated regional extent"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "the projected chart coordinates are nonnegative"
        )]
        Some(point.map(|v| (v * super::TICKS_PER_METRE).round().min(extent) as u32))
    }
}

impl TerrainGpu {
    /// Generate resident samples at explicit chart addresses instead of
    /// an axis-aligned collision grid. Output occupies the same sample
    /// buffer and order as the request; consumers supply their own topology.
    ///
    /// The grid dimension still determines the fixed slot capacity.
    /// All requests must fill one slot exactly. Collision-grid snapshots
    /// and rigid-body solvers deliberately reject these irregular slots.
    ///
    /// # Errors
    /// Rejects out-of-range coordinates, wrong sample counts, unordered or
    /// duplicate slots and device limits before changing resident ownership.
    pub fn encode_points(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        requests: &[TerrainPointRequest<'_>],
    ) -> io::Result<()> {
        if requests.is_empty() {
            return Ok(());
        }
        let count = u32::try_from(requests.len()).map_err(io::Error::other)?;
        if count > device.limits().max_compute_workgroups_per_dimension
            || !requests.windows(2).all(|pair| pair[0].slot < pair[1].slot)
            || requests.iter().any(|request| {
                request.slot as usize >= self.resident.len()
                    || request.addresses.len() != self.samples_per_slot() as usize
                    || request
                        .addresses
                        .iter()
                        .flatten()
                        .any(|&v| v > self.config.extent_ticks)
            })
        {
            return Err(invalid("invalid explicit terrain point batch"));
        }
        // The common job ABI retains origin/spacing for normal grids;
        // explicit-point dispatch reads only the slot identifier.
        let jobs: Vec<_> = requests
            .iter()
            .map(|request| TerrainRequest {
                origin_ticks: [0; 2],
                spacing_ticks: 0,
                slot: request.slot,
            })
            .collect();
        let points: Vec<_> = requests
            .iter()
            .flat_map(|request| request.addresses.iter().copied())
            .collect();
        let job_buffer = storage(
            device,
            "explicit terrain point jobs",
            bytemuck::cast_slice(&jobs),
        )?;
        let point_buffer = storage(
            device,
            "canonical terrain point addresses",
            bytemuck::cast_slice(&points),
        )?;
        let groups = [self.samples_per_slot().div_ceil(64), count, 1];
        let pipeline = self.points_pipeline.get_or_insert_with(|| {
            Kernel::new(
                device,
                "explicit terrain point generation",
                WGSL,
                "generate_points",
            )
        });
        pipeline.encode(
            device,
            encoder,
            "generate explicit terrain points",
            &[
                &self.source.config,
                &self.source.controls,
                &self.source.reaches,
                &self.source.bins,
                &job_buffer,
                &self.samples,
                &point_buffer,
            ],
            groups,
        )?;
        for request in requests {
            self.resident[request.slot as usize] = Some(Resident::Points);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_ticks_is_canonical_and_rejects_outside() {
        let chart = RegionChart {
            radius_m: 600_000.0,
            latitude_rad: 0.1,
            longitude_rad: 0.2,
            spacing_m: 250.0,
            cells: 8,
        };
        let point = [801.25, 1200.125];
        assert_eq!(
            chart.project_ticks(chart.direction(point)),
            Some([12820, 19202])
        );
        assert!(
            chart
                .project_ticks(chart.direction([-10.0, 12.0]))
                .is_none()
        );
    }
}
