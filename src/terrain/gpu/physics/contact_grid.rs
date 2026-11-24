//! Canonical contact-grid sources. Regional detail stays on the GPU;
//! external authored height fields can upload exact immutable lattice
//! samples without building a second procedural terrain generator.
use super::{TerrainGpu, TerrainRequest, invalid};
use crate::terrain::{
    RegionChart,
    gpu::{GridProjection, TerrainSample},
};
use std::io;
use wgpu::util::DeviceExt;

/// A resident physical lattice with a stable slot identity and projection.
/// The caller must retain its ownership throughout every in-flight batch.
pub trait ContactGrid {
    fn samples(&self) -> &wgpu::Buffer;
    fn grid_request(&self, slot: u32) -> Option<TerrainRequest>;
    /// # Errors
    /// Rejects a slot without a valid contact projection.
    fn grid_projection(&self, slot: u32) -> io::Result<GridProjection>;
    fn side(&self) -> u32;
}

impl ContactGrid for TerrainGpu {
    fn samples(&self) -> &wgpu::Buffer {
        self.samples()
    }
    fn grid_request(&self, slot: u32) -> Option<TerrainRequest> {
        self.grid_request(slot)
    }
    fn grid_projection(&self, slot: u32) -> io::Result<GridProjection> {
        self.grid_projection(slot)
    }
    fn side(&self) -> u32 {
        self.side()
    }
}

/// One immutable canonical sample grid provided by an authored terrain field.
pub struct PinnedGrid {
    samples: wgpu::Buffer,
    request: TerrainRequest,
    projection: GridProjection,
    side: u32,
}

impl PinnedGrid {
    /// Upload the precise canonical samples; no hidden procedural perturbation.
    ///
    /// # Errors
    /// Rejects invalid bounds, counts, heights or adapter storage limits.
    pub fn upload(
        device: &wgpu::Device,
        chart: RegionChart,
        request: TerrainRequest,
        side: u32,
        samples: &[TerrainSample],
    ) -> io::Result<Self> {
        if !(2..=257).contains(&side)
            || request.slot != 0
            || request.spacing_ticks == 0
            || usize::try_from(side.pow(2)).map_err(io::Error::other)? != samples.len()
            || !chart.valid()
        {
            return Err(invalid("invalid pinned terrain dimensions"));
        }
        let span = request
            .spacing_ticks
            .checked_mul(side - 1)
            .ok_or_else(|| invalid("terrain extent overflow"))?;
        if !request.origin_ticks.iter().all(|&v| {
            v.checked_add(span)
                .is_some_and(|end| f64::from(end) <= chart.extent() * 16.0)
        }) || samples
            .iter()
            .any(|s| !s.bed_m.is_finite() || !s.water_m.is_finite() || s.water_kind > 2)
        {
            return Err(invalid("invalid pinned terrain sample or address"));
        }
        let elevation = samples
            .iter()
            .map(|s| f64::from(s.bed_m.abs()))
            .fold(0.0, f64::max);
        let projection = GridProjection::new(chart, request, side, elevation)?;
        let bytes = bytemuck::cast_slice(samples);
        if u64::try_from(bytes.len()).map_err(io::Error::other)?
            > device
                .limits()
                .max_buffer_size
                .min(device.limits().max_storage_buffer_binding_size)
        {
            return Err(invalid("pinned terrain exceeds adapter storage limit"));
        }
        Ok(Self {
            samples: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("immutable authored contact grid"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_SRC,
            }),
            request,
            projection,
            side,
        })
    }
}

impl ContactGrid for PinnedGrid {
    fn samples(&self) -> &wgpu::Buffer {
        &self.samples
    }
    fn grid_request(&self, slot: u32) -> Option<TerrainRequest> {
        (slot == 0).then_some(self.request)
    }
    fn grid_projection(&self, slot: u32) -> io::Result<GridProjection> {
        if slot != 0 {
            return Err(invalid("unknown pinned terrain slot"));
        }
        Ok(self.projection)
    }
    fn side(&self) -> u32 {
        self.side
    }
}
