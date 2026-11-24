//! GPU-generated regional terrain.
//!
//! Render/scatter consumers can bind the resident sample buffer directly;
//! diagnostic consumers can request asynchronous snapshots.
//! The chart lattice is integer-addressed (1/16 m), not camera-relative.
use super::Region;
use std::io;
mod kernel;
mod points;
pub use points::TerrainPointRequest;
mod readback;
use kernel::Kernel;
pub use readback::{PendingReadback, RecordedReadback};
pub mod physics;
use wgpu::util::DeviceExt;

/// Shared GPU terrain sample ABI for downstream storage-buffer consumers.
pub const SAMPLE_WGSL: &str = include_str!("gpu/sample.wgsl");

pub const WGSL: &str = concat!(
    include_str!("gpu/sample.wgsl"),
    include_str!("gpu/field.wgsl")
);
const TICKS_PER_METRE: f64 = 16.0;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

/// One sample, shared by the renderer, scatter and collision readback.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainSample {
    pub bed_m: f32,
    pub water_m: f32,
    /// Linear RGBA8, little-endian channels.
    pub rgba: u32,
    /// 0 dry, 1 standing water, 2 river.
    pub water_kind: u32,
}

/// Square grid request in 1/16 metre chart ticks. Adjacent or nested grids
/// must use matching integer addresses. Each job replaces one resident slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainRequest {
    pub origin_ticks: [u32; 2],
    pub spacing_ticks: u32,
    pub slot: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Config {
    extent_ticks: u32,
    control_side: u32,
    bin_side: u32,
    grid_side: u32,
    spacing_m: f32,
    padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedReach {
    ends: [f32; 4],
    stage_width: [f32; 4],
    bend_valley: [f32; 4],
}

struct Source {
    controls: wgpu::Buffer,
    reaches: wgpu::Buffer,
    bins: wgpu::Buffer,
    config: wgpu::Buffer,
}

fn storage(device: &wgpu::Device, label: &str, contents: &[u8]) -> io::Result<wgpu::Buffer> {
    if contents.is_empty()
        || u64::try_from(contents.len()).map_err(io::Error::other)?
            > device
                .limits()
                .max_storage_buffer_binding_size
                .min(device.limits().max_buffer_size)
    {
        return Err(invalid("terrain storage exceeds device binding limits"));
    }
    Ok(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        }),
    )
}

fn compact_index(buckets: &[Vec<usize>]) -> io::Result<Vec<u32>> {
    // Absolute offsets into one flat word array: [offsets..., reach IDs...].
    let mut offset = buckets
        .len()
        .checked_add(1)
        .ok_or_else(|| invalid("channel index overflow"))?;
    let mut result = Vec::with_capacity(offset);
    for bucket in buckets {
        result.push(u32::try_from(offset).map_err(io::Error::other)?);
        offset = offset
            .checked_add(bucket.len())
            .ok_or_else(|| invalid("channel index overflow"))?;
    }
    result.push(u32::try_from(offset).map_err(io::Error::other)?);
    for bucket in buckets {
        for &index in bucket {
            result.push(u32::try_from(index).map_err(io::Error::other)?);
        }
    }
    Ok(result)
}

impl Source {
    fn new(device: &wgpu::Device, region: &Region, config: Config) -> io::Result<Self> {
        let cells: Vec<_> = region.gpu_controls().collect();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "validated regional metric coordinates and heights fit f32 GPU storage"
        )]
        let mut reaches: Vec<_> = region
            .reaches()
            .iter()
            .map(|r| PackedReach {
                ends: [r.a.xy[0], r.a.xy[1], r.b.xy[0], r.b.xy[1]].map(|v| v as f32),
                stage_width: [r.a.stage, r.b.stage, r.a.width, r.b.width].map(|v| v as f32),
                bend_valley: [r.a.bend, r.b.bend, r.valley, 0.0].map(|v| v as f32),
            })
            .collect();
        if reaches.is_empty() {
            reaches.push(bytemuck::Zeroable::zeroed());
        }
        let (_, buckets) = region.gpu_channels().gpu_buckets();
        let bins = compact_index(buckets)?;
        Ok(Self {
            controls: storage(
                device,
                "terrain macro controls",
                bytemuck::cast_slice(&cells),
            )?,
            reaches: storage(
                device,
                "terrain drainage reaches",
                bytemuck::cast_slice(&reaches),
            )?,
            bins: storage(device, "terrain channel index", bytemuck::cast_slice(&bins))?,
            config: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain field configuration"),
                contents: bytemuck::bytes_of(&config),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
        })
    }
}

#[derive(Clone, Copy)]
enum Resident {
    Grid(TerrainRequest),
    Points,
}

/// Fixed-size resident sample pool; both draw and scatter may consume it
/// without any copy to the CPU. The caller owns slot lifetime and scheduling.
pub struct TerrainGpu {
    source: Source,
    pipeline: Kernel,
    points_pipeline: Option<Kernel>,
    samples: wgpu::Buffer,
    config: Config,
    resident: Vec<Option<Resident>>,
}

impl TerrainGpu {
    /// Create a generator and resident pool for one region and grid size.
    /// A different collision grid resolution may use the same field in a
    /// separate pool on the same device. Both evaluate the canonical region.
    ///
    /// # Errors
    /// Rejects invalid configuration, chart extent, or device storage limits.
    pub fn new(device: &wgpu::Device, region: &Region, side: u16, slots: u32) -> io::Result<Self> {
        if !(2..=257).contains(&side) || slots == 0 {
            return Err(invalid("invalid GPU terrain pool dimensions"));
        }
        let chart = region.chart();
        let ticks = chart.extent() * TICKS_PER_METRE;
        if !(1.0..=4_000_000.0).contains(&ticks) {
            return Err(invalid("invalid GPU terrain chart extent"));
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "validated chart extent and finite regional spacing"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "validated nonnegative lattice extent"
        )]
        let config = Config {
            extent_ticks: ticks.floor() as u32,
            control_side: u32::from(chart.cells) + 1,
            bin_side: u32::try_from(region.gpu_channels().gpu_buckets().0)
                .map_err(io::Error::other)?,
            grid_side: u32::from(side),
            spacing_m: chart.spacing_m as f32,
            padding: [0; 3],
        };
        let stride = u64::from(side).pow(2) * 16;
        let bytes = stride * u64::from(slots);
        let limits = device.limits();
        if bytes
            > limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size)
            || u32::from(side).pow(2).div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(invalid("GPU terrain pool exceeds device limits"));
        }
        let source = Source::new(device, region, config)?;
        let pipeline = Kernel::new(device, "resident terrain generation", WGSL, "generate");
        let samples = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident generated terrain"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(Self {
            source,
            pipeline,
            points_pipeline: None,
            samples,
            config,
            resident: vec![None; usize::try_from(slots).map_err(io::Error::other)?],
        })
    }

    #[must_use = "Bind the resident terrain to downstream GPU consumers"]
    pub const fn samples(&self) -> &wgpu::Buffer {
        &self.samples
    }

    #[must_use = "Use the grid side to address resident samples"]
    pub const fn side(&self) -> u32 {
        self.config.grid_side
    }

    #[must_use = "Use the sample count to address a resident slot"]
    pub const fn samples_per_slot(&self) -> u32 {
        self.side() * self.side()
    }

    /// The canonical outward-facing solid grid topology. Reuse one index
    /// buffer for all slots having the same side length.
    #[must_use = "Render exactly the triangles used by terrain collision"]
    pub fn triangles(&self) -> impl Iterator<Item = [u32; 3]> {
        collision::grid_triangles(self.side())
    }

    /// Encode a batch of unique sorted slot replacements. No CPU evaluation
    /// of terrain detail and no implicit readback. Submit encoders in order.
    ///
    /// # Errors
    /// Rejects invalid coordinates, duplicate/unsorted slots, and device limits.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        requests: &[TerrainRequest],
    ) -> io::Result<()> {
        if requests.is_empty() {
            return Ok(());
        }
        let count = u32::try_from(requests.len()).map_err(io::Error::other)?;
        if count > device.limits().max_compute_workgroups_per_dimension
            || !requests.windows(2).all(|pair| pair[0].slot < pair[1].slot)
            || requests.iter().any(|request| !self.valid_request(*request))
        {
            return Err(invalid("invalid GPU terrain request batch"));
        }
        let jobs = storage(
            device,
            "terrain generation requests",
            bytemuck::cast_slice(requests),
        )?;
        let buffers = [
            &self.source.config,
            &self.source.controls,
            &self.source.reaches,
            &self.source.bins,
            &jobs,
            &self.samples,
        ];
        self.pipeline.encode(
            device,
            encoder,
            "generate terrain slots",
            &buffers,
            [self.samples_per_slot().div_ceil(64), count, 1],
        )?;
        for request in requests {
            self.resident[request.slot as usize] = Some(Resident::Grid(*request));
        }
        Ok(())
    }

    fn valid_request(&self, request: TerrainRequest) -> bool {
        (request.slot as usize) < self.resident.len()
            && request.spacing_ticks > 0
            && request
                .spacing_ticks
                .checked_mul(self.side() - 1)
                .is_some_and(|span| {
                    request.origin_ticks.iter().all(|&origin| {
                        origin
                            .checked_add(span)
                            .is_some_and(|end| end <= self.config.extent_ticks)
                    })
                })
    }

    fn grid_request(&self, slot: u32) -> Option<TerrainRequest> {
        match self.resident.get(slot as usize).copied().flatten()? {
            Resident::Grid(request) => Some(request),
            Resident::Points => None,
        }
    }

    /// Capture raw samples from either kind of resident slot, in request order.
    ///
    /// # Errors
    /// Rejects uninitialized or out-of-range slots.
    pub fn snapshot_samples(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        slot: u32,
    ) -> io::Result<RecordedReadback<TerrainSample>> {
        self.resident
            .get(slot as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| invalid("terrain slot not initialized"))?;
        let bytes = u64::from(self.samples_per_slot()) * 16;
        Ok(RecordedReadback::copy(
            device,
            encoder,
            &self.samples,
            u64::from(slot) * bytes,
            bytes,
        ))
    }

    /// Copy exactly one initialized slot into its own staging buffer.
    /// The request is captured with the data, so later slot reuse cannot
    /// change the address attached to this collision mesh snapshot.
    ///
    /// # Errors
    /// Rejects uninitialized or out-of-range slots.
    pub fn snapshot(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        slot: u32,
    ) -> io::Result<TerrainSnapshot> {
        let request = self
            .grid_request(slot)
            .ok_or_else(|| invalid("collision snapshot requires a regular terrain grid"))?;
        let data = self.snapshot_samples(device, encoder, slot)?;
        Ok(TerrainSnapshot {
            request,
            side: self.side(),
            data,
        })
    }
}

/// A recorded copy. Submit its encoder before starting the asynchronous map.
pub struct TerrainSnapshot {
    request: TerrainRequest,
    side: u32,
    data: RecordedReadback<TerrainSample>,
}

impl TerrainSnapshot {
    #[must_use = "Poll the pending collision snapshot after submission"]
    pub fn begin(self) -> PendingTerrain {
        PendingTerrain {
            request: self.request,
            side: self.side,
            data: self.data.begin(),
        }
    }
}

pub struct PendingTerrain {
    request: TerrainRequest,
    side: u32,
    data: PendingReadback<TerrainSample>,
}

/// Immutable CPU-side collision grid, produced by the same GPU field.
#[derive(Debug)]
pub struct CollisionGrid {
    pub request: TerrainRequest,
    pub side: u32,
    pub samples: Vec<TerrainSample>,
}

impl PendingTerrain {
    /// Nonblocking completion probe; the application polls its device.
    ///
    /// # Errors
    /// Returns a mapping or readback error if the snapshot cannot complete.
    pub fn try_take(&mut self) -> io::Result<Option<CollisionGrid>> {
        Ok(self.data.try_take()?.map(|samples| CollisionGrid {
            request: self.request,
            side: self.side,
            samples,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn channel_index_has_absolute_offsets() -> io::Result<()> {
        assert_eq!(
            compact_index(&[vec![3, 4], vec![], vec![7]])?,
            [4, 6, 6, 7, 3, 4, 7]
        );
        Ok(())
    }
}

mod collision;
