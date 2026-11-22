//! GPU-only, independently timestamped presentation poses.
use super::{PhysicsGpu, RigidBody};
use crate::terrain::gpu::{Kernel, RecordedReadback, invalid};
use std::io;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderPose {
    pub position: [f32; 4],
    pub rotation: [f32; 4],
}

/// Requested render timestamp in integral physics ticks plus a fractional
/// tick; it is independent of the number of steps encoded per submission.
#[derive(Clone, Copy, Debug)]
pub struct PresentationTime {
    pub tick: u64,
    pub fraction: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Query {
    time: [u32; 2],
    fraction: f32,
    count: u32,
}

pub struct PresentationHistory {
    pipeline: Kernel,
    source: wgpu::Buffer,
    states: [wgpu::Buffer; 2],
    poses: wgpu::Buffer,
    count: u32,
    current: Option<usize>,
    previous: Option<usize>,
}

impl PresentationHistory {
    /// Allocate two GPU snapshots and a separate render pose buffer.
    ///
    /// # Errors
    /// Rejects unsupported buffer sizes.
    pub fn new(device: &wgpu::Device, physics: &PhysicsGpu) -> io::Result<Self> {
        let count = physics.params.count;
        let source = physics.states();
        let bytes =
            u64::from(count) * u64::try_from(size_of::<RenderPose>()).map_err(io::Error::other)?;
        if bytes
            > device
                .limits()
                .max_storage_buffer_binding_size
                .min(device.limits().max_buffer_size)
        {
            return Err(invalid("render poses exceed GPU buffer limits"));
        }
        let states = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamped physics state on GPU"),
                size: source.size(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let poses = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interpolated vessel render poses"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pipeline = Kernel::new(
            device,
            "interpolate presentation poses",
            concat!(include_str!("body.wgsl"), include_str!("presentation.wgsl")),
            "interpolate",
        );
        Ok(Self {
            pipeline,
            source: source.clone(),
            states,
            poses,
            count,
            current: None,
            previous: None,
        })
    }

    /// Capture the current physics buffer in GPU queue order. This copies
    /// state GPU-to-GPU, not to the CPU. The captured state includes the
    /// actual per-body completed time, including stopped bodies.
    pub fn capture(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let index = self.current.map_or(0, |current| 1 - current);
        encoder.copy_buffer_to_buffer(&self.source, 0, &self.states[index], 0, self.source.size());
        self.previous = self.current.or(Some(index));
        self.current = Some(index);
    }

    /// Compute poses at a requested timestamp, clamped to available states.
    /// Consecutive renders may choose different timestamps without changing
    /// the simulation clock or issuing any physics steps.
    ///
    /// # Errors
    /// Rejects requests before the first capture or invalid fractional ticks.
    pub fn interpolate(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        time: PresentationTime,
    ) -> io::Result<()> {
        if !(0.0..1.0).contains(&time.fraction) {
            return Err(invalid("invalid fractional presentation tick"));
        }
        let previous = self
            .previous
            .ok_or_else(|| invalid("no presentation history"))?;
        let current = self
            .current
            .ok_or_else(|| invalid("no presentation history"))?;
        let [lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3] = time.tick.to_le_bytes();
        let query = Query {
            time: [
                u32::from_le_bytes([lo0, lo1, lo2, lo3]),
                u32::from_le_bytes([hi0, hi1, hi2, hi3]),
            ],
            fraction: time.fraction,
            count: self.count,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("presentation timestamp"),
            contents: bytemuck::bytes_of(&query),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        self.pipeline.encode(
            device,
            encoder,
            "sample presentation time",
            &[
                &self.states[previous],
                &self.states[current],
                &self.poses,
                &uniform,
            ],
            [self.count.div_ceil(64), 1, 1],
        )
    }

    #[must_use = "Bind the interpolated GPU poses in the renderer"]
    pub const fn poses(&self) -> &wgpu::Buffer {
        &self.poses
    }

    /// Optional diagnostic readback; rendering binds `poses()` directly.
    #[must_use = "Submit the encoder and begin the diagnostic map"]
    pub fn snapshot(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RecordedReadback<RenderPose> {
        RecordedReadback::copy(device, encoder, &self.poses, 0, self.poses.size())
    }
}

// The WGSL layout is mirrored as a buffer of seven vec4 fields.
const _: () = assert!(size_of::<RigidBody>() == 112);
