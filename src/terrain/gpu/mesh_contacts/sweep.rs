//! GPU continuous candidate search over known rigid pose intervals.
use super::{MeshBatch, MeshCatalog, MeshContact, invalid, output_buffer};
use std::{io, marker::PhantomData, sync::Arc};
use wgpu::util::DeviceExt;

/// A rigid position and unit quaternion in a shared local metric frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RigidPose {
    pub position: [f32; 4],
    pub rotation: [f32; 4],
}

/// A pair of linearly translated, shortest-arc rotated meshes.
/// The four poses are A start/end followed by B start/end.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SweepPair {
    pub meshes: [u32; 2],
    pub padding: [u32; 2],
    pub poses: [RigidPose; 4],
}

/// Bounds for a normalized forecast interval. Nonrigid or curved
/// animation paths must supply a certified maximum deviation in metres
/// from the interpolated poses, summed over both meshes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SweepSettings {
    pub margin: f32,
    pub time_tolerance: f32,
    pub trajectory_deviation_m: f32,
    pub query_budget: u32,
}

/// Earliest unresolved interval, not a fabricated exact impact time.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SweepResult {
    /// Proven clear-before time, possible-until time, midpoint, radius.
    pub interval: [f32; 4],
    /// 0 clear, 1 candidate bracket, 2 invalid, 3 budget exhausted.
    /// Second word: number of distance queries.
    pub status: [u32; 4],
    /// Closest features under the inflated midpoint query radius.
    pub contact: MeshContact,
}

pub type MeshSweeps = MeshBatch<SweepResult>;

impl MeshCatalog {
    /// Bind a resident GPU-written or CPU-uploaded sweep array.
    ///
    /// # Errors
    /// Rejects invalid buffer dimensions or search settings.
    pub fn sweeps(
        &self,
        device: &wgpu::Device,
        pairs: &wgpu::Buffer,
        settings: SweepSettings,
    ) -> io::Result<MeshSweeps> {
        let stride = std::mem::size_of::<SweepPair>() as u64;
        let count = u32::try_from(pairs.size() / stride).map_err(io::Error::other)?;
        if count == 0
            || count > device.limits().max_compute_workgroups_per_dimension
            || !pairs.size().is_multiple_of(stride)
            || !pairs.usage().contains(wgpu::BufferUsages::STORAGE)
            || !settings.margin.is_finite()
            || settings.margin < 0.0
            || !settings.trajectory_deviation_m.is_finite()
            || settings.trajectory_deviation_m < 0.0
            || !(1.0 / 1_048_576.0..=1.0).contains(&settings.time_tolerance)
            || settings.query_budget == 0
        {
            return Err(invalid("invalid mesh sweep settings"));
        }
        let output = output_buffer::<SweepResult>(device, count, "mesh sweep results")?;
        let kernel = self.sweep_kernel.get_or_init(|| {
            Arc::new(super::Kernel::new(
                device,
                "mesh swept query",
                concat!(
                    include_str!("common.wgsl"),
                    include_str!("containment.wgsl"),
                    include_str!("sweep.wgsl")
                ),
                "sweep",
            ))
        });
        let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh sweep settings"),
            contents: bytemuck::bytes_of(&settings),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = kernel.bind(
            device,
            "resident mesh sweeps",
            &[
                &self.nodes,
                &self.triangles,
                &self.meshes,
                pairs,
                &output,
                &config,
            ],
        )?;
        Ok(MeshSweeps {
            bind,
            output,
            count,
            kernel: Arc::clone(kernel),
            marker: PhantomData,
            label: "mesh continuous candidate search",
        })
    }
}
