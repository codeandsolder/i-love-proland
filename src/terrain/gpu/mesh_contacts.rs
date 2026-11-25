//! Shared immutable local-space collision meshes and GPU surface queries.
//!
//! The hierarchy never replaces the authored triangles. A sparse occupancy
//! mask at each BVH node rejects empty subvolumes conservatively. Independent
//! resident affine transforms allow animated and articulated parts to move
//! without rebuilding their collision acceleration.
use super::{Kernel, RecordedReadback, invalid, storage};
use std::{
    io,
    marker::PhantomData,
    sync::{Arc, OnceLock},
};
use wgpu::util::DeviceExt;
mod body_response;
mod build;
pub use body_response::{BodyMeshInstance, BodyMeshPoses, MeshImpact, MeshImpulses};
mod sweep;
mod topology;
pub use sweep::{MeshSweeps, RigidPose, SweepPair, SweepResult, SweepSettings};

/// One indexed collision mesh in its authored local coordinate system.
pub struct MeshSource<'a> {
    pub positions: &'a [[f32; 3]],
    pub indices: &'a [u32],
}

/// Two mesh instances and their indices in a resident matrix array.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshPair {
    pub meshes: [u32; 2],
    pub transforms: [u32; 2],
}

/// Query mode. Both modes use identical original triangle narrow phases.
#[derive(Clone, Copy, Debug)]
pub enum Midphase {
    Bvh,
    VoxelBvh,
    DualBvh,
    DualVoxelBvh,
}

/// Surface-only distance or physical closed-shell containment as well.
#[derive(Clone, Copy, Debug)]
pub enum Domain {
    Surface,
    ClosedSolid,
}

/// Per-pair reduced contact candidate and instrumentation.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshContact {
    /// Closest point on A, squared separation (in world metres).
    pub point_a: [f32; 4],
    /// Corresponding point on B, reserved.
    pub point_b: [f32; 4],
    /// B-to-A unit separation normal and separation in metres. Zero
    /// normal means coincident witnesses, not a penetration axis.
    pub normal: [f32; 4],
    /// Authored triangle IDs on A and B, count of near pairs, status.
    /// Status flags: 1 invalid transform, 2 traversal overflow,
    /// 4 closed-shell containment, 8 unresolved containment classification.
    pub features: [u32; 4],
    /// BVH visits, triangle tests, mask rejections, reserved.
    pub statistics: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Triangle {
    a: [f32; 3],
    id: u32,
    b: [f32; 4],
    c: [f32; 3],
    representative: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Node {
    lo: [f32; 3],
    first: u32,
    hi: [f32; 3],
    count: u32,
    escape: u32,
    mask: [u32; 2],
    end: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Mesh {
    nodes: [u32; 2],
    triangles: [u32; 2],
    closed: u32,
    padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Config {
    margin: f32,
    mode: u32,
    count: u32,
    transform_count: u32,
}

/// GPU-resident mesh catalog, shared by every instance of the same asset.
pub struct MeshCatalog {
    nodes: wgpu::Buffer,
    triangles: wgpu::Buffer,
    meshes: wgpu::Buffer,
    count: u32,
    kernels: [OnceLock<Arc<Kernel>>; 4],
    sweep_kernel: OnceLock<Arc<Kernel>>,
}

impl MeshCatalog {
    /// Preprocess immutable meshes and upload their acceleration.
    ///
    /// # Errors
    /// Rejects invalid mesh geometry and device storage overflows.
    pub fn new(device: &wgpu::Device, sources: &[MeshSource<'_>]) -> io::Result<Self> {
        let (nodes, triangles, meshes) = build::catalog(sources)?;
        Ok(Self {
            nodes: storage(device, "mesh collision nodes", bytemuck::cast_slice(&nodes))?,
            triangles: storage(
                device,
                "mesh collision triangles",
                bytemuck::cast_slice(&triangles),
            )?,
            meshes: storage(
                device,
                "mesh collision ranges",
                bytemuck::cast_slice(&meshes),
            )?,
            count: u32::try_from(meshes.len()).map_err(io::Error::other)?,
            kernels: std::array::from_fn(|_| OnceLock::new()),
            sweep_kernel: OnceLock::new(),
        })
    }

    /// Acceleration plus original triangle storage in bytes.
    #[must_use = "Use to account for resident collision storage"]
    pub fn resident_bytes(&self) -> u64 {
        self.nodes.size() + self.triangles.size() + self.meshes.size()
    }

    /// Allocate a reusable query set. Transforms are not copied or mapped:
    /// pass the authoritative storage buffer of column-major affine matrices.
    ///
    /// # Errors
    /// Rejects invalid indices, margins and dispatch or binding sizes.
    pub fn queries(
        &self,
        device: &wgpu::Device,
        transforms: &wgpu::Buffer,
        pairs: &[MeshPair],
        margin: f32,
        mode: Midphase,
        domain: Domain,
    ) -> io::Result<MeshQueries> {
        let count = u32::try_from(pairs.len()).map_err(io::Error::other)?;
        let transform_count = u32::try_from(transforms.size() / 64).map_err(io::Error::other)?;
        if count == 0
            || count > device.limits().max_compute_workgroups_per_dimension
            || !transforms.size().is_multiple_of(64)
            || !transforms.usage().contains(wgpu::BufferUsages::STORAGE)
            || !margin.is_finite()
            || margin < 0.0
            || pairs.iter().any(|p| {
                p.meshes.iter().any(|&i| i >= self.count)
                    || p.transforms.iter().any(|&i| i >= transform_count)
            })
        {
            return Err(invalid("invalid mesh contact query"));
        }
        let pairs = storage(device, "mesh contact pairs", bytemuck::cast_slice(pairs))?;
        let output = output_buffer::<MeshContact>(device, count, "mesh contact results")?;
        let mode_index = match mode {
            Midphase::Bvh => 0,
            Midphase::VoxelBvh => 1,
            Midphase::DualBvh => 2,
            Midphase::DualVoxelBvh => 3,
        };
        let kernel = self.kernels[mode_index].get_or_init(|| {
            Arc::new(Kernel::specialized(
                device,
                "specialized mesh contact query",
                concat!(
                    include_str!("mesh_contacts/common.wgsl"),
                    include_str!("mesh_contacts/containment.wgsl"),
                    include_str!("mesh_contacts/query.wgsl")
                ),
                "query",
                &[
                    ("DUAL", f64::from(u32::from(mode_index & 2 != 0))),
                    ("VOXELS", f64::from(u32::from(mode_index & 1 != 0))),
                ],
            ))
        });
        let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh contact config"),
            contents: bytemuck::bytes_of(&Config {
                margin,
                mode: if matches!(domain, Domain::ClosedSolid) {
                    4
                } else {
                    0
                },
                count,
                transform_count,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = kernel.bind(
            device,
            "resident mesh contact queries",
            &[
                &self.nodes,
                &self.triangles,
                &self.meshes,
                transforms,
                &pairs,
                &output,
                &config,
            ],
        )?;
        Ok(MeshQueries {
            bind,
            output,
            count,
            kernel: Arc::clone(kernel),
            marker: PhantomData,
            label: "mesh contact midphase/narrowphase",
        })
    }
}

fn output_buffer<T>(device: &wgpu::Device, count: u32, label: &str) -> io::Result<wgpu::Buffer> {
    let size = u64::from(count) * std::mem::size_of::<T>() as u64;
    if size
        > device
            .limits()
            .max_storage_buffer_binding_size
            .min(device.limits().max_buffer_size)
    {
        return Err(invalid("mesh output exceeds device limit"));
    }
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
}

/// Reusable resident query resources, typed by the output record ABI.
pub struct MeshBatch<T> {
    bind: wgpu::BindGroup,
    output: wgpu::Buffer,
    count: u32,
    kernel: Arc<Kernel>,
    marker: PhantomData<T>,
    label: &'static str,
}
impl<T: bytemuck::Pod> MeshBatch<T> {
    #[must_use = "Bind the authoritative results to the next GPU stage"]
    pub const fn results(&self) -> &wgpu::Buffer {
        &self.output
    }
    /// Record this batch after its GPU input updates.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.kernel
            .dispatch(encoder, self.label, &self.bind, [self.count, 1, 1]);
    }

    /// Record a diagnostic readback without changing resident results.
    #[must_use = "Submit and poll the recorded snapshot"]
    pub fn snapshot(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RecordedReadback<T> {
        RecordedReadback::copy(device, encoder, &self.output, 0, self.output.size())
    }
}
pub type MeshQueries = MeshBatch<MeshContact>;
