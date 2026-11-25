//! Shared immutable local-space collision meshes and GPU surface queries.
//!
//! The hierarchy never replaces the authored triangles. A sparse occupancy
//! mask at each BVH node rejects empty subvolumes conservatively. Independent
//! resident affine transforms allow animated and articulated parts to move
//! without rebuilding their collision acceleration.
use super::{Kernel, RecordedReadback, invalid, storage};
use std::io;
use wgpu::util::DeviceExt;
mod build;

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
}

/// Per-pair reduced contact candidate and instrumentation.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshContact {
    /// Closest point on A, squared separation (in world metres).
    pub point_a: [f32; 4],
    /// Corresponding point on B, reserved.
    pub point_b: [f32; 4],
    /// Authored triangle IDs on A and B, count of near pairs, status.
    /// Status 0 is valid; 1 indicates an invalid or singular transform.
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
    c: [f32; 4],
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
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Mesh {
    nodes: [u32; 2],
    triangles: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Config {
    margin: f32,
    voxels: u32,
    count: u32,
    transform_count: u32,
}

/// GPU-resident mesh catalog, shared by every instance of the same asset.
pub struct MeshCatalog {
    nodes: wgpu::Buffer,
    triangles: wgpu::Buffer,
    meshes: wgpu::Buffer,
    count: u32,
    kernel: Kernel,
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
            kernel: Kernel::new(
                device,
                "mesh contact query",
                include_str!("mesh_contacts/query.wgsl"),
                "query",
            ),
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
        let size = u64::from(count) * std::mem::size_of::<MeshContact>() as u64;
        if size
            > device
                .limits()
                .max_storage_buffer_binding_size
                .min(device.limits().max_buffer_size)
        {
            return Err(invalid("mesh contact output exceeds device limit"));
        }
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh contact results"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh contact config"),
            contents: bytemuck::bytes_of(&Config {
                margin,
                voxels: u32::from(matches!(mode, Midphase::VoxelBvh)),
                count,
                transform_count,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = self.kernel.bind(
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
        })
    }

    /// Record queries after any GPU updates to the transform buffer.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, queries: &MeshQueries) {
        self.kernel.dispatch(
            encoder,
            "mesh contact midphase/narrowphase",
            &queries.bind,
            [queries.count, 1, 1],
        );
    }
}

/// Reusable resident query resources. Results may feed another compute stage.
pub struct MeshQueries {
    bind: wgpu::BindGroup,
    output: wgpu::Buffer,
    count: u32,
}
impl MeshQueries {
    #[must_use = "Bind the authoritative contact results to the next GPU stage"]
    pub const fn results(&self) -> &wgpu::Buffer {
        &self.output
    }

    /// Record a diagnostic readback without changing the resident result.
    #[must_use = "Submit and poll the recorded snapshot"]
    pub fn snapshot(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RecordedReadback<MeshContact> {
        RecordedReadback::copy(device, encoder, &self.output, 0, self.output.size())
    }
}
