//! GPU body transforms and predictive, separated-surface impact impulses.
//!
//! This stage is intentionally not an overlap correction solver. Coincident
//! witnesses and contained solids request refinement instead of guessing
//! penetration axes. The existing articulated state buffer is authoritative.
use super::{Kernel, MeshContact, RecordedReadback, invalid, output_buffer, storage};
use crate::terrain::gpu::physics::RigidBody;
use std::io;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BodyMeshInstance {
    /// Body index, remaining words reserved.
    pub metadata: [u32; 4],
    /// Mesh-local to body-local transform, including animated node pose.
    pub local: [[f32; 4]; 4],
}

/// Compose resident physics bodies and animated local matrices directly
/// into the same matrix ABI consumed by the mesh query kernel.
pub struct BodyMeshPoses {
    kernel: Kernel,
    bind: wgpu::BindGroup,
    matrices: wgpu::Buffer,
    groups: u32,
}
impl BodyMeshPoses {
    /// # Errors
    /// Rejects invalid buffer dimensions, usage or dispatch size.
    pub fn new(
        device: &wgpu::Device,
        bodies: &wgpu::Buffer,
        instances: &wgpu::Buffer,
    ) -> io::Result<Self> {
        let count = buffer_count::<BodyMeshInstance>(device, instances)?;
        buffer_count::<RigidBody>(device, bodies)?;
        let matrices =
            output_buffer::<[[f32; 4]; 4]>(device, count, "resident physical mesh transforms")?;
        let kernel = Kernel::new(
            device,
            "compose body mesh poses",
            concat!(
                include_str!("../physics/body.wgsl"),
                include_str!("../physics/body_math.wgsl"),
                include_str!("body_poses.wgsl")
            ),
            "compose",
        );
        let bind = kernel.bind(device, "body mesh poses", &[bodies, instances, &matrices])?;
        Ok(Self {
            kernel,
            bind,
            matrices,
            groups: count.div_ceil(64),
        })
    }
    #[must_use = "Bind the authoritative mesh matrices"]
    pub const fn matrices(&self) -> &wgpu::Buffer {
        &self.matrices
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.kernel.dispatch(
            encoder,
            "compose physical mesh poses",
            &self.bind,
            [self.groups, 1, 1],
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshImpact {
    /// World-space point and pre-solve closing normal speed (m/s).
    pub point_speed: [f32; 4],
    /// Actual impulse on A (N s), normal impulse magnitude.
    pub impulse_normal: [f32; 4],
    /// 0 no contact, 1 impulse, 2 separating, 3 refine/unresolved.
    /// Second and third words retain the original triangle IDs.
    pub metadata: [u32; 4],
}

const SOURCE: &str = concat!(
    include_str!("../physics/body.wgsl"),
    include_str!("../physics/body_math.wgsl"),
    include_str!("../physics/sphere_pair_math.wgsl"),
    include_str!("impulse.wgsl")
);

fn buffer_count<T>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> io::Result<u32> {
    let stride = std::mem::size_of::<T>() as u64;
    let count = u32::try_from(buffer.size() / stride).map_err(io::Error::other)?;
    if count == 0
        || !buffer.size().is_multiple_of(stride)
        || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
        || count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
    {
        return Err(invalid("invalid resident mesh physics buffer"));
    }
    Ok(count)
}

fn adjacency(count: u32, links: &[[u32; 2]]) -> io::Result<(Vec<u32>, Vec<u32>)> {
    let mut lists = vec![Vec::new(); count as usize];
    for (i, &[a, b]) in links.iter().enumerate() {
        if a == b || a >= count || b >= count {
            return Err(invalid("invalid mesh response body mapping"));
        }
        let index = u32::try_from(i).map_err(io::Error::other)?;
        lists[a as usize].push(index);
        lists[b as usize].push(index);
    }
    let mut offsets = vec![0];
    let mut ids = Vec::new();
    for list in lists {
        ids.extend(list);
        offsets.push(u32::try_from(ids.len()).map_err(io::Error::other)?);
    }
    Ok((offsets, ids))
}

/// One symmetric Jacobi impact sweep. Body poses remain unchanged;
/// impulses are written back to their authoritative state buffer.
pub struct MeshImpulses {
    prediction: Kernel,
    gather: Kernel,
    predict_bind: wgpu::BindGroup,
    gather_bind: wgpu::BindGroup,
    states: wgpu::Buffer,
    scratch: wgpu::Buffer,
    impacts: wgpu::Buffer,
    pair_groups: u32,
    body_groups: u32,
}
impl MeshImpulses {
    /// Bind query results in the same order as the two-body link array.
    ///
    /// # Errors
    /// Rejects invalid mappings and buffer sizes. States require COPY_DST
    /// for the final immutable-snapshot-to-authoritative GPU copy.
    pub fn new(
        device: &wgpu::Device,
        states: &wgpu::Buffer,
        contacts: &wgpu::Buffer,
        links: &[[u32; 2]],
    ) -> io::Result<Self> {
        let bodies = buffer_count::<RigidBody>(device, states)?;
        let pairs = buffer_count::<MeshContact>(device, contacts)?;
        if links.len() != pairs as usize || !states.usage().contains(wgpu::BufferUsages::COPY_DST) {
            return Err(invalid("invalid mesh impact buffer mapping"));
        }
        let (offsets, ids) = adjacency(bodies, links)?;
        let links = storage(
            device,
            "mesh impact body links",
            bytemuck::cast_slice(links),
        )?;
        let offsets = storage(
            device,
            "mesh body adjacency offsets",
            bytemuck::cast_slice(&offsets),
        )?;
        let ids = storage(
            device,
            "mesh body adjacency IDs",
            bytemuck::cast_slice(&ids),
        )?;
        let raw = output_buffer::<MeshImpact>(device, pairs, "unrelaxed mesh impulses")?;
        let impacts = output_buffer::<MeshImpact>(device, pairs, "applied mesh impulses")?;
        let scratch = output_buffer::<RigidBody>(device, bodies, "mesh impulse body snapshot")?;
        let prediction = Kernel::new(device, "predict mesh impulses", SOURCE, "predict_impulses");
        let gather = Kernel::new(device, "gather mesh impulses", SOURCE, "gather_impulses");
        let predict_bind = prediction.bind(
            device,
            "mesh impulse inputs",
            &[states, contacts, &links, &raw],
        )?;
        let gather_bind = gather.bind_numbered(
            device,
            "mesh impulse gather",
            &[
                (0, states),
                (2, &links),
                (3, &raw),
                (4, &offsets),
                (5, &ids),
                (6, &scratch),
                (7, &impacts),
            ],
        );
        Ok(Self {
            prediction,
            gather,
            predict_bind,
            gather_bind,
            states: states.clone(),
            scratch,
            impacts,
            pair_groups: pairs.div_ceil(64),
            body_groups: bodies.div_ceil(64),
        })
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.prediction.dispatch(
            encoder,
            "predict mesh impulses",
            &self.predict_bind,
            [self.pair_groups, 1, 1],
        );
        self.gather.dispatch(
            encoder,
            "gather reciprocal mesh impulses",
            &self.gather_bind,
            [self.body_groups, 1, 1],
        );
        encoder.copy_buffer_to_buffer(&self.scratch, 0, &self.states, 0, self.states.size());
    }
    /// Snapshot authoritative body states for regression validation.
    #[must_use = "Submit and poll the state snapshot"]
    pub fn snapshot_states(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RecordedReadback<RigidBody> {
        RecordedReadback::copy(device, encoder, &self.states, 0, self.states.size())
    }
    #[must_use = "Consume actual impulses and unresolved contacts"]
    pub const fn impacts(&self) -> &wgpu::Buffer {
        &self.impacts
    }
    #[must_use = "Submit and poll the impact snapshot"]
    pub fn snapshot(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RecordedReadback<MeshImpact> {
        RecordedReadback::copy(device, encoder, &self.impacts, 0, self.impacts.size())
    }
}
