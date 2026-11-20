//! GPU visibility, LOD selection, and indirect-draw compaction.
//!
//! This optional module deliberately never creates canonical objects. The
//! independent, deterministic CPU generator produces each stable ID and
//! variant; the GPU handles only camera-dependent visibility and mesh versus
//! impostor selection. GPU atomic append order has no persistence semantics.

use crate::{Error, Instance, Result};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

pub const WGSL_COMPACTION: &str = include_str!("compact.wgsl");

const INSTANCE_BYTES: u64 = 96;
const WORKGROUP_SIZE: u32 = 64;

/// An explicit, stable, shader-compatible 96-byte instance layout. All
/// positions are camera-relative; large planetary coordinates remain in the
/// caller's precise domain until the final render projection.
///
/// Identifiers and variants survive the GPU pipeline without rehashing.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedInstance {
    pub position_radius: [f32; 4],
    /// Unit quaternion in XYZW order, supporting terrain alignment/tilt.
    pub rotation: [f32; 4],
    /// Per-axis physical radii for a source normalized within a unit sphere;
    /// fourth component is deterministic appearance variation.
    pub axis_radii: [f32; 4],
    pub object_id: [u32; 4],
    pub metadata: [u32; 4],
    /// Independent per-object RGBA tint in linear source color space.
    /// The material's vertex tint is baked into its source mesh/atlas.
    pub tint: [f32; 4],
}

/// A visual pose computed from whichever terrain LOD is being displayed.
///
/// Source positions must already be normalized about their source sphere;
/// axis radii are the corresponding physical XYZ scale factors in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderPose {
    pub camera_relative_position: [f32; 3],
    pub rotation_xyzw: [f32; 4],
    pub axis_radii_m: [f32; 3],
}

impl RenderPose {
    /// Upright source pose, with one physical radius and yaw in full turns.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a canonical f64 yaw is projected to f32 only for GPU visual orientation"
    )]
    pub fn upright(position: [f32; 3], radius: f32, yaw_turns: f64) -> Self {
        let half = yaw_turns * core::f64::consts::PI;
        let (sin, cos) = half.sin_cos();
        Self {
            camera_relative_position: position,
            rotation_xyzw: [0.0, sin as f32, 0.0, cos as f32],
            axis_radii_m: [radius; 3],
        }
    }
}

impl PackedInstance {
    /// Convert a canonical object into renderer-friendly data.
    ///
    /// # Errors
    /// Returns an error on non-finite projection or invalid radius/scale.
    pub fn from_instance(instance: &Instance, pose: RenderPose) -> Result<Self> {
        if !pose.camera_relative_position.iter().all(|v| v.is_finite())
            || !pose.axis_radii_m.iter().all(|v| v.is_finite() && *v > 0.0)
            || !pose.rotation_xyzw.iter().all(|v| v.is_finite())
        {
            return Err(Error::InvalidConfiguration(
                "GPU render pose needs finite position, positive axis radii and unit quaternion",
            ));
        }
        let norm_sq = pose.rotation_xyzw.iter().map(|x| x * x).sum::<f32>();
        if (norm_sq - 1.0).abs() > 1.0e-3 {
            return Err(Error::InvalidConfiguration(
                "GPU source quaternion must be normalized",
            ));
        }
        let radius = pose
            .axis_radii_m
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(0.0);
        if !radius.is_finite() {
            return Err(Error::InvalidConfiguration(
                "invalid GPU bounding sphere radius",
            ));
        }

        let id = instance.candidate.id;
        let model = instance.variant_id;
        let layer = instance.candidate.layer_id;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "u128/u64 stable IDs are explicitly serialized into low-to-high u32 limbs"
        )]
        let words = Self {
            position_radius: [
                pose.camera_relative_position[0],
                pose.camera_relative_position[1],
                pose.camera_relative_position[2],
                radius,
            ],
            rotation: pose.rotation_xyzw,
            axis_radii: [
                pose.axis_radii_m[0],
                pose.axis_radii_m[1],
                pose.axis_radii_m[2],
                instance.candidate.appearance_unit as f32,
            ],
            object_id: [
                id as u32,
                (id >> 32) as u32,
                (id >> 64) as u32,
                (id >> 96) as u32,
            ],
            metadata: [
                model as u32,
                (model >> 32) as u32,
                layer as u32,
                (layer >> 32) as u32,
            ],
            tint: [1.0; 4],
        };
        Ok(words)
    }

    /// Return a copy with validated per-object tint, without touching
    /// canonical IDs, variant metadata, world pose or visual LOD.
    ///
    /// RGB may exceed 1 for physically meaningful bright tints.
    /// Alpha is a conventional [0, 1] transparency multiplier.
    ///
    /// # Errors
    /// Rejects NaN, infinity, negative RGB or out-of-range alpha.
    pub fn with_tint(mut self, tint: [f32; 4]) -> Result<Self> {
        if !tint.iter().all(|x| x.is_finite())
            || tint[..3].iter().any(|x| *x < 0.0)
            || !(0.0..=1.0).contains(&tint[3])
        {
            return Err(Error::InvalidConfiguration(
                "GPU per-instance tint requires finite nonnegative RGB and alpha in 0..1",
            ));
        }
        self.tint = tint;
        Ok(self)
    }

    #[must_use]
    pub const fn stable_id(self) -> u128 {
        (self.object_id[0] as u128)
            | ((self.object_id[1] as u128) << 32)
            | ((self.object_id[2] as u128) << 64)
            | ((self.object_id[3] as u128) << 96)
    }

    #[must_use]
    pub fn variant_id(self) -> u64 {
        u64::from(self.metadata[0]) | (u64::from(self.metadata[1]) << 32)
    }
}

/// Conservative camera frustum and LOD parameters.
///
/// The plane equation is `dot(normal, camera_relative_position) + offset >= -crown_radius`.
/// Supply six inward-facing, normalized planes, or a zero plane to disable
/// that particular culling boundary.
#[derive(Clone, Copy, Debug)]
pub struct GpuView {
    focal_length_px: f32,
    mesh_threshold_px: f32,
    impostor_threshold_px: f32,
    forward: [f32; 3],
    planes: [[f32; 4]; 6],
    source_center: [f32; 3],
}

impl GpuView {
    /// # Errors
    /// Rejects non-finite or malformed projection, camera basis, and planes.
    pub fn new(
        focal_length_px: f32,
        mesh_threshold_px: f32,
        impostor_threshold_px: f32,
        forward: [f32; 3],
        planes: [[f32; 4]; 6],
    ) -> Result<Self> {
        if !focal_length_px.is_finite()
            || focal_length_px <= 0.0
            || !mesh_threshold_px.is_finite()
            || mesh_threshold_px <= 0.0
            || !impostor_threshold_px.is_finite()
            || impostor_threshold_px <= 0.0
            || mesh_threshold_px <= impostor_threshold_px
            || !forward.iter().all(|v| v.is_finite())
        {
            return Err(Error::InvalidConfiguration("invalid GPU LOD view"));
        }
        let forward_norm = forward.iter().map(|v| v * v).sum::<f32>().sqrt();
        if (forward_norm - 1.0).abs() > 1.0e-4 {
            return Err(Error::InvalidConfiguration(
                "GPU forward basis must be unit length",
            ));
        }
        for plane in &planes {
            if !plane.iter().all(|v| v.is_finite()) {
                return Err(Error::InvalidConfiguration(
                    "GPU frustum plane is not finite",
                ));
            }
            let normal = plane[..3].iter().map(|v| v * v).sum::<f32>().sqrt();
            if normal == 0.0 && plane[3] == 0.0 {
                continue;
            }
            if (normal - 1.0).abs() > 1.0e-4 {
                return Err(Error::InvalidConfiguration(
                    "GPU frustum plane normals must be unit length",
                ));
            }
        }
        Ok(Self {
            focal_length_px,
            mesh_threshold_px,
            impostor_threshold_px,
            forward,
            planes,
            source_center: [0.0; 3],
        })
    }

    /// Supply the original mesh bounds centre relative to its placement anchor,
    /// in unit-sphere mesh coordinates. Distance LOD selection then uses
    /// the reconstructed physical anchor even when the source is offset,
    /// rotated, or nonuniformly scaled.
    ///
    /// # Errors
    /// Rejects non-finite source coordinates.
    pub fn with_source_center(mut self, source_center: [f32; 3]) -> Result<Self> {
        if source_center.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidConfiguration("nonfinite source center"));
        }
        self.source_center = source_center;
        Ok(self)
    }

    /// Disable frustum plane rejection while keeping correct size-based LOD.
    /// Suitable for tests or when the host has already done tile-level culling.
    ///
    /// # Errors
    /// Rejects invalid focal length or size thresholds.
    pub fn unculled(
        focal_length_px: f32,
        mesh_threshold_px: f32,
        impostor_threshold_px: f32,
    ) -> Result<Self> {
        Self::new(
            focal_length_px,
            mesh_threshold_px,
            impostor_threshold_px,
            [0.0, 0.0, 1.0],
            [[0.0; 4]; 6],
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuUniform {
    counts: [u32; 4],
    lod: [f32; 4],
    forward: [f32; 4],
    source_center: [f32; 4],
    planes: [[f32; 4]; 6],
}

/// GPU-indexed indirect drawing arguments. The instance count is an atomic
/// in the WGSL compute stage, and a normal u32 in an indirect draw.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndexedDraw {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// An authored mesh tier begins at a distance in metres from the physical
/// object's placement anchor. The first tier always starts at zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshLod {
    pub start_m: f32,
    pub index_count: u32,
}

/// Validated, strictly ordered authored mesh tiers. The final mesh tier
/// extends until the screen-size-based impostor crossover.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshLods {
    tiers: Vec<MeshLod>,
}

impl MeshLods {
    /// # Errors
    /// Rejects an empty set, nonzero first boundary, duplicate/nonfinite
    /// distances, overflowing squared boundaries, or empty geometry.
    pub fn new(tiers: Vec<MeshLod>) -> Result<Self> {
        if tiers.first().is_none_or(|t| t.start_m != 0.0)
            || tiers.iter().any(|t| {
                !t.start_m.is_finite()
                    || t.start_m < 0.0
                    || !(t.start_m * t.start_m).is_finite()
                    || t.index_count == 0
            })
            || tiers
                .windows(2)
                .any(|w| w[0].start_m * w[0].start_m >= w[1].start_m * w[1].start_m)
        {
            return Err(Error::InvalidConfiguration(
                "invalid ordered mesh LOD ranges",
            ));
        }
        Ok(Self { tiers })
    }

    #[must_use]
    pub fn tiers(&self) -> &[MeshLod] {
        &self.tiers
    }

    #[must_use]
    pub const fn impostor_bin(&self) -> usize {
        self.tiers.len()
    }

    /// The last start boundary at or below the squared anchor distance.
    fn tier_for_distance_squared(&self, distance_squared: f32) -> usize {
        self.tiers
            .partition_point(|t| t.start_m * t.start_m <= distance_squared)
            .saturating_sub(1)
    }
}

/// Stable reference classification for unit tests and GPU parity checks.
/// Placement identity is independent of this visual classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisualLod {
    Mesh(usize),
    Impostor,
    TerrainCanopy,
    OutsideFrustum,
}

/// Rotate a scaled source-center offset by the physical world quaternion.
fn source_anchor(object: &PackedInstance, view: GpuView) -> [f32; 3] {
    let v = core::array::from_fn(|i| view.source_center[i] * object.axis_radii[i]);
    let q = [object.rotation[0], object.rotation[1], object.rotation[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[2].mul_add(-b[1], a[1] * b[2]),
            a[0].mul_add(-b[2], a[2] * b[0]),
            a[1].mul_add(-b[0], a[0] * b[1]),
        ]
    };
    let t = cross(q, v).map(|x| x * 2.0);
    let u = cross(q, t);
    core::array::from_fn(|i| {
        object.position_radius[i] - (object.rotation[3].mul_add(t[i], v[i]) + u[i])
    })
}

/// Reference classifier for parity testing with the WGSL shader.
///
/// Both paths use the same f32 inputs and cross-multiplied pixel thresholds;
/// GPU math can still differ slightly near boundaries, affecting visual
/// LOD only, never canonical object identity.
#[must_use]
pub fn classify_cpu(object: &PackedInstance, view: GpuView, mesh_lods: &MeshLods) -> VisualLod {
    let pos = &object.position_radius[..3];
    let radius = object.position_radius[3];
    if view
        .planes
        .iter()
        .any(|p| p[0].mul_add(pos[0], p[1].mul_add(pos[1], p[2] * pos[2])) + p[3] < -radius)
    {
        return VisualLod::OutsideFrustum;
    }
    let depth = view.forward[0].mul_add(
        pos[0],
        view.forward[1].mul_add(pos[1], view.forward[2] * pos[2]),
    );
    if depth + radius <= 0.0 {
        return VisualLod::OutsideFrustum;
    }
    let projected_crown = (2.0 * radius) * view.focal_length_px;
    let safe_depth = depth.max(0.001);
    if projected_crown >= view.mesh_threshold_px * safe_depth {
        let anchor = source_anchor(object, view);
        let distance_squared = anchor.iter().map(|x| x * x).sum();
        VisualLod::Mesh(mesh_lods.tier_for_distance_squared(distance_squared))
    } else if projected_crown >= view.impostor_threshold_px * safe_depth {
        VisualLod::Impostor
    } else {
        VisualLod::TerrainCanopy
    }
}

/// One compacted instance-bin binding and its indexed indirect record.
///
/// Each bin is an aligned, independently bindable slice of one storage
/// buffer. The same instance record is written at most once per dispatch.
pub struct CompactedBin<'a> {
    pub instances: wgpu::BufferBinding<'a>,
    pub indirect: &'a wgpu::Buffer,
    pub indirect_offset: u64,
    pub index_count: u32,
}

/// GPU classification for ordered mesh tiers and one impostor bin.
///
/// Canonical IDs and attributes survive unchanged; output order within each
/// bin is intentionally unspecified.
pub struct GpuCompactor {
    capacity: u32,
    mesh_lods: MeshLods,
    input: wgpu::Buffer,
    visible: wgpu::Buffer,
    indirect: wgpu::Buffer,
    bin_stride: u64,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    templates: Vec<IndexedDraw>,
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// A bin offsets whole instance records, and also meets WGPU storage
/// binding offset alignment. Padding is private, never counted/drawn.
fn aligned_bin_stride(capacity: u32, alignment: u64) -> Option<u64> {
    let alignment = alignment.max(1);
    let records_alignment = alignment / gcd(INSTANCE_BYTES, alignment);
    u64::from(capacity)
        .div_ceil(records_alignment)
        .checked_mul(records_alignment)?
        .checked_mul(INSTANCE_BYTES)
}

fn create_compaction_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("i-love-proland visibility + LOD compaction WGSL"),
        source: wgpu::ShaderSource::Wgsl(WGSL_COMPACTION.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("i-love-proland GPU compaction"),
        layout: None,
        module: &shader,
        entry_point: Some("classify_and_compact"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn compaction_sizes(
    device: &wgpu::Device,
    capacity: u32,
    mesh_lods: &MeshLods,
    impostor_index_count: u32,
) -> Result<(u64, u64, u64)> {
    let limits = device.limits();
    let bytes = u64::from(capacity) * INSTANCE_BYTES;
    let bins = mesh_lods
        .tiers
        .len()
        .checked_add(1)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(Error::InvalidConfiguration("GPU LOD bin count overflow"))?;
    let bin_stride = aligned_bin_stride(
        capacity,
        u64::from(limits.min_storage_buffer_offset_alignment),
    )
    .ok_or(Error::InvalidConfiguration("GPU LOD stride overflow"))?;
    let visible_bytes = bin_stride
        .checked_mul(bins)
        .ok_or(Error::InvalidConfiguration("GPU LOD storage overflow"))?;
    if capacity == 0
        || impostor_index_count == 0
        || bytes > limits.max_storage_buffer_binding_size
        || bytes > limits.max_buffer_size
        || visible_bytes > limits.max_storage_buffer_binding_size
        || visible_bytes > limits.max_buffer_size
        || visible_bytes / INSTANCE_BYTES > u64::from(u32::MAX)
        || capacity.div_ceil(WORKGROUP_SIZE) > limits.max_compute_workgroups_per_dimension
    {
        return Err(Error::InvalidConfiguration(
            "GPU compactor capacity exceeds device limits",
        ));
    }
    Ok((bytes, bin_stride, visible_bytes))
}

impl GpuCompactor {
    /// # Errors
    /// Rejects capacity, tier count, storage or workgroup requirements
    /// exceeding device limits, and a zero impostor index count.
    pub fn new(
        device: &wgpu::Device,
        capacity: u32,
        mesh_lods: MeshLods,
        impostor_index_count: u32,
    ) -> Result<Self> {
        let (bytes, bin_stride, visible_bytes) =
            compaction_sizes(device, capacity, &mesh_lods, impostor_index_count)?;
        let create_instance_buffer = |name, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let input = create_instance_buffer("i-love-proland candidate input", bytes);
        let visible = create_instance_buffer("i-love-proland compacted LOD bins", visible_bytes);
        let templates = mesh_lods
            .tiers
            .iter()
            .map(|tier| tier.index_count)
            .chain(core::iter::once(impostor_index_count))
            .map(|index_count| IndexedDraw {
                index_count,
                instance_count: 0,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            })
            .collect::<Vec<_>>();
        let indirect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("i-love-proland LOD indirect arguments"),
            contents: bytemuck::cast_slice(&templates),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let distance_squared = mesh_lods
            .tiers
            .iter()
            .map(|tier| tier.start_m * tier.start_m)
            .collect::<Vec<_>>();
        let thresholds = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("i-love-proland authored squared mesh LOD boundaries"),
            contents: bytemuck::cast_slice(&distance_squared),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("i-love-proland GPU view parameters"),
            size: core::mem::size_of::<GpuUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = create_compaction_pipeline(device);
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("i-love-proland GPU buffers"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indirect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: thresholds.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            capacity,
            mesh_lods,
            input,
            visible,
            indirect,
            bin_stride,
            uniforms,
            bind_group,
            pipeline,
            templates,
        })
    }

    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    #[must_use]
    pub const fn mesh_lods(&self) -> &MeshLods {
        &self.mesh_lods
    }

    #[must_use]
    pub const fn bin_count(&self) -> usize {
        self.templates.len()
    }

    /// Bind or copy the exact capacity-sized instance slice and the
    /// corresponding indirect record. Mesh bins precede the impostor bin.
    #[must_use]
    pub fn bin(&self, index: usize) -> Option<CompactedBin<'_>> {
        let template = self.templates.get(index)?;
        let index = u64::try_from(index).ok()?;
        Some(CompactedBin {
            instances: wgpu::BufferBinding {
                buffer: &self.visible,
                offset: index * self.bin_stride,
                size: NonZeroU64::new(u64::from(self.capacity) * INSTANCE_BYTES),
            },
            indirect: &self.indirect,
            indirect_offset: index * core::mem::size_of::<IndexedDraw>() as u64,
            index_count: template.index_count,
        })
    }

    /// Write candidates and reset every indexed indirect record, then
    /// record one classification/compaction compute pass.
    ///
    /// # Errors
    /// Rejects candidate batches exceeding the fixed instance capacity.
    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        candidates: &[PackedInstance],
        view: GpuView,
    ) -> Result<()> {
        let count = u32::try_from(candidates.len())
            .map_err(|_| Error::InvalidConfiguration("too many GPU instances"))?;
        if count > self.capacity {
            return Err(Error::InvalidConfiguration(
                "GPU instance input exceeds buffer capacity",
            ));
        }
        let uniform = GpuUniform {
            counts: [
                count,
                u32::try_from(self.bin_stride / INSTANCE_BYTES)
                    .map_err(|_| Error::InvalidConfiguration("GPU LOD bin stride overflow"))?,
                u32::try_from(self.mesh_lods.tiers.len())
                    .map_err(|_| Error::InvalidConfiguration("GPU mesh LOD count overflow"))?,
                0,
            ],
            lod: [
                view.focal_length_px,
                view.mesh_threshold_px,
                view.impostor_threshold_px,
                0.0,
            ],
            forward: [view.forward[0], view.forward[1], view.forward[2], 0.0],
            source_center: [
                view.source_center[0],
                view.source_center[1],
                view.source_center[2],
                0.0,
            ],
            planes: view.planes,
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(&self.indirect, 0, bytemuck::cast_slice(&self.templates));
        if count == 0 {
            return Ok(());
        }
        queue.write_buffer(&self.input, 0, bytemuck::cast_slice(candidates));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("i-love-proland compact and classify visible instances"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Distribution, Domain, Generator, GeneratorConfig, Layer, TileKey, Variant};

    #[test]
    fn gpu_layout_and_identity_roundtrip() {
        assert_eq!(core::mem::size_of::<PackedInstance>(), 96);
        assert_eq!(core::mem::size_of::<GpuUniform>(), 160);
        assert_eq!(core::mem::size_of::<IndexedDraw>(), 20);
        let g = Generator::new(
            Domain::Planar { cell_size_m: 10.0 },
            GeneratorConfig::default(),
        )
        .unwrap();
        let layer = Layer::new(
            111,
            0.01,
            Distribution::Poisson,
            vec![Variant {
                id: u64::MAX - 1,
                weight: 1.0,
            }],
        )
        .unwrap();
        let original = g.tile(TileKey::planar(1, 2), &layer).unwrap()[0];
        let packed = PackedInstance::from_instance(
            &original,
            RenderPose::upright([4.0, -6.0, 50.0], 5.0, 0.3),
        )
        .unwrap();
        assert_eq!(packed.stable_id(), original.candidate.id);
        assert_eq!(packed.variant_id(), original.variant_id);
        assert_eq!(packed.metadata[2], 111);
        assert_eq!(packed.tint, [1.0; 4]);
        let colored = packed.with_tint([0.4, 0.7, 1.5, 0.75]).unwrap();
        assert_eq!(colored.tint, [0.4, 0.7, 1.5, 0.75]);
        assert_eq!(colored.stable_id(), packed.stable_id());
        assert_eq!(colored.metadata, packed.metadata);
        assert_eq!(colored.position_radius, packed.position_radius);
        assert!(packed.with_tint([f32::NAN, 1.0, 1.0, 1.0]).is_err());
        assert!(packed.with_tint([-0.1, 1.0, 1.0, 1.0]).is_err());
        assert!(packed.with_tint([1.0, 1.0, 1.0, 1.1]).is_err());
        assert!(
            PackedInstance::from_instance(
                &original,
                RenderPose::upright([f32::INFINITY, 0.0, 0.0], 1.0, 0.0)
            )
            .is_err()
        );
    }

    #[test]
    fn arbitrary_orientation_and_anisotropic_source_radii_are_preserved() {
        let world = Generator::new(
            Domain::Planar { cell_size_m: 10.0 },
            GeneratorConfig::default(),
        )
        .unwrap();
        let layer = Layer::new(
            77,
            0.2,
            Distribution::Poisson,
            vec![Variant {
                id: 314,
                weight: 1.0,
            }],
        )
        .unwrap();
        let original = world.tile(TileKey::planar(0, 0), &layer).unwrap()[0];
        let xyz = [0.18_f32, -0.34, 0.25];
        let w = (1.0 - xyz.iter().map(|v| v * v).sum::<f32>()).sqrt();
        let q = [xyz[0], xyz[1], xyz[2], w];
        let pose = RenderPose {
            camera_relative_position: [-14.0, 20.0, 88.0],
            rotation_xyzw: q,
            axis_radii_m: [3.0, 7.0, 2.0],
        };
        let packed = PackedInstance::from_instance(&original, pose).unwrap();
        assert_eq!(packed.rotation, q);
        assert_eq!(packed.axis_radii[..3], [3.0, 7.0, 2.0]);
        assert_eq!(packed.position_radius, [-14.0, 20.0, 88.0, 7.0]);
        assert_eq!(packed.stable_id(), original.candidate.id);
        assert_eq!(packed.variant_id(), original.variant_id);
        assert_eq!(core::mem::size_of::<PackedInstance>(), 96);
        let mut invalid = pose;
        invalid.rotation_xyzw[3] = 2.0;
        assert!(PackedInstance::from_instance(&original, invalid).is_err());
        invalid = pose;
        invalid.axis_radii_m[1] = 0.0;
        assert!(PackedInstance::from_instance(&original, invalid).is_err());
        invalid = pose;
        invalid.axis_radii_m[2] = f32::INFINITY;
        assert!(PackedInstance::from_instance(&original, invalid).is_err());
    }

    #[test]
    fn reference_classification_respects_crown_pixels_and_frustum() {
        let view = GpuView::unculled(540.0, 32.0, 1.0).unwrap();
        let lods = MeshLods::new(vec![MeshLod {
            start_m: 0.0,
            index_count: 3,
        }])
        .unwrap();
        let object = |z, radius| PackedInstance {
            position_radius: [0.0, 0.0, z, radius],
            rotation: [0.0, 0.0, 0.0, 1.0],
            axis_radii: [radius, radius, radius, 0.0],
            object_id: [0; 4],
            metadata: [0; 4],
            tint: [1.0; 4],
        };
        assert_eq!(
            classify_cpu(&object(100.0, 10.0), view, &lods),
            VisualLod::Mesh(0)
        );
        assert_eq!(
            classify_cpu(&object(500.0, 10.0), view, &lods),
            VisualLod::Impostor
        );
        assert_eq!(
            classify_cpu(&object(25_000.0, 10.0), view, &lods),
            VisualLod::TerrainCanopy
        );
        assert_eq!(
            classify_cpu(&object(-100.0, 10.0), view, &lods),
            VisualLod::OutsideFrustum
        );
        let plane = [1.0, 0.0, 0.0, 0.0];
        let clipped = GpuView::new(540.0, 32.0, 1.0, [0.0, 0.0, 1.0], [plane; 6]).unwrap();
        let mut behind = object(100.0, 1.0);
        behind.position_radius[0] = -20.0;
        assert_eq!(
            classify_cpu(&behind, clipped, &lods),
            VisualLod::OutsideFrustum
        );
    }
    #[test]
    fn arbitrary_lod_ranges_and_alignment() {
        let tier = |start_m| MeshLod {
            start_m,
            index_count: 3,
        };
        assert!(MeshLods::new(vec![]).is_err());
        assert!(MeshLods::new(vec![tier(1.0)]).is_err());
        assert!(MeshLods::new(vec![tier(0.0), tier(0.0)]).is_err());
        assert!(MeshLods::new(vec![tier(0.0), tier(f32::INFINITY)]).is_err());
        assert!(MeshLods::new(vec![tier(0.0), tier(f32::MAX)]).is_err());
        assert!(
            MeshLods::new(vec![MeshLod {
                start_m: 0.0,
                index_count: 0
            }])
            .is_err()
        );
        let lods = MeshLods::new(vec![tier(0.0), tier(10.0), tier(20.0), tier(90.0)]).unwrap();
        assert_eq!(lods.impostor_bin(), 4);
        for (d, expected) in [
            (0.0, 0),
            (9.0, 0),
            (10.0, 1),
            (20.0, 2),
            (90.0, 3),
            (190.0, 3),
        ] {
            assert_eq!(lods.tier_for_distance_squared(d * d), expected);
        }
        for capacity in [1, 3, 8, 9, 91, 20_013] {
            for alignment in [1, 16, 32, 64, 256, 4096] {
                let stride = aligned_bin_stride(capacity, alignment).unwrap();
                assert_eq!(stride % alignment, 0);
                assert_eq!(stride % INSTANCE_BYTES, 0);
                assert!(stride >= u64::from(capacity) * INSTANCE_BYTES);
            }
        }
    }

    #[test]
    fn mesh_tier_distance_uses_physical_anchor_not_crown_or_depth() {
        let lods = MeshLods::new(vec![
            MeshLod {
                start_m: 0.0,
                index_count: 9,
            },
            MeshLod {
                start_m: 10.0,
                index_count: 3,
            },
            MeshLod {
                start_m: 20.0,
                index_count: 6,
            },
        ])
        .unwrap();
        let view = GpuView::unculled(540.0, 16.0, 0.15)
            .unwrap()
            .with_source_center([0.0, 0.0, 1.0])
            .unwrap();
        // Sphere center is 21m away but the physical anchor is only 8m.
        let mut object = PackedInstance {
            position_radius: [0.0, 0.0, 21.0, 13.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            axis_radii: [3.0, 5.0, 13.0, 0.0],
            object_id: [1; 4],
            metadata: [2; 4],
            tint: [1.0; 4],
        };
        assert_eq!(classify_cpu(&object, view, &lods), VisualLod::Mesh(0));
        // Off-axis distance, not view-forward depth, chooses the next tiers.
        object.position_radius[0] = 14.0;
        assert_eq!(classify_cpu(&object, view, &lods), VisualLod::Mesh(1));
        object.position_radius[0] = 22.0;
        assert_eq!(classify_cpu(&object, view, &lods), VisualLod::Mesh(2));
    }
}
