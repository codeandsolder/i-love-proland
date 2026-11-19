//! GPU visibility, LOD selection, and indirect-draw compaction.
//!
//! This optional module deliberately never creates canonical objects. The
//! independent, deterministic CPU generator produces each stable ID and
//! variant; the GPU handles only camera-dependent visibility and mesh versus
//! impostor selection. GPU atomic append order has no persistence semantics.

use crate::{Error, Instance, Result};
use wgpu::util::DeviceExt;

pub const WGSL_COMPACTION: &str = include_str!("compact.wgsl");

const INSTANCE_BYTES: u64 = 80;
const WORKGROUP_SIZE: u32 = 64;

/// An explicit, stable, shader-compatible 80-byte instance layout. All
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
    pub fn upright(position: [f32; 3], radius: f32, yaw_turns: f32) -> Self {
        let half = yaw_turns * core::f32::consts::PI;
        Self {
            camera_relative_position: position,
            rotation_xyzw: [0.0, half.sin(), 0.0, half.cos()],
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
        };
        Ok(words)
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
        })
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

/// Stable reference classification for unit tests and GPU parity checks.
/// Placement identity is independent of this visual classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisualLod {
    Mesh,
    Impostor,
    TerrainCanopy,
    OutsideFrustum,
}

/// Reference classifier for parity testing with the WGSL shader.
///
/// Both paths use the same f32 inputs and cross-multiplied pixel thresholds;
/// GPU math can still differ slightly near boundaries, affecting visual
/// LOD only, never canonical object identity.
#[must_use]
pub fn classify_cpu(object: &PackedInstance, view: GpuView) -> VisualLod {
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
        VisualLod::Mesh
    } else if projected_crown >= view.impostor_threshold_px * safe_depth {
        VisualLod::Impostor
    } else {
        VisualLod::TerrainCanopy
    }
}

/// Buffers can be bound directly by the host render pass; use the matching
/// indirect draw buffer and instance storage buffer for each geometry LOD.
///
/// Every GPU invocation rewrites the indirect instance counts. Compaction
/// preserves exact IDs/variants but not output order across workgroups.
pub struct GpuCompactor {
    capacity: u32,
    input: wgpu::Buffer,
    mesh: wgpu::Buffer,
    impostor: wgpu::Buffer,
    mesh_indirect: wgpu::Buffer,
    impostor_indirect: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    mesh_template: IndexedDraw,
    impostor_template: IndexedDraw,
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

impl GpuCompactor {
    /// # Errors
    /// Rejects capacity exceeding the GPU's storage/workgroup limits.
    pub fn new(
        device: &wgpu::Device,
        capacity: u32,
        mesh_index_count: u32,
        impostor_index_count: u32,
    ) -> Result<Self> {
        let limits = device.limits();
        let bytes = u64::from(capacity) * INSTANCE_BYTES;
        if capacity == 0
            || bytes > limits.max_storage_buffer_binding_size
            || bytes > limits.max_buffer_size
            || capacity.div_ceil(WORKGROUP_SIZE) > limits.max_compute_workgroups_per_dimension
        {
            return Err(Error::InvalidConfiguration(
                "GPU compactor capacity exceeds device limits",
            ));
        }

        let create_instance_buffer = |name| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let input = create_instance_buffer("i-love-proland candidate input");
        let mesh = create_instance_buffer("i-love-proland compacted meshes");
        let impostor = create_instance_buffer("i-love-proland compacted impostors");
        let mesh_template = IndexedDraw {
            index_count: mesh_index_count,
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        };
        let impostor_template = IndexedDraw {
            index_count: impostor_index_count,
            ..mesh_template
        };
        let create_indirect = |name, record: &IndexedDraw| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::bytes_of(record),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let mesh_indirect = create_indirect("i-love-proland mesh indirect", &mesh_template);
        let impostor_indirect =
            create_indirect("i-love-proland impostor indirect", &impostor_template);
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
                    resource: mesh.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: impostor.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mesh_indirect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: impostor_indirect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            capacity,
            input,
            mesh,
            impostor,
            mesh_indirect,
            impostor_indirect,
            uniforms,
            bind_group,
            pipeline,
            mesh_template,
            impostor_template,
        })
    }

    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    #[must_use]
    pub const fn mesh_index_count(&self) -> u32 {
        self.mesh_template.index_count
    }

    #[must_use]
    pub const fn impostor_index_count(&self) -> u32 {
        self.impostor_template.index_count
    }

    #[must_use]
    pub const fn mesh_instances(&self) -> &wgpu::Buffer {
        &self.mesh
    }

    #[must_use]
    pub const fn impostor_instances(&self) -> &wgpu::Buffer {
        &self.impostor
    }

    #[must_use]
    pub const fn mesh_indirect(&self) -> &wgpu::Buffer {
        &self.mesh_indirect
    }

    #[must_use]
    pub const fn impostor_indirect(&self) -> &wgpu::Buffer {
        &self.impostor_indirect
    }

    /// Write candidate data and reset both indexed indirect draws, then
    /// record a single GPU classification/compaction compute pass.
    ///
    /// Call before a render pass that consumes the returned storage buffers
    /// via `@builtin(instance_index)` and the indirect draw parameters.
    ///
    /// # Errors
    /// Rejects candidate batches larger than the fixed instance capacity.
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
            counts: [count, 0, 0, 0],
            lod: [
                view.focal_length_px,
                view.mesh_threshold_px,
                view.impostor_threshold_px,
                0.0,
            ],
            forward: [view.forward[0], view.forward[1], view.forward[2], 0.0],
            planes: view.planes,
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(
            &self.mesh_indirect,
            0,
            bytemuck::bytes_of(&self.mesh_template),
        );
        queue.write_buffer(
            &self.impostor_indirect,
            0,
            bytemuck::bytes_of(&self.impostor_template),
        );
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
        assert_eq!(core::mem::size_of::<PackedInstance>(), 80);
        assert_eq!(core::mem::size_of::<GpuUniform>(), 144);
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
        assert_eq!(core::mem::size_of::<PackedInstance>(), 80);
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
        let object = |z, radius| PackedInstance {
            position_radius: [0.0, 0.0, z, radius],
            rotation: [0.0, 0.0, 0.0, 1.0],
            axis_radii: [radius, radius, radius, 0.0],
            object_id: [0; 4],
            metadata: [0; 4],
        };
        assert_eq!(classify_cpu(&object(100.0, 10.0), view), VisualLod::Mesh);
        assert_eq!(
            classify_cpu(&object(500.0, 10.0), view),
            VisualLod::Impostor
        );
        assert_eq!(
            classify_cpu(&object(25_000.0, 10.0), view),
            VisualLod::TerrainCanopy
        );
        assert_eq!(
            classify_cpu(&object(-100.0, 10.0), view),
            VisualLod::OutsideFrustum
        );
        let plane = [1.0, 0.0, 0.0, 0.0];
        let clipped = GpuView::new(540.0, 32.0, 1.0, [0.0, 0.0, 1.0], [plane; 6]).unwrap();
        let mut behind = object(100.0, 1.0);
        behind.position_radius[0] = -20.0;
        assert_eq!(classify_cpu(&behind, clipped), VisualLod::OutsideFrustum);
    }
}
