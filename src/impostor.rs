//! Actual instanced source-mesh and multi-angle impostor rendering.
//! Consumes compacted GPU buffers without reading back their contents.

use crate::gpu::{CompactedBin, GpuCompactor, IndexedDraw};
use crate::{
    Error, Result,
    atlas_bake::to_byte,
    billboard::Billboard,
    material::{linear_to_srgb, srgb_to_linear},
    wind::{WindFrame, WindParameters},
};
use wgpu::util::DeviceExt;

pub const AZIMUTH_VIEWS: u32 = 8;
pub const ELEVATION_VIEWS: u32 = 4;

const SHADER: &str = include_str!("impostor.wgsl");

/// Source geometry normalized to a unit-radius sphere at the object center.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TreeVertex {
    pub position: [f32; 4],
    pub color: [f32; 4],
    /// Source UVs are normalized texture coordinates (may repeat outside 0..1).
    pub uv: [f32; 2],
}

impl TreeVertex {
    /// Validate floating-point source-material color, not a normalized
    /// vertex attribute: linear RGB multipliers may legitimately exceed 1.
    /// The alpha channel remains ordinary fractional coverage.
    #[must_use]
    pub fn valid_color(color: [f32; 4]) -> bool {
        color[..3].iter().all(|c| c.is_finite() && *c >= 0.0)
            && color[3].is_finite()
            && (0.0..=1.0).contains(&color[3])
    }
}

/// Cloning shares the immutable GPU vertex and index buffers.
#[derive(Clone)]
pub struct TreeMesh {
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    index_count: u32,
}

impl TreeMesh {
    /// # Errors
    /// Rejects missing triangles, invalid index values or non-finite vertices.
    pub fn new(device: &wgpu::Device, vertices: &[TreeVertex], indices: &[u32]) -> Result<Self> {
        if vertices.is_empty() || indices.is_empty() || !indices.len().is_multiple_of(3) {
            return Err(Error::InvalidConfiguration(
                "tree mesh requires triangle geometry",
            ));
        }
        if vertices.iter().any(|v| {
            !v.position.iter().all(|p| p.is_finite())
                || !v.uv.iter().all(|p| p.is_finite())
                || !TreeVertex::valid_color(v.color)
        }) || indices
            .iter()
            .any(|i| usize::try_from(*i).map_or(true, |i| i >= vertices.len()))
        {
            return Err(Error::InvalidConfiguration(
                "invalid tree mesh vertex/index",
            ));
        }
        let index_count = u32::try_from(indices.len()).map_err(|_| Error::IndexOverflow)?;
        let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("i-love-proland source mesh"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("i-love-proland source index buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Ok(Self {
            vertex,
            index,
            index_count,
        })
    }

    #[must_use]
    pub const fn index_count(&self) -> u32 {
        self.index_count
    }
}

/// Linear row-major atlas: eight azimuth views times four elevation views.
/// Each square tile has its own padding to prevent neighbouring view bleed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtlasImage {
    tile_size: u32,
    gutter: u32,
    rgba: Vec<u8>,
}

impl AtlasImage {
    /// # Errors
    /// Rejects inconsistent image byte lengths and inadequate tile padding.
    pub fn new(tile_size: u32, gutter: u32, rgba: Vec<u8>) -> Result<Self> {
        if !(8..=512).contains(&tile_size)
            || gutter == 0
            || u64::from(gutter) * 4 >= u64::from(tile_size)
        {
            return Err(Error::InvalidConfiguration(
                "invalid impostor tile size/gutter",
            ));
        }
        let pixels =
            u64::from(tile_size).pow(2) * u64::from(AZIMUTH_VIEWS) * u64::from(ELEVATION_VIEWS) * 4;
        if usize::try_from(pixels).ok() != Some(rgba.len()) {
            return Err(Error::InvalidConfiguration(
                "impostor atlas RGBA length mismatch",
            ));
        }
        Ok(Self {
            tile_size,
            gutter,
            rgba,
        })
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.tile_size * AZIMUTH_VIEWS
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.tile_size * ELEVATION_VIEWS
    }

    #[must_use]
    pub const fn tile_size(&self) -> u32 {
        self.tile_size
    }

    #[must_use]
    pub const fn gutter(&self) -> u32 {
        self.gutter
    }

    #[must_use]
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Upload each angular view as an independent, coverage-filtered
    /// texture-array layer. The disk/CPU atlas retains its original layout.
    ///
    /// # Errors
    /// Rejects dimensions unsupported by the selected device.
    pub fn upload(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<AtlasTexture> {
        let limits = device.limits();
        let side = self.tile_size - 2 * self.gutter;
        let layers = AZIMUTH_VIEWS * ELEVATION_VIEWS;
        if side > limits.max_texture_dimension_2d || layers > limits.max_texture_array_layers {
            return Err(Error::InvalidConfiguration(
                "impostor view array exceeds GPU texture limit",
            ));
        }
        let (mip_level_count, data) = self.isolated_mips();
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("i-love-proland independently filtered angular views"),
                size: wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: layers,
                },
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &data,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Ok(AtlasTexture {
            _texture: texture,
            view,
        })
    }

    /// Crop the duplicated atlas gutters. Each output texel is the exact
    /// area average of the original content in linear premultiplied RGBA,
    /// including arbitrary (non-power-of-two and odd) source dimensions.
    /// Derive every level from the unquantized base, never from a quantized
    /// intermediate. No angular neighbours can contribute to a view's mip.
    fn isolated_mips(&self) -> (u32, Vec<u8>) {
        let side = self.tile_size - 2 * self.gutter;
        let pitch = self.width() as usize;
        let levels = side.bit_width();
        let mut data = Vec::new();
        let mut linear = [0.0; 256];
        for (slot, byte) in linear.iter_mut().zip(0..=u8::MAX) {
            *slot = srgb_to_linear(byte);
        }
        let mut base = Vec::with_capacity((side * side) as usize);
        for elevation in 0..ELEVATION_VIEWS {
            for azimuth in 0..AZIMUTH_VIEWS {
                base.clear();
                let x0 = azimuth * self.tile_size + self.gutter;
                let y0 = elevation * self.tile_size + self.gutter;
                for y in y0..y0 + side {
                    for x in x0..x0 + side {
                        let i = 4 * (y as usize * pitch + x as usize);
                        let p = &self.rgba[i..i + 4];
                        let a = f32::from(p[3]) / 255.0;
                        base.push([
                            linear[p[0] as usize] * a,
                            linear[p[1] as usize] * a,
                            linear[p[2] as usize] * a,
                            a,
                        ]);
                    }
                }
                for level in 0..levels {
                    append_area_mip(&base, side, side >> level, &mut data);
                }
            }
        }
        (levels, data)
    }
}

// Coordinates use exact integer overlap lengths in units of 1/dest_side
// of an original texel. Each output square has total weight source_side².
fn append_area_mip(base: &[[f32; 4]], source_side: u32, dest_side: u32, data: &mut Vec<u8>) {
    let norm = f64::from(source_side).powi(2).recip();
    for y in 0..dest_side {
        let y0 = y * source_side;
        let y1 = y0 + source_side;
        for x in 0..dest_side {
            let x0 = x * source_side;
            let x1 = x0 + source_side;
            let mut sum = [0.0_f64; 4];
            for sy in y0 / dest_side..y1.div_ceil(dest_side) {
                let wy = y1.min((sy + 1) * dest_side) - y0.max(sy * dest_side);
                for sx in x0 / dest_side..x1.div_ceil(dest_side) {
                    let wx = x1.min((sx + 1) * dest_side) - x0.max(sx * dest_side);
                    let weight = f64::from(wx * wy);
                    let p = base[(sy * source_side + sx) as usize];
                    for (sum, channel) in sum.iter_mut().zip(p) {
                        *sum = weight.mul_add(f64::from(channel), *sum);
                    }
                }
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "normalized area averages are converted to the f32 color transfer function before 8-bit quantization"
            )]
            let average = sum.map(|v| (v * norm) as f32);
            data.extend_from_slice(&[
                to_byte(linear_to_srgb(average[0])),
                to_byte(linear_to_srgb(average[1])),
                to_byte(linear_to_srgb(average[2])),
                to_byte(average[3]),
            ]);
        }
    }
}

/// Owns the texture used by a live GPU renderer.
pub struct AtlasTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// Explicit camera-relative projection. The application owns view rotation.
#[derive(Clone, Copy, Debug)]
pub struct TreeCamera {
    projection: [[f32; 4]; 4],
    forward: [f32; 3],
    wind: Option<WindFrame>,
}

impl TreeCamera {
    /// # Errors
    /// Rejects non-finite camera matrices or a non-unit forward vector.
    pub fn new(projection: [[f32; 4]; 4], forward: [f32; 3]) -> Result<Self> {
        if projection.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::InvalidConfiguration("non-finite tree projection"));
        }
        let length_squared: f32 = forward.iter().map(|v| v * v).sum();
        if !length_squared.is_finite() || (length_squared - 1.0).abs() > 1e-4 {
            return Err(Error::InvalidConfiguration("invalid tree camera forward"));
        }
        Ok(Self {
            projection,
            forward,
            wind: None,
        })
    }

    /// Attach stable wind coordinates and elapsed animation time.
    #[must_use]
    pub const fn with_wind(mut self, frame: WindFrame) -> Self {
        self.wind = Some(frame);
        self
    }

    /// Left-handed forward-positive-Z perspective with depth in [0, 1].
    ///
    /// # Errors
    /// Rejects invalid FOV, aspect ratio, near and far planes.
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Result<Self> {
        if !fov_y.is_finite()
            || fov_y <= 0.0
            || fov_y >= core::f32::consts::PI
            || !aspect.is_finite()
            || aspect <= 0.0
            || !near.is_finite()
            || !far.is_finite()
            || near <= 0.0
            || far <= near
        {
            return Err(Error::InvalidConfiguration(
                "invalid tree perspective camera",
            ));
        }
        let focal = 1.0 / (fov_y * 0.5).tan();
        let depth = far / (far - near);
        Self::new(
            [
                [focal / aspect, 0.0, 0.0, 0.0],
                [0.0, focal, 0.0, 0.0],
                [0.0, 0.0, depth, 1.0],
                [0.0, 0.0, -near * depth, 0.0],
            ],
            [0.0, 0.0, 1.0],
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawUniform {
    projection: [[f32; 4]; 4],
    material: [f32; 4],
    wind_parameters: [f32; 4],
    wind_source: [f32; 4],
    field_axes: [[f32; 4]; 3],
    field_origin: [f32; 4],
    field_phase: [f32; 4],
    camera_forward: [f32; 4],
    billboard_source: [f32; 4],
}

/// Render targets and material alpha-cutoff semantics.
#[derive(Clone, Copy, Debug)]
pub struct TreeRenderConfig {
    pub color_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
    pub sample_count: u32,
}

#[derive(Clone, Copy)]
enum PassKind {
    Mesh,
    Impostor,
}

/// Optional texture-driven wind on a single authored material.
pub struct MeshWind<'a> {
    pub parameters: WindParameters,
    pub map: &'a wgpu::TextureView,
    /// Unnormalized source-space bounds centre Y and radius.
    pub source_center_y: f32,
    pub source_radius: f32,
}

#[derive(Clone, Copy)]
struct MaterialWind {
    parameters: WindParameters,
    source_center_y: f32,
    source_radius: f32,
}

impl MeshWind<'_> {
    fn validate(&self) -> Result<()> {
        self.parameters.validate()?;
        if !self.source_center_y.is_finite()
            || !self.source_radius.is_finite()
            || self.source_radius <= 0.0
        {
            return Err(Error::InvalidConfiguration("invalid wind source bounds"));
        }
        self.parameters
            .max_displacement(self.source_center_y + self.source_radius)?;
        Ok(())
    }
}

/// One textured source mesh. The renderer takes ownership of its GPU mesh.
pub struct TreeMaterial<'a> {
    pub mesh: TreeMesh,
    pub source_view: &'a wgpu::TextureView,
    pub alpha_cutoff: f32,
    pub wind: Option<MeshWind<'a>>,
    pub billboard: Option<Billboard>,
}

struct MeshDraw {
    mesh: TreeMesh,
    bind: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    indirect: wgpu::Buffer,
    source_indirect_offset: u64,
    cutoff: f32,
    wind: Option<MaterialWind>,
    billboard: Option<Billboard>,
}

/// Every authored mesh tier shares one GPU-visible instance list across its
/// material draws, and the family emits one composite impostor draw.
pub struct TreeRenderer {
    mesh_pipeline: wgpu::RenderPipeline,
    impostor_pipeline: wgpu::RenderPipeline,
    mesh_draws: Vec<MeshDraw>,
    impostor_bind: wgpu::BindGroup,
    impostor_uniforms: wgpu::Buffer,
    source_indirect: wgpu::Buffer,
    impostor_indirect_offset: u64,
    quad_indices: wgpu::Buffer,
    alpha_to_coverage: bool,
}

fn render_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    pass_kind: PassKind,
    vertex_buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    config: TreeRenderConfig,
) -> wgpu::RenderPipeline {
    let (name, vertex_entry, fragment_entry, alpha_to_coverage) = match pass_kind {
        PassKind::Mesh => (
            "i-love-proland instanced source trees",
            "mesh_vs",
            "mesh_fs",
            false,
        ),
        PassKind::Impostor => (
            "i-love-proland instanced textured impostors",
            "impostor_vs",
            "impostor_fs",
            config.sample_count > 1,
        ),
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: vertex_buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: config.depth_format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: config.sample_count,
            alpha_to_coverage_enabled: alpha_to_coverage,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn render_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("i-love-proland tree render bindings"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

struct MaterialBindings<'a> {
    atlas: &'a AtlasTexture,
    atlas_sampler: &'a wgpu::Sampler,
    source_view: &'a wgpu::TextureView,
    source_sampler: &'a wgpu::Sampler,
    wind_view: &'a wgpu::TextureView,
}

fn make_draw_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    instances: wgpu::BufferBinding<'_>,
    uniforms: &wgpu::Buffer,
    materials: &MaterialBindings<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("i-love-proland tree indirect draw bindings"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(instances),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&materials.atlas.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(materials.atlas_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(materials.source_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(materials.source_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(materials.wind_view),
            },
        ],
    })
}

fn make_source_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("i-love-proland source UV repeat and linear sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    })
}

fn new_uniform(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("i-love-proland material camera uniforms"),
        size: core::mem::size_of::<DrawUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn build_mesh_draw(
    device: &wgpu::Device,
    bin: &CompactedBin<'_>,
    source: TreeMaterial<'_>,
    layout: &wgpu::BindGroupLayout,
    atlas: &AtlasTexture,
    atlas_sampler: &wgpu::Sampler,
    source_sampler: &wgpu::Sampler,
) -> MeshDraw {
    let uniforms = new_uniform(device);
    let bind = make_draw_bind(
        device,
        layout,
        bin.instances.clone(),
        &uniforms,
        &MaterialBindings {
            atlas,
            atlas_sampler,
            source_view: source.source_view,
            source_sampler,
            wind_view: source
                .wind
                .as_ref()
                .map_or(source.source_view, |wind| wind.map),
        },
    );
    let indirect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("i-love-proland material indirect arguments"),
        contents: bytemuck::bytes_of(&IndexedDraw {
            index_count: source.mesh.index_count(),
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }),
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
    });
    MeshDraw {
        mesh: source.mesh,
        bind,
        uniforms,
        indirect,
        source_indirect_offset: bin.indirect_offset,
        cutoff: source.alpha_cutoff,
        billboard: source.billboard,
        wind: source.wind.map(|wind| MaterialWind {
            parameters: wind.parameters,
            source_center_y: wind.source_center_y,
            source_radius: wind.source_radius,
        }),
    }
}

fn make_impostor_bind(
    device: &wgpu::Device,
    bin: &CompactedBin<'_>,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
    atlas: &AtlasTexture,
    atlas_sampler: &wgpu::Sampler,
    source_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    // The impostor shader uses only the composite atlas, while the common
    // render layout still requires a valid source-albedo binding.
    let placeholder = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("i-love-proland unused impostor source texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let source_view = placeholder.create_view(&wgpu::TextureViewDescriptor::default());
    make_draw_bind(
        device,
        layout,
        bin.instances.clone(),
        uniforms,
        &MaterialBindings {
            atlas,
            atlas_sampler,
            source_view: &source_view,
            source_sampler,
            wind_view: &source_view,
        },
    )
}

fn validate_material_tiers(
    compactor: &GpuCompactor,
    sources: &[Vec<TreeMaterial<'_>>],
    config: TreeRenderConfig,
) -> Result<()> {
    if sources.len() != compactor.mesh_lods().tiers().len()
        || config.sample_count == 0
        || !config.sample_count.is_power_of_two()
        || sources.iter().any(|tier| {
            tier.is_empty()
                || tier
                    .iter()
                    .any(|s| !s.alpha_cutoff.is_finite() || !(0.0..=1.0).contains(&s.alpha_cutoff))
        })
    {
        return Err(Error::InvalidConfiguration(
            "invalid forest material tiers or sample count",
        ));
    }
    for source in sources.iter().flatten() {
        if let Some(wind) = &source.wind {
            wind.validate()?;
        }
    }
    for (tier, source) in sources.iter().enumerate() {
        if compactor
            .bin(tier)
            .is_none_or(|bin| bin.index_count != source[0].mesh.index_count())
        {
            return Err(Error::InvalidConfiguration(
                "incompatible mesh LOD draw indexes",
            ));
        }
    }
    if compactor
        .bin(compactor.mesh_lods().impostor_bin())
        .is_none_or(|bin| bin.index_count != 6)
    {
        return Err(Error::InvalidConfiguration(
            "incompatible impostor draw indexes",
        ));
    }
    Ok(())
}

impl TreeRenderer {
    /// Compile ordered authored mesh tiers and one shared middle impostor pass.
    /// Each tier's first mesh provides its base indirect index count. All
    /// material draws within a tier share that tier's compacted instances.
    ///
    /// # Errors
    /// Rejects empty/invalid materials, invalid sample count and incompatible
    /// compactor mesh/impostor arguments.
    pub fn new(
        device: &wgpu::Device,
        compactor: &GpuCompactor,
        sources: Vec<Vec<TreeMaterial<'_>>>,
        atlas: &AtlasTexture,
        config: TreeRenderConfig,
    ) -> Result<Self> {
        validate_material_tiers(compactor, &sources, config)?;
        let impostor_bin = compactor
            .bin(compactor.mesh_lods().impostor_bin())
            .ok_or(Error::InvalidConfiguration("missing impostor bin"))?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("i-love-proland angular view trilinear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let source_sampler = make_source_sampler(device);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("i-love-proland mesh and 32-angle impostor shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = render_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("i-love-proland tree render pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut mesh_draws = Vec::new();
        for (tier, materials) in sources.into_iter().enumerate() {
            let bin = compactor
                .bin(tier)
                .ok_or(Error::InvalidConfiguration("missing mesh LOD bin"))?;
            mesh_draws.extend(materials.into_iter().map(|source| {
                build_mesh_draw(
                    device,
                    &bin,
                    source,
                    &layout,
                    atlas,
                    &sampler,
                    &source_sampler,
                )
            }));
        }
        let impostor_uniforms = new_uniform(device);
        let impostor_bind = make_impostor_bind(
            device,
            &impostor_bin,
            &layout,
            &impostor_uniforms,
            atlas,
            &sampler,
            &source_sampler,
        );
        let vertex_layout = [Some(wgpu::VertexBufferLayout {
            array_stride: core::mem::size_of::<TreeVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x2],
        })];
        let mesh_pipeline = render_pipeline(
            device,
            &pipeline_layout,
            &shader,
            PassKind::Mesh,
            &vertex_layout,
            config,
        );
        let impostor_pipeline = render_pipeline(
            device,
            &pipeline_layout,
            &shader,
            PassKind::Impostor,
            &[],
            config,
        );
        let quad_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("i-love-proland billboard quad 6 indices"),
            contents: bytemuck::cast_slice(&[0_u32, 1, 2, 2, 1, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });
        Ok(Self {
            mesh_pipeline,
            impostor_pipeline,
            mesh_draws,
            impostor_bind,
            impostor_uniforms,
            source_indirect: impostor_bin.indirect.clone(),
            impostor_indirect_offset: impostor_bin.indirect_offset,
            quad_indices,
            alpha_to_coverage: config.sample_count > 1,
        })
    }

    /// Upload one camera matrix, keeping original alpha cutoffs per material.
    pub fn set_camera(&self, queue: &wgpu::Queue, camera: TreeCamera) {
        let mut uniform = DrawUniform {
            projection: camera.projection,
            material: [0.0, f32::from(u8::from(self.alpha_to_coverage)), 0.0, 0.0],
            wind_parameters: [0.0; 4],
            wind_source: [0.0; 4],
            field_axes: [[0.0; 4]; 3],
            field_origin: [0.0; 4],
            field_phase: [0.0; 4],
            camera_forward: [camera.forward[0], camera.forward[1], camera.forward[2], 0.0],
            billboard_source: [0.0; 4],
        };
        for material in &self.mesh_draws {
            uniform.material[0] = material.cutoff;
            uniform.billboard_source = material.billboard.map_or([0.0; 4], |b| {
                let [x, y, z] = b.center();
                [x, y, z, b.radius()]
            });
            uniform.wind_parameters = [0.0; 4];
            if let (Some(wind), Some(frame)) = (material.wind, camera.wind) {
                let p = wind.parameters;
                uniform.wind_parameters = [
                    p.spatial_scale,
                    p.height_start,
                    p.height_exponent,
                    p.intensity,
                ];
                uniform.wind_source = [wind.source_center_y, wind.source_radius, 0.0, 0.0];
                (
                    uniform.field_axes,
                    uniform.field_origin,
                    uniform.field_phase,
                ) = frame.uniforms(p);
            }
            queue.write_buffer(&material.uniforms, 0, bytemuck::bytes_of(&uniform));
        }
        uniform.material[0] = 0.0;
        uniform.wind_parameters = [0.0; 4];
        queue.write_buffer(&self.impostor_uniforms, 0, bytemuck::bytes_of(&uniform));
    }

    /// Copy the GPU-computed tier instance count into each material's indirect
    /// draw arguments, preserving its original index count. This must happen
    /// after compaction and before drawing, in the same encoder.
    pub fn encode_mesh_draws(&self, encoder: &mut wgpu::CommandEncoder) {
        for material in &self.mesh_draws {
            encoder.copy_buffer_to_buffer(
                &self.source_indirect,
                material.source_indirect_offset + 4,
                &material.indirect,
                4,
                4,
            );
        }
    }

    /// Draw every mesh tier's materials with its compacted instance set,
    /// then emit precisely one composite middle-distance impostor draw.
    pub fn draw<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>) {
        pass.set_pipeline(&self.mesh_pipeline);
        for material in &self.mesh_draws {
            pass.set_bind_group(0, &material.bind, &[]);
            pass.set_vertex_buffer(0, material.mesh.vertex.slice(..));
            pass.set_index_buffer(material.mesh.index.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed_indirect(&material.indirect, 0);
        }
        pass.set_pipeline(&self.impostor_pipeline);
        pass.set_bind_group(0, &self.impostor_bind, &[]);
        pass.set_index_buffer(self.quad_indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(&self.source_indirect, self.impostor_indirect_offset);
    }

    #[must_use]
    pub const fn material_count(&self) -> usize {
        self.mesh_draws.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_size_validation() {
        let len = 8 * 4 * 16 * 16 * 4;
        let image = AtlasImage::new(16, 2, vec![0; len]).unwrap();
        assert_eq!((image.width(), image.height()), (128, 64));
        assert_eq!(image.rgba().len(), len);
        assert!(AtlasImage::new(4, 1, vec![]).is_err());
        assert!(AtlasImage::new(16, 4, vec![]).is_err());
        assert!(AtlasImage::new(16, 2, vec![]).is_err());
    }

    #[test]
    fn isolated_mips_preserve_odd_sized_view_coverage_and_color() {
        let tile = 13_u32;
        let gutter = 2_u32;
        let side = tile - 2 * gutter;
        let width = tile * AZIMUTH_VIEWS;
        let mut rgba = vec![255; (width * tile * ELEVATION_VIEWS * 4) as usize];
        // Deliberately unrelated opaque white gutters and transparent
        // saturated magenta content must not leak into neighbouring views.
        for e in 0..ELEVATION_VIEWS {
            for a in 0..AZIMUTH_VIEWS {
                for y in 0..side {
                    for x in 0..side {
                        let i = (4 * ((e * tile + gutter + y) * width + a * tile + gutter + x))
                            as usize;
                        let visible = x == y;
                        rgba[i..i + 4].copy_from_slice(if !visible {
                            &[255, 0, 255, 0]
                        } else if a == 0 && e == 0 {
                            &[255, 0, 0, 255]
                        } else {
                            &[0, 255, 0, 255]
                        });
                    }
                }
            }
        }
        let atlas = AtlasImage::new(tile, gutter, rgba).unwrap();
        let (levels, bytes) = atlas.isolated_mips();
        assert_eq!(levels, 4);
        let layer_bytes: usize = (0..levels).map(|l| ((side >> l).pow(2) * 4) as usize).sum();
        assert_eq!(bytes.len(), layer_bytes * 32);
        for (view, layer) in bytes.chunks_exact(layer_bytes).enumerate() {
            let mut offset = 0;
            for level in 0..levels {
                let size = (side >> level).pow(2) as usize;
                let pixels = &layer[offset..offset + size * 4];
                let mut mean = 0.0;
                for pixel in pixels.as_chunks::<4>().0 {
                    let alpha = f64::from(pixel[3]) / 255.0;
                    mean += alpha;
                    let channel = if view == 0 { pixel[0] } else { pixel[1] };
                    let other = if view == 0 { pixel[1] } else { pixel[0] };
                    assert_eq!((other, pixel[2]), (0, 0));
                    assert!((f64::from(srgb_to_linear(channel)) - alpha).abs() < 0.007);
                }
                mean /= f64::from((side >> level).pow(2));
                assert!((mean - 1.0 / f64::from(side)).abs() <= 0.5 / 255.0 + 1e-9);
                offset += size * 4;
            }
        }
    }

    #[test]
    fn perspective_validation() {
        assert!(
            TreeCamera::perspective(core::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.5, 50_000.0,)
                .is_ok()
        );
        assert!(TreeCamera::perspective(0.0, 1.0, 0.1, 100.0).is_err());
        assert!(TreeCamera::perspective(core::f32::consts::FRAC_PI_2, 1.0, 0.1, 0.1,).is_err());
        assert!(TreeCamera::new([[f32::NAN; 4]; 4], [0., 0., 1.]).is_err());
    }
}
