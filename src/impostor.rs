//! Actual instanced source-mesh and multi-angle impostor rendering.
//! Consumes compacted GPU buffers without reading back their contents.

use crate::gpu::GpuCompactor;
use crate::{Error, Result};
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
}

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
                || !v
                    .color
                    .iter()
                    .all(|c| c.is_finite() && (0.0..=1.0).contains(c))
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

    /// Upload actual sRGB texture data to a GPU using a single mip level.
    ///
    /// # Errors
    /// Rejects a texture too large for the selected device.
    pub fn upload(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<AtlasTexture> {
        let limits = device.limits();
        if self.width() > limits.max_texture_dimension_2d
            || self.height() > limits.max_texture_dimension_2d
        {
            return Err(Error::InvalidConfiguration(
                "impostor atlas exceeds GPU texture limit",
            ));
        }
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("i-love-proland 32-view impostor atlas"),
                size: wgpu::Extent3d {
                    width: self.width(),
                    height: self.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &self.rgba,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let tile = f32::from(u16::try_from(self.tile_size).map_err(|_| Error::IndexOverflow)?);
        let gutter = f32::from(u16::try_from(self.gutter).map_err(|_| Error::IndexOverflow)?);
        Ok(AtlasTexture {
            _texture: texture,
            view,
            gutter_fraction: gutter / tile,
        })
    }
}

/// Owns the texture used by a live GPU renderer.
pub struct AtlasTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    gutter_fraction: f32,
}

/// Explicit camera-relative projection. The application owns view rotation.
#[derive(Clone, Copy, Debug)]
pub struct TreeCamera {
    projection: [[f32; 4]; 4],
}

impl TreeCamera {
    /// # Errors
    /// Rejects non-finite camera matrices.
    pub fn new(projection: [[f32; 4]; 4]) -> Result<Self> {
        if projection.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::InvalidConfiguration("non-finite tree projection"));
        }
        Ok(Self { projection })
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
        Self::new([
            [focal / aspect, 0.0, 0.0, 0.0],
            [0.0, focal, 0.0, 0.0],
            [0.0, 0.0, depth, 1.0],
            [0.0, 0.0, -near * depth, 0.0],
        ])
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawUniform {
    projection: [[f32; 4]; 4],
    atlas: [f32; 4],
}
/// Draw two indexed-indirect batches directly from GPU-compacted instances.
pub struct TreeRenderer {
    mesh_pipeline: wgpu::RenderPipeline,
    impostor_pipeline: wgpu::RenderPipeline,
    mesh_bind: wgpu::BindGroup,
    impostor_bind: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    quad_indices: wgpu::Buffer,
    gutter_fraction: f32,
}

fn render_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    name: &'static str,
    entry_points: (&'static str, &'static str),
    vertex_buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    formats: (wgpu::TextureFormat, wgpu::TextureFormat),
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(entry_points.0),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: vertex_buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: formats.1,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(entry_points.1),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: formats.0,
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
                    view_dimension: wgpu::TextureViewDimension::D2,
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
        ],
    })
}

fn make_draw_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    instances: &wgpu::Buffer,
    uniforms: &wgpu::Buffer,
    atlas: &AtlasTexture,
    sampler: &wgpu::Sampler,
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
                resource: instances.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&atlas.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

impl TreeRenderer {
    /// Compile the near source-mesh and middle billboard pipelines.
    ///
    /// # Errors
    /// Rejects mismatched compactor geometry index counts.
    pub fn new(
        device: &wgpu::Device,
        compactor: &GpuCompactor,
        mesh: &TreeMesh,
        atlas: &AtlasTexture,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        if compactor.mesh_index_count() != mesh.index_count()
            || compactor.impostor_index_count() != 6
        {
            return Err(Error::InvalidConfiguration(
                "compactor draw index counts must match tree mesh and billboard quad",
            ));
        }
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("i-love-proland impostor atlas bilinear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
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
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("i-love-proland tree view/projection uniform"),
            size: core::mem::size_of::<DrawUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mesh_bind = make_draw_bind(
            device,
            &layout,
            compactor.mesh_instances(),
            &uniforms,
            atlas,
            &sampler,
        );
        let impostor_bind = make_draw_bind(
            device,
            &layout,
            compactor.impostor_instances(),
            &uniforms,
            atlas,
            &sampler,
        );
        let vertex_layout = [Some(wgpu::VertexBufferLayout {
            array_stride: core::mem::size_of::<TreeVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4],
        })];
        let mesh_pipeline = render_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "i-love-proland instanced source trees",
            ("mesh_vs", "mesh_fs"),
            &vertex_layout,
            (color_format, depth_format),
        );
        let impostor_pipeline = render_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "i-love-proland instanced impostors",
            ("impostor_vs", "impostor_fs"),
            &[],
            (color_format, depth_format),
        );
        let quad_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("i-love-proland billboard quad 6 indices"),
            contents: bytemuck::cast_slice(&[0_u32, 1, 2, 2, 1, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });
        Ok(Self {
            mesh_pipeline,
            impostor_pipeline,
            mesh_bind,
            impostor_bind,
            uniforms,
            quad_indices,
            gutter_fraction: atlas.gutter_fraction,
        })
    }

    /// Upload the camera view before dispatch/render recording.
    pub fn set_camera(&self, queue: &wgpu::Queue, camera: TreeCamera) {
        let uniform = DrawUniform {
            projection: camera.projection,
            atlas: [8.0, 4.0, self.gutter_fraction, 0.0],
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniform));
    }

    /// Record near-mesh and billboard indexed-indirect draws.
    ///
    /// Encode the compactor compute pass before the rendering pass. The host
    /// owns target clearing, MSAA, depth and all render-pass synchronization.
    pub fn draw<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        mesh: &'pass TreeMesh,
        compactor: &'pass GpuCompactor,
    ) {
        pass.set_pipeline(&self.mesh_pipeline);
        pass.set_bind_group(0, &self.mesh_bind, &[]);
        pass.set_vertex_buffer(0, mesh.vertex.slice(..));
        pass.set_index_buffer(mesh.index.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(compactor.mesh_indirect(), 0);

        pass.set_pipeline(&self.impostor_pipeline);
        pass.set_bind_group(0, &self.impostor_bind, &[]);
        pass.set_index_buffer(self.quad_indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(compactor.impostor_indirect(), 0);
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
    fn perspective_validation() {
        assert!(
            TreeCamera::perspective(core::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.5, 50_000.0,)
                .is_ok()
        );
        assert!(TreeCamera::perspective(0.0, 1.0, 0.1, 100.0).is_err());
        assert!(TreeCamera::perspective(core::f32::consts::FRAC_PI_2, 1.0, 0.1, 0.1,).is_err());
        assert!(TreeCamera::new([[f32::NAN; 4]; 4]).is_err());
    }
}
