//! Headless Vulkan comparison of procedural vertex synthesis, shared foliage
//! modules and expanded precomputed variants. This is a geometry/rendering
//! experiment, not a new live KSP foliage renderer.
use std::{error::Error, fmt::Write as _, fs, path::Path, sync::mpsc, time::Instant};
use wgpu::util::DeviceExt;

const W: u32 = 960;
const H: u32 = 600;
const KINDS: u32 = 3;
const VARIANTS: u32 = 8;
const MODELS: u32 = KINDS * VARIANTS;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const QUERIES: u32 = 32;
const MODES: [&str; 4] = [
    "gpu_unique",
    "gpu_24_variants",
    "shared_modules",
    "expanded_24_variants",
];
type AnyResult<T> = Result<T, Box<dyn Error>>;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Tree {
    position_scale: [f32; 4],
    ids: [u32; 4],
    tint: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Patch {
    center_radius: [f32; 4],
    orient: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BakedVertex {
    position_uvx: [f32; 4],
    uvy_brightness_material: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Scene {
    view: [f32; 4],
    counts: [u32; 4],
    switches: [f32; 4],
}

const fn hash(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}
#[expect(
    clippy::cast_precision_loss,
    reason = "16-bit fraction is exact in f32"
)]
fn random(seed: u32, salt: u32) -> f32 {
    (hash(seed ^ salt) & 0xffff) as f32 / 65_536.0
}
fn patch_for(seed: u32, part: u32, kind: u32, clumps: u32) -> Patch {
    if part == 0 {
        return Patch {
            center_radius: [0.0, 0.45, 0.0, 0.065],
            orient: [0.0, 0.9, 1.0, 1.0],
        };
    }
    let cluster = (part - 1) / 2;
    let side = (part - 1) % 2;
    // The first eight parent clusters survive every LOD. Finer tiers only
    // split each parent's shape into deterministic, spatially related cards.
    let parent = cluster % 8;
    let children = (clumps + 7 - parent) / 8;
    let azimuth_random = random(seed, parent.wrapping_mul(0x9e37_79b9).wrapping_add(21));
    let radial_random = random(seed, parent.wrapping_mul(0x85eb_ca6b).wrapping_add(77));
    let height_random = random(seed, parent.wrapping_mul(0xc2b2_ae35).wrapping_add(135));
    let size_random = random(seed, parent.wrapping_mul(0x27d4_eb2f).wrapping_add(389));
    let angle = core::f32::consts::TAU * azimuth_random;
    let (radial, altitude) = if kind == 1 {
        (
            0.48 * (1.0 - height_random),
            height_random.mul_add(0.68, 0.19),
        )
    } else {
        (
            0.48 * radial_random.sqrt(),
            height_random.mul_add(0.50, 0.39),
        )
    };
    let (radial, altitude) = if kind == 2 {
        (
            0.44 * radial_random.sqrt(),
            height_random.mul_add(0.36, 0.27),
        )
    } else {
        (radial, altitude)
    };
    let (sin, cos) = angle.sin_cos();
    let mut center_x = radial * cos;
    let mut center_y = altitude;
    let mut center_z = radial * sin;
    let base_radius = if kind == 1 { 0.235 } else { 0.36 };
    if children > 1 {
        let child_angle = core::f32::consts::TAU
            * random(seed, cluster.wrapping_mul(0xbf58_476d).wrapping_add(531));
        let displacement = base_radius
            * 0.5
            * random(seed, cluster.wrapping_mul(0x94d0_49bb).wrapping_add(227)).sqrt();
        let (sin, cos) = child_angle.sin_cos();
        center_x = displacement.mul_add(cos, center_x);
        center_z = displacement.mul_add(sin, center_z);
        center_y = (displacement * 0.35).mul_add(
            random(seed, cluster.wrapping_mul(0x87f53).wrapping_add(155)) - 0.5,
            center_y,
        );
    }
    let coverage_scale = match children {
        1 => 1.0,
        2..=3 => 1.45,
        _ => 2.20,
    };
    #[expect(
        clippy::cast_precision_loss,
        reason = "the foliage cluster count is capped at 64"
    )]
    let radius = base_radius / (children as f32).sqrt() * coverage_scale;
    Patch {
        center_radius: [center_x, center_y, center_z, radius],
        orient: [
            angle
                + if side == 1 {
                    core::f32::consts::FRAC_PI_2
                } else {
                    0.0
                },
            radius * size_random.mul_add(0.45, 1.5),
            radial_random.mul_add(0.34, 0.78),
            0.0,
        ],
    }
}
fn expanded(patch: Patch, i: u32) -> BakedVertex {
    let uv: [f32; 2] = match i {
        0 => [-1.0, -1.0],
        1 | 4 => [1.0, -1.0],
        2 | 3 => [-1.0, 1.0],
        _ => [1.0, 1.0],
    };
    let (sin, cos) = patch.orient[0].sin_cos();
    let r = patch.center_radius[3];
    BakedVertex {
        position_uvx: [
            uv[0].mul_add(r * cos, patch.center_radius[0]),
            (uv[1] * patch.orient[1] * 0.5).mul_add(1.0, patch.center_radius[1]),
            uv[0].mul_add(r * sin, patch.center_radius[2]),
            uv[0],
        ],
        uvy_brightness_material: [uv[1], patch.orient[2], patch.orient[3], 0.0],
    }
}
fn make_modules(clumps: u32) -> (Vec<Patch>, Vec<BakedVertex>) {
    let mut models = Vec::new();
    let mut expanded_models = Vec::new();
    for variant in 0..MODELS {
        for part in 0..=(clumps * 2) {
            let data = patch_for(variant ^ 0x56cb_1a93, part, variant / VARIANTS, clumps);
            models.push(data);
            for vertex in 0..6 {
                expanded_models.push(expanded(data, vertex));
            }
        }
    }
    (models, expanded_models)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Camera {
    Ground,
    Aerial,
}

/// Smooth two-dimensional biome field from stable grid points, not a random
/// species assignment per tree. The eight-clump morphology is still stable.
#[expect(
    clippy::cast_precision_loss,
    reason = "benchmark habitat grid coordinates are at most 88 by 47"
)]
fn habitat_kind(i: u32) -> u32 {
    let gx = i % 88;
    let gz = i / 88;
    let bx = gx / 18;
    let bz = gz / 10;
    let mut tx = (gx % 18) as f32 / 18.0;
    let mut tz = (gz % 10) as f32 / 10.0;
    tx = tx * tx * tx.mul_add(-2.0, 3.0);
    tz = tz * tz * tz.mul_add(-2.0, 3.0);
    let sample = |x: u32, z: u32| {
        random(
            x.wrapping_mul(0x9e37_79b9) ^ z.wrapping_mul(0x85eb_ca6b),
            0x1bd3,
        )
    };
    let n00 = sample(bx, bz);
    let n10 = sample(bx + 1, bz);
    let n01 = sample(bx, bz + 1);
    let n11 = sample(bx + 1, bz + 1);
    let moisture = (n10 - n00).mul_add(tx, n00);
    let moisture2 = (n11 - n01).mul_add(tx, n01);
    let climate = (moisture2 - moisture).mul_add(tz, moisture);
    // Species-specific groves, with occasional out-of-habitat colonizers.
    if random(i, 0x3f42_8a1d) < 0.08 {
        return hash(i) % KINDS;
    }
    if climate < 0.39 {
        1 // conifer
    } else if climate < 0.49 {
        2 // shrub
    } else {
        0 // broadleaf
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "benchmark grid has <2^16 positions"
)]
fn make_trees(count: u32, camera: Camera) -> Vec<Tree> {
    (0..count)
        .map(|i| {
            let kind = if camera == Camera::Aerial {
                habitat_kind(i)
            } else {
                hash(i) % KINDS
            };
            let model = kind * VARIANTS + hash(i ^ 0x9e37_79b9) % VARIANTS;
            let (spacing_x, spacing_z, start_z) = match camera {
                Camera::Ground => (1.65, 3.4, 14.0),
                Camera::Aerial => (4.3, 7.5, 50.0),
            };
            let x = ((i % 88) as f32 - 44.0)
                .mul_add(spacing_x, (random(i, 55) - 0.5) * spacing_x * 0.55);
            let z = random(i, 72).mul_add(
                spacing_z * 0.42,
                ((i / 88) as f32).mul_add(spacing_z, start_z),
            );
            let size = if kind == 2 {
                4.4
            } else {
                random(i, 91).mul_add(5.0, 7.5)
            };
            let tint = random(i, 999);
            Tree {
                position_scale: [x, 0.0, z, size],
                ids: [hash(i ^ 0x7f4a_7c15), model, kind, 0],
                tint: [
                    0.3_f32.mul_add(tint, 0.80),
                    0.25_f32.mul_add(tint, 0.9),
                    0.25_f32.mul_add(tint, 0.8),
                    1.0,
                ],
            }
        })
        .collect()
}
fn gpu_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &'static str,
    value: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(value),
        usage: wgpu::BufferUsages::STORAGE,
    })
}
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::RenderPipeline>,
    color: wgpu::Texture,
    view: wgpu::TextureView,
    depth: wgpu::TextureView,
    timestamp_supported: bool,
    name: String,
}
/// Scene resources stay uniform or storage-backed across all four draw modes.
fn foliage_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("foliage scene"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// All paths share exactly the same depth, color and fragment configuration.
fn foliage_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> Vec<wgpu::RenderPipeline> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("foliage pipeline layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let mut pipelines = Vec::new();
    for entry in ["vs_generated", "vs_limited", "vs_modules", "vs_expanded"] {
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        pipelines.push(pipeline);
    }
    pipelines
}

fn open_gpu() -> AnyResult<Gpu> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))?;
    let info = adapter.get_info();
    let timestamp_supported = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("foliage procedural benchmark"),
        required_features: if timestamp_supported {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        },
        ..Default::default()
    }))?;
    let layout = foliage_layout(&device);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("modular GPU foliage"),
        source: wgpu::ShaderSource::Wgsl(include_str!("foliage_lab.wgsl").into()),
    });
    let pipelines = foliage_pipelines(&device, &layout, &shader);
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("foliage offscreen color"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("foliage offscreen depth"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    Ok(Gpu {
        name: info.name,
        device,
        queue,
        layout,
        pipelines,
        view: color.create_view(&wgpu::TextureViewDescriptor::default()),
        depth: depth.create_view(&wgpu::TextureViewDescriptor::default()),
        color,
        timestamp_supported,
    })
}
struct DrawCase {
    _uniform: wgpu::Buffer,
    _trees: wgpu::Buffer,
    _modules: wgpu::Buffer,
    _expanded: wgpu::Buffer,
    bind: wgpu::BindGroup,
    vertices_per_tree: u32,
    trees_count: u32,
    geometry_bytes: usize,
    module_bytes: usize,
}
impl DrawCase {
    #[expect(
        clippy::cast_precision_loss,
        reason = "fixed W=960 and H=600 are exactly representable as f32"
    )]
    fn new(gpu: &Gpu, clumps: u32, count: u32, fade: f32, side: u32, camera: Camera) -> Self {
        let models = make_modules(clumps);
        let trees = make_trees(count, camera);
        let (eye_height, pitch) = match camera {
            Camera::Ground => (2.5, 0.0),
            Camera::Aerial => (40.0, 0.22),
        };
        let uniform_data = Scene {
            view: [1.73, W as f32 / H as f32, eye_height, pitch],
            counts: [clumps, MODELS, side, count],
            switches: [fade, W as f32, H as f32, 0.0],
        };
        let uniform = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("foliage scene params"),
                contents: bytemuck::bytes_of(&uniform_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let trees_buf = gpu_buffer(&gpu.device, "forested instances", &trees);
        let module_buf = gpu_buffer(&gpu.device, "compressed foliage modules", &models.0);
        let expanded_buf = gpu_buffer(&gpu.device, "expanded foliage variants", &models.1);
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("foliage scene binding"),
            layout: &gpu.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: trees_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: module_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: expanded_buf.as_entire_binding(),
                },
            ],
        });
        Self {
            _uniform: uniform,
            _trees: trees_buf,
            _modules: module_buf,
            _expanded: expanded_buf,
            bind,
            vertices_per_tree: (clumps * 2 + 1) * 6,
            trees_count: count,
            geometry_bytes: models.1.len() * size_of::<BakedVertex>(),
            module_bytes: models.0.len() * size_of::<Patch>(),
        }
    }
}
fn draw(
    gpu: &Gpu,
    encoder: &mut wgpu::CommandEncoder,
    case: &DrawCase,
    mode: usize,
    first: bool,
    timestamp: Option<(&wgpu::QuerySet, u32)>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("foliage lab rasterization"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &gpu.view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: if first {
                    wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.57,
                        g: 0.70,
                        b: 0.86,
                        a: 1.0,
                    })
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &gpu.depth,
            depth_ops: Some(wgpu::Operations {
                load: if first {
                    wgpu::LoadOp::Clear(1.0)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        occlusion_query_set: None,
        timestamp_writes: timestamp.map(|(queries, start)| wgpu::RenderPassTimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: Some(start),
            end_of_pass_write_index: Some(start + 1),
        }),
        multiview_mask: None,
    });
    pass.set_pipeline(&gpu.pipelines[mode]);
    pass.set_bind_group(0, &case.bind, &[]);
    pass.draw(0..case.vertices_per_tree, 0..case.trees_count);
}
fn map_read(gpu: &Gpu, buffer: &wgpu::Buffer) -> AnyResult<Vec<u8>> {
    let (tx, rx) = mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let view = buffer.slice(..).get_mapped_range()?;
    let bytes = view.to_vec();
    drop(view);
    buffer.unmap();
    Ok(bytes)
}
fn capture(gpu: &Gpu, pieces: &[(&DrawCase, usize)], path: &Path) -> AnyResult<Vec<u8>> {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("foliage image capture"),
        });
    for (index, &(case, mode)) in pieces.iter().enumerate() {
        draw(gpu, &mut encoder, case, mode, index == 0, None);
    }
    let stride = W * 4;
    assert!(stride.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    let bytes = u64::from(stride) * u64::from(H);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("captured pixels"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &gpu.color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let rgba = map_read(gpu, &staging)?;
    let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
    ppm.reserve(usize::try_from(W)? * usize::try_from(H)? * 3);
    for pixel in rgba.as_chunks::<4>().0 {
        ppm.extend_from_slice(&pixel[..3]);
    }
    fs::write(path, ppm)?;
    Ok(rgba)
}
#[derive(Clone, Copy)]
struct Timing {
    min: f64,
    median: f64,
    p90: f64,
}
#[expect(
    clippy::cast_precision_loss,
    reason = "timestamp interval is bounded to a short GPU render pass, well within 2^53 integer ticks"
)]
fn measure(gpu: &Gpu, case: &DrawCase, mode: usize) -> AnyResult<Timing> {
    const FRAMES: usize = 16;
    const WARMUP: usize = 4;
    for _ in 0..WARMUP {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("warmup trees"),
            });
        draw(gpu, &mut encoder, case, mode, true, None);
        gpu.queue.submit([encoder.finish()]);
    }
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    let start = Instant::now();
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU foliage timestamp timing"),
        });
    let qset = gpu.timestamp_supported.then(|| {
        gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("tree draw timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERIES,
        })
    });
    for frame in 0..FRAMES {
        let frame_index = u32::try_from(frame)? * 2;
        let marker = qset.as_ref().map(|q| (q, frame_index));
        draw(gpu, &mut encoder, case, mode, true, marker);
    }
    let mapping = qset.as_ref().map(|qs| {
        let size = u64::from(QUERIES) * 8;
        let gpu_times = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resolved foliage draw timings"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let host = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback foliage draw timings"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.resolve_query_set(qs, 0..QUERIES, &gpu_times, 0);
        encoder.copy_buffer_to_buffer(&gpu_times, 0, &host, 0, size);
        (gpu_times, host)
    });
    gpu.queue.submit([encoder.finish()]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    let cpu_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(u32::try_from(FRAMES)?);
    let mut times = if let Some((_, host)) = mapping {
        let bytes = map_read(gpu, &host)?;
        let ticks: Vec<u64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|eight| u64::from_le_bytes(*eight))
            .collect();
        let period = f64::from(gpu.queue.get_timestamp_period()) * 1.0e-6;
        ticks
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| pair[1].saturating_sub(pair[0]) as f64 * period)
            .collect::<Vec<_>>()
    } else {
        vec![cpu_ms; FRAMES]
    };
    times.sort_unstable_by(f64::total_cmp);
    Ok(Timing {
        min: times[0],
        median: times[FRAMES / 2],
        p90: times[FRAMES * 9 / 10],
    })
}
fn image_difference(a: &[u8], b: &[u8]) -> f64 {
    let total = a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0.iter())
        .map(|(left, right)| {
            left[..3]
                .iter()
                .zip(&right[..3])
                .map(|(l, r)| f64::from(l.abs_diff(*r)))
                .sum::<f64>()
        })
        .sum::<f64>();
    total / (f64::from(W) * f64::from(H) * 3.0)
}
fn green_pixels(data: &[u8]) -> usize {
    data.as_chunks::<4>()
        .0
        .iter()
        .filter(|rgba| rgba[1] > rgba[0] && rgba[1] > rgba[2])
        .count()
}
/// The same source trees in two different LODs compete for every pixel;
/// complementary hashes reduce coverage pops without remapping stable IDs.
#[expect(
    clippy::cast_precision_loss,
    reason = "green-pixel counts are bounded by W*H=576000"
)]
fn capture_transitions(
    gpu: &Gpu,
    output: &Path,
    camera: Camera,
    trees: u32,
    pair: (u32, u32),
    label: &str,
) -> AnyResult<()> {
    let mut transition = String::from("fade,green_pixels,coverage_ratio_to_near\n");
    let mut original_count = None;
    for fade in [0.0_f32, 0.20, 0.40, 0.60, 0.80, 1.0] {
        let near = DrawCase::new(gpu, pair.0, trees, fade, 1, camera);
        let far = DrawCase::new(gpu, pair.1, trees, fade, 2, camera);
        let path = output.join(format!("{label}-{fade:.2}.ppm"));
        let rgba = capture(gpu, &[(&near, 0), (&far, 0)], &path)?;
        let count = green_pixels(&rgba);
        let original = *original_count.get_or_insert(count);
        if original == 0 {
            return Err("no visible foliage in LOD transition".into());
        }
        let relative = count as f64 / original as f64;
        println!(
            "FOLIAGE_TRANSITION scenario={label} fade={fade:.2} green_pixels={count} ratio_to_near={relative:.5}"
        );
        writeln!(&mut transition, "{fade:.2},{count},{relative:.6}")?;
    }
    fs::write(output.join(format!("{label}.csv")), transition)?;
    Ok(())
}

/// Compare both ALU-heavy procedural and bandwidth-heavy shared geometry
/// at actual intermediate LOD projected sizes, not just a dense ground view.
fn benchmark_aerial(gpu: &Gpu, output: &Path) -> AnyResult<()> {
    const TREES: u32 = 4096;
    let mut csv = String::from(
        "mode,trees,clumps,triangles,geometry_bytes,patch_bytes,min_ms,p50_ms,p90_ms\n",
    );
    for clumps in [8, 24, 64] {
        let case = DrawCase::new(gpu, clumps, TREES, 0.0, 0, Camera::Aerial);
        for (mode, name) in MODES.iter().enumerate() {
            let ms = measure(gpu, &case, mode)?;
            let bytes = match mode {
                0 | 1 => 0,
                2 => case.module_bytes,
                _ => case.geometry_bytes,
            };
            println!(
                "FOLIAGE_AERIAL mode={name} trees={TREES} clumps={clumps} triangles={} gpu_ms_p50={:.5} p90={:.5} bytes={bytes}",
                (clumps * 4 + 2) * TREES,
                ms.median,
                ms.p90
            );
            writeln!(
                &mut csv,
                "{name},{TREES},{clumps},{},{bytes},{},{:.6},{:.6},{:.6}",
                (clumps * 4 + 2) * TREES,
                case.module_bytes,
                ms.min,
                ms.median,
                ms.p90
            )?;
        }
    }
    fs::write(output.join("aerial-timings.csv"), csv)?;
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "population counts are bounded by 4096; exact as f64"
)]
fn population_summary(trees: &[Tree]) {
    let mut counts = [0_usize; 3];
    let mut same_neighbor = 0_usize;
    let mut neighbors = 0_usize;
    for (i, tree) in trees.iter().enumerate() {
        counts[tree.ids[2] as usize] += 1;
        for other in [i + 1, i + 88] {
            if other >= trees.len() || other == i + 1 && i % 88 == 87 {
                continue;
            }
            neighbors += 1;
            same_neighbor += usize::from(tree.ids[2] == trees[other].ids[2]);
        }
    }
    let expected = counts
        .iter()
        .map(|count| (*count as f64 / trees.len() as f64).powi(2))
        .sum::<f64>();
    println!(
        "FOLIAGE_BIOME trees={} broadleaf={} conifer={} shrub={} adjacent_same={:.4} random_expected={expected:.4}",
        trees.len(),
        counts[0],
        counts[1],
        counts[2],
        same_neighbor as f64 / neighbors as f64
    );
}

/// Rotate method order between rounds to limit GPU clock/thermal ordering
/// artifacts. First two rounds stabilize the driver; five later runs feed
/// median-of-medians. All geometry, pixel shaders and instances are shared.
fn benchmark_aerial_interleaved(gpu: &Gpu, output: &Path) -> AnyResult<()> {
    let mut csv = String::from("clumps,round,mode,min_ms,p50_ms,p90_ms\n");
    for clumps in [8_u32, 24, 64] {
        let case = DrawCase::new(gpu, clumps, 4096, 0.0, 0, Camera::Aerial);
        let mut times: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for round in 0..7_usize {
            for order in 0..4 {
                let mode = (order + round * 3) % 4;
                let timing = measure(gpu, &case, mode)?;
                writeln!(
                    &mut csv,
                    "{clumps},{round},{},{:.6},{:.6},{:.6}",
                    MODES[mode], timing.min, timing.median, timing.p90
                )?;
                if round >= 2 {
                    times[mode].push(timing.median);
                }
            }
        }
        for (mode, values) in times.iter_mut().enumerate() {
            values.sort_unstable_by(f64::total_cmp);
            let median = values[values.len() / 2];
            let min = values[0];
            let max = values[values.len() - 1];
            println!(
                "FOLIAGE_INTERLEAVED clumps={clumps} mode={} p50_of_runs_ms={median:.6} min_round_ms={min:.6} max_round_ms={max:.6}",
                MODES[mode]
            );
        }
    }
    fs::write(output.join("aerial-interleaved.csv"), csv)?;
    Ok(())
}

fn run_aerial(gpu: &Gpu, output: &Path) -> AnyResult<()> {
    let trees = make_trees(4096, Camera::Aerial);
    population_summary(&trees);
    benchmark_aerial(gpu, output)?;
    let case = DrawCase::new(gpu, 24, 4096, 0.0, 0, Camera::Aerial);
    let mut images = Vec::new();
    for (mode, name) in MODES.iter().enumerate() {
        let path = output.join(format!("aerial-{name}.ppm"));
        let rgba = capture(gpu, &[(&case, mode)], &path)?;
        println!(
            "FOLIAGE_AERIAL_IMAGE mode={name} green_pixels={} file={}",
            green_pixels(&rgba),
            path.display()
        );
        images.push(rgba);
    }
    println!(
        "FOLIAGE_AERIAL_MATCH generated_vs_modules_mae={:.5} generated_vs_expanded_mae={:.5} unique_vs_shared_mae={:.5}",
        image_difference(&images[1], &images[2]),
        image_difference(&images[1], &images[3]),
        image_difference(&images[0], &images[2])
    );
    capture_transitions(gpu, output, Camera::Aerial, 4096, (24, 8), "aerial-24-to-8")?;
    capture_transitions(
        gpu,
        output,
        Camera::Aerial,
        4096,
        (64, 24),
        "aerial-64-to-24",
    )?;
    Ok(())
}

fn run() -> AnyResult<()> {
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: foliage_lab OUTPUT_DIRECTORY")?;
    let output = Path::new(&output);
    fs::create_dir_all(output)?;
    let gpu = open_gpu()?;
    println!(
        "FOLIAGE_GPU name={:?} gpu_timestamps={}",
        gpu.name, gpu.timestamp_supported
    );
    if std::env::args().nth(2).as_deref() == Some("--repeat-aerial") {
        return benchmark_aerial_interleaved(&gpu, output);
    }
    let mut report = format!(
        "gpu={}\nmode,trees,clumps,triangles,geometry_bytes,patch_bytes,min_ms,p50_ms,p90_ms\n",
        gpu.name
    );
    for trees in [1024, 4096] {
        for clumps in [8, 24, 64] {
            let case = DrawCase::new(&gpu, clumps, trees, 0.0, 0, Camera::Ground);
            for (mode, name) in MODES.iter().enumerate() {
                let sample = measure(&gpu, &case, mode)?;
                let bytes = match mode {
                    0 | 1 => 0,
                    2 => case.module_bytes,
                    _ => case.geometry_bytes,
                };
                println!(
                    "FOLIAGE_BENCH mode={name} trees={trees} clumps={clumps} triangles={} gpu_ms_p50={:.5} p90={:.5} geometry_bytes={bytes}",
                    (clumps * 4 + 2) * trees,
                    sample.median,
                    sample.p90
                );
                writeln!(
                    &mut report,
                    "{name},{trees},{clumps},{},{bytes},{},{:.6},{:.6},{:.6}",
                    (clumps * 4 + 2) * trees,
                    case.module_bytes,
                    sample.min,
                    sample.median,
                    sample.p90
                )?;
            }
        }
    }
    fs::write(output.join("timings.csv"), &report)?;
    let case = DrawCase::new(&gpu, 24, 2048, 0.0, 0, Camera::Ground);
    let mut pictures = Vec::new();
    for (mode, name) in MODES.iter().enumerate() {
        let path = output.join(format!("{name}.ppm"));
        let rgba = capture(&gpu, &[(&case, mode)], &path)?;
        println!(
            "FOLIAGE_IMAGE mode={name} green_pixels={} file={}",
            green_pixels(&rgba),
            path.display()
        );
        pictures.push(rgba);
    }
    println!(
        "FOLIAGE_MATCH limited_vs_modules_rgb_mae={:.4} limited_vs_baked_rgb_mae={:.4} unique_vs_limited_rgb_mae={:.4}",
        image_difference(&pictures[1], &pictures[2]),
        image_difference(&pictures[1], &pictures[3]),
        image_difference(&pictures[0], &pictures[1])
    );
    // Two LODs are rendered together with complementary stochastic alpha.
    capture_transitions(
        &gpu,
        output,
        Camera::Ground,
        2048,
        (24, 8),
        "ground-24-to-8",
    )?;
    run_aerial(&gpu, output)?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("FOLIAGE_ERROR: {error}");
        std::process::exit(1);
    }
}
